"""Torch-free OpenAI CLIP text encoder for the Jittor port.

InterDist uses only CLIP's text path.  This module mirrors the small public
surface used by the original OpenAI ``clip`` package: ``tokenize()``, ``load()``,
``model.convert_weights()``, and the returned module's text attributes.
"""

import gzip
import html
import math
import os
import pickle
import types
from functools import lru_cache
from pathlib import Path

import ftfy
import numpy as np
import regex as re

from utils.jittor_compat import jt, nn

_CONTEXT_LENGTH = 77
_VOCAB_SIZE = 49408
_SOT_TOKEN = 49406
_EOT_TOKEN = 49407
_CLIP_VERSION = "ViT-L/14@336px"
_DEFAULT_TEXT_CKPT = "clip_vit_l_14_336px_text.pth"
_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer:
    def __init__(self, bpe_path=_VOCAB_PATH):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


@lru_cache()
def _get_tokenizer():
    return SimpleTokenizer()


def tokenize(texts, context_length=_CONTEXT_LENGTH, truncate=False):
    if isinstance(texts, str):
        texts = [texts]
    tokenizer = _get_tokenizer()
    tokens = []
    for text in texts:
        ids = [_SOT_TOKEN] + tokenizer.encode(text) + [_EOT_TOKEN]
        if len(ids) > context_length:
            if not truncate:
                raise RuntimeError(f"Input text is too long for context length {context_length}")
            ids = ids[:context_length]
            ids[-1] = _EOT_TOKEN
        ids = ids + [0] * (context_length - len(ids))
        tokens.append(ids)
    return jt.array(tokens, dtype="int32")


class QuickGELU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.in_proj_weight = jt.zeros((width * 3, width))
        self.in_proj_bias = jt.zeros((width * 3,))
        self.out_proj = nn.Linear(width, width)

    def execute(self, x, attn_mask=None):
        seq_len, batch_size, width = x.shape
        flat = x.reshape(seq_len * batch_size, width)
        qkv = jt.matmul(flat, self.in_proj_weight.transpose(0, 1)) + self.in_proj_bias
        qkv = qkv.reshape(seq_len, batch_size, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(1, 2, 0, 3)
        k = qkv[:, :, 1].permute(1, 2, 0, 3)
        v = qkv[:, :, 2].permute(1, 2, 0, 3)
        scores = jt.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_mask is not None:
            scores = scores + attn_mask.reshape(1, 1, seq_len, seq_len)
        weights = nn.softmax(scores, dim=-1)
        out = jt.matmul(weights, v)
        out = out.permute(2, 0, 1, 3).reshape(seq_len * batch_size, width)
        out = self.out_proj(out).reshape(seq_len, batch_size, width)
        return out


class CLIPMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = nn.Linear(width, width * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(width * 4, width)

    def execute(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads, attn_mask=None):
        super().__init__()
        self.attn = CLIPAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = CLIPMLP(width)
        self.ln_2 = nn.LayerNorm(width)
        self.attn_mask = attn_mask

    def attention(self, x):
        return self.attn(x, attn_mask=self.attn_mask)

    def execute(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTransformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def execute(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class CLIPTextModel(nn.Module):
    def __init__(
        self,
        width=768,
        layers=12,
        heads=12,
        context_length=_CONTEXT_LENGTH,
        vocab_size=_VOCAB_SIZE,
        output_dim=768,
    ):
        super().__init__()
        self.dtype = "float32"
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = jt.zeros((context_length, width))
        self.transformer = CLIPTransformer(width, layers, heads, self.build_attention_mask(context_length))
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = jt.zeros((width, output_dim))
        self.logit_scale = jt.array(np.log(1 / 0.07), dtype="float32")

    @staticmethod
    def build_attention_mask(context_length):
        mask = np.empty((context_length, context_length), dtype=np.float32)
        mask.fill(float("-inf"))
        mask = np.triu(mask, 1)
        return jt.array(mask, dtype="float32")

    def encode_text(self, text):
        x = self.token_embedding(text).float32()
        x = x + self.positional_embedding.float32()
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = self.ln_final(x).permute(1, 0, 2).float32()
        eot = [int(v) for v in text.argmax(dim=-1).numpy().tolist()]
        pooled = jt.cat([x[i, idx].unsqueeze(0) for i, idx in enumerate(eot)], dim=0)
        return jt.matmul(pooled, self.text_projection.float32())


def _default_checkpoint_path():
    return Path(__file__).resolve().parents[1] / "checkpoints" / _DEFAULT_TEXT_CKPT


def _resolve_checkpoint_path(name):
    env_path = os.environ.get("INTERDIST_CLIP_TEXT_PTH")
    if env_path:
        return Path(env_path)
    if name not in (None, _CLIP_VERSION):
        path = Path(str(name))
        if path.exists():
            return path
    return _default_checkpoint_path()


def _load_pickle_checkpoint(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj.get("metadata", {}), obj["state_dict"]
    return {}, obj


def _array(value):
    return jt.array(np.asarray(value, dtype=np.float32))


def _assign(var, value):
    var.assign(_array(value))


def _load_text_state(model, state_dict):
    required = [
        "token_embedding.weight",
        "positional_embedding",
        "text_projection",
        "ln_final.weight",
        "ln_final.bias",
    ]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(f"Converted CLIP text checkpoint is missing keys: {missing}")

    _assign(model.token_embedding.weight, state_dict["token_embedding.weight"])
    _assign(model.positional_embedding, state_dict["positional_embedding"])
    _assign(model.text_projection, state_dict["text_projection"])
    if "logit_scale" in state_dict:
        _assign(model.logit_scale, state_dict["logit_scale"])
    _assign(model.ln_final.weight, state_dict["ln_final.weight"])
    _assign(model.ln_final.bias, state_dict["ln_final.bias"])

    for idx, block in enumerate(model.transformer.resblocks):
        prefix = f"transformer.resblocks.{idx}."
        for name, target in [
            ("attn.in_proj_weight", block.attn.in_proj_weight),
            ("attn.in_proj_bias", block.attn.in_proj_bias),
            ("attn.out_proj.weight", block.attn.out_proj.weight),
            ("attn.out_proj.bias", block.attn.out_proj.bias),
            ("ln_1.weight", block.ln_1.weight),
            ("ln_1.bias", block.ln_1.bias),
            ("mlp.c_fc.weight", block.mlp.c_fc.weight),
            ("mlp.c_fc.bias", block.mlp.c_fc.bias),
            ("mlp.c_proj.weight", block.mlp.c_proj.weight),
            ("mlp.c_proj.bias", block.mlp.c_proj.bias),
            ("ln_2.weight", block.ln_2.weight),
            ("ln_2.bias", block.ln_2.bias),
        ]:
            key = prefix + name
            if key not in state_dict:
                raise RuntimeError(f"Converted CLIP text checkpoint is missing key: {key}")
            _assign(target, state_dict[key])


def load(name=_CLIP_VERSION, device="cpu", jit=False, warn=True):
    if name not in (None, _CLIP_VERSION):
        path = Path(str(name))
        if not path.exists():
            raise ValueError(f"Only {_CLIP_VERSION!r} is supported by the Jittor text port, got {name!r}")
    ckpt_path = _resolve_checkpoint_path(name)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Converted CLIP text weights not found: {ckpt_path}. "
            "Run `python tools/convert_clip_text_checkpoint.py --src /path/to/ViT-L-14-336px.pt "
            f"--dst {ckpt_path}` or set INTERDIST_CLIP_TEXT_PTH."
        )

    metadata, state_dict = _load_pickle_checkpoint(ckpt_path)
    model = CLIPTextModel(
        width=int(metadata.get("width", 768)),
        layers=int(metadata.get("layers", 12)),
        heads=int(metadata.get("heads", 12)),
        context_length=int(metadata.get("context_length", _CONTEXT_LENGTH)),
        vocab_size=int(metadata.get("vocab_size", _VOCAB_SIZE)),
        output_dim=int(metadata.get("output_dim", 768)),
    )
    _load_text_state(model, state_dict)
    model.eval()
    if str(device) != "cpu":
        model.to(device)
    return model, None


model = types.SimpleNamespace(convert_weights=lambda clip_model: clip_model)
