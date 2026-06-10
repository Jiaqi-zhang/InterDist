"""Small PyTorch-compatibility layer backed by Jittor.

The project was originally written against PyTorch.  This module centralizes
Jittor bootstrapping plus the few PyTorch-style aliases that keep the migrated
code readable without depending on the real torch package.
"""

import math
import os
import pickle
import copy
from contextlib import contextmanager

# Jittor writes config/cache files under HOME before honoring JITTOR_HOME.  In
# restricted containers HOME can be read-only, so move it to a writable place.
_home = os.path.expanduser("~")
if not os.access(_home, os.W_OK):
    os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("JITTOR_HOME", os.path.abspath(os.environ.get("INTERDIST_JITTOR_HOME", ".jittor_cache")))

import numpy as np
import jittor as jt
from jittor import nn, optim
import jittor.lr_scheduler as lr_scheduler
import jittor.linalg as linalg

try:
    import jittor.misc  # noqa: F401 - registers PyTorch-like Var helpers.
except Exception:
    pass

_orig = {}
_INITIALIZED = False
_ORIG_MODULE_TRAIN = None
_ORIG_MODULE_EVAL = None
_ORIG_VAR_TRANSPOSE = None
_ORIG_VAR_EXPAND = None


def _remember(name):
    if name not in _orig and hasattr(jt, name):
        _orig[name] = getattr(jt, name)


def _dtype(dtype, default=None):
    if dtype is None:
        return default
    if dtype is bool:
        return "bool"
    if dtype is int:
        return "int32"
    if dtype is float:
        return "float32"
    text = str(dtype)
    if "float64" in text or text == "double":
        return "float64"
    if "float16" in text or text == "half":
        return "float16"
    if "float" in text:
        return "float32"
    if "int64" in text or "long" in text:
        # Jittor 1.x commonly maps long to int32 for indexing.
        return "int32"
    if "int32" in text or text == "int":
        return "int32"
    if "bool" in text:
        return "bool"
    return dtype


def _set_grad(var, requires_grad):
    if hasattr(var, "start_grad") and hasattr(var, "stop_grad"):
        if requires_grad:
            var.start_grad()
        else:
            var.stop_grad()
    return var


def tensor(data, dtype=None, device=None, requires_grad=False):
    var = jt.array(data, dtype=_dtype(dtype))
    return _set_grad(var, requires_grad)


def from_numpy(arr):
    return jt.array(arr)


def as_tensor(data, dtype=None, device=None):
    return tensor(data, dtype=dtype, device=device)


def _wrap_creator(name, default_dtype=None):
    _remember(name)
    fn = _orig[name]

    def wrapped(*args, dtype=None, device=None, requires_grad=False, **kwargs):
        args = tuple(args)
        kwargs.pop("device", None)
        kwargs.pop("requires_grad", None)
        if dtype is None and "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
        # Older Jittor internals call creators as jt.empty(shape, dtype).
        if dtype is None and len(args) >= 2 and str(args[1]).lower() in {
            "float16", "float32", "float64", "int8", "uint8", "int16", "int32", "int64", "bool"
        }:
            dtype = args[1]
            args = (args[0],) + args[2:]
        if name in {"zeros", "ones", "empty", "rand", "randn"} and len(args) > 1 and all(isinstance(v, (int, np.integer)) for v in args):
            args = (tuple(int(v) for v in args),)
        dtype = _dtype(dtype, default_dtype)
        if dtype is not None:
            try:
                var = fn(*args, dtype=dtype, **kwargs)
            except TypeError:
                var = fn(*args, **kwargs).cast(dtype)
        else:
            var = fn(*args, **kwargs)
        return _set_grad(var, requires_grad)

    return wrapped


def _zeros_like(x, dtype=None, device=None, requires_grad=False):
    var = _orig["zeros_like"](x)
    if dtype is not None:
        var = var.cast(_dtype(dtype))
    return _set_grad(var, requires_grad)


def _ones_like(x, dtype=None, device=None, requires_grad=False):
    var = _orig["ones_like"](x)
    if dtype is not None:
        var = var.cast(_dtype(dtype))
    return _set_grad(var, requires_grad)


def _full_like(x, fill_value, dtype=None, device=None, requires_grad=False):
    dtype = _dtype(dtype) or x.dtype
    var = jt.full(x.shape, fill_value, dtype=dtype)
    return _set_grad(var, requires_grad)


def _empty_like(x, dtype=None, device=None, requires_grad=False):
    dtype = _dtype(dtype) or x.dtype
    var = jt.empty(x.shape, dtype=dtype)
    return _set_grad(var, requires_grad)


def _randint_like(x, high, low=0, dtype=None, device=None):
    return jt.randint(low, high, x.shape, dtype=_dtype(dtype, "int32"))


def _arange(*args, dtype=None, device=None, **kwargs):
    _remember("arange")
    kwargs.pop("device", None)
    if dtype is not None:
        kwargs["dtype"] = _dtype(dtype)
    return _orig["arange"](*args, **kwargs)


def _eye(n, m=None, dtype=None, device=None, requires_grad=False):
    if m is None:
        m = n
    try:
        var = jt.init.eye((n, m), dtype=_dtype(dtype, "float32"))
    except Exception:
        var = jt.array(np.eye(n, m), dtype=_dtype(dtype, "float32"))
    return _set_grad(var, requires_grad)


def _linspace(start, end, steps, dtype=None, device=None):
    _remember("linspace")
    try:
        out = _orig["linspace"](start, end, steps)
    except Exception:
        out = jt.array(np.linspace(float(start), float(end), int(steps), dtype=np.float32))
    if dtype is not None:
        out = out.cast(_dtype(dtype))
    return out


def _max(x, y=None, dim=None, keepdim=False, keepdims=False):
    if dim is None and isinstance(y, int):
        dim = y
        y = None
    if y is not None:
        return x.maximum(y) if isinstance(x, jt.Var) else y.maximum(x)
    if dim is None:
        return x.max()
    if keepdims and not keepdim:
        return x.max(dim, True)
    return x.max(dim, keepdim or keepdims), x.argmax(dim)


def _min(x, y=None, dim=None, keepdim=False, keepdims=False):
    if dim is None and isinstance(y, int):
        dim = y
        y = None
    if y is not None:
        return x.minimum(y) if isinstance(x, jt.Var) else y.minimum(x)
    if dim is None:
        return x.min()
    if keepdims and not keepdim:
        return x.min(dim, True)
    return x.min(dim, keepdim or keepdims), x.argmin(dim)


def _argmax(x, dim=None, keepdim=False):
    res = _orig["argmax"](x, dim) if dim is not None else _orig["argmax"](x)
    idx = res[0] if isinstance(res, (tuple, list)) else res
    return idx.unsqueeze(dim) if keepdim and dim is not None else idx


def _argmin(x, dim=None, keepdim=False):
    res = _orig["argmin"](x, dim) if dim is not None else _orig["argmin"](x)
    idx = res[0] if isinstance(res, (tuple, list)) else res
    return idx.unsqueeze(dim) if keepdim and dim is not None else idx


def _argsort(x, dim=-1, descending=False):
    try:
        res = _orig["argsort"](x, dim, descending=descending)
    except TypeError:
        res = _orig["argsort"](x, dim)
        if descending:
            res = (res[0].flip(dim), res[1].flip(dim)) if isinstance(res, tuple) else res.flip(dim)
    return res[0] if isinstance(res, (tuple, list)) else res


def _sum(x, dim=None, dims=None, keepdim=False, keepdims=False):
    if dim is None:
        dim = dims
    if dim is None:
        return _orig["sum"](x)
    return _orig["sum"](x, dim, keepdim or keepdims)


def _mean(x, dim=None, dims=None, keepdim=False, keepdims=False):
    if dim is None:
        dim = dims
    if dim is None:
        return _orig["mean"](x)
    return _orig["mean"](x, dim, keepdim or keepdims)


def _norm(x, p=2, dim=None, keepdim=False, keepdims=False):
    keep = keepdim or keepdims
    if p == 2:
        return _sum(x * x, dim=dim, keepdim=keep).sqrt()
    if p == 1:
        return _sum(jt.abs(x), dim=dim, keepdim=keep)
    return _sum(jt.abs(x) ** p, dim=dim, keepdim=keep) ** (1.0 / p)


class _TopKResult:
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def __iter__(self):
        yield self.values
        yield self.indices

    def __getitem__(self, index):
        return (self.values, self.indices)[index]


def _topk(x, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    values, indices = _orig["topk"](x, k, dim=dim, largest=largest, sorted=sorted)
    return _TopKResult(values, indices)


def _load(path, *args, **kwargs):
    kwargs.pop("map_location", None)
    try:
        return _orig["load"](path)
    except RuntimeError as exc:
        msg = str(exc)
        if "Invalid magic number" not in msg and "corrupt file" not in msg and "pytorch need to be installed" not in msg:
            raise
        with open(path, "rb") as f:
            return pickle.load(f)


def _save(obj, path, *args, **kwargs):
    return _orig["save"](obj, path)


def _no_grad_class():
    base = jt.no_grad

    class NoGrad(base):
        def __call__(self, func):
            def wrapper(*args, **kwargs):
                with base():
                    return func(*args, **kwargs)
            wrapper.__name__ = getattr(func, "__name__", "wrapped")
            wrapper.__doc__ = getattr(func, "__doc__", None)
            return wrapper

    return NoGrad


def _module_to(self, device=None, *args, **kwargs):
    if device is not None and "cuda" in str(device):
        jt.flags.use_cuda = 1
    elif device is not None and "cpu" in str(device):
        jt.flags.use_cuda = 0
    return self


def _module_cpu(self):
    jt.flags.use_cuda = 0
    return self


def _module_cuda(self, device=None):
    return self


def _iter_child_modules(module):
    for value in getattr(module, "__dict__", {}).values():
        if isinstance(value, nn.Module):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, nn.Module):
                    yield item
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, nn.Module):
                    yield item


def _set_training_recursive(module, mode, seen=None):
    if seen is None:
        seen = set()
    if id(module) in seen:
        return
    seen.add(id(module))
    try:
        setattr(module, "training", bool(mode))
    except Exception:
        pass
    for child in _iter_child_modules(module):
        _set_training_recursive(child, mode, seen)


def _module_train(self, mode=True):
    try:
        if mode:
            _ORIG_MODULE_TRAIN(self)
        elif _ORIG_MODULE_EVAL is not None:
            _ORIG_MODULE_EVAL(self)
    except Exception:
        pass
    _set_training_recursive(self, mode)
    return self


def _module_eval(self):
    return _module_train(self, False)


def _module_register_parameter(self, name, value):
    setattr(self, name, value)
    return value


def _module_register_buffer(self, name, value, persistent=True):
    setattr(self, name, value)
    return value


def _convert_state_dict(params):
    if not hasattr(params, "items"):
        return params
    converted = dict(params)
    suffixes = {
        "in_proj_weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
        "in_proj_bias": ("q_proj.bias", "k_proj.bias", "v_proj.bias"),
    }
    for key in list(converted.keys()):
        for suffix, targets in suffixes.items():
            needle = "." + suffix
            if not key.endswith(needle):
                continue
            value = converted.pop(key)
            size0 = value.shape[0]
            if size0 % 3 != 0:
                converted[key] = value
                continue
            base = key[:-len(suffix)]
            split = size0 // 3
            converted[base + targets[0]] = value[:split]
            converted[base + targets[1]] = value[split:2 * split]
            converted[base + targets[2]] = value[2 * split:]
    return converted


def _load_state_dict(self, params, strict=True):
    converted = _convert_state_dict(params)
    model_keys = set(self.state_dict().keys())
    loaded_keys = set(converted.keys())
    missing = sorted(model_keys - loaded_keys)
    unexpected = sorted(loaded_keys - model_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Missing keys: {missing}; unexpected keys: {unexpected}")
    self.load_parameters(converted)
    return missing, unexpected


def _param_requires_grad_get(self):
    try:
        return not self.is_stop_grad()
    except Exception:
        return True


def _param_requires_grad_set(self, value):
    _set_grad(self, bool(value))


def _var_device(self):
    return "cuda" if getattr(jt.flags, "use_cuda", 0) else "cpu"


def _item(self):
    arr = self.numpy()
    return arr.item() if hasattr(arr, "item") else arr


def _tolist(self):
    return self.numpy().tolist()


def _masked_select(x, mask):
    return x[mask]


def _masked_fill(x, mask, value):
    if tuple(mask.shape) != tuple(x.shape):
        mask = mask.broadcast(x.shape)
    fill = jt.full(x.shape, value, dtype=x.dtype)
    mask = mask.float32()
    return x * (1.0 - mask) + fill * mask


def _where(cond, x=None, y=None):
    if x is None and y is None:
        return _orig["where"](cond)
    shape = tuple(cond.shape)
    if isinstance(x, jt.Var):
        shape = tuple(x.shape)
        dtype = x.dtype
    elif isinstance(y, jt.Var):
        shape = tuple(y.shape)
        dtype = y.dtype
    elif isinstance(x, float) or isinstance(y, float):
        dtype = "float32"
    else:
        dtype = "int32"
    if tuple(cond.shape) != shape:
        cond = cond.expand(shape)
    if not isinstance(x, jt.Var):
        x = jt.full(shape, x, dtype=dtype)
    elif tuple(x.shape) != shape:
        x = x.expand(shape)
    if not isinstance(y, jt.Var):
        y = jt.full(shape, y, dtype=dtype)
    elif tuple(y.shape) != shape:
        y = y.expand(shape)
    cond = cond.cast(dtype)
    return cond * x + (1 - cond) * y


def _scatter_(x, dim, index, src, reduce=None):
    if not isinstance(src, jt.Var):
        src = jt.full(index.shape, src, dtype=x.dtype)
    if reduce is None:
        return x.assign(x.scatter(dim, index, src))
    return x.assign(x.scatter(dim, index, src, reduce))


def _is_tensor(x):
    return isinstance(x, jt.Var)


def _bernoulli(input, p=None):
    if p is None:
        prob = input
    else:
        prob = jt.ones_like(input).float32() * p
    out = jt.rand(prob.shape) < prob
    return out if "bool" in str(input.dtype) else out.float32()


def _clamp(x, min=None, max=None):
    if min is not None:
        x = x.maximum(min)
    if max is not None:
        x = x.minimum(max)
    return x


def _clip(x, min=None, max=None):
    return _clamp(x, min, max)


def _maximum(x, y):
    return x.maximum(y) if isinstance(x, jt.Var) else jt.maximum(x, y)


def _minimum(x, y):
    return x.minimum(y) if isinstance(x, jt.Var) else jt.minimum(x, y)


def _logical_not(x):
    return x == 0


def _var_transpose(self, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        axes = list(dims[0])
    elif len(dims) == 2:
        ndim = len(self.shape)
        dim0 = int(dims[0]) % ndim
        dim1 = int(dims[1]) % ndim
        axes = list(range(ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    elif len(dims) == 0:
        axes = list(reversed(range(len(self.shape))))
    else:
        axes = list(dims)
    return _ORIG_VAR_TRANSPOSE(self, axes)


def _var_expand(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    new_shape = []
    for idx, size in enumerate(shape):
        if int(size) == -1:
            new_shape.append(self.shape[idx])
        else:
            new_shape.append(int(size))
    return _ORIG_VAR_EXPAND(self, tuple(new_shape))


def _einsum(pattern, operands, *extra_operands):
    if extra_operands:
        operands = (operands,) + extra_operands
    arrays = [item.numpy() if isinstance(item, jt.Var) else item for item in operands]
    return jt.array(np.einsum(pattern, *arrays))


class _CudaCompat:
    @staticmethod
    def set_device(device):
        if int(device) >= 0:
            try:
                jt.flags.use_cuda = 1
            except RuntimeError:
                jt.flags.use_cuda = 0

    @staticmethod
    def is_available():
        return bool(getattr(jt.flags, "use_cuda", 0))

    @staticmethod
    def manual_seed_all(seed):
        jt.set_global_seed(int(seed))

    @staticmethod
    def empty_cache():
        try:
            jt.gc()
        except Exception:
            pass


class _AutogradCompat:
    @staticmethod
    def set_detect_anomaly(*args, **kwargs):
        return None


class Categorical:
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        from ops.categorical import categorical_sample

        return categorical_sample(self.probs)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x):
        b, t, c = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def execute(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        transposed = False
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
            transposed = True
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))
        scores = jt.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            if attn_mask.dtype == "bool":
                scores = scores.masked_fill(attn_mask, -1e9)
            else:
                scores = scores + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        attn = nn.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = jt.matmul(attn, v).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.embed_dim)
        out = self.out_proj(out)
        if transposed:
            out = out.transpose(0, 1)
        if need_weights:
            return out, attn.mean(dim=1)
        return out, None


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def execute(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        attn, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def execute(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        attn, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn))
        attn, _ = self.multihead_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False)
        x = self.norm2(x + self.dropout2(attn))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(ff))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def execute(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def execute(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


def _reduce_loss(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


class L1Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def execute(self, input, target):
        return _reduce_loss(jt.abs(input - target), self.reduction)


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def execute(self, input, target):
        return _reduce_loss((input - target) ** 2, self.reduction)


class SmoothL1Loss(nn.Module):
    def __init__(self, reduction="mean", beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def execute(self, input, target):
        diff = jt.abs(input - target)
        loss = jt.where(diff < self.beta, 0.5 * diff * diff / self.beta, diff - 0.5 * self.beta)
        return _reduce_loss(loss, self.reduction)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = jt.randn((num_embeddings, embedding_dim)) * 0.02
        if padding_idx is not None:
            self.weight[int(padding_idx)] = 0

    def execute(self, x):
        flat = x.reshape(-1).int32()
        out = self.weight[flat]
        return out.reshape(tuple(x.shape) + (self.embedding_dim,))


def _embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    flat = input.reshape(-1).int32()
    out = weight[flat]
    return out.reshape(tuple(input.shape) + (weight.shape[-1],))


def _sigmoid(x):
    return jt.sigmoid(x) if hasattr(jt, "sigmoid") else 1.0 / (1.0 + jt.exp(-x))


def _relu(x):
    return jt.maximum(x, 0)


def _gelu(x):
    return 0.5 * x * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _silu(x):
    return x * _sigmoid(x)


def _softplus(x):
    return jt.log(jt.exp(x) + 1.0)


def _missing_function(name):
    def inner(*args, **kwargs):
        raise NotImplementedError(f"jittor.nn.{name} is not available in this runtime")
    return inner


_NN_RELU = getattr(nn, "relu", _relu)
_NN_GELU = getattr(nn, "gelu", _gelu)
_NN_SILU = getattr(nn, "silu", _silu)
_NN_SOFTPLUS = getattr(nn, "softplus", _softplus)


class _Functional:
    relu = staticmethod(_NN_RELU)
    gelu = staticmethod(_NN_GELU)
    silu = staticmethod(_NN_SILU)
    softplus = staticmethod(_NN_SOFTPLUS)
    softmax = staticmethod(getattr(nn, "softmax", _missing_function("softmax")))
    log_softmax = staticmethod(getattr(nn, "log_softmax", _missing_function("log_softmax")))
    mse_loss = staticmethod(getattr(nn, "mse_loss", lambda input, target, reduction="mean": _reduce_loss((input - target) ** 2, reduction)))
    l1_loss = staticmethod(getattr(nn, "l1_loss", lambda input, target, reduction="mean": _reduce_loss(jt.abs(input - target), reduction)))
    smooth_l1_loss = staticmethod(getattr(nn, "smooth_l1_loss", lambda input, target, reduction="mean": SmoothL1Loss(reduction)(input, target)))
    one_hot = staticmethod(getattr(nn, "one_hot", _missing_function("one_hot")))
    embedding = staticmethod(getattr(nn, "embedding", _embedding))
    pad = staticmethod(getattr(nn, "pad", _missing_function("pad")))
    interpolate = staticmethod(getattr(nn, "interpolate", _missing_function("interpolate")))

    @staticmethod
    def cross_entropy(input, target, ignore_index=None, reduction="mean"):
        if len(input.shape) > 2:
            dims = [0] + list(range(2, len(input.shape))) + [1]
            input = input.permute(*dims).reshape(-1, input.shape[1])
            target = target.reshape(-1)
        if ignore_index is None:
            ignore_index = -100
        mask = target != ignore_index
        safe_target = target.masked_fill(~mask, 0).int32()
        log_prob = nn.log_softmax(input, dim=1)
        losses = -log_prob.gather(1, safe_target.reshape(-1, 1)).squeeze(1)
        losses = losses.masked_select(mask)
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        return losses.mean()

    @staticmethod
    def normalize(x, p=2, dim=1, eps=1e-12):
        return x / jt.norm(x, p=p, dim=dim, keepdim=True).maximum(eps)

    @staticmethod
    def pairwise_distance(x1, x2, p=2, eps=1e-6, keepdim=False):
        return jt.norm(x1 - x2 + eps, p=p, dim=-1, keepdim=keepdim)

    @staticmethod
    def glu(x, dim=-1):
        a, b = x.chunk(2, dim=dim)
        return a * jt.sigmoid(b)


F = _Functional()
Tensor = jt.Var


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    grads = []
    for p in parameters:
        try:
            if p.opt_grad() is not None:
                grads.append(p.opt_grad().flatten())
        except Exception:
            pass
    if not grads:
        return jt.array(0.0)
    total_norm = jt.norm(jt.concat(grads), norm_type)
    coef = jt.minimum(max_norm / (total_norm + 1e-6), 1.0)
    for p in parameters:
        try:
            g = p.opt_grad()
            if g is not None:
                g.update(g * coef)
        except Exception:
            pass
    return total_norm


def _constant_(var, value):
    var.assign(jt.ones_like(var) * value)
    return var


def _xavier_normal_(var):
    fan_in = var.shape[1] if len(var.shape) > 1 else var.shape[0]
    fan_out = var.shape[0]
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    var.assign(jt.randn(var.shape, dtype=var.dtype) * std)
    return var


def init_jittor():
    global _INITIALIZED, _ORIG_MODULE_TRAIN, _ORIG_MODULE_EVAL, _ORIG_VAR_TRANSPOSE, _ORIG_VAR_EXPAND
    if _INITIALIZED:
        return jt

    for name in ["zeros", "ones", "full", "empty", "rand", "randn", "randint"]:
        if hasattr(jt, name):
            setattr(jt, name, _wrap_creator(name, "float32" if name not in ("randint",) else "int32"))
    for name in ["zeros_like", "ones_like", "full_like", "empty_like", "randint_like", "load", "save"]:
        _remember(name)
    jt.zeros_like = _zeros_like
    jt.ones_like = _ones_like
    jt.full_like = _full_like
    jt.empty_like = _empty_like
    jt.randint_like = _randint_like
    jt.load = _load
    jt.save = _save

    if hasattr(jt, "arange"):
        _remember("arange")
        jt.arange = _arange
    if hasattr(jt, "linspace"):
        _remember("linspace")
        jt.linspace = _linspace
    jt.eye = _eye
    for name in ["argmax", "argmin", "argsort", "topk"]:
        if hasattr(jt, name):
            _remember(name)
    jt.argmax = _argmax
    jt.argmin = _argmin
    if "topk" in _orig:
        jt.topk = _topk

    jt.tensor = tensor
    jt.Tensor = tensor
    jt.FloatTensor = lambda data: tensor(data, dtype="float32")
    jt.LongTensor = lambda data: tensor(data, dtype="int32")
    jt.from_numpy = from_numpy
    jt.as_tensor = as_tensor
    jt.is_tensor = _is_tensor
    jt.is_var = getattr(jt, "is_var", _is_tensor)
    jt.cat = getattr(jt, "concat", None)
    jt.matmul = nn.matmul
    jt.bmm = nn.bmm
    jt.einsum = getattr(linalg, "einsum", _einsum)
    jt.softmax = nn.softmax
    for name in ["sum", "mean"]:
        if hasattr(jt, name):
            _remember(name)
    jt.sum = _sum
    jt.mean = _mean
    jt.norm = _norm
    jt.max = _max
    jt.min = _min
    jt.clamp = _clamp
    jt.clip = _clip
    jt.bernoulli = _bernoulli
    if hasattr(jt, "where"):
        _remember("where")
    jt.where = _where
    jt.maximum = getattr(jt, "maximum", _maximum)
    jt.minimum = getattr(jt, "minimum", _minimum)
    jt.no_grad = _no_grad_class()
    jt.cuda = _CudaCompat()
    jt.autograd = _AutogradCompat()
    jt.device = lambda value: str(value)
    jt.manual_seed = lambda seed: jt.set_global_seed(int(seed))
    jt.Size = lambda values=(): tuple(values)
    jt.logical_not = _logical_not
    jt.chunk = lambda x, chunks, dim=0: x.chunk(chunks, dim=dim)
    jt.float = getattr(jt, "float32", "float32")
    jt.double = getattr(jt, "float64", "float64")
    jt.long = getattr(jt, "int32", "int32")
    jt.bool = getattr(jt, "bool", "bool")

    if hasattr(nn, "Module"):
        if _ORIG_MODULE_TRAIN is None:
            _ORIG_MODULE_TRAIN = nn.Module.train
        if _ORIG_MODULE_EVAL is None and hasattr(nn.Module, "eval"):
            _ORIG_MODULE_EVAL = nn.Module.eval
        nn.Module.to = _module_to
        nn.Module.cpu = _module_cpu
        nn.Module.cuda = _module_cuda
        nn.Module.train = _module_train
        nn.Module.eval = _module_eval
        nn.Module.register_parameter = _module_register_parameter
        nn.Module.register_buffer = _module_register_buffer
        nn.Module.load_state_dict = _load_state_dict
    jt.Module.to = _module_to
    jt.Module.cpu = _module_cpu
    jt.Module.cuda = _module_cuda
    jt.Module.train = _module_train
    jt.Module.eval = _module_eval
    jt.Module.register_parameter = _module_register_parameter
    jt.Module.register_buffer = _module_register_buffer
    jt.Module.load_state_dict = _load_state_dict

    try:
        jt.Var.requires_grad = property(_param_requires_grad_get, _param_requires_grad_set)
        jt.Var.device = property(_var_device)
    except Exception:
        pass
    if _ORIG_VAR_TRANSPOSE is None and hasattr(jt.Var, "transpose"):
        _ORIG_VAR_TRANSPOSE = jt.Var.transpose
    if _ORIG_VAR_EXPAND is None and hasattr(jt.Var, "expand"):
        _ORIG_VAR_EXPAND = jt.Var.expand
    for name, func in {
        "item": _item,
        "tolist": _tolist,
        "detach": lambda self: self.stop_grad(),
        "to": lambda self, device=None, *args, **kwargs: self,
        "cpu": lambda self: self,
        "cuda": lambda self, device=None: self,
        "contiguous": lambda self: self,
        "transpose": _var_transpose,
        "permute": lambda self, *dims: _var_transpose(self, *dims),
        "expand": _var_expand,
        "type": lambda self, dtype=None: self.cast(_dtype(dtype)) if dtype is not None else self,
        "bool": lambda self: self.cast("bool"),
        "clamp": lambda self, min=None, max=None: _clamp(self, min=min, max=max),
        "masked_select": lambda self, mask: _masked_select(self, mask),
        "masked_fill": lambda self, mask, value: _masked_fill(self, mask, value),
        "softmax": lambda self, dim=None: nn.softmax(self, dim=dim),
        "sigmoid": lambda self: _sigmoid(self),
        "scatter_": _scatter_,
        "__invert__": lambda self: _logical_not(self),
        "sum": lambda self, dim=None, dims=None, keepdim=False, keepdims=False: _sum(self, dim=dim, dims=dims, keepdim=keepdim, keepdims=keepdims),
        "mean": lambda self, dim=None, dims=None, keepdim=False, keepdims=False: _mean(self, dim=dim, dims=dims, keepdim=keepdim, keepdims=keepdims),
        "norm": lambda self, p=2, dim=None, keepdim=False, keepdims=False: _norm(self, p=p, dim=dim, keepdim=keepdim, keepdims=keepdims),
        "ne": lambda self, other: self != other,
        "eq": lambda self, other: self == other,
        "lt": lambda self, other: self < other,
        "le": lambda self, other: self <= other,
        "gt": lambda self, other: self > other,
        "ge": lambda self, other: self >= other,
        "argmax": lambda self, dim=None, keepdim=False: _argmax(self, dim=dim, keepdim=keepdim),
        "argmin": lambda self, dim=None, keepdim=False: _argmin(self, dim=dim, keepdim=keepdim),
        "argsort": lambda self, dim=-1, descending=False: _argsort(self, dim=dim, descending=descending),
        "topk": lambda self, k, dim=None, largest=True, sorted=True: _topk(self, k, dim=dim, largest=largest, sorted=sorted),
    }.items():
        try:
            setattr(jt.Var, name, func)
        except Exception:
            pass

    nn.MultiheadAttention = MultiheadAttention
    nn.TransformerEncoderLayer = TransformerEncoderLayer
    nn.TransformerDecoderLayer = TransformerDecoderLayer
    nn.TransformerEncoder = TransformerEncoder
    nn.TransformerDecoder = TransformerDecoder
    if not hasattr(nn, "Parameter"):
        nn.Parameter = lambda value: value
    nn.Embedding = Embedding
    nn.L1Loss = L1Loss
    nn.MSELoss = MSELoss
    nn.SmoothL1Loss = SmoothL1Loss
    nn.functional = F
    if not hasattr(nn, "SiLU"):
        nn.SiLU = jt.make_module(_NN_SILU)
    if not hasattr(nn, "init"):
        nn.init = type("InitCompat", (), {})()
    nn.init.constant_ = _constant_
    nn.init.xavier_normal_ = _xavier_normal_
    optim.lr_scheduler = lr_scheduler

    _INITIALIZED = True
    return jt


init_jittor()
