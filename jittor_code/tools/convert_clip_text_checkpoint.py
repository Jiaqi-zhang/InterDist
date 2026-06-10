#!/usr/bin/env python3
"""Convert OpenAI CLIP ViT-L/14@336px TorchScript weights to a Jittor text checkpoint.

The Jittor eval runtime is torch-free, but this one-time converter intentionally
uses PyTorch because OpenAI distributes CLIP as a TorchScript archive.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np

CLIP_URL = "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
TEXT_KEYS = {
    "token_embedding.weight",
    "positional_embedding",
    "text_projection",
    "ln_final.weight",
    "ln_final.bias",
    "logit_scale",
}


def convert(src: Path, dst: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required only for this converter; run it in an environment with torch installed.") from exc

    model = torch.jit.load(str(src), map_location="cpu").eval()
    source_state = model.state_dict()
    text_state = OrderedDict()
    for key, value in source_state.items():
        if key in TEXT_KEYS or key.startswith("transformer.resblocks."):
            tensor = value.detach().cpu()
            if tensor.is_floating_point():
                tensor = tensor.float()
            text_state[key] = np.array(tensor.numpy(), copy=True)

    metadata = {
        "clip_version": "ViT-L/14@336px",
        "source_url": CLIP_URL,
        "context_length": 77,
        "vocab_size": 49408,
        "width": 768,
        "heads": 12,
        "layers": 12,
        "output_dim": 768,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump({"metadata": metadata, "state_dict": text_state}, f, protocol=4)

    total = sum(arr.nbytes for arr in text_state.values())
    print(f"saved {dst} keys={len(text_state)} tensor_bytes={total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="OpenAI ViT-L-14-336px.pt TorchScript checkpoint")
    parser.add_argument("--dst", type=Path, default=Path("checkpoints/clip_vit_l_14_336px_text.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise SystemExit(f"Source checkpoint not found: {args.src}\nDownload it from: {CLIP_URL}")
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
