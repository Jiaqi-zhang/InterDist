#!/usr/bin/env python3
"""Convert PyTorch zip checkpoints to Jittor-readable pickle checkpoints.

This intentionally does not import torch.  It supports the checkpoint format
produced by torch.save in the official InterDist release: a zip archive with
`data.pkl` metadata and raw tensor storages under `data/<id>`.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import shutil
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np


OFFICIAL_CHECKPOINTS = OrderedDict(
    [
        ("interh_vq_model.pth", "interh_vq_model.pth"),
        ("inter_x_vq_model.pth", "inter_x_vq_model.pth"),
        ("interh_t2m_model.pth", "interh_t2m_model.pth"),
        ("inter_x_t2m_model.pth", "inter_x_t2m_model.pth"),
    ]
)

_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("float32"),
    "DoubleStorage": np.dtype("float64"),
    "HalfStorage": np.dtype("float16"),
    "LongStorage": np.dtype("int64"),
    "IntStorage": np.dtype("int32"),
    "ShortStorage": np.dtype("int16"),
    "CharStorage": np.dtype("int8"),
    "ByteStorage": np.dtype("uint8"),
    "BoolStorage": np.dtype("bool"),
}


@dataclass(frozen=True)
class StorageType:
    name: str
    dtype: np.dtype


@dataclass
class StorageRef:
    key: str
    dtype: np.dtype
    array: np.ndarray


def _rebuild_tensor(storage: StorageRef, storage_offset: int, size: Iterable[int], stride: Iterable[int]) -> np.ndarray:
    shape = tuple(int(v) for v in size)
    strides = tuple(int(v) * storage.dtype.itemsize for v in stride)
    base = storage.array[int(storage_offset):]
    if not shape:
        return np.array(base[0], dtype=storage.dtype)
    view = np.lib.stride_tricks.as_strided(base, shape=shape, strides=strides)
    # Copy into a normal contiguous ndarray so Jittor load_parameters can ingest it directly.
    return np.array(view, copy=True)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, *rest):
    return _rebuild_tensor(storage, storage_offset, size, stride)


def _rebuild_tensor_plain(storage, storage_offset, size, stride):
    return _rebuild_tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, file, archive: zipfile.ZipFile, prefix: str, byteorder: str):
        super().__init__(file, encoding="latin1")
        self.archive = archive
        self.prefix = prefix
        self.byteorder = byteorder
        self._storages: Dict[str, StorageRef] = {}

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_tensor":
            return _rebuild_tensor_plain
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return StorageType(name, _STORAGE_DTYPES[name])
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        typename = saved_id[0]
        if typename != "storage":
            raise RuntimeError(f"Unsupported persistent id: {saved_id!r}")
        storage_type, key, location, numel = saved_id[1:5]
        if not isinstance(storage_type, StorageType):
            raise RuntimeError(f"Unsupported storage type in id: {saved_id!r}")
        key = str(key)
        if key not in self._storages:
            raw = self.archive.read(f"{self.prefix}/data/{key}")
            dtype = storage_type.dtype
            if self.byteorder == "big":
                dtype = dtype.newbyteorder(">")
            arr = np.frombuffer(raw, dtype=dtype, count=int(numel))
            if self.byteorder == "big":
                arr = arr.astype(storage_type.dtype, copy=False)
            self._storages[key] = StorageRef(key=key, dtype=storage_type.dtype, array=arr)
        return self._storages[key]


def load_torch_checkpoint(path: os.PathLike[str] | str) -> Any:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        data_name = next((name for name in names if name.endswith("/data.pkl")), None)
        if data_name is None:
            raise RuntimeError(f"{path} does not look like a torch zip checkpoint")
        prefix = data_name.rsplit("/", 1)[0]
        byteorder_name = f"{prefix}/byteorder"
        byteorder = archive.read(byteorder_name).decode().strip() if byteorder_name in names else sys.byteorder
        payload = archive.read(data_name)
        return TorchZipUnpickler(io.BytesIO(payload), archive, prefix, byteorder).load()


def _iter_arrays(obj: Any, prefix: str = ""):
    if isinstance(obj, np.ndarray):
        yield prefix, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_arrays(value, child)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _iter_arrays(value, f"{prefix}[{i}]")


def summarize_checkpoint(obj: Any) -> str:
    arrays = list(_iter_arrays(obj))
    total = sum(arr.nbytes for _, arr in arrays)
    top = ", ".join(f"{name}:{tuple(arr.shape)}" for name, arr in arrays[:5])
    return f"arrays={len(arrays)}, tensor_bytes={total}, first=[{top}]"


def save_jittor_checkpoint(obj: Any, path: os.PathLike[str] | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)


def convert_one(src: Path, dst: Path, dry_run: bool = False) -> None:
    obj = load_torch_checkpoint(src)
    print(f"{src} -> {dst}")
    print(f"  {summarize_checkpoint(obj)}")
    if not dry_run:
        save_jittor_checkpoint(obj, dst)


def copy_official(src_dir: Path, dst_dir: Path, dry_run: bool = False) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_CHECKPOINTS:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        print(f"copy {src} -> {dst}")
        if not dry_run:
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", help="single torch checkpoint to convert")
    parser.add_argument("dst", nargs="?", help="single output checkpoint path")
    parser.add_argument("--official-dir", type=Path, help="directory containing official InterDist *.pth files")
    parser.add_argument("--copy-to", type=Path, help="copy official torch checkpoints to this directory before conversion")
    parser.add_argument("--convert-to", type=Path, help="convert all official checkpoints into this directory")
    parser.add_argument("--dry-run", action="store_true", help="inspect/copy plan without writing converted files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.official_dir:
        if args.copy_to:
            copy_official(args.official_dir, args.copy_to, dry_run=args.dry_run)
        if args.convert_to:
            for name in OFFICIAL_CHECKPOINTS:
                convert_one(args.official_dir / name, args.convert_to / name, dry_run=args.dry_run)
        return
    if not args.src or not args.dst:
        raise SystemExit("Provide SRC DST, or --official-dir with --convert-to/--copy-to")
    convert_one(Path(args.src), Path(args.dst), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
