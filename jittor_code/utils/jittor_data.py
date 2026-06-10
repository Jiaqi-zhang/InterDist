"""Minimal DataLoader utilities that do not depend on PyTorch."""

import random
from collections.abc import Mapping, Sequence

import numpy as np
from .jittor_compat import jt


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def _to_batch(values):
    first = values[0]
    if isinstance(first, jt.Var):
        return jt.stack(values, dim=0)
    if isinstance(first, np.ndarray):
        return jt.array(np.stack(values, axis=0))
    if isinstance(first, (np.number, int, float, bool)):
        return jt.array(values)
    if isinstance(first, str):
        return list(values)
    if isinstance(first, Mapping):
        return {k: _to_batch([v[k] for v in values]) for k in first}
    if isinstance(first, tuple):
        return tuple(_to_batch(items) for items in zip(*values))
    if isinstance(first, Sequence):
        return [_to_batch(items) for items in zip(*values)]
    return list(values)


def default_collate(batch):
    return _to_batch(batch)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        batch = []
        for index in indices:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
