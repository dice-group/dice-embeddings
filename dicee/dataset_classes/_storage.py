"""Numeric dataset storage that does not copy Python objects on worker access.

Tensor buffers stay shared when DataLoader uses spawn/forkserver as well as
fork. No per-query lists, tuples, dictionaries, or tensor objects are retained.
"""

import operator
from dataclasses import dataclass

import numpy as np
import torch

_PAIR_DTYPE = np.dtype([('first', np.int64), ('second', np.int64)])


@dataclass
class _TensorArray:
    tensor: torch.Tensor


@dataclass
class _MemmapArray:
    filename: str
    dtype: str
    shape: tuple
    offset: int


class WorkerDataset(torch.utils.data.Dataset):
    """Keep read-only NumPy data shared when a loader pickles its dataset.

    Fork already shares numeric array buffers. For spawn/forkserver, reopen
    contiguous file mappings and let PyTorch's multiprocessing reducers share
    ordinary arrays as tensor storage. Cache the tensor wrapper so subsequent
    workers reuse the same shared storage instead of making another copy.
    Dataset buffers must not be modified while workers are using them.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        cache = state.pop('_worker_array_cache', {})
        for name, array in state.items():
            if not isinstance(array, np.ndarray):
                continue
            if array.dtype.hasobject:
                raise TypeError('Worker datasets require numeric arrays, not dtype=object')
            cached = cache.get(name)
            if cached is None or cached[0] is not array:
                storage: _MemmapArray | _TensorArray
                root = array
                while isinstance(root.base, np.ndarray):
                    root = root.base
                if isinstance(root, np.memmap) and root.filename and array.flags.c_contiguous:
                    offset = root.offset + array.ctypes.data - root.ctypes.data
                    storage = _MemmapArray(str(root.filename), array.dtype.str, array.shape, offset)
                else:
                    storage = _TensorArray(torch.from_numpy(array))
                cached = cache[name] = (array, storage)
            state[name] = cached[1]
        self._worker_array_cache = cache
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            if isinstance(value, _MemmapArray):
                state[name] = np.memmap(value.filename, dtype=value.dtype, mode='r',
                                        shape=value.shape, offset=value.offset)
            elif isinstance(value, _TensorArray):
                state[name] = value.tensor.numpy()
        self.__dict__.update(state)


class RaggedIndices:
    """Variable-length integer rows stored as two contiguous CPU tensors."""

    def __init__(self, values, offsets):
        self.values = torch.as_tensor(values, dtype=torch.long).contiguous()
        self.offsets = torch.as_tensor(offsets, dtype=torch.long).contiguous()

    @classmethod
    def from_rows(cls, rows):
        if isinstance(rows, cls):
            return rows
        lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        values = np.empty(int(offsets[-1]), dtype=np.int64)
        for i, row in enumerate(rows):
            values[offsets[i]:offsets[i + 1]] = row
        return cls(values, offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # numpy() is a view of the tensor buffer, including in spawned workers.
        start, end = self.offsets.numpy()[index:index + 2]
        return self.values[int(start):int(end)]

    @property
    def max_length(self):
        return int(np.diff(self.offsets.numpy()).max(initial=0))

    def padded_rows(self, rows, padding):
        """Materialize padding only for the current batch, never the full KG."""
        offsets = self.offsets.numpy()
        starts = offsets[rows]
        lengths = offsets[rows + 1] - starts
        positions = np.arange(lengths.max(initial=0))[None, :]
        valid = positions < lengths[:, None]
        indices = starts[:, None] + positions
        padded = np.full(valid.shape, padding, dtype=np.int64)
        padded[valid] = self.values.numpy()[indices[valid]]
        return padded, lengths


class PairIndex:
    """Sorted integer pairs and their ragged targets, without a Python dict."""

    def __init__(self, keys, targets: RaggedIndices, max_input_count=None):
        self.keys = torch.as_tensor(keys, dtype=torch.long).contiguous()
        self.targets = targets
        self.max_input_count = targets.max_length if max_input_count is None else max_input_count

    @classmethod
    def from_triples(cls, triples, columns=(0, 1, 2), *, sort_targets=False, unique=False):
        # lexsort is stable: by default targets retain their original ordering
        # within each pair, matching the old dictionary-of-lists implementation.
        first, second, target = (np.asarray(triples[:, col]) for col in columns)
        sort_keys = (target, second, first) if sort_targets or unique else (second, first)
        order = np.lexsort(sort_keys)
        keys = np.column_stack((first[order], second[order])).astype(np.int64, copy=False)
        values = np.asarray(target[order], dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)]) if len(keys) else np.empty(0, dtype=np.int64)
        max_input_count = int(np.diff(np.r_[starts, len(values)]).max(initial=0))
        if unique and len(values):
            keep = np.r_[True, (keys[1:] != keys[:-1]).any(axis=1) | (values[1:] != values[:-1])]
            keys, values = keys[keep], values[keep]
            starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])
        offsets = np.r_[starts, len(values)]
        return cls(keys[starts], RaggedIndices(values, offsets), max_input_count)

    def find_rows(self, pairs):
        """Vectorized lexicographic lookup, without overflowing packed integer IDs."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        keys = self.keys.numpy()
        rows = np.searchsorted(keys.view(_PAIR_DTYPE).reshape(-1),
                               np.ascontiguousarray(pairs).view(_PAIR_DTYPE).reshape(-1))
        if np.any(rows >= len(keys)) or np.any(keys[rows] != pairs):
            raise KeyError('Pair is absent from the training graph')
        return rows

    def __getitem__(self, pair):
        return self.targets[int(self.find_rows([pair])[0])]
