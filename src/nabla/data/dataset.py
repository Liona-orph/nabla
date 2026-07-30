"""Datasets and the loader that batches them.

# The map-style contract

A dataset is `__len__` plus `__getitem__(i)`, returning a tuple of NumPy arrays
for one example. That is the whole interface. The iterable-style alternative -
a dataset that is a generator - is not offered, because it cannot be shuffled,
cannot be split, and cannot report its own length, and every one of those is
needed by something below.

# Determinism

`DataLoader` owns a `nabla.random.Generator` and draws a fresh permutation from
it at the start of each epoch. Two loaders built with the same seed therefore
produce the same batches in the same order, epoch after epoch, and a run is
reproducible from its seeds. The permutation advances the generator, so epoch 2
differs from epoch 1 - which is the point of shuffling - while the *sequence* of
epochs is fixed.

The one thing this does not survive is drawing from the same generator
elsewhere. That is why the loader takes its own generator rather than sharing
the model's: `Generator.spawn` exists to hand out independent streams, and a
loader is exactly the kind of component that should get one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .._typing import Array
from ..errors import ShapeError
from ..random import Generator
from ..tensor import Tensor

__all__ = ["ArrayDataset", "DataLoader", "Dataset", "random_split"]


@runtime_checkable
class Dataset(Protocol):
    """What a `DataLoader` requires: a length and indexed access to examples.

    A `Protocol` rather than a base class, so that anything with the two
    methods works without inheriting from this package. The runtime-checkable
    form is used only for a construction-time error message; `isinstance`
    against a protocol checks method *names* and not signatures, which is
    enough to catch "you passed a list of arrays" and not enough to be trusted
    for anything else.
    """

    def __len__(self) -> int:
        """Number of examples."""
        ...

    def __getitem__(self, index: int) -> tuple[Array, ...]:
        """One example, as a tuple of arrays - typically `(inputs, target)`."""
        ...


class ArrayDataset:
    """A dataset backed by parallel NumPy arrays, sliced along axis 0.

    The common case: features and labels already in memory. Indexing returns
    *views* into the underlying arrays, so constructing a dataset from a 100 MB
    feature matrix copies nothing; the copy happens once per batch in
    `DataLoader`, where `np.stack` has to produce a contiguous block anyway.

    >>> import numpy as np
    >>> from nabla.data import ArrayDataset
    >>> ds = ArrayDataset(np.arange(6).reshape(3, 2), np.array([0, 1, 0]))
    >>> len(ds), ds[1][0].tolist(), int(ds[1][1])
    (3, [2, 3], 1)
    """

    def __init__(self, *arrays: Array) -> None:
        if not arrays:
            msg = "ArrayDataset needs at least one array"
            raise ShapeError(msg)
        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            msg = f"ArrayDataset arrays disagree about the number of examples: {sorted(lengths)}"
            raise ShapeError(msg)
        self.arrays = tuple(np.asarray(a) for a in arrays)

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, index: int) -> tuple[Array, ...]:
        return tuple(a[index] for a in self.arrays)

    def __repr__(self) -> str:
        shapes = ", ".join(str(a.shape) for a in self.arrays)
        return f"ArrayDataset({shapes})"


class Subset:
    """A view of a dataset restricted to the given indices.

    Holds indices rather than copying examples, so splitting a dataset costs one
    integer array however large the examples are.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int] | Array) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Array, ...]:
        return self.dataset[int(self.indices[index])]

    def __repr__(self) -> str:
        return f"Subset({self.dataset!r}, n={len(self.indices)})"


def random_split(
    dataset: Dataset, fractions: Sequence[float], generator: Generator
) -> list[Subset]:
    """Split a dataset into disjoint subsets by fraction, using `generator`.

    Sizes are floored and the remainder goes to the last part, so the parts
    always cover the dataset exactly. A split that silently dropped a few
    examples would be invisible until somebody compared two counts.

    >>> import numpy as np
    >>> import nabla
    >>> from nabla.data import ArrayDataset, random_split
    >>> parts = random_split(
    ...     ArrayDataset(np.arange(10)), [0.8, 0.2], nabla.random.Generator(seed=0)
    ... )
    >>> [len(p) for p in parts]
    [8, 2]
    """
    if not fractions or any(f <= 0 for f in fractions):
        msg = f"fractions must all be positive, got {list(fractions)}"
        raise ValueError(msg)
    total = float(sum(fractions))
    if abs(total - 1.0) > 1e-6:
        msg = f"fractions must sum to 1, got {total}"
        raise ValueError(msg)
    count = len(dataset)
    order = generator.permutation(count)
    sizes = [int(f * count) for f in fractions[:-1]]
    sizes.append(count - sum(sizes))
    parts: list[Subset] = []
    start = 0
    for size in sizes:
        parts.append(Subset(dataset, order[start : start + size]))
        start += size
    return parts


class DataLoader:
    """Batches from a dataset, optionally shuffled, as tensors.

    Collation is `np.stack` per field, which requires every example's arrays to
    have the same shape. Ragged data - variable-length sequences - needs padding
    before it reaches here, and doing that padding inside a collate function
    would hide from the caller that a sequence was truncated.

    `drop_last` exists because the final short batch changes the shape of every
    activation in the model, which for a fixed-shape model (a transformer with a
    precomputed mask, for instance) is either an error or a silent recompilation
    of nothing. It defaults to False because dropping data by default is worse.

    >>> import numpy as np
    >>> import nabla
    >>> from nabla.data import ArrayDataset, DataLoader
    >>> loader = DataLoader(
    ...     ArrayDataset(np.arange(10.0), np.arange(10)),
    ...     batch_size=4,
    ...     shuffle=True,
    ...     generator=nabla.random.Generator(seed=0),
    ... )
    >>> [tuple(int(v) for v in batch[1].numpy()) for batch in loader]
    [(4, 6, 2, 7), (3, 5, 9, 0), (8, 1)]
    >>> len(loader)
    3
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool = False,
        generator: Generator | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)
        if shuffle and generator is None:
            msg = (
                "shuffle=True needs a generator. There is no global random stream in this "
                "package, so a shuffled loader with no generator would have no defined order."
            )
            raise ValueError(msg)
        # The annotation is a promise; this is the check for callers who do not
        # run a type checker. It probes one example rather than testing
        # `isinstance(dataset, Dataset)`, because a plain list satisfies the
        # protocol's two method names and then fails much later with a message
        # about stacking. The `object` annotation is what stops a type checker
        # from deciding the branch is unreachable and deleting the diagnostic.
        if len(dataset) > 0:
            probe: object = dataset[0]
            # A scalar field is legitimate and common - indexing a 1-d label
            # array gives a NumPy scalar, not a 0-d array - so `np.generic` is
            # accepted alongside `ndarray`. `np.stack` treats the two alike.
            if not isinstance(probe, tuple) or not all(
                isinstance(field, (np.ndarray, np.generic)) for field in probe
            ):
                msg = (
                    f"{type(dataset).__name__}[0] returned {type(probe).__name__}; a dataset "
                    "must return a tuple of NumPy arrays per example, e.g. (features, label)"
                )
                raise TypeError(msg)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.drop_last = drop_last

    def __len__(self) -> int:
        """Number of batches this loader will yield per epoch."""
        count = len(self.dataset)
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """One epoch of batches, drawing a fresh permutation if shuffling."""
        count = len(self.dataset)
        if self.shuffle and self.generator is not None:
            order = self.generator.permutation(count)
        else:
            order = np.arange(count)
        limit = (count // self.batch_size) * self.batch_size if self.drop_last else count
        for start in range(0, limit, self.batch_size):
            indices = order[start : start + self.batch_size]
            yield _collate([self.dataset[int(i)] for i in indices])

    def __repr__(self) -> str:
        return (
            f"DataLoader({self.dataset!r}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last})"
        )


def _collate(examples: list[tuple[Array, ...]]) -> tuple[Tensor, ...]:
    """Stack a list of examples into one tensor per field.

    Integer fields stay integer. A label array cast to float on the way into a
    batch is a label array that cannot be used as an index, and
    `nabla.nn.loss.cross_entropy` takes indices.
    """
    fields = len(examples[0])
    if any(len(example) != fields for example in examples):
        msg = "the dataset returned examples with different numbers of fields"
        raise ShapeError(msg)
    return tuple(
        Tensor(np.stack([example[field] for example in examples])) for field in range(fields)
    )
