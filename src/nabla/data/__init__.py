"""Datasets, batching, and the two synthetic problems the examples use.

Nothing in this package reaches the network, at import time or at run time. The
synthetic generators are pure functions of a seed and the one checked-in corpus
lives in `testdata/`, so every number in the examples' READMEs can be reproduced
on a machine with no internet connection.

>>> import numpy as np
>>> import nabla
>>> from nabla import data
>>> x, y = data.make_moons(64, nabla.random.Generator(seed=0))
>>> loader = data.DataLoader(data.ArrayDataset(x, y), batch_size=16, shuffle=False)
>>> [tuple(t.shape) for t in next(iter(loader))]
[(16, 2), (16,)]
"""

from __future__ import annotations

from .dataset import ArrayDataset, DataLoader, Dataset, Subset, random_split
from .synthetic import DIGIT_GLYPHS, make_digits, make_moons
from .text import CharVocabulary, NextCharDataset

__all__ = [
    "DIGIT_GLYPHS",
    "ArrayDataset",
    "CharVocabulary",
    "DataLoader",
    "Dataset",
    "NextCharDataset",
    "Subset",
    "make_digits",
    "make_moons",
    "random_split",
]
