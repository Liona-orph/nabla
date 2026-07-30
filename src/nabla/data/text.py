"""Character-level text: a vocabulary and a next-token dataset.

Character level, not sub-word. A byte-pair encoder would give shorter sequences
and a better loss per character, and it would also be several hundred lines of
merge-table construction with no gradient in it - which is a lot of code to add
to a package whose subject is gradients. The cost is stated in
`examples/char_transformer`: the model has to spend capacity learning that `q`
is followed by `u`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .._typing import Array

__all__ = ["CharVocabulary", "NextCharDataset"]


class CharVocabulary:
    """A bijection between the characters of a corpus and `range(size)`.

    Built from the sorted set of characters that actually occur, so the mapping
    is a pure function of the text and two runs over the same file agree. An
    insertion-ordered vocabulary would depend on where in the file each
    character first appeared, which makes a checkpoint's embedding table
    silently incompatible with a vocabulary built from a different sample of the
    same corpus.

    There is no unknown-token slot. Encoding a character the vocabulary has
    never seen raises, because the alternative - mapping it to a shared `<unk>` -
    is a decision about the modelling problem that a tokeniser should not be
    making silently.

    >>> from nabla.data import CharVocabulary
    >>> vocab = CharVocabulary("banana")
    >>> vocab.characters
    ('a', 'b', 'n')
    >>> vocab.encode("nab").tolist()
    [2, 0, 1]
    >>> vocab.decode([2, 0, 1])
    'nab'
    """

    def __init__(self, text: str) -> None:
        if not text:
            msg = "cannot build a vocabulary from an empty text"
            raise ValueError(msg)
        self.characters: tuple[str, ...] = tuple(sorted(set(text)))
        self._to_index = {character: index for index, character in enumerate(self.characters)}

    @property
    def size(self) -> int:
        """Number of distinct characters."""
        return len(self.characters)

    def __len__(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> Array:
        """Text to an int64 array of indices."""
        try:
            return np.array([self._to_index[c] for c in text], dtype=np.int64)
        except KeyError as exc:
            msg = f"character {exc.args[0]!r} is not in this vocabulary of {self.size} characters"
            raise KeyError(msg) from exc

    def decode(self, indices: Iterable[int] | Array) -> str:
        """Indices back to text."""
        return "".join(self.characters[int(i)] for i in indices)

    def __repr__(self) -> str:
        return f"CharVocabulary(size={self.size})"


class NextCharDataset:
    """Fixed-length windows of a token stream, paired with the same window shifted by one.

    Example `i` is `(tokens[i : i+L], tokens[i+1 : i+L+1])`, so a model that maps
    a window to a same-length sequence of logits is trained on `L` next-character
    predictions per example rather than one. That is the whole reason a
    causal mask exists: without it position `t` could see position `t+1`, which
    is the answer, and the loss would go to zero without the model learning
    anything.

    Windows overlap and are stored as views, so a 96,000-character corpus with
    `length=64` yields about 96,000 examples and costs one array.

    >>> import numpy as np
    >>> from nabla.data import NextCharDataset
    >>> ds = NextCharDataset(np.arange(10), length=4)
    >>> len(ds)
    6
    >>> [a.tolist() for a in ds[0]]
    [[0, 1, 2, 3], [1, 2, 3, 4]]
    """

    def __init__(self, tokens: Sequence[int] | Array, *, length: int) -> None:
        self.tokens: Array = np.asarray(tokens, dtype=np.int64)
        if length < 1:
            msg = f"length must be at least 1, got {length}"
            raise ValueError(msg)
        if self.tokens.size < length + 1:
            msg = (
                f"a corpus of {self.tokens.size} tokens is too short for windows of "
                f"{length} plus one target"
            )
            raise ValueError(msg)
        self.length = length

    def __len__(self) -> int:
        return int(self.tokens.size) - self.length

    def __getitem__(self, index: int) -> tuple[Array, ...]:
        window = self.tokens[index : index + self.length + 1]
        return window[:-1], window[1:]

    def __repr__(self) -> str:
        return f"NextCharDataset(tokens={self.tokens.size}, length={self.length})"
