"""Seeded random number generation.

There is no module-level generator and no equivalent of `np.random.seed()`. A
global stream means a run is only reproducible if every piece of code that
touches it draws in the same order - so adding a dropout layer, or evaluating a
model between two training steps, changes the initialisation of everything that
comes after. The failure is silent and looks like variance between runs.

A `Generator` is therefore an explicit object that has to be passed around.
`spawn()` derives independent child streams from a parent, which is how a
subcomponent gets its own randomness without consuming draws from anyone else's.
The independence comes from NumPy's `SeedSequence`, whose entropy mixing is
designed for exactly this and does not rely on the children being drawn from in
any particular order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._typing import DEFAULT_FLOAT_DTYPE, Array, DTypeLike, Shape
from .tensor import Tensor

__all__ = ["Generator"]


def _resolve(dtype: DTypeLike | None) -> np.dtype[Any]:
    """Resolve a requested dtype, refusing anything that cannot hold noise."""
    resolved: np.dtype[Any] = np.dtype(DEFAULT_FLOAT_DTYPE if dtype is None else dtype)
    if resolved.kind != "f":
        msg = f"random values need a floating dtype, got {resolved}"
        raise TypeError(msg)
    return resolved


class Generator:
    """A reproducible source of random tensors.

    Wraps `numpy.random.Generator` (PCG64) rather than the legacy
    `RandomState`: the legacy one is the global that this module exists to
    avoid, and its algorithm is kept frozen for backwards compatibility rather
    than for quality.

    Values are generated in float64 - PCG64's output is - and cast to the
    requested dtype afterwards. For float32 that cast is exact for the uniform
    case and correctly rounded for the normal case; generating directly in
    float32 would be faster and would consume a different number of bits per
    draw, making the stream depend on the dtype.

    >>> import nabla
    >>> g = nabla.random.Generator(seed=0)
    >>> x = g.normal((2, 3))
    >>> x.shape, str(x.dtype)
    ((2, 3), 'float32')
    >>> other = nabla.random.Generator(seed=0)
    >>> bool((other.normal((2, 3)).numpy() == x.numpy()).all())
    True
    """

    __slots__ = ("_rng", "_seed_sequence")

    def __init__(self, seed: int | np.random.SeedSequence = 0) -> None:
        self._seed_sequence = (
            seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        )
        self._rng = np.random.default_rng(self._seed_sequence)

    def __repr__(self) -> str:
        return f"Generator(entropy={self._seed_sequence.entropy!r})"

    @property
    def numpy_generator(self) -> np.random.Generator:
        """The underlying NumPy generator, for draws this class does not wrap."""
        return self._rng

    def spawn(self, count: int) -> list[Generator]:
        """Derive `count` independent child generators.

        Independent in the sense that matters: the children's streams are
        statistically unrelated to each other and to the parent's remaining
        stream, so a component handed child 3 produces the same values however
        much anyone else draws.
        """
        return [Generator(child) for child in self._seed_sequence.spawn(count)]

    def normal(
        self,
        shape: Shape | int = (),
        *,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: DTypeLike | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Normally distributed values."""
        size = (shape,) if isinstance(shape, int) else tuple(shape)
        values: Array = self._rng.normal(mean, std, size).astype(_resolve(dtype))
        return _wrap(values, requires_grad)

    def uniform(
        self,
        shape: Shape | int = (),
        *,
        low: float = 0.0,
        high: float = 1.0,
        dtype: DTypeLike | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Uniformly distributed values on `[low, high)`."""
        size = (shape,) if isinstance(shape, int) else tuple(shape)
        values: Array = self._rng.uniform(low, high, size).astype(_resolve(dtype))
        return _wrap(values, requires_grad)

    def bernoulli(
        self,
        shape: Shape | int = (),
        *,
        p: float = 0.5,
        dtype: DTypeLike | None = None,
    ) -> Tensor:
        """A 0/1 mask, 1 with probability `p`.

        Floating rather than boolean, because its only use is to be multiplied
        by an activation, and a boolean mask would force a cast at every call
        site. Never requires a gradient: the mask is a sample, not a parameter.
        """
        if not 0.0 <= p <= 1.0:
            msg = f"p must be a probability, got {p}"
            raise ValueError(msg)
        size = (shape,) if isinstance(shape, int) else tuple(shape)
        values: Array = (self._rng.random(size) < p).astype(_resolve(dtype))
        return _wrap(values, requires_grad=False)

    def permutation(self, n: int) -> Array:
        """A random permutation of `range(n)`, as an ndarray of indices.

        Returned as an array rather than a tensor: it is used to index, and an
        integer tensor would only have to be unwrapped again.
        """
        out: Array = self._rng.permutation(n)
        return out

    def integers(self, low: int, high: int | None = None, size: Shape | int = ()) -> Array:
        """Random integers, NumPy's `integers` semantics. Also an ndarray."""
        out: Array = self._rng.integers(low, high, size)
        return out


def _wrap(values: Array, requires_grad: bool) -> Tensor:
    """Take ownership of a freshly generated array without copying it again."""
    tensor = Tensor._owning(values)
    tensor.requires_grad = requires_grad
    return tensor
