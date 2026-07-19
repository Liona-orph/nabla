"""Reductions.

A reduction turns many values into few, so its vjp turns few cotangents into
many: every backward pass here is a *scatter*. Two shapes of scatter appear.

* `sum`, `mean`, `var`, `logsumexp` spread the cotangent over every element that
  contributed, weighted by that element's share. The spreading is done with
  `np.broadcast_to`, which allocates nothing - the result is a zero-stride view -
  so the cost is deferred to whoever consumes it.
* `max` and `min` send the whole cotangent to one element per reduced group and
  zero to the rest. That needs a real buffer and an `argmax`, which is why they
  are the only reductions here that allocate in the backward pass.

This module shadows the builtins `sum`, `max` and `min` on purpose; the API is
meant to read like NumPy's. Inside the module, use `np.sum`, `np.max`, `np.min`.
"""

from __future__ import annotations

import numpy as np

from .._typing import Array, Axis, Shape, TensorLike
from ..errors import ShapeError
from ..tensor import Tensor, as_tensor, record
from .basic import sqrt

__all__ = ["logsumexp", "max", "mean", "min", "std", "sum", "var"]


def _normalise_axis(axis: Axis, ndim: int) -> tuple[int, ...]:
    """Turn NumPy's several axis spellings into a sorted tuple of non-negative axes."""
    if axis is None:
        return tuple(range(ndim))
    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    seen: set[int] = set()
    for a in raw:
        if ndim == 0 or not -ndim <= a < ndim:
            msg = f"axis {a} is out of range for a tensor of rank {ndim}"
            raise ShapeError(msg)
        normalised = a % ndim
        if normalised in seen:
            msg = f"axis {a} appears twice in {axis!r}"
            raise ShapeError(msg)
        seen.add(normalised)
    return tuple(sorted(seen))


def _restore_reduced(g: Array, axes: tuple[int, ...], *, keepdims: bool) -> Array:
    """Give `g` back the axes the reduction removed, as size-1 dimensions.

    With `keepdims=True` they are already there. Without it the cotangent
    arrives one rank short per reduced axis, and broadcasting it back against
    the input would silently align the wrong dimensions.
    """
    if keepdims or not axes:
        return g
    return np.expand_dims(g, axes)


def _reduced_count(shape: Shape, axes: tuple[int, ...]) -> int:
    """Number of elements collapsed into each output element."""
    count = 1
    for a in axes:
        count *= shape[a]
    return count


def sum(a: TensorLike, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
    """Sum over `axis` (all axes when `None`).

    The cotangent of a sum is the same value for every contributing element, so
    the backward pass is a broadcast and allocates nothing.

    >>> import nabla
    >>> x = nabla.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> nabla.sum(x, axis=0).numpy().tolist()
    [4.0, 6.0]
    >>> nabla.sum(x).backward()
    >>> x.grad.tolist()
    [[1.0, 1.0], [1.0, 1.0]]
    """
    ta = as_tensor(a)
    axes = _normalise_axis(axis, ta.ndim)
    out: Array = np.sum(ta.data, axis=axis, keepdims=keepdims)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (np.broadcast_to(_restore_reduced(g, axes, keepdims=keepdims), ta.shape),)

    return record(out, "sum", (ta,), vjp)


def mean(a: TensorLike, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
    """Arithmetic mean over `axis`.

    Implemented directly rather than as `sum(x) / n` to save one tape node per
    call; the mean appears once per loss and once per normalisation layer, so
    the nodes add up.

    The mean of an empty axis is `nan` (NumPy's answer, and the only honest one);
    its gradient is `nan` too, since there is no element to receive it.
    """
    ta = as_tensor(a)
    axes = _normalise_axis(axis, ta.ndim)
    count = _reduced_count(ta.shape, axes)
    out: Array = np.mean(ta.data, axis=axis, keepdims=keepdims)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        spread = _restore_reduced(g, axes, keepdims=keepdims) / count
        return (np.broadcast_to(spread, ta.shape),)

    return record(out, "mean", (ta,), vjp)


def max(a: TensorLike, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
    """Maximum over `axis`.

    The gradient goes entirely to the first maximal element of each reduced
    group, matching `np.argmax`. At a tie the function is not differentiable and
    any convention is a choice; this one at least matches what `argmax` reports,
    so a caller inspecting the two sees a consistent story.
    """
    return _extremum(a, axis, keepdims=keepdims, op="max")


def min(a: TensorLike, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
    """Minimum over `axis`. Ties resolve to the first minimal element."""
    return _extremum(a, axis, keepdims=keepdims, op="min")


def _extremum(a: TensorLike, axis: Axis, *, keepdims: bool, op: str) -> Tensor:
    ta = as_tensor(a)
    axes = _normalise_axis(axis, ta.ndim)
    reduce_fn = np.max if op == "max" else np.min
    arg_fn = np.argmax if op == "max" else np.argmin
    out: Array = reduce_fn(ta.data, axis=axis, keepdims=keepdims)

    # The reduced axes are moved to the end and flattened so that one
    # `argmax(..., axis=-1)` handles any combination of axes, including all of
    # them. The permutation is a view; only the final scatter buffer allocates.
    kept = tuple(i for i in range(ta.ndim) if i not in axes)
    perm = kept + axes
    inverse = np.argsort(perm)
    kept_shape = tuple(ta.shape[i] for i in kept)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        moved = ta.data.transpose(perm)
        flat = moved.reshape((*kept_shape, -1))
        winners = arg_fn(flat, axis=-1)[..., None]
        seed = _restore_reduced(g, axes, keepdims=keepdims)
        seed_flat = seed.transpose(perm).reshape((*kept_shape, 1))
        scattered = np.zeros(flat.shape, dtype=g.dtype)
        np.put_along_axis(scattered, winners, seed_flat, axis=-1)
        return (scattered.reshape(moved.shape).transpose(inverse),)

    return record(out, op, (ta,), vjp)


def var(a: TensorLike, axis: Axis = None, *, keepdims: bool = False, ddof: int = 0) -> Tensor:
    """Variance over `axis`, with `ddof` degrees of freedom removed.

    `ddof=0` is the biased estimator (divide by `n`) and matches NumPy; `ddof=1`
    is Bessel's correction. Both appear in real models - batch normalisation
    uses the biased one for its statistics - so neither is hidden behind a
    boolean.

    The gradient is `2 (x - mean) / (n - ddof)`, which uses the *same* `n - ddof`
    the forward pass did: differentiating the biased estimator and dividing by
    the unbiased count is a classic off-by-one that only shows up as a slightly
    wrong gradient.
    """
    ta = as_tensor(a)
    axes = _normalise_axis(axis, ta.ndim)
    count = _reduced_count(ta.shape, axes)
    denom = count - ddof
    if denom <= 0:
        msg = f"variance over {count} element(s) with ddof={ddof} has no degrees of freedom left"
        raise ShapeError(msg)
    out: Array = np.var(ta.data, axis=axis, keepdims=keepdims, ddof=ddof)
    centred: Array = ta.data - np.mean(ta.data, axis=axis, keepdims=True)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        spread = _restore_reduced(g, axes, keepdims=keepdims)
        return ((2.0 / denom) * centred * spread,)

    return record(out, "var", (ta,), vjp)


def std(a: TensorLike, axis: Axis = None, *, keepdims: bool = False, ddof: int = 0) -> Tensor:
    """Standard deviation over `axis`.

    Composed as `sqrt(var(...))` rather than given its own node. That costs one
    extra tape node and one extra buffer per call, and buys the guarantee that
    `std` and `var` can never disagree about `ddof` or about their gradients -
    a guarantee worth more than the node, since the two formulas differing by a
    factor of `2*std` is precisely the mistake being avoided.

    At zero variance the gradient is `inf`, inherited from `sqrt`.
    """
    return sqrt(var(a, axis, keepdims=keepdims, ddof=ddof))


def logsumexp(a: TensorLike, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
    """`log(sum(exp(x)))`, computed without overflowing.

    The naive form overflows for inputs above about 88 in float32 and 709 in
    float64, which ordinary logits reach during a diverging run - the moment you
    most need the number to be readable. Subtracting the per-group maximum first
    makes the largest exponent exactly `exp(0) = 1`, and adding it back afterwards
    is exact in the log domain.

    The gradient is the softmax of the input, which is a probability
    distribution and therefore bounded no matter how extreme the logits were.

    >>> import nabla
    >>> x = nabla.tensor([1000.0, 1000.0], dtype="float64")
    >>> round(nabla.logsumexp(x).item() - 1000.0, 12)
    0.69314718056
    """
    ta = as_tensor(a)
    axes = _normalise_axis(axis, ta.ndim)
    shift: Array = np.max(ta.data, axis=axis, keepdims=True)
    # An all-`-inf` group would give `-inf - -inf = nan`; leaving the shift at 0
    # there keeps the arithmetic finite and the answer (-inf) correct.
    shift = np.where(np.isfinite(shift), shift, 0.0)
    stable: Array = np.exp(ta.data - shift)
    total: Array = np.sum(stable, axis=axis, keepdims=True)
    out_keepdims: Array = np.log(total) + shift
    out: Array = out_keepdims if keepdims else np.squeeze(out_keepdims, axis=axes)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        # The softmax weights are formed here rather than in the forward pass.
        # They are one division of an array that is already retained, and doing
        # it eagerly would compute `0 / 0` for an all-`-inf` group whose value
        # (`-inf`) is perfectly well defined and often never differentiated.
        spread = _restore_reduced(g, axes, keepdims=keepdims)
        return (stable / total * spread,)

    return record(out, "logsumexp", (ta,), vjp)
