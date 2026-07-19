"""Shape manipulation: rearranging values without changing any of them.

Every op here has the same character - the forward pass moves data and the
backward pass moves the cotangent back the way it came - but the reverse of a
shape op is not always another shape op:

* `reshape`, `transpose`, `squeeze` are bijections. Their vjps are the inverse
  rearrangement, and cost nothing beyond a view or a copy.
* `broadcast_to`, `getitem` with repeated indices, and `pad` are not injective in
  the forward direction. Their vjps have to *add* - a value read twice
  contributes twice - which is where `unbroadcast` and `np.add.at` come in.
* `concat`, `stack` and `split` change how many tensors there are, so their vjps
  slice or gather.

NumPy will return a view where it can; Nabla's tensors are read-only, so a view
into a frozen buffer is still frozen and safe to hand out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

from .._typing import Array, Axis, IndexLike, Shape, TensorLike
from ..errors import ShapeError
from ..tensor import Tensor, as_tensor, record, unbroadcast

__all__ = [
    "broadcast_to",
    "concat",
    "expand_dims",
    "getitem",
    "pad",
    "permute",
    "reshape",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "transpose",
]

#: `pad(x, 1)`, `pad(x, (1, 2))` and `pad(x, [(1, 2), (0, 0)])` are all accepted,
#: matching `np.pad`'s first three spellings.
PadWidth: TypeAlias = int | Sequence[int] | Sequence[Sequence[int]]


def _flatten_shape_args(shape: tuple[int | Sequence[int], ...]) -> Shape:
    """Accept both `reshape(x, 2, 3)` and `reshape(x, (2, 3))`."""
    if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
        return tuple(int(d) for d in shape[0])
    return tuple(int(d) for d in shape)  # type: ignore[arg-type]


def reshape(a: TensorLike, *shape: int | Sequence[int]) -> Tensor:
    """Reinterpret the values under a new shape. One dimension may be `-1`.

    >>> import nabla
    >>> x = nabla.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    >>> nabla.reshape(x, 3, 2).shape
    (3, 2)
    >>> nabla.reshape(x, -1).shape
    (6,)
    """
    ta = as_tensor(a)
    target = _flatten_shape_args(shape)
    out: Array = ta.data.reshape(target)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g.reshape(ta.shape),)

    return record(out, "reshape", (ta,), vjp)


def transpose(a: TensorLike, axes: Sequence[int] | None = None) -> Tensor:
    """Permute axes; with `axes=None`, reverse them all (NumPy's `.T`).

    The vjp is the inverse permutation, not the same one. They coincide for rank
    2 and for a full reversal, which is exactly why using the forward
    permutation in the backward pass survives every test written on matrices and
    then fails on the first rank-3 tensor.
    """
    ta = as_tensor(a)
    perm = tuple(range(ta.ndim)[::-1]) if axes is None else tuple(int(i) for i in axes)
    if len(perm) != ta.ndim:
        msg = f"transpose needs {ta.ndim} axes for a rank-{ta.ndim} tensor, got {len(perm)}"
        raise ShapeError(msg)
    out: Array = ta.data.transpose(perm)
    inverse = tuple(int(i) for i in np.argsort(perm))

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g.transpose(inverse),)

    return record(out, "transpose", (ta,), vjp)


def permute(a: TensorLike, *axes: int | Sequence[int]) -> Tensor:
    """Reorder axes explicitly: `permute(x, 2, 0, 1)`.

    The same operation as `transpose` with an argument; it exists because
    `transpose(x)` silently means something different for rank 3 than the
    two-axis swap most readers picture.
    """
    return transpose(a, _flatten_shape_args(axes))


def swapaxes(a: TensorLike, axis0: int, axis1: int) -> Tensor:
    """Exchange two axes, leaving the rest in place."""
    ta = as_tensor(a)
    perm = list(range(ta.ndim))
    try:
        perm[axis0], perm[axis1] = perm[axis1], perm[axis0]
    except IndexError as exc:
        msg = f"cannot swap axes {axis0} and {axis1} of a rank-{ta.ndim} tensor"
        raise ShapeError(msg) from exc
    return transpose(ta, perm)


def broadcast_to(a: TensorLike, shape: Shape) -> Tensor:
    """Expand to `shape` by NumPy's broadcasting rules.

    The forward pass allocates nothing - the result is a zero-stride view - and
    the backward pass is a pure `unbroadcast`. Made explicit as an op because
    doing it implicitly inside arithmetic is how the reduction gets forgotten.
    """
    ta = as_tensor(a)
    target = tuple(int(d) for d in shape)
    try:
        out: Array = np.broadcast_to(ta.data, target)
    except ValueError as exc:
        msg = f"cannot broadcast shape {ta.shape} to {target}"
        raise ShapeError(msg) from exc

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (unbroadcast(g, ta.shape),)

    return record(out, "broadcast_to", (ta,), vjp)


def expand_dims(a: TensorLike, axis: int) -> Tensor:
    """Insert a size-1 axis at `axis`."""
    ta = as_tensor(a)
    out: Array = np.expand_dims(ta.data, axis)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g.reshape(ta.shape),)

    return record(out, "expand_dims", (ta,), vjp)


def squeeze(a: TensorLike, axis: Axis = None) -> Tensor:
    """Drop size-1 axes; with `axis=None`, drop all of them."""
    ta = as_tensor(a)
    out: Array = np.squeeze(ta.data, axis=axis)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g.reshape(ta.shape),)

    return record(out, "squeeze", (ta,), vjp)


def concat(tensors: Sequence[TensorLike], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis.

    The vjp slices the cotangent back into the original pieces. Those slices are
    views, so the backward pass of a concatenation of `k` tensors allocates
    nothing at all.
    """
    parts = tuple(as_tensor(t) for t in tensors)
    if not parts:
        msg = "concat needs at least one tensor"
        raise ShapeError(msg)
    try:
        out: Array = np.concatenate([p.data for p in parts], axis=axis)
    except ValueError as exc:
        shapes = [p.shape for p in parts]
        msg = f"cannot concatenate shapes {shapes} along axis {axis}: {exc}"
        raise ShapeError(msg) from exc
    sizes = [p.shape[axis] for p in parts]
    offsets = np.cumsum([0, *sizes])

    def vjp(g: Array) -> tuple[Array | None, ...]:
        cuts = []
        for i in range(len(parts)):
            index: list[slice] = [slice(None)] * g.ndim
            index[axis] = slice(int(offsets[i]), int(offsets[i + 1]))
            cuts.append(g[tuple(index)])
        return tuple(cuts)

    return record(out, "concat", parts, vjp)


def stack(tensors: Sequence[TensorLike], axis: int = 0) -> Tensor:
    """Join tensors along a new axis. All shapes must match exactly."""
    parts = tuple(as_tensor(t) for t in tensors)
    if not parts:
        msg = "stack needs at least one tensor"
        raise ShapeError(msg)
    try:
        out: Array = np.stack([p.data for p in parts], axis=axis)
    except ValueError as exc:
        shapes = [p.shape for p in parts]
        msg = f"cannot stack shapes {shapes} along a new axis {axis}: {exc}"
        raise ShapeError(msg) from exc

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return tuple(np.take(g, i, axis=axis) for i in range(len(parts)))

    return record(out, "stack", parts, vjp)


def split(a: TensorLike, sections: int | Sequence[int], axis: int = 0) -> tuple[Tensor, ...]:
    """Cut a tensor into pieces along `axis`.

    `sections` is either a number of equal parts or the explicit size of each
    part.

    Each piece gets its own tape node, so backward through a `k`-way split
    allocates `k` zero buffers the size of the input and adds them. A
    multiple-output node would allocate one, but every other operation in the
    package is single-output, and a second node protocol is a lot of machinery
    to buy back one allocation in the one place it matters least.
    """
    ta = as_tensor(a)
    length = ta.shape[axis]
    if isinstance(sections, int):
        if sections <= 0 or length % sections != 0:
            msg = f"cannot split axis {axis} of length {length} into {sections} equal parts"
            raise ShapeError(msg)
        sizes = [length // sections] * sections
    else:
        sizes = [int(s) for s in sections]
        if sum(sizes) != length:
            msg = f"split sizes {sizes} do not add up to the axis length {length}"
            raise ShapeError(msg)

    pieces: list[Tensor] = []
    start = 0
    for size in sizes:
        pieces.append(_slice_axis(ta, axis, start, start + size))
        start += size
    return tuple(pieces)


def _slice_axis(ta: Tensor, axis: int, start: int, stop: int) -> Tensor:
    """One contiguous slice along a single axis, with a scatter vjp."""
    index: list[slice] = [slice(None)] * ta.ndim
    index[axis] = slice(start, stop)
    key = tuple(index)
    out: Array = ta.data[key]

    def vjp(g: Array) -> tuple[Array | None, ...]:
        # A plain assignment rather than `np.add.at`: a contiguous slice touches
        # every position at most once, so there is nothing to accumulate.
        scattered = np.zeros(ta.shape, dtype=g.dtype)
        scattered[key] = g
        return (scattered,)

    return record(out, "slice", (ta,), vjp)


def _unwrap_index(index: IndexLike) -> IndexLike:
    """Replace any tensor inside an index expression with its array.

    Index tensors are integer or boolean, so they never carry a gradient; the
    gradient flows to the tensor being indexed, not to the coordinates.
    """
    if isinstance(index, Tensor):
        return index.data
    if isinstance(index, tuple):
        return tuple(_unwrap_index(part) for part in index)
    return index


def getitem(a: TensorLike, index: IndexLike) -> Tensor:
    """NumPy indexing - ints, slices, ellipsis, index arrays, boolean masks.

    The vjp scatters the cotangent back into a zero buffer with `np.add.at`,
    which is unbuffered and therefore slow - roughly an order of magnitude
    slower than `buffer[index] = value` on large arrays. It is used anyway
    because `=` overwrites: for `x[[0, 0]]`, element 0 was read twice and must
    receive both cotangents, and the assignment form would silently keep only
    the last one. Correctness first; the fast path for the non-repeating case is
    `nabla.split`, which knows its slices are disjoint.

    >>> import nabla
    >>> x = nabla.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> x[[0, 0, 2]].sum().backward()
    >>> x.grad.tolist()  # index 0 was read twice
    [2.0, 0.0, 1.0]
    """
    ta = as_tensor(a)
    key = _unwrap_index(index)
    try:
        out: Array = ta.data[key]
    except IndexError as exc:
        msg = f"index {index!r} is not valid for a tensor of shape {ta.shape}: {exc}"
        raise ShapeError(msg) from exc
    # Indexing down to a single element gives a NumPy scalar, not an array;
    # asarray on something that is already an array is free.
    materialised: Array = np.asarray(out)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        scattered = np.zeros(ta.shape, dtype=g.dtype)
        np.add.at(scattered, key, g)
        return (scattered,)

    return record(materialised, "getitem", (ta,), vjp)


def _normalise_pad(pad_width: PadWidth, ndim: int) -> tuple[tuple[int, int], ...]:
    """Turn any of `np.pad`'s three width spellings into one pair per axis."""
    if isinstance(pad_width, (int, np.integer)):
        return ((int(pad_width), int(pad_width)),) * ndim
    # Deliberately untyped from here: the three accepted spellings differ in
    # nesting depth, and a precise type would be a union the reader has to
    # decode to learn something the error messages below already say.
    items: list[Any] = list(pad_width)
    if len(items) == 2 and all(isinstance(v, (int, np.integer)) for v in items):
        return ((int(items[0]), int(items[1])),) * ndim
    if len(items) != ndim:
        msg = f"pad_width has {len(items)} entries for a rank-{ndim} tensor"
        raise ShapeError(msg)
    pairs: list[tuple[int, int]] = []
    for entry in items:
        pair = tuple(int(v) for v in entry)
        if len(pair) != 2:
            msg = f"each pad_width entry must be (before, after), got {entry!r}"
            raise ShapeError(msg)
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def pad(a: TensorLike, pad_width: PadWidth, value: float = 0.0) -> Tensor:
    """Pad with a constant.

    The gradient of the padding is discarded, which is right: the padded
    positions are a constant, and a constant has no input to send a gradient
    back to. The vjp is therefore a slice of the interior - a view, no
    allocation.
    """
    ta = as_tensor(a)
    pairs = _normalise_pad(pad_width, ta.ndim)
    if any(before < 0 or after < 0 for before, after in pairs):
        msg = f"pad widths must be non-negative, got {pairs}"
        raise ShapeError(msg)
    out: Array = np.pad(ta.data, pairs, mode="constant", constant_values=value)
    interior = tuple(
        slice(before, before + dim) for (before, _), dim in zip(pairs, ta.shape, strict=True)
    )

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g[interior],)

    return record(out, "pad", (ta,), vjp)
