"""The tensor and the tape it records itself onto.

# Why reverse mode

A gradient can be computed in two directions. Forward mode propagates one
input's perturbation through the program and reads off the effect on every
output; it costs one pass per *input*. Reverse mode propagates one output's
sensitivity backwards and reads off the effect of every input; it costs one pass
per *output*.

Training a model is the extreme case of the second shape: millions of
parameters, one scalar loss. Forward mode would need one pass per parameter.
Reverse mode needs one. The price is memory - every intermediate value that some
backward function needs must survive until the backward pass runs - and that
price is why a training step's peak memory is dominated by activations rather
than by weights. Nabla pays it explicitly: each operation stores exactly the
arrays its vjp closure captured, and those arrays stay alive as long as the
output tensor does.

# The tape

There is no separate tape object. Each non-leaf tensor holds a `Node` naming the
operation that produced it, the input tensors, and a closure computing the
vector-Jacobian product. The graph is therefore the tensors themselves, which
means it is freed by refcounting the moment the loss goes out of scope, and it
means there is no global mutable state to reset between iterations or to get
confused by two models training in the same process.

`backward()` topologically sorts that graph (see `nabla.grad`) rather than
relying on creation order. Creation order happens to be a valid topological
order for straight-line code, and relying on it is the traditional way for a
hand-rolled engine to be subtly wrong the first time somebody reuses a tensor
computed earlier.

# Buffers are immutable

`Tensor.data` is a read-only NumPy array. An operation's vjp closes over the
forward values it needs; if a caller mutated one of those buffers afterwards,
`backward()` would compute the exact gradient of a function that was never
evaluated, and it would do it silently. Marking buffers read-only turns that
into a NumPy error at the point of the mutation. The cost is one array copy when
a tensor is built from a caller's array, and one allocation per `assign_`;
`detach()` and `numpy(copy=False)` are free precisely because nothing can write
through them.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import numpy as np

from ._typing import (
    DEFAULT_FLOAT_DTYPE,
    Array,
    BackwardFn,
    DTypeLike,
    IndexLike,
    Shape,
    TensorLike,
)
from .errors import DTypeError, InPlaceOperationError, ShapeError

if TYPE_CHECKING:  # pragma: no cover
    from ._typing import Axis

__all__ = [
    "Node",
    "Tensor",
    "arange",
    "eye",
    "from_numpy",
    "full",
    "is_grad_enabled",
    "linspace",
    "no_grad",
    "ones",
    "ones_like",
    "record",
    "tensor",
    "unbroadcast",
    "zeros",
    "zeros_like",
]


# --------------------------------------------------------------------------- #
# grad mode
# --------------------------------------------------------------------------- #


class _GradMode(threading.local):
    """Per-thread grad-mode flag.

    Thread-local rather than a plain module global: two threads evaluating two
    models is a reasonable thing to do, and a global flag would let one thread's
    `no_grad()` silently disable the other's training step.
    """

    enabled: bool = True


_GRAD_MODE = _GradMode()


def is_grad_enabled() -> bool:
    """True if operations are currently being recorded on the tape.

    >>> import nabla
    >>> nabla.is_grad_enabled()
    True
    >>> with nabla.no_grad():
    ...     nabla.is_grad_enabled()
    False
    """
    return _GRAD_MODE.enabled


class no_grad:  # noqa: N801 - reads as a verb at the call site, as in `with no_grad():`
    """Suspend tape recording for the duration of the block.

    Operations still compute their forward values; they simply do not attach a
    `Node`, so their results are leaves and nothing downstream can be
    differentiated. This is what evaluation and parameter updates want: without
    it, a validation loop would build a graph nobody ever calls `backward()` on
    and hold every activation alive until the next garbage collection.

    Re-entrant, and restores the previous state rather than unconditionally
    enabling: nesting `no_grad()` inside `no_grad()` must not re-enable
    recording on the way out.

    >>> import nabla
    >>> x = nabla.tensor([1.0, 2.0], requires_grad=True)
    >>> with nabla.no_grad():
    ...     y = x * 2
    >>> y.requires_grad, y.node is None
    (False, True)
    """

    __slots__ = ("_previous",)

    def __init__(self) -> None:
        self._previous: bool = True

    def __enter__(self) -> no_grad:
        self._previous = _GRAD_MODE.enabled
        _GRAD_MODE.enabled = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _GRAD_MODE.enabled = self._previous


# --------------------------------------------------------------------------- #
# tape nodes
# --------------------------------------------------------------------------- #


class Node:
    """One recorded operation: what produced a tensor, and how to differentiate it.

    `backward_fn` receives the cotangent of the operation's output and returns
    one cotangent per entry of `parents`, in order. It may return `None` for a
    parent whose gradient it chose not to compute; see `nabla.ops.basic.pow`,
    where the derivative with respect to the exponent is `nan` for a negative
    base and computing it unconditionally would poison a perfectly well-defined
    backward pass.
    """

    __slots__ = ("backward_fn", "op", "parents")

    def __init__(self, op: str, parents: tuple[Tensor, ...], backward_fn: BackwardFn) -> None:
        self.op = op
        self.parents = parents
        self.backward_fn = backward_fn

    def __repr__(self) -> str:
        shapes = ", ".join(str(p.shape) for p in self.parents)
        return f"Node({self.op}, parents=[{shapes}])"


# --------------------------------------------------------------------------- #
# broadcasting
# --------------------------------------------------------------------------- #


def unbroadcast(grad: Array, shape: Shape) -> Array:
    """Sum `grad` back down to `shape`, undoing a forward broadcast.

    Broadcasting is a *copy*: `x` of shape `(3,)` used against `(4, 3)` behaves
    as four copies of `x`. Four copies means four contributions to the loss, so
    the gradient of the original `x` is the sum of the four rows. Getting this
    wrong is the single most common defect in hand-written autodiff engines,
    because the forward pass keeps working and only the gradient is wrong - by a
    factor equal to the broadcast width, which looks exactly like a badly tuned
    learning rate.

    Two cases, applied in order:

    1. `grad` has more axes than `shape`. Those leading axes did not exist in the
       input at all, so they are summed away entirely.
    2. `shape` has a 1 where `grad` has more. That axis was stretched, so it is
       summed with `keepdims=True` to leave the 1 in place.

    A dimension of size 0 needs no special case: NumPy broadcasts 1 against 0 to
    0, and summing an empty axis with `keepdims=True` yields the 0.0 that the
    empty product should be.

    Cost: at most two `sum` calls, each allocating its own result. No copy is
    made when the shapes already agree, which is the common path.

    >>> import numpy as np
    >>> from nabla.tensor import unbroadcast
    >>> unbroadcast(np.ones((4, 3)), (3,)).tolist()
    [4.0, 4.0, 4.0]
    >>> unbroadcast(np.ones((4, 3)), (1, 3)).tolist()
    [[4.0, 4.0, 4.0]]
    >>> unbroadcast(np.ones((2, 3)), ()).tolist()
    6.0
    """
    if grad.shape == shape:
        return grad

    extra = grad.ndim - len(shape)
    if extra < 0:
        msg = (
            f"cannot reduce a cotangent of shape {grad.shape} to {shape}: "
            "the target has more axes than the gradient"
        )
        raise ShapeError(msg)

    reduced: Array = grad
    if extra > 0:
        reduced = reduced.sum(axis=tuple(range(extra)))

    stretched = tuple(i for i, dim in enumerate(shape) if dim == 1 and reduced.shape[i] != 1)
    if stretched:
        reduced = reduced.sum(axis=stretched, keepdims=True)

    if reduced.shape != shape:
        msg = (
            f"cannot reduce a cotangent of shape {grad.shape} to {shape}: "
            f"got {reduced.shape}, which means the forward pass did not broadcast"
        )
        raise ShapeError(msg)
    return reduced


# --------------------------------------------------------------------------- #
# tensor
# --------------------------------------------------------------------------- #


def _coerce(data: TensorLike, dtype: DTypeLike | None) -> Array:
    """Turn arbitrary input into an owned, validated array.

    dtype policy, and the reasoning for it:

    * A Python scalar or nested list carries no dtype intent - NumPy would make
      it float64 - so it becomes float32. Neural networks are trained in float32
      almost universally, and defaulting to float64 would double every buffer to
      buy precision that stochastic gradient descent does not use.
    * An ndarray *does* carry intent. `np.zeros(3, np.float64)` stays float64.
      Silently halving the precision of an array a caller explicitly widened is
      the kind of surprise this package exists to avoid.
    * Integers and booleans are preserved as they are. They are legal tensor
      contents (indices, masks, labels) but never differentiable: see
      `Tensor.requires_grad`.
    """
    if isinstance(data, Tensor):
        base: Array = data.data
    elif isinstance(data, np.ndarray):
        base = data
    else:
        base = np.asarray(data)
        if dtype is None and base.dtype == np.float64:
            dtype = DEFAULT_FLOAT_DTYPE

    if base.dtype.kind not in "fiub":
        msg = f"cannot build a tensor from data of dtype {base.dtype!r}: expected a numeric array"
        raise DTypeError(msg)

    # copy=True unconditionally: the tensor freezes what it owns, and freezing a
    # caller's array in place would be an unpleasant surprise.
    owned: Array = np.array(base, dtype=dtype, copy=True)
    if owned.dtype.kind not in "fiub":
        msg = f"cannot build a tensor of dtype {owned.dtype!r}: expected a numeric array"
        raise DTypeError(msg)
    owned.setflags(write=False)
    return owned


def _is_float(dtype: np.dtype[Any]) -> bool:
    return bool(np.issubdtype(dtype, np.floating))


class Tensor:
    """An n-dimensional array that remembers how it was computed.

    A tensor is a leaf if nothing on the tape produced it - a parameter, an
    input batch, a constant. Leaves accumulate into `.grad`; interior tensors do
    not, unless asked to with `retain_grad()`, because keeping every
    intermediate gradient alive would multiply the memory cost of a backward
    pass by the depth of the graph for information almost nobody reads.

    >>> import nabla
    >>> x = nabla.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> y = (x * x).sum()
    >>> y.backward()
    >>> x.grad.tolist()  # d(sum x^2)/dx = 2x
    [[2.0, 4.0], [6.0, 8.0]]
    """

    __slots__ = ("_data", "_node", "_requires_grad", "_retains_grad", "grad")

    #: Opting out of NumPy's ufunc protocol. Without this, `np.ones(3) * tensor`
    #: would let ndarray win the dispatch, convert the tensor through
    #: `__array__` and return a plain ndarray - dropping the tape silently, and
    #: only for operands written in that order. Setting it to None makes NumPy
    #: return NotImplemented so Python falls back to `Tensor.__rmul__`.
    __array_ufunc__: ClassVar[None] = None

    def __init__(
        self,
        data: TensorLike,
        *,
        requires_grad: bool = False,
        dtype: DTypeLike | None = None,
    ) -> None:
        self._data: Array = _coerce(data, dtype)
        self._node: Node | None = None
        self._retains_grad: bool = False
        #: Accumulated gradient, or None if none has been computed. `None` and a
        #: buffer of zeros mean different things: the first says nothing reached
        #: this tensor, which is how a detached parameter announces itself.
        self.grad: Array | None = None
        self._requires_grad = False
        if requires_grad:
            self.requires_grad = True

    # -- construction ------------------------------------------------------- #

    @classmethod
    def _owning(cls, array: Array) -> Tensor:
        """Wrap an array this package just allocated, without copying it.

        Only safe for arrays no other code holds a reference to, which is why it
        is private: every op result comes through here, and every user-facing
        entry point goes through `__init__` and copies.
        """
        out = cls.__new__(cls)
        array.setflags(write=False)
        out._data = array
        out._node = None
        out._requires_grad = False
        out._retains_grad = False
        out.grad = None
        return out

    # -- plain properties --------------------------------------------------- #

    @property
    def data(self) -> Array:
        """The underlying array. Read-only; see the module docstring."""
        return self._data

    @property
    def shape(self) -> Shape:
        """Tuple of dimensions. `()` for a 0-d tensor."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """The element dtype."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Number of elements."""
        return self._data.size

    @property
    def node(self) -> Node | None:
        """The tape node that produced this tensor, or None if it is a leaf.

        Exposed rather than private because "did this build a graph?" is a
        question worth being able to answer in a test, and answering it by
        checking whether a gradient came out is exactly the kind of indirect
        assertion that passes for the wrong reason.
        """
        return self._node

    @property
    def is_leaf(self) -> bool:
        """True if no recorded operation produced this tensor."""
        return self._node is None

    @property
    def requires_grad(self) -> bool:
        """Whether gradients flow to this tensor.

        Only floating tensors may set it. An integer gradient is not a rounding
        problem, it is a category error: the derivative of a function of a
        discrete variable does not exist, and the two candidate behaviours
        (truncate to int, or silently promote the tensor to float) are both worse
        than refusing.
        """
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if value and not _is_float(self.dtype):
            msg = (
                f"a tensor of dtype {self.dtype} cannot require grad; "
                "only floating-point tensors are differentiable"
            )
            raise DTypeError(msg)
        if self._node is not None:
            msg = (
                "cannot change requires_grad on a non-leaf tensor: it was produced by "
                f"{self._node.op}, and detaching it here would silently orphan that "
                "part of the graph. Use detach() to get an independent leaf."
            )
            raise InPlaceOperationError(msg)
        self._requires_grad = bool(value)

    @property
    def T(self) -> Tensor:  # noqa: N802 - matches numpy's attribute name
        """Transpose, reversing every axis (NumPy's rule, not just a 2-d swap).

        For rank 3 and above this is rarely what a reader expects; prefer
        `nabla.permute` or `nabla.swapaxes`, which say which axes they mean.
        """
        from .ops.shape import transpose

        return transpose(self)

    # -- interop ------------------------------------------------------------ #

    def numpy(self, *, copy: bool = True) -> Array:
        """Return the values as a NumPy array.

        Copies by default so the result is writable and independent. Pass
        `copy=False` for the tensor's own read-only buffer when the caller only
        reads - that is free, and NumPy will refuse any write through it.
        """
        if not copy:
            return self._data
        out: Array = np.array(self._data, copy=True)
        return out

    def item(self) -> float:
        """The single value of a one-element tensor, as a Python float.

        >>> import nabla
        >>> nabla.tensor([[3.5]]).item()
        3.5
        """
        if self._data.size != 1:
            msg = f"item() needs a tensor with exactly one element, got shape {self.shape}"
            raise ShapeError(msg)
        return float(self._data.reshape(()).item())

    def __array__(self, dtype: DTypeLike = None, copy: bool | None = None) -> Array:
        """NumPy conversion protocol, so `np.asarray(tensor)` works."""
        if copy is False:
            if dtype is not None and np.dtype(dtype) != self._data.dtype:
                msg = "cannot convert dtype without copying"
                raise ValueError(msg)
            return self._data
        out: Array = np.array(self._data, dtype=dtype, copy=True)
        return out

    def __repr__(self) -> str:
        body = np.array2string(self._data, separator=", ", prefix="Tensor(")
        flags = f", dtype={self.dtype}"
        if self._requires_grad:
            flags += ", requires_grad=True"
        if self._node is not None:
            flags += f", node={self._node.op}"
        return f"Tensor({body}{flags})"

    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of a 0-d tensor"
            raise ShapeError(msg)
        return self.shape[0]

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over the leading axis. Each slice keeps its gradient path."""
        for i in range(len(self)):
            yield self[i]

    def __bool__(self) -> bool:
        if self._data.size != 1:
            msg = (
                f"the truth value of a tensor with shape {self.shape} is ambiguous; reduce it first"
            )
            raise ShapeError(msg)
        return bool(self._data.reshape(()).item())

    def __float__(self) -> float:
        return self.item()

    def __int__(self) -> int:
        return int(self.item())

    # Identity hashing and identity equality are kept deliberately. The tape
    # keys tensors by identity in sets and dicts during the topological sort,
    # and an elementwise `__eq__` returning a tensor would make `x in [x]` and
    # `{x: g}` behave in ways no reader would predict. Elementwise comparison is
    # available on `.data`.
    __hash__ = object.__hash__

    # -- autograd ----------------------------------------------------------- #

    def backward(self, gradient: TensorLike | None = None) -> None:
        """Accumulate gradients into every leaf this tensor depends on.

        See `nabla.grad.backward`, which this forwards to.
        """
        from .grad import backward

        backward(self, gradient)

    def zero_grad(self) -> None:
        """Drop the accumulated gradient.

        Sets `.grad` back to `None` rather than to zeros, because the two states
        answer different questions: `None` means nothing reached this tensor at
        all, which is how a parameter that was accidentally detached from the
        loss makes itself visible. Zeroing in place would hide that.
        """
        self.grad = None

    def retain_grad(self) -> None:
        """Ask an interior tensor to keep its gradient in `.grad`.

        A no-op on leaves, which always keep theirs. On an interior tensor this
        costs one buffer the size of the tensor, held until the next
        `zero_grad()`.
        """
        self._retains_grad = True

    @property
    def retains_grad(self) -> bool:
        """True if `.grad` will be populated for this tensor by a backward pass."""
        return self.is_leaf or self._retains_grad

    def detach(self) -> Tensor:
        """A tensor with the same values, cut off from the tape.

        Shares the buffer instead of copying it, which is safe only because
        buffers are read-only. The result is a leaf with `requires_grad=False`;
        gradients stop here.
        """
        return Tensor._owning(self._data)

    # -- the in-place policy ------------------------------------------------ #

    def assign_(self, value: TensorLike) -> None:
        """Replace this tensor's values, outside the tape.

        The one sanctioned mutation, and the one an optimiser needs. Only legal
        on a leaf: an interior tensor's value is a claim about a computation
        that already happened, and changing it would make the graph describe a
        program that never ran.

        It rebinds the buffer rather than writing into it, so any vjp closure
        that captured the old values still sees the values it was given. That
        costs one allocation per call; the alternative - writing in place - is
        the bug this package refuses to make possible.

        Shape and dtype must match, because a parameter that changes shape
        mid-run invalidates every optimiser state keyed to it.

        >>> import nabla
        >>> p = nabla.tensor([1.0, 2.0], requires_grad=True)
        >>> p.assign_(p.data - 0.5)
        >>> p.numpy().tolist()
        [0.5, 1.5]
        """
        if self._node is not None:
            msg = (
                f"cannot assign to a non-leaf tensor produced by {self._node.op}: "
                "the tape holds its value and the gradients would no longer describe "
                "the computation that ran"
            )
            raise InPlaceOperationError(msg)
        replacement = _coerce(value, None)
        if replacement.shape != self.shape:
            msg = f"assign_ expects shape {self.shape}, got {replacement.shape}"
            raise ShapeError(msg)
        if replacement.dtype != self.dtype:
            replacement = np.array(replacement, dtype=self.dtype, copy=True)
            replacement.setflags(write=False)
        self._data = replacement

    def _no_in_place(self, op: str) -> NoReturn:
        msg = (
            f"in-place {op} is not supported. A tensor's buffer is captured by the "
            "vjp closures of every operation that consumed it, so mutating it would "
            "silently change the function backward() differentiates. Write `x = x "
            f"{op} y` for a new tensor, or Tensor.assign_ to update a leaf in place."
        )
        raise InPlaceOperationError(msg)

    def __iadd__(self, other: TensorLike) -> NoReturn:
        self._no_in_place("+=")

    def __isub__(self, other: TensorLike) -> NoReturn:
        self._no_in_place("-=")

    def __imul__(self, other: TensorLike) -> NoReturn:
        self._no_in_place("*=")

    def __itruediv__(self, other: TensorLike) -> NoReturn:
        self._no_in_place("/=")

    def __ipow__(self, other: TensorLike) -> NoReturn:
        self._no_in_place("**=")

    def __imatmul__(self, other: TensorLike) -> NoReturn:
        self._no_in_place("@=")

    def __setitem__(self, index: IndexLike, value: TensorLike) -> NoReturn:
        msg = (
            "tensors do not support item assignment. Build the result functionally "
            "(nabla.concat, nabla.pad, arithmetic with a mask) so the tape can see it, "
            "or use Tensor.assign_ to replace a leaf's values wholesale."
        )
        raise InPlaceOperationError(msg)

    # -- operators ---------------------------------------------------------- #
    #
    # Each of these imports its op inside the function body. `nabla.ops` needs
    # `Tensor` at import time to build results, and `Tensor` needs `nabla.ops`
    # to implement `+`; one of the two directions has to be deferred. Doing it
    # here costs a `sys.modules` dictionary lookup per operator call, which is
    # noise next to the NumPy work that follows, and it keeps the ops package
    # importable on its own.

    def __add__(self, other: TensorLike) -> Tensor:
        from .ops.basic import add

        return add(self, other)

    def __radd__(self, other: TensorLike) -> Tensor:
        from .ops.basic import add

        return add(other, self)

    def __sub__(self, other: TensorLike) -> Tensor:
        from .ops.basic import sub

        return sub(self, other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        from .ops.basic import sub

        return sub(other, self)

    def __mul__(self, other: TensorLike) -> Tensor:
        from .ops.basic import mul

        return mul(self, other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        from .ops.basic import mul

        return mul(other, self)

    def __truediv__(self, other: TensorLike) -> Tensor:
        from .ops.basic import div

        return div(self, other)

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        from .ops.basic import div

        return div(other, self)

    def __pow__(self, other: TensorLike) -> Tensor:
        from .ops.basic import pow

        return pow(self, other)

    def __rpow__(self, other: TensorLike) -> Tensor:
        from .ops.basic import pow

        return pow(other, self)

    def __matmul__(self, other: TensorLike) -> Tensor:
        from .ops.linalg import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        from .ops.linalg import matmul

        return matmul(other, self)

    def __neg__(self) -> Tensor:
        from .ops.basic import neg

        return neg(self)

    def __pos__(self) -> Tensor:
        return self

    def __abs__(self) -> Tensor:
        from .ops.basic import abs

        return abs(self)

    def __getitem__(self, index: IndexLike) -> Tensor:
        from .ops.shape import getitem

        return getitem(self, index)

    # -- method forms of the common ops ------------------------------------- #
    #
    # Only the ones that read better as methods (`loss.sum()`, `x.reshape(...)`)
    # are mirrored here. The full set lives in `nabla.ops`; duplicating all of it
    # would double the surface area that has to stay consistent.

    def sum(self, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
        """Sum over `axis`. See `nabla.ops.reduce.sum`."""
        from .ops.reduce import sum as sum_op

        return sum_op(self, axis, keepdims=keepdims)

    def mean(self, axis: Axis = None, *, keepdims: bool = False) -> Tensor:
        """Arithmetic mean over `axis`. See `nabla.ops.reduce.mean`."""
        from .ops.reduce import mean

        return mean(self, axis, keepdims=keepdims)

    def reshape(self, *shape: int | Sequence[int]) -> Tensor:
        """Reshape. See `nabla.ops.shape.reshape`."""
        from .ops.shape import reshape

        return reshape(self, *shape)

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        """Transpose. See `nabla.ops.shape.transpose`."""
        from .ops.shape import transpose

        return transpose(self, axes)


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #


def record(
    data: Array,
    op: str,
    parents: tuple[Tensor, ...],
    backward_fn: BackwardFn,
) -> Tensor:
    """Wrap the result of a forward computation and attach it to the tape.

    This is the extension point: an operation is a forward NumPy expression plus
    a closure that maps the output cotangent to input cotangents. Nothing else
    in the package needs to know the op exists.

    No node is attached when recording is off, when no input requires a
    gradient, or when the result is not floating point - the third case covers
    comparisons and integer arithmetic, where a gradient is meaningless and
    attaching a node would only keep memory alive.

    >>> import numpy as np
    >>> import nabla
    >>> from nabla.tensor import record
    >>> x = nabla.tensor([3.0], requires_grad=True)
    >>> y = record(np.sin(x.data), "sin", (x,), lambda g: (g * np.cos(x.data),))
    >>> y.backward()
    >>> bool(np.allclose(x.grad, np.cos(x.data)))
    True
    """
    out = Tensor._owning(data)
    if not _GRAD_MODE.enabled or not _is_float(out.dtype):
        return out
    if not any(p.requires_grad for p in parents):
        return out
    out._requires_grad = True
    out._node = Node(op, parents, backward_fn)
    return out


def as_tensor(value: TensorLike, *, dtype: DTypeLike | None = None) -> Tensor:
    """Return `value` unchanged if it is already a tensor, otherwise wrap it.

    The identity case matters: wrapping would copy the buffer and, worse, would
    cut the tape, so `x + y` would stop differentiating with respect to `y`.
    """
    if isinstance(value, Tensor) and (dtype is None or value.dtype == np.dtype(dtype)):
        return value
    return Tensor(value, dtype=dtype)


def _is_python_scalar(value: TensorLike) -> bool:
    """True for a bare Python number, which carries no dtype of its own."""
    return isinstance(value, (bool, int, float)) and not isinstance(value, np.generic)


def _weak_dtype(value: TensorLike, other: np.dtype[Any]) -> DTypeLike | None:
    """The dtype a Python scalar should adopt from the tensor it is used with.

    `None` means "no sensible adoption; use the ordinary rules", which happens
    when the adoption would lose information - a Python float alongside an
    integer tensor, for instance.
    """
    if other.kind == "f":
        return other
    if isinstance(value, bool):
        return other if other.kind == "b" else None
    if isinstance(value, int) and other.kind in "iu":
        return other
    return None


def as_operands(a: TensorLike, b: TensorLike) -> tuple[Tensor, Tensor]:
    """Prepare the two arguments of a binary op, keeping Python scalars weak.

    A Python `int` or `float` has no dtype; NumPy calls it *weakly typed* (NEP
    50) and lets it take the dtype of whatever it is combined with. Wrapping it
    in a tensor first would give it one - `1` would become an int64 array - and
    `float32_tensor + 1` would come back as float64, quietly doubling the memory
    of every model that wrote `x + 1` anywhere. Adopting the other operand's
    dtype reproduces NumPy's own behaviour, deliberately rather than by accident.

    A scalar that cannot be represented in the other operand's dtype - a float
    alongside an integer tensor - falls back to the ordinary rules, so nothing
    is silently truncated.
    """
    a_scalar, b_scalar = _is_python_scalar(a), _is_python_scalar(b)
    if a_scalar == b_scalar:
        return as_tensor(a), as_tensor(b)
    if b_scalar:
        ta = as_tensor(a)
        return ta, Tensor(b, dtype=_weak_dtype(b, ta.dtype))
    tb = as_tensor(b)
    return Tensor(a, dtype=_weak_dtype(a, tb.dtype)), tb


# --------------------------------------------------------------------------- #
# creation
# --------------------------------------------------------------------------- #


def tensor(
    data: TensorLike,
    *,
    requires_grad: bool = False,
    dtype: DTypeLike | None = None,
) -> Tensor:
    """Build a tensor from data. The ordinary entry point.

    >>> import nabla
    >>> x = nabla.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> x.shape, str(x.dtype), x.requires_grad
    ((3,), 'float32', True)
    """
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def from_numpy(array: Array, *, requires_grad: bool = False) -> Tensor:
    """Build a tensor from an ndarray, preserving its dtype.

    Copies: the tensor owns a frozen buffer, and the caller keeps a writable one.
    """
    return Tensor(array, requires_grad=requires_grad)


def zeros(
    *shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Zero-filled tensor. Accepts `zeros(2, 3)` or `zeros((2, 3))`."""
    return _created(np.zeros(_normalise_shape(shape), dtype=_creation_dtype(dtype)), requires_grad)


def ones(
    *shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """One-filled tensor. Accepts `ones(2, 3)` or `ones((2, 3))`."""
    return _created(np.ones(_normalise_shape(shape), dtype=_creation_dtype(dtype)), requires_grad)


def full(
    shape: int | Sequence[int],
    value: float,
    *,
    dtype: DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Tensor filled with `value`."""
    resolved = _creation_dtype(dtype)
    return _created(np.full(_normalise_shape((shape,)), value, dtype=resolved), requires_grad)


def zeros_like(other: Tensor, *, requires_grad: bool = False) -> Tensor:
    """Zeros with the shape and dtype of `other`."""
    return _created(np.zeros_like(other.data), requires_grad)


def ones_like(other: Tensor, *, requires_grad: bool = False) -> Tensor:
    """Ones with the shape and dtype of `other`."""
    return _created(np.ones_like(other.data), requires_grad)


def arange(
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    *,
    dtype: DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Evenly spaced values, NumPy's `arange` semantics."""
    resolved = _creation_dtype(dtype)
    array = (
        np.arange(start, stop, step, dtype=resolved)
        if stop is not None
        else np.arange(start, dtype=resolved)
    )
    return _created(array, requires_grad)


def linspace(
    start: float,
    stop: float,
    num: int = 50,
    *,
    dtype: DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """`num` evenly spaced values from `start` to `stop`, inclusive."""
    array = np.linspace(start, stop, num, dtype=_creation_dtype(dtype))
    return _created(array, requires_grad)


def eye(n: int, m: int | None = None, *, dtype: DTypeLike | None = None) -> Tensor:
    """Identity matrix, or the `n x m` rectangular version of one."""
    return _created(np.eye(n, m, dtype=_creation_dtype(dtype)), requires_grad=False)


def _creation_dtype(dtype: DTypeLike | None) -> np.dtype[Any]:
    """float32 unless asked otherwise.

    Creation functions have no input array to take a dtype from, so they cannot
    apply the rule in `_coerce`; they take the package default instead.
    """
    resolved: np.dtype[Any] = np.dtype(DEFAULT_FLOAT_DTYPE if dtype is None else dtype)
    return resolved


def _created(array: Array, requires_grad: bool) -> Tensor:
    """Take ownership of a freshly allocated array and set its grad flag.

    Goes through the `requires_grad` property rather than the private field, so
    `zeros(3, dtype=int, requires_grad=True)` is refused here for the same
    reason and with the same message as everywhere else.
    """
    out = Tensor._owning(array)
    out.requires_grad = requires_grad
    return out


def _normalise_shape(shape: tuple[int | Sequence[int], ...]) -> Shape:
    """Accept both `f(2, 3)` and `f((2, 3))`, the way NumPy's creation functions do."""
    if len(shape) == 1 and not isinstance(shape[0], int):
        return tuple(int(d) for d in shape[0])
    return tuple(int(d) for d in shape)  # type: ignore[arg-type]
