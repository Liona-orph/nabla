"""Matrix products.

`matmul` is the only primitive here; `dot`, `outer` and `trace` are either thin
wrappers around it or compositions of ops that already have gradients, so there
is one contraction whose vjp has to be right rather than four.

There is no `einsum`. An autodiff `einsum` means parsing the subscript grammar
and, for the backward pass, rewriting each term's subscripts into the
contraction that produces its cotangent - a few hundred lines of index algebra
whose failure mode is a silently transposed gradient. Everything the rest of
this package needs is `matmul` plus `permute` and `sum`, both of which are
already tested against numerical differentiation. If a caller needs a
contraction badly enough, composing it from those three is legible; adding a
subscript parser is not.
"""

from __future__ import annotations

import numpy as np

from .._typing import Array, TensorLike
from ..errors import ShapeError
from ..tensor import Tensor, as_tensor, record, unbroadcast
from .shape import reshape

__all__ = ["dot", "matmul", "outer", "trace"]


def _swap_last_two(x: Array) -> Array:
    return np.swapaxes(x, -1, -2)


def matmul(a: TensorLike, b: TensorLike) -> Tensor:
    """Matrix product with NumPy's `@` semantics.

    That includes the two rules that make the vjp fiddly:

    * A 1-d operand is promoted for the duration of the product - prepended as a
      row on the left, appended as a column on the right - and the promoted axis
      is removed from the result. The backward pass has to reinstate it before
      it can use the matrix formulas, and remove it again afterwards.
    * Leading axes broadcast. `(8, 4, 3) @ (3, 5)` treats the right operand as
      eight copies, so its gradient is the sum over those eight - the same
      `unbroadcast` every elementwise op uses, which is why this one gets no
      special-purpose reduction of its own.

    The gradients themselves are the standard pair, `dA = G B^T` and
    `dB = A^T G`, and their cost is two more matrix products of the same size as
    the forward one: a backward pass costs about twice a forward pass, which is
    the usual rule of thumb for the whole framework.

    >>> import nabla
    >>> a = nabla.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> b = nabla.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    >>> (a @ b).sum().backward()
    >>> a.grad.tolist()  # dA = G B^T, with G all ones
    [[1.0, 1.0], [1.0, 1.0]]
    """
    ta, tb = as_tensor(a), as_tensor(b)
    if ta.ndim == 0 or tb.ndim == 0:
        msg = "matmul needs tensors of rank 1 or more; use `*` to scale by a scalar"
        raise ShapeError(msg)
    try:
        out: Array = np.matmul(ta.data, tb.data)
    except ValueError as exc:
        msg = f"cannot matmul shapes {ta.shape} and {tb.shape}: {exc}"
        raise ShapeError(msg) from exc

    # The shapes the operands and the output take once 1-d operands have been
    # promoted to matrices. Computed once here rather than per backward pass.
    a2_shape = ta.shape if ta.ndim > 1 else (1, ta.shape[0])
    b2_shape = tb.shape if tb.ndim > 1 else (tb.shape[0], 1)
    # np.matmul drops one trailing axis per 1-d operand; whatever is left in
    # front of the remaining ones is the broadcast batch shape.
    dropped = int(ta.ndim == 1) + int(tb.ndim == 1)
    batch_shape = out.shape[: out.ndim - (2 - dropped)]
    out2_shape = (*batch_shape, a2_shape[-2], b2_shape[-1])

    def vjp(g: Array) -> tuple[Array | None, ...]:
        g2 = g.reshape(out2_shape)
        a2 = ta.data.reshape(a2_shape) if ta.ndim == 1 else ta.data
        b2 = tb.data.reshape(b2_shape) if tb.ndim == 1 else tb.data
        grad_a = unbroadcast(np.matmul(g2, _swap_last_two(b2)), a2_shape).reshape(ta.shape)
        grad_b = unbroadcast(np.matmul(_swap_last_two(a2), g2), b2_shape).reshape(tb.shape)
        return grad_a, grad_b

    return record(out, "matmul", (ta, tb), vjp)


def dot(a: TensorLike, b: TensorLike) -> Tensor:
    """Inner product of two vectors, returning a 0-d tensor.

    A named case of `matmul` rather than a separate op: NumPy's `dot` means four
    different things depending on rank, and only the vector case is unambiguous
    enough to be worth a name.
    """
    ta, tb = as_tensor(a), as_tensor(b)
    if ta.ndim != 1 or tb.ndim != 1:
        msg = f"dot is for vectors; got ranks {ta.ndim} and {tb.ndim}. Use matmul."
        raise ShapeError(msg)
    return matmul(ta, tb)


def outer(a: TensorLike, b: TensorLike) -> Tensor:
    """Outer product of two vectors.

    Composed from `reshape` and broadcast multiplication, so its gradient is
    whatever those two already do. Two tape nodes instead of one, in exchange
    for no new vjp to get wrong.
    """
    ta, tb = as_tensor(a), as_tensor(b)
    if ta.ndim != 1 or tb.ndim != 1:
        msg = f"outer is for vectors; got ranks {ta.ndim} and {tb.ndim}"
        raise ShapeError(msg)
    return reshape(ta, ta.shape[0], 1) * reshape(tb, 1, tb.shape[0])


def trace(a: TensorLike) -> Tensor:
    """Sum of the diagonal of a matrix.

    The gradient is the identity scaled by the cotangent, since each diagonal
    element enters the sum exactly once and nothing else enters at all.
    """
    ta = as_tensor(a)
    if ta.ndim != 2:
        msg = f"trace needs a matrix, got a rank-{ta.ndim} tensor"
        raise ShapeError(msg)
    out: Array = np.trace(ta.data)
    rows, cols = ta.shape

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (np.eye(rows, cols, dtype=g.dtype) * g,)

    return record(out, "trace", (ta,), vjp)
