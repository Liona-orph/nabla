"""Elementwise arithmetic.

Every binary op here broadcasts, and every one of them therefore ends its vjp
with `unbroadcast`. That is not incidental: an elementwise op's Jacobian is
diagonal, so the whole of its backward pass is "multiply by the local
derivative, then sum away whatever broadcasting duplicated". The first half is
op-specific and one line; the second half is identical everywhere and lives in
exactly one function, because writing it out per-op is how these engines end up
with one op that reduces over the wrong axis.

This module deliberately shadows the builtins `abs` and `pow` so the API reads
like NumPy's. Inside the module, use `np.abs` and `np.power`.
"""

from __future__ import annotations

import numpy as np

from .._typing import Array, TensorLike
from ..tensor import Tensor, as_operands, as_tensor, record, unbroadcast

__all__ = [
    "abs",
    "add",
    "div",
    "exp",
    "log",
    "maximum",
    "minimum",
    "mul",
    "neg",
    "pow",
    "sqrt",
    "sub",
]


def add(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise `a + b`, with broadcasting.

    Addition copies its cotangent to both inputs unchanged; all the work is in
    the broadcast reduction.

    >>> import nabla
    >>> x = nabla.tensor([[1.0], [2.0]], requires_grad=True)  # (2, 1)
    >>> y = nabla.tensor([10.0, 20.0, 30.0], requires_grad=True)  # (3,)
    >>> z = nabla.add(x, y)
    >>> z.shape
    (2, 3)
    >>> z.sum().backward()
    >>> x.grad.tolist()  # three broadcast copies per row
    [[3.0], [3.0]]
    >>> y.grad.tolist()  # two broadcast copies per column
    [2.0, 2.0, 2.0]
    """
    ta, tb = as_operands(a, b)
    out: Array = np.add(ta.data, tb.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return unbroadcast(g, ta.shape), unbroadcast(g, tb.shape)

    return record(out, "add", (ta, tb), vjp)


def sub(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise `a - b`, with broadcasting."""
    ta, tb = as_operands(a, b)
    out: Array = np.subtract(ta.data, tb.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return unbroadcast(g, ta.shape), unbroadcast(-g, tb.shape)

    return record(out, "sub", (ta, tb), vjp)


def mul(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise `a * b`, with broadcasting.

    Both cotangents need the *other* operand's forward values, so the closure
    keeps both input buffers alive until the backward pass. That is the memory
    cost of reverse mode in its smallest form.
    """
    ta, tb = as_operands(a, b)
    out: Array = np.multiply(ta.data, tb.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (
            unbroadcast(g * tb.data, ta.shape),
            unbroadcast(g * ta.data, tb.shape),
        )

    return record(out, "mul", (ta, tb), vjp)


def div(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise `a / b`, with broadcasting.

    A denominator at or near zero produces `inf`/`nan` in both the forward pass
    and the gradient, exactly as NumPy would. Nabla does not add an epsilon: a
    hidden epsilon changes the function being differentiated, and a caller who
    wants one should be able to see it in their own code.
    """
    ta, tb = as_operands(a, b)
    out: Array = np.divide(ta.data, tb.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        grad_a = unbroadcast(g / tb.data, ta.shape)
        # -a/b^2 written as -out/b, which reuses the forward result and keeps
        # one fewer temporary of the broadcast shape alive.
        grad_b = unbroadcast(-g * out / tb.data, tb.shape)
        return grad_a, grad_b

    return record(out, "div", (ta, tb), vjp)


def neg(a: TensorLike) -> Tensor:
    """Elementwise `-a`."""
    ta = as_tensor(a)
    out: Array = np.negative(ta.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (-g,)

    return record(out, "neg", (ta,), vjp)


def pow(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise `a ** b`, with broadcasting.

    The derivative with respect to the base, `b * a**(b-1)`, is real for any
    base as long as the exponent is an integer, which is the overwhelmingly
    common case (`x ** 2`). The derivative with respect to the *exponent*,
    `a**b * log(a)`, is not real for `a <= 0`.

    So the exponent's cotangent is computed only when the exponent actually
    requires a gradient, and it is `nan` where the base is non-positive rather
    than a plausible-looking zero. Computing it unconditionally would flood a
    perfectly well-defined backward pass - `(-x) ** 2` - with NumPy invalid-value
    warnings for a value that is then discarded.

    >>> import nabla
    >>> x = nabla.tensor([-2.0, 3.0], requires_grad=True)
    >>> (x**2).sum().backward()
    >>> x.grad.tolist()
    [-4.0, 6.0]
    """
    ta, tb = as_operands(a, b)
    out: Array = np.power(ta.data, tb.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        grad_a: Array | None = None
        if ta.requires_grad:
            grad_a = unbroadcast(g * tb.data * np.power(ta.data, tb.data - 1), ta.shape)
        grad_b: Array | None = None
        if tb.requires_grad:
            positive = ta.data > 0
            log_a = np.where(positive, np.log(np.where(positive, ta.data, 1.0)), np.nan)
            grad_b = unbroadcast(g * out * log_a, tb.shape)
        return grad_a, grad_b

    return record(out, "pow", (ta, tb), vjp)


def exp(a: TensorLike) -> Tensor:
    """Elementwise `e ** a`.

    The vjp reuses the forward result instead of recomputing `exp`, which halves
    the transcendental work at the cost of keeping the output buffer alive - a
    trade every framework makes the same way.
    """
    ta = as_tensor(a)
    out: Array = np.exp(ta.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g * out,)

    return record(out, "exp", (ta,), vjp)


def log(a: TensorLike) -> Tensor:
    """Elementwise natural logarithm.

    `log(0)` is `-inf` and `log(x < 0)` is `nan`, with gradients `inf` and `nan`
    respectively. Both propagate; neither is clamped.
    """
    ta = as_tensor(a)
    out: Array = np.log(ta.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g / ta.data,)

    return record(out, "log", (ta,), vjp)


def sqrt(a: TensorLike) -> Tensor:
    """Elementwise square root.

    The gradient at zero is `inf`, which is correct: the square root has a
    vertical tangent there. Code that reaches this (an L2 norm of a zero vector,
    a standard deviation of identical values) needs an epsilon inside the
    radicand, and needs to have chosen it deliberately.
    """
    ta = as_tensor(a)
    out: Array = np.sqrt(ta.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g / (2.0 * out),)

    return record(out, "sqrt", (ta,), vjp)


def abs(a: TensorLike) -> Tensor:
    """Elementwise absolute value.

    Not differentiable at zero. The subgradient 0 is used there, which is the
    convention every framework picked and the only one that keeps `abs(0)`
    from moving under gradient descent.
    """
    ta = as_tensor(a)
    out: Array = np.abs(ta.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g * np.sign(ta.data),)

    return record(out, "abs", (ta,), vjp)


def maximum(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise maximum, with broadcasting.

    On a tie the whole cotangent goes to `a`. Splitting it evenly is equally
    defensible - and is what a symmetric reading of the subdifferential
    suggests - but it makes `maximum(x, x)` return a gradient of `0.5` for each
    argument, which surprises everyone the first time. Ties are a measure-zero
    event for continuous data either way.
    """
    ta, tb = as_operands(a, b)
    out: Array = np.maximum(ta.data, tb.data)
    return _select_vjp(out, "maximum", ta, tb, ta.data >= tb.data)


def minimum(a: TensorLike, b: TensorLike) -> Tensor:
    """Elementwise minimum, with broadcasting. Ties go to `a`, as in `maximum`."""
    ta, tb = as_operands(a, b)
    out: Array = np.minimum(ta.data, tb.data)
    return _select_vjp(out, "minimum", ta, tb, ta.data <= tb.data)


def _select_vjp(out: Array, op: str, ta: Tensor, tb: Tensor, chose_a: Array) -> Tensor:
    """Record an op that routes its cotangent to whichever operand it selected."""

    def vjp(g: Array) -> tuple[Array | None, ...]:
        mask = chose_a.astype(g.dtype)
        return (
            unbroadcast(g * mask, ta.shape),
            unbroadcast(g * (1.0 - mask), tb.shape),
        )

    return record(out, op, (ta, tb), vjp)
