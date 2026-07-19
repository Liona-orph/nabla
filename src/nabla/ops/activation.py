"""Activation functions.

These are the ops where numerical stability is not a detail. `sigmoid`,
`softmax` and `log_softmax` all contain an `exp`, and `exp` overflows at 88 in
float32 - a value ordinary logits reach the moment a run starts to diverge. Each
one here is written in the form that cannot overflow, and each says which form
it is avoiding, because "it works on my inputs" is how the naive version
survives review.

Every function is implemented as a single primitive with a hand-written vjp
rather than composed from `exp` and `div`. Composition would be correct, but the
composed backward pass recomputes quantities the forward pass already had
(`sigmoid` needs only its own output; `softmax` needs only its own output) and
allocates a temporary per intermediate node.
"""

from __future__ import annotations

import numpy as np

from .._typing import Array, TensorLike
from ..tensor import Tensor, as_tensor, record

__all__ = ["gelu", "log_softmax", "relu", "sigmoid", "softmax", "tanh"]

#: sqrt(2/pi), the constant in the tanh form of GELU.
_SQRT_2_OVER_PI = 0.7978845608028654


def relu(a: TensorLike) -> Tensor:
    """`max(x, 0)`.

    Not differentiable at zero; the subgradient 0 is used there. The alternative
    conventions (1, or 1/2) all make `relu` behave differently from the identity
    on a set of measure zero, and 0 is the one that keeps a dead unit dead
    instead of letting a exactly-zero pre-activation drift.
    """
    ta = as_tensor(a)
    out: Array = np.maximum(ta.data, 0.0)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (np.where(ta.data > 0, g, 0.0),)

    return record(out, "relu", (ta,), vjp)


def sigmoid(a: TensorLike) -> Tensor:
    """`1 / (1 + exp(-x))`, computed branch-wise so it cannot overflow.

    Written naively, a large negative input makes `exp(-x)` overflow to `inf`
    and the result comes back as exactly 0 with a warning; the branch-wise form
    evaluates `exp(x) / (1 + exp(x))` there instead, where the exponent is
    negative and the arithmetic is harmless. Both branches are computed and one
    is selected, so the cost is two `exp` calls over the whole array rather than
    one - the price of avoiding a data-dependent branch in NumPy.

    The gradient `s (1 - s)` needs only the output, so the input buffer is not
    retained.
    """
    ta = as_tensor(a)
    x = ta.data
    positive = x >= 0
    exp_neg_abs: Array = np.exp(-np.abs(x))
    out: Array = np.where(positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g * out * (1.0 - out),)

    return record(out, "sigmoid", (ta,), vjp)


def tanh(a: TensorLike) -> Tensor:
    """Hyperbolic tangent.

    `np.tanh` is already saturating and overflow-free, so there is nothing to
    stabilise. The gradient `1 - t^2` underflows to zero for `|x|` beyond about
    10, which is the vanishing gradient itself and not an implementation defect.
    """
    ta = as_tensor(a)
    out: Array = np.tanh(ta.data)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g * (1.0 - out * out),)

    return record(out, "tanh", (ta,), vjp)


def gelu(a: TensorLike) -> Tensor:
    """Gaussian error linear unit, in its tanh form.

    The exact definition is `x * Phi(x)`, and `Phi` needs `erf`. NumPy has no
    vectorised `erf`; `math.erf` is scalar, so using it would mean a Python-level
    loop over every element, which is two orders of magnitude slower than the
    rest of this package and would make GELU the most expensive operation in any
    model that used it. The tanh form used here is the standard substitute and
    stays within about 3e-4 of the exact function everywhere - well below float32
    resolution at the magnitudes where activations live.

    That approximation is the only place in Nabla where a *forward* value is
    knowingly inexact. The gradient below is the exact derivative of the
    approximation, not an approximation of the exact derivative, so gradient
    checking still passes to full float64 tolerance.
    """
    ta = as_tensor(a)
    x = ta.data
    inner: Array = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    t: Array = np.tanh(inner)
    out: Array = 0.5 * x * (1.0 + t)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        d_inner = _SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x**2)
        local = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
        return (g * local,)

    return record(out, "gelu", (ta,), vjp)


def softmax(a: TensorLike, axis: int = -1) -> Tensor:
    """Softmax along `axis`, with the maximum subtracted first.

    Subtracting the per-group maximum leaves the result unchanged - the constant
    cancels between numerator and denominator - and bounds every exponent at 0,
    so the largest term is exactly 1 and nothing can overflow. Without it,
    logits of 1000 give `inf / inf = nan`; with it they give the right answer.
    Underflow at the other end is harmless: a term that rounds to 0 was smaller
    than the largest by a factor of 1e38 and could not have affected the sum.

    The vjp uses the identity `dx = s * (g - sum(g * s))`, which is the full
    Jacobian `diag(s) - s s^T` applied to `g` without ever forming the `n x n`
    matrix. Forming it would cost `O(n^2)` memory per row for no gain.

    >>> import nabla
    >>> p = nabla.softmax(nabla.tensor([1000.0, 1000.0, 1000.0]))
    >>> [round(v, 6) for v in p.numpy().tolist()]
    [0.333333, 0.333333, 0.333333]
    """
    ta = as_tensor(a)
    shifted: Array = ta.data - np.max(ta.data, axis=axis, keepdims=True)
    exponentiated: Array = np.exp(shifted)
    out: Array = exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        weighted = np.sum(g * out, axis=axis, keepdims=True)
        return (out * (g - weighted),)

    return record(out, "softmax", (ta,), vjp)


def log_softmax(a: TensorLike, axis: int = -1) -> Tensor:
    """`log(softmax(x))`, computed as `x - logsumexp(x)`.

    Taking the log of a softmax that has already underflowed gives `-inf`, and
    that is exactly what happens to the unlikely classes in any real
    classification problem. Going through `logsumexp` keeps every value finite
    and accurate, which is why cross-entropy losses are always written on top of
    this function rather than on `log(softmax(x))`.

    The vjp is `g - softmax(x) * sum(g)`: the gradient of the shift, subtracted
    from the gradient of the identity.
    """
    ta = as_tensor(a)
    shifted: Array = ta.data - np.max(ta.data, axis=axis, keepdims=True)
    log_denominator: Array = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out: Array = shifted - log_denominator
    probabilities: Array = np.exp(out)

    def vjp(g: Array) -> tuple[Array | None, ...]:
        return (g - probabilities * np.sum(g, axis=axis, keepdims=True),)

    return record(out, "log_softmax", (ta,), vjp)
