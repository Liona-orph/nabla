"""Loss functions.

# Every classification loss here takes logits

Not probabilities. That is a deliberate restriction and it is about numerics
rather than taste.

A cross-entropy written on probabilities is `-log(p[target])`. To get `p` the
caller had to run a softmax, and the softmax of a confidently wrong prediction
underflows to exactly 0 in float32 for any logit gap beyond about 88. `log(0)`
is `-inf`, the loss is `inf`, the gradient is `nan`, and the run is over. The
information needed to avoid that - the logit gap - was destroyed by the softmax
one line earlier and cannot be recovered.

Written on logits, the same quantity is `logsumexp(z) - z[target]`, which is a
difference of two finite numbers for any input whatsoever. The test in
`tests/test_nn_loss.py` feeds logits of 1e4 and asserts a finite loss and a
finite gradient; the naive form overflows on those inputs, and the test says so
by computing the naive form alongside.

The same argument applies to the binary case, where the stable identity is

    -[y log s(z) + (1-y) log(1-s(z))] = max(z, 0) - z y + log(1 + exp(-|z|))

with every term finite for any `z`.

# Reductions

`"mean"`, `"sum"` or `"none"`. `"mean"` is the default because it makes the
learning rate independent of the batch size, which is the property that lets a
batch size be changed without retuning anything. `"sum"` is what gradient
accumulation over micro-batches wants. `"none"` returns the per-example loss,
for weighting or for inspection.
"""

from __future__ import annotations

import numpy as np

from .._typing import Array
from ..errors import ShapeError
from ..ops.activation import log_softmax
from ..ops.basic import abs as abs_op
from ..ops.basic import exp, log, maximum
from ..ops.shape import getitem
from ..tensor import Tensor, as_tensor

__all__ = ["binary_cross_entropy", "cross_entropy", "mse", "nll", "reduce_loss"]


def reduce_loss(values: Tensor, reduction: str) -> Tensor:
    """Apply `"mean"`, `"sum"` or `"none"` to a per-element loss.

    Shared by every loss in this module so that they cannot disagree about what
    the words mean - which sounds trivial until one of them averages over the
    class axis as well as the batch axis.
    """
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    msg = f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}"
    raise ValueError(msg)


def mse(prediction: Tensor, target: Tensor | Array, *, reduction: str = "mean") -> Tensor:
    """Mean squared error.

    The target is detached on the way in. A target that carried a gradient would
    let the model reduce the loss by moving the labels, which is never what a
    caller means; when it *is* what they mean - a distillation objective, a
    reconstruction against another network's output - they can pass the tensor
    through explicitly by computing the difference themselves.

    >>> import nabla
    >>> from nabla.nn import loss
    >>> loss.mse(nabla.tensor([1.0, 2.0]), nabla.tensor([1.0, 4.0])).item()
    2.0
    """
    truth = _detached(target, prediction)
    if truth.shape != prediction.shape:
        msg = f"mse: prediction has shape {prediction.shape} but target has {truth.shape}"
        raise ShapeError(msg)
    difference = prediction - truth
    return reduce_loss(difference * difference, reduction)


def nll(log_probabilities: Tensor, targets: Tensor | Array, *, reduction: str = "mean") -> Tensor:
    """Negative log likelihood of `targets` under already-normalised log-probabilities.

    Takes *log* probabilities, not probabilities: this is the second half of
    `cross_entropy`, split out so that a model which produces its own normalised
    log-probabilities (a mixture, a flow) can use the same objective without
    exponentiating and re-logging them.

    Targets are integer class indices of shape `logits.shape[:-1]`. A one-hot
    matrix would be the same arithmetic with a `C`-times-larger multiply and an
    extra `C`-sized buffer per example, for a vector that is zero everywhere but
    one place; the gather below is the same operation written honestly.
    """
    indices = _class_indices(targets, log_probabilities)
    flat = log_probabilities.reshape(-1, log_probabilities.shape[-1])
    picked = getitem(flat, (np.arange(flat.shape[0]), indices.reshape(-1)))
    return reduce_loss(-picked.reshape(indices.shape), reduction)


def cross_entropy(logits: Tensor, targets: Tensor | Array, *, reduction: str = "mean") -> Tensor:
    """Softmax cross-entropy from logits.

    Defined as `nll(log_softmax(logits), targets)` rather than as its own
    expression, so the two can never disagree and so the stability argument
    lives in exactly one place - `nabla.log_softmax`, which subtracts the
    per-row maximum before exponentiating.

    The gradient of this loss with respect to the logits is `softmax(z) - onehot`,
    which is bounded by 1 in absolute value however extreme the logits are. That
    is the second reason to compute it this way: the naive composition's
    gradient contains a `1/p` that is unbounded.

    >>> import nabla
    >>> from nabla.nn import loss
    >>> import numpy as np
    >>> huge = nabla.tensor([[1e4, 0.0, 0.0]])
    >>> bool(loss.cross_entropy(huge, np.array([0])).item() == 0.0)
    True
    >>> bool(np.isfinite(loss.cross_entropy(huge, np.array([1])).item()))
    True
    """
    return nll(log_softmax(logits, axis=-1), targets, reduction=reduction)


def binary_cross_entropy(
    logits: Tensor,
    targets: Tensor | Array,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Binary cross-entropy from logits, elementwise.

    Computed as `max(z, 0) - z y + log(1 + exp(-|z|))`. The identity is exact,
    and every term in it is finite: the exponent is never positive, so nothing
    overflows, and the `max(z, 0)` term carries the magnitude that the naive
    `-y log sigmoid(z)` would have lost when the sigmoid saturated to exactly 0
    or 1.

    The remaining inaccuracy is that `log(1 + u)` for `u` below the machine
    epsilon rounds to 0 rather than to `u`. That is the right answer to within
    the resolution of the result, because the `max(z, 0)` term it is being added
    to is then of order 20 or more; a `log1p` would be more accurate in a place
    where accuracy has no consumer.

    Targets are 0 or 1 (or anything in between, for a soft label) and are
    detached, as in `mse`.

    >>> import nabla
    >>> from nabla.nn import loss
    >>> import numpy as np
    >>> value = loss.binary_cross_entropy(nabla.tensor([0.0]), np.array([1.0])).item()
    >>> round(value, 6)  # -log(0.5)
    0.693147
    """
    truth = _detached(targets, logits)
    if truth.shape != logits.shape:
        msg = f"binary_cross_entropy: logits have shape {logits.shape}, targets {truth.shape}"
        raise ShapeError(msg)
    stable = maximum(logits, 0.0) - logits * truth + log(1.0 + exp(-abs_op(logits)))
    return reduce_loss(stable, reduction)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _detached(value: Tensor | Array, like: Tensor) -> Tensor:
    """`value` as a tensor of `like`'s dtype, carrying no gradient."""
    tensor = as_tensor(value)
    detached = tensor.detach()
    if detached.dtype != like.dtype:
        return Tensor(detached.numpy(copy=False), dtype=like.dtype)
    return detached


def _class_indices(targets: Tensor | Array, log_probabilities: Tensor) -> Array:
    """Validate integer class targets against the shape of the log-probabilities."""
    indices: Array = np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    if indices.dtype.kind not in "iu":
        msg = (
            f"class targets must be integers, got dtype {indices.dtype}. "
            "A float target is either a one-hot matrix (use its argmax) or a mistake."
        )
        raise ShapeError(msg)
    expected = log_probabilities.shape[:-1]
    if indices.shape != expected:
        msg = (
            f"targets have shape {indices.shape}, but log-probabilities of shape "
            f"{log_probabilities.shape} need one class index per example, i.e. {expected}"
        )
        raise ShapeError(msg)
    classes = log_probabilities.shape[-1]
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= classes):
        msg = (
            f"class index out of range: got [{int(indices.min())}, {int(indices.max())}] "
            f"for {classes} classes"
        )
        raise ShapeError(msg)
    return indices
