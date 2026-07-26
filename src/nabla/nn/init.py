r"""Weight initialisers, and the fan-in/fan-out argument behind each one.

# The problem

Consider a stack of linear layers with no activation, weights drawn
independently with mean 0 and variance `s`, and an input whose components have
variance `v`. One layer's output component is a sum of `fan_in` independent
products, so its variance is `fan_in * s * v`. After `L` layers the variance is
multiplied by `(fan_in * s)^L`. Anything other than `fan_in * s = 1` is an
exponential in the depth: too small and the signal is gone by layer twenty, too
large and it overflows. This is not a subtle effect - at `fan_in * s = 1.5` and
`L = 30` the activations are up to 1e5 times the input.

The backward pass runs the same argument through the transpose, where the sum
is over `fan_out` terms, so the gradient variance is multiplied by
`fan_out * s` per layer.

# The two answers

**Xavier/Glorot** (Glorot and Bengio, 2010) cannot satisfy `fan_in * s = 1` and
`fan_out * s = 1` at once unless the layer is square, so it splits the
difference and takes the harmonic-mean-like compromise
`s = 2 / (fan_in + fan_out)`. It was derived for activations that are
approximately linear near the origin - `tanh`, and the logistic - where the
"no activation" assumption above is roughly true.

**Kaiming/He** (He et al., 2015) drops the compromise and observes that ReLU
zeroes half its inputs, which halves the output variance. Correcting for that
gives `s = 2 / fan_in`. The factor of 2 is not a tuning constant; it is
`1 / E[relu(z)^2 / E[z^2]]` for a standard normal `z`. Choosing `fan_in`
preserves the forward signal, and choosing `fan_out` would preserve the
backward one; the forward choice is the usual default because a dead forward
pass produces no gradient to preserve.

The `gain` argument is the general form of that factor: it multiplies the
standard deviation, so `gain = sqrt(2)` is the ReLU correction and `gain = 1` is
the linear one. `recommended_gain` names the ones worth having.

# Fan-in and fan-out in *this* package

Nabla's `Linear` stores its weight as `(in_features, out_features)` and computes
`x @ w`, with no transpose in the forward pass. That is the opposite of the
convention most frameworks use, so:

    fan_in  = shape[0]
    fan_out = shape[1]

Getting this backwards is a real and quiet failure: for a 784 -> 128 layer it
scales the initial weights by `sqrt(784/128) = 2.5`, which does not crash and
does not obviously diverge, it just trains worse. `fan_in_out` is the one place
that knows the layout, and every initialiser goes through it.

# Uniform or normal

For a given target variance the two differ only in the tails, and no result in
the literature depends on which is used. Both are provided because both appear
in papers, and the uniform bound is `sqrt(3) * std` because a uniform
distribution on `[-b, b]` has variance `b^2 / 3`.
"""

from __future__ import annotations

import math

from .._typing import Array, DTypeLike, Shape
from ..errors import ShapeError
from ..random import Generator
from ..tensor import Tensor

__all__ = [
    "fan_in_out",
    "kaiming_normal",
    "kaiming_uniform",
    "recommended_gain",
    "truncated_normal",
    "xavier_normal",
    "xavier_uniform",
]


def fan_in_out(shape: Shape) -> tuple[int, int]:
    """The number of inputs and outputs a weight of `shape` connects.

    Only ranks 1 and 2 occur in this package - there are no convolutions - so
    the rule is short: a matrix is `(fan_in, fan_out)` in Nabla's layout, and a
    vector (a bias, a normalisation scale) has the same count on both sides
    because it is applied elementwise.

    >>> from nabla.nn import init
    >>> init.fan_in_out((784, 128))
    (784, 128)
    """
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    msg = (
        f"cannot infer fan-in and fan-out for a rank-{len(shape)} weight of shape {shape}: "
        "this package initialises matrices and vectors only"
    )
    raise ShapeError(msg)


def recommended_gain(nonlinearity: str) -> float:
    """The multiplier on the standard deviation for a given activation.

    Each value is `1 / sqrt(E[f(z)^2])` for a standard normal `z`, which is
    exactly the factor that restores unit output variance after `f`:

    * `linear`, `sigmoid` - 1. The logistic is not variance-preserving either,
      but it is not variance-*destroying* the way ReLU is, and no standard
      correction for it exists.
    * `relu` - `sqrt(2)`. ReLU zeroes half the mass, halving the second moment.
    * `leaky_relu` and `gelu` - close enough to ReLU's that the same factor is
      used; GELU's exact value is about 1.7, against ReLU's 1.414, and the
      difference is smaller than the spread between reasonable learning rates.
    * `tanh` - 5/3, the value Glorot and Bengio report from the empirical
      second moment of `tanh(z)` for a standard normal `z`.
    """
    gains = {
        "linear": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0),
        "gelu": math.sqrt(2.0),
    }
    if nonlinearity not in gains:
        msg = f"no recommended gain for {nonlinearity!r}; known: {sorted(gains)}"
        raise ValueError(msg)
    return gains[nonlinearity]


def xavier_normal(
    shape: Shape,
    generator: Generator,
    *,
    gain: float = 1.0,
    dtype: DTypeLike | None = None,
) -> Tensor:
    """Normal weights with variance `gain^2 * 2 / (fan_in + fan_out)`.

    The compromise between preserving the forward variance and preserving the
    backward one; see the module docstring. Exact for a square layer, off by at
    most `sqrt(fan_out / fan_in)` for a rectangular one.
    """
    fan_in, fan_out = fan_in_out(shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return generator.normal(shape, std=std, dtype=dtype, requires_grad=True)


def xavier_uniform(
    shape: Shape,
    generator: Generator,
    *,
    gain: float = 1.0,
    dtype: DTypeLike | None = None,
) -> Tensor:
    """Uniform weights on `+-gain * sqrt(6 / (fan_in + fan_out))`.

    The same variance as `xavier_normal`: a uniform on `[-b, b]` has variance
    `b^2 / 3`, and `6 = 3 * 2` is where the 6 in the usual formula comes from.
    """
    fan_in, fan_out = fan_in_out(shape)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return generator.uniform(shape, low=-bound, high=bound, dtype=dtype, requires_grad=True)


def kaiming_normal(
    shape: Shape,
    generator: Generator,
    *,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
    dtype: DTypeLike | None = None,
) -> Tensor:
    """Normal weights with variance `gain^2 / fan`, `fan` chosen by `mode`.

    `mode="fan_in"` preserves the variance of the forward activations,
    `mode="fan_out"` that of the backward gradients. They coincide for a square
    layer. The default is `fan_in` because a forward pass whose activations have
    collapsed produces no gradient worth preserving.
    """
    fan_in, fan_out = fan_in_out(shape)
    fan = _select_fan(fan_in, fan_out, mode)
    std = recommended_gain(nonlinearity) / math.sqrt(fan)
    return generator.normal(shape, std=std, dtype=dtype, requires_grad=True)


def kaiming_uniform(
    shape: Shape,
    generator: Generator,
    *,
    nonlinearity: str = "relu",
    mode: str = "fan_in",
    dtype: DTypeLike | None = None,
) -> Tensor:
    """Uniform weights with the same variance as `kaiming_normal`."""
    fan_in, fan_out = fan_in_out(shape)
    fan = _select_fan(fan_in, fan_out, mode)
    bound = math.sqrt(3.0) * recommended_gain(nonlinearity) / math.sqrt(fan)
    return generator.uniform(shape, low=-bound, high=bound, dtype=dtype, requires_grad=True)


def _select_fan(fan_in: int, fan_out: int, mode: str) -> int:
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    msg = f"mode must be 'fan_in' or 'fan_out', got {mode!r}"
    raise ValueError(msg)


def truncation_scale(limit: float) -> float:
    """How much a normal's standard deviation shrinks when truncated at `+-limit`.

    A standard normal conditioned on `|z| <= a` has variance

        1 - 2 a phi(a) / (2 Phi(a) - 1)

    where `phi` and `Phi` are the standard normal density and cumulative. This
    returns the square root of that, so dividing by it restores the requested
    standard deviation.

    At the usual `limit = 2` the factor is about 0.8796, so an implementation
    that resamples the tail and then does *not* rescale delivers weights 12%
    smaller than the caller asked for. That is not fatal, but it means the
    initialiser's promise is false, and an initialiser whose promise is false is
    worse than no initialiser at all - the whole point of the fan-in argument
    above is that the number is exact.

    >>> from nabla.nn.init import truncation_scale
    >>> round(truncation_scale(2.0), 6)
    0.879626
    """
    if limit <= 0:
        msg = f"limit must be positive, got {limit}"
        raise ValueError(msg)
    density = math.exp(-0.5 * limit * limit) / math.sqrt(2.0 * math.pi)
    mass = math.erf(limit / math.sqrt(2.0))
    return math.sqrt(1.0 - 2.0 * limit * density / mass)


def truncated_normal(
    shape: Shape,
    generator: Generator,
    *,
    std: float = 1.0,
    limit: float = 2.0,
    dtype: DTypeLike | None = None,
    requires_grad: bool = True,
) -> Tensor:
    """Normal weights with no value further than `limit` standard deviations out.

    Two things a caller wants from this, and both are delivered:

    * **No outliers.** A single weight five standard deviations out in the first
      layer of a deep network is a single unit that saturates for every input,
      and in a `Linear` feeding a softmax it is a logit that starts the run
      already confident. Truncation removes them at the source, which is cheaper
      than the gradient clipping that would otherwise be papering over them.
    * **The variance it asked for.** The samples are divided by
      `truncation_scale(limit)`, so the returned tensor has standard deviation
      `std` and not the 88% of it that naive rejection sampling produces.

    Implemented by rejection: draw, redraw only the entries that fell outside,
    repeat. The alternative - the inverse CDF of the truncated normal - needs
    `erfinv`, which NumPy does not have. Rejection at `limit = 2` accepts 95% of
    draws, so the expected number of rounds is small and the loop is bounded;
    the bound is what turns a pathological `limit` into an error rather than a
    hang.

    >>> import nabla
    >>> from nabla.nn import init
    >>> w = init.truncated_normal((256, 256), nabla.random.Generator(seed=0), std=0.05)
    >>> bool(abs(w.numpy()).max() <= 2.0 * 0.05 / init.truncation_scale(2.0) + 1e-9)
    True
    """
    if std < 0:
        msg = f"std must be non-negative, got {std}"
        raise ValueError(msg)
    scale = std / truncation_scale(limit)
    values: Array = generator.normal(shape, std=scale, dtype=dtype).numpy()
    bound = limit * scale
    for _ in range(64):
        outside = abs(values) > bound
        count = int(outside.sum())
        if count == 0:
            break
        values[outside] = generator.normal((count,), std=scale, dtype=dtype).numpy()
    else:  # pragma: no cover - only reachable with a limit so small nothing is accepted
        msg = (
            f"truncated_normal did not converge in 64 rounds at limit={limit}; "
            "the acceptance region is too narrow to sample by rejection"
        )
        raise ValueError(msg)
    out = Tensor(values)
    out.requires_grad = requires_grad
    return out
