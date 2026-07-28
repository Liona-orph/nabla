"""RMSProp: Adam's denominator without Adam's numerator."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .._typing import Array
from ..tensor import Tensor
from .optimizer import Optimizer, _positive, _unit_interval

__all__ = ["RMSProp"]


class RMSProp(Optimizer):
    r"""Divide the gradient by a running root-mean-square of itself.

        v <- a v + (1 - a) g^2
        p <- p - lr g / (sqrt(v) + eps)

    Historically this came first (Tieleman and Hinton, 2012, in a lecture slide
    that is still the canonical citation) and Adam is it plus a momentum term
    and bias correction. It is kept here because the difference is instructive:
    RMSProp rescales the *current* gradient, so it reacts immediately to a
    change in gradient magnitude, where Adam's numerator averages over roughly
    `1/(1-b1)` steps and therefore lags.

    There is no bias correction, deliberately: without a first moment there is
    no ratio for the two corrections to distort, and `v` starting at zero makes
    the first few steps large rather than small - the opposite direction from
    Adam's bias, and one that a warmup handles.

    `centred=True` subtracts the square of a running mean, so the denominator
    estimates the gradient's *variance* rather than its second moment. That is
    more expensive by one buffer and one multiply per parameter, and it matters
    when the gradient has a large constant component that would otherwise
    suppress the step for the whole run.

    `momentum` adds a velocity on the *rescaled* gradient, which is where it
    goes in every implementation and is not the same as adding it before the
    division.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-2,
        *,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centred: bool = False,
    ) -> None:
        super().__init__(parameters, lr)
        self.alpha = _unit_interval("alpha", alpha)
        self.eps = _positive("eps", eps)
        self.weight_decay = _positive("weight_decay", weight_decay, allow_zero=True)
        self.momentum = _unit_interval("momentum", momentum)
        self.centred = centred

    def hyperparameters(self) -> dict[str, Any]:
        """Learning rate, smoothing constant, epsilon, decay, momentum, centring."""
        return {
            "lr": self.lr,
            "alpha": self.alpha,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "centred": self.centred,
        }

    def update(self, parameter: Tensor, gradient: Array, state: dict[str, Array]) -> Array:
        """One RMSProp step for a single parameter."""
        values = parameter.data
        if "square_avg" not in state:
            state["square_avg"] = np.zeros_like(values)
            if self.centred:
                state["grad_avg"] = np.zeros_like(values)
            if self.momentum:
                state["velocity"] = np.zeros_like(values)

        g = gradient
        if self.weight_decay:
            g = g + self.weight_decay * values

        state["square_avg"] = self.alpha * state["square_avg"] + (1.0 - self.alpha) * (g * g)
        if self.centred:
            state["grad_avg"] = self.alpha * state["grad_avg"] + (1.0 - self.alpha) * g
            variance = state["square_avg"] - state["grad_avg"] * state["grad_avg"]
            # The estimate is a difference of two running averages and can go
            # very slightly negative through rounding; clipping at zero keeps
            # the square root real without pretending the variance was larger.
            denominator = np.sqrt(np.maximum(variance, 0.0)) + self.eps
        else:
            denominator = np.sqrt(state["square_avg"]) + self.eps

        step = g / denominator
        if self.momentum:
            state["velocity"] = self.momentum * state["velocity"] + step
            step = state["velocity"]
        out: Array = values - self.lr * step
        return out
