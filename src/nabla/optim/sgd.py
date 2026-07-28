"""Stochastic gradient descent, with momentum, Nesterov and weight decay."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .._typing import Array
from ..errors import OptimizerError
from ..tensor import Tensor
from .optimizer import Optimizer, _positive, _unit_interval

__all__ = ["SGD"]


class SGD(Optimizer):
    r"""Gradient descent with an optional velocity.

    # Momentum

    Plain descent takes a step along the current gradient. On a quadratic whose
    condition number is `k`, that means oscillating across the narrow direction
    while creeping along the shallow one, and the number of steps needed grows
    linearly in `k`. Momentum keeps a running average of the gradients, so the
    oscillating components cancel and the consistent one accumulates; the buffer
    is

        v <- mu v + (1 - tau) g
        p <- p - lr v

    with `tau` the dampening. `tau = 0` is the usual choice and makes the
    steady-state velocity `g / (1 - mu)`, which is why switching momentum on at
    `mu = 0.9` without lowering the learning rate multiplies the effective step
    by ten and diverges. That is the single most common surprise with this
    optimiser and it is arithmetic, not instability.

    # Nesterov

    Nesterov's variant evaluates the gradient at the point the momentum is
    already carrying the parameters to, rather than where they currently are.
    Written in the form above that is `g + mu v` in place of `v`: one extra add,
    no extra state. It is a genuine improvement on convex problems and a small
    one in practice. It requires `tau = 0` - with dampening the "look ahead"
    point is not where the update lands - so that combination is refused rather
    than silently approximated.

    # Weight decay

    Added to the gradient, `g <- g + lambda p`, which for plain SGD is *exactly*
    the same thing as adding `lambda/2 ||p||^2` to the loss. That equivalence is
    what makes "L2 regularisation" and "weight decay" synonyms here - and it is
    precisely the equivalence that Adam breaks; see `nabla.optim.adam.AdamW`.

    >>> import nabla
    >>> from nabla import optim
    >>> p = nabla.tensor([1.0], requires_grad=True)
    >>> opt = optim.SGD([p], lr=0.1)
    >>> (p * p).sum().backward()  # gradient 2p = 2
    >>> opt.step()
    >>> p.numpy().tolist()  # 1 - 0.1 * 2
    [0.800000011920929]
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        *,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(parameters, lr)
        self.momentum = _unit_interval("momentum", momentum)
        self.dampening = _unit_interval("dampening", dampening)
        self.weight_decay = _positive("weight_decay", weight_decay, allow_zero=True)
        if nesterov and (momentum == 0.0 or dampening != 0.0):
            msg = "nesterov momentum needs momentum > 0 and dampening == 0"
            raise OptimizerError(msg)
        self.nesterov = nesterov

    def hyperparameters(self) -> dict[str, Any]:
        """Learning rate, momentum, dampening, decay and the Nesterov flag."""
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "dampening": self.dampening,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }

    def update(self, parameter: Tensor, gradient: Array, state: dict[str, Array]) -> Array:
        """One SGD step for a single parameter."""
        g = gradient
        if self.weight_decay:
            g = g + self.weight_decay * parameter.data
        if self.momentum:
            if "velocity" not in state:
                # Initialised from the first gradient rather than from zero.
                # Starting at zero makes the first step `(1 - tau) lr g`, which
                # for the usual tau = 0 is the same thing; starting from the
                # gradient keeps it the same for any dampening as well.
                state["velocity"] = np.array(g, copy=True)
            else:
                state["velocity"] = self.momentum * state["velocity"] + (1.0 - self.dampening) * g
            g = g + self.momentum * state["velocity"] if self.nesterov else state["velocity"]
        out: Array = parameter.data - self.lr * g
        return out
