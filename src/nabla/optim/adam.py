r"""Adam and AdamW, and the one difference between them that matters.

# Adam

Two running averages per parameter: of the gradient, and of its square.

    m <- b1 m + (1 - b1) g
    v <- b2 v + (1 - b2) g^2
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)

The division makes the step size roughly `lr` regardless of the gradient's
magnitude, which is what lets one learning rate work across parameters whose
gradients differ by orders of magnitude - the embedding and the final layer of a
transformer, for instance.

# Bias correction

`m` and `v` start at zero, so early on they are biased towards zero: after one
step `m = (1 - b1) g`, which for `b1 = 0.9` is a tenth of the gradient. The
correction divides by `1 - b1^t`, which is exactly the sum of the weights that
have actually been applied, so `m_hat` is a proper weighted average from step 1.

Without it the first steps are tiny and, worse, the *ratio* `m/sqrt(v)` is
wrong: `1 - b1` over `sqrt(1 - b2)` is 0.1 / 0.0316 = 3.2, so the first update
is more than three times too large. Implementations that skip the correction
compensate with a warmup, which is a way of hiding a known bug behind a
hyperparameter.

# eps outside the square root

`sqrt(v_hat) + eps`, not `sqrt(v_hat + eps)`. The two differ by a factor of
`sqrt(eps)` in the regime where they matter, and the outside form is what the
paper specifies and what every other implementation does; a checkpoint's
hyperparameters would not transfer otherwise. The role of `eps` is to bound the
step at `lr / eps` when the gradient has been zero for a while, and at 1e-8 that
bound is 1e8 * lr, which is to say it is not really a bound at all - it is a
guard against dividing by exactly zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .._typing import Array
from ..tensor import Tensor
from .optimizer import Optimizer, _positive, _unit_interval

__all__ = ["Adam", "AdamW"]


class Adam(Optimizer):
    """Adaptive moment estimation (Kingma and Ba, 2015).

    `weight_decay` here is **coupled**: it is added to the gradient, so it is
    L2 regularisation in the loss. That is the classic form, and
    `nabla.optim.AdamW` explains why it does not do what its name suggests.

    >>> import nabla
    >>> from nabla import optim
    >>> p = nabla.tensor([5.0], requires_grad=True)
    >>> opt = optim.Adam([p], lr=0.1)
    >>> for _ in range(200):
    ...     opt.zero_grad()
    ...     (p * p).sum().backward()
    ...     opt.step()
    >>> bool(abs(p.item()) < 1e-3)
    True
    """

    #: Whether weight decay is applied to the parameter directly (AdamW) or
    #: folded into the gradient (Adam). A field rather than two near-identical
    #: classes, because the two forms differ by three lines and the point of
    #: this module is that a reader can see the three lines side by side.
    decoupled_weight_decay: bool = False

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = _unit_interval("beta1", beta1)
        self.beta2 = _unit_interval("beta2", beta2)
        self.eps = _positive("eps", eps)
        self.weight_decay = _positive("weight_decay", weight_decay, allow_zero=True)

    def hyperparameters(self) -> dict[str, Any]:
        """Learning rate, both decay rates, epsilon and the weight decay."""
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def update(self, parameter: Tensor, gradient: Array, state: dict[str, Array]) -> Array:
        """One Adam step for a single parameter."""
        if "exp_avg" not in state:
            state["exp_avg"] = np.zeros_like(parameter.data)
            state["exp_avg_sq"] = np.zeros_like(parameter.data)

        g = gradient
        values = parameter.data
        if self.weight_decay and not self.decoupled_weight_decay:
            g = g + self.weight_decay * values

        state["exp_avg"] = self.beta1 * state["exp_avg"] + (1.0 - self.beta1) * g
        state["exp_avg_sq"] = self.beta2 * state["exp_avg_sq"] + (1.0 - self.beta2) * (g * g)

        # The step counter is the optimiser's, not the parameter's: every
        # parameter is stepped on every call, including the ones that were
        # skipped for want of a gradient, so a per-parameter counter would only
        # differ for parameters whose correction nobody is applying anyway.
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        step = (
            self.lr
            * (state["exp_avg"] / correction1)
            / (np.sqrt(state["exp_avg_sq"] / correction2) + self.eps)
        )
        if self.weight_decay and self.decoupled_weight_decay:
            step = step + self.lr * self.weight_decay * values
        out: Array = values - step
        return out


class AdamW(Adam):
    r"""Adam with **decoupled** weight decay (Loshchilov and Hutter, 2019).

    # The difference that actually matters

    Adam's `weight_decay` adds `lambda p` to the gradient. That term then goes
    through the same adaptive rescaling as everything else, so the decay
    actually applied to a parameter is

        lr * lambda p / (sqrt(v_hat) + eps)

    and it is divided by the parameter's own gradient history. Two consequences,
    both bad and both easy to observe:

    * A parameter whose gradients have been large gets *less* regularisation
      than one whose gradients have been small - the opposite of what a uniform
      prior on the weights is supposed to express.
    * In the limit where the data gradient is zero, `m` and `v` both come
      entirely from the decay term, they cancel in the ratio, and the update
      becomes `lr * sign(p)` - a step of fixed size, independent of `lambda` and
      independent of how large `p` is. The regularisation coefficient stops
      controlling anything.

    AdamW applies the decay to the parameter directly, outside the adaptive
    step:

        p <- p - lr * m_hat / (sqrt(v_hat) + eps) - lr * lambda * p

    so the decay is multiplicative, proportional to `lambda`, and identical for
    every parameter. `tests/test_optim.py` asserts both halves of this: that
    AdamW shrinks two parameters of very different magnitude by the same
    *factor* while Adam+L2 shrinks them by the same *absolute amount*, and that
    doubling `lambda` doubles AdamW's decay while barely moving Adam's.

    The default `weight_decay` is 0.01 rather than 0, because an AdamW
    constructed with no decay is an Adam with extra steps.
    """

    decoupled_weight_decay = True

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(
            parameters,
            lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
