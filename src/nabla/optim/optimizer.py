"""The optimiser base class.

# What an optimiser is here

A list of parameters, a learning rate, and one dictionary of per-parameter state
for each of them. `step()` reads `.grad`, computes an update, and writes it back
with `Tensor.assign_` - the one sanctioned mutation in this package, which
rebinds a leaf's buffer rather than writing into it, so no vjp closure that
captured the old values is invalidated.

# Why the state is keyed by position

Adam's moments have to be associated with the parameter they belong to. Three
candidate keys:

* The tensor's `id()`. Works within a process and is meaningless across one, so
  a checkpoint written with it cannot be loaded.
* A name. The optimiser is given tensors, not a model, so it does not have one -
  and inventing one would couple the optimiser to `nabla.nn`, which it otherwise
  has no reason to know about.
* The position in the parameter list. Stable across processes, requires nothing
  from the caller, and is correct as long as the list is built the same way
  twice - which it is, because `Module.parameters()` walks the registries in
  insertion order and that order is fixed by the constructor.

Position it is, and the cost is stated plainly: a checkpoint is only loadable
into a model whose `parameters()` yields the same tensors in the same order. A
model that gained a layer cannot load an old optimiser state, and
`load_state_dict` says so rather than silently pairing the wrong moments with
the wrong weights.

# Why `step()` skips a parameter with no gradient

`.grad` of `None` means nothing reached that parameter - it was detached, or it
is not used by the loss that was backpropagated. Treating that as a zero
gradient would still apply weight decay and still advance the momentum buffer,
so a frozen parameter would drift. Skipping leaves it exactly where it was.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .._typing import Array
from ..errors import OptimizerError
from ..tensor import Tensor

__all__ = ["Optimizer"]


class Optimizer:
    """Base class: parameter bookkeeping, state serialisation, `zero_grad`.

    Subclasses implement `update`, which is handed one parameter, its gradient
    and its state dictionary, and returns the array to assign. Everything else -
    iteration, skipping, the step counter, saving and loading - happens here so
    that four optimisers cannot have four subtly different ideas of what a step
    is.
    """

    def __init__(self, parameters: Iterable[Tensor], lr: float) -> None:
        self.parameters: list[Tensor] = list(parameters)
        if not self.parameters:
            msg = "an optimiser needs at least one parameter; got an empty iterable"
            raise OptimizerError(msg)
        seen: set[int] = set()
        for parameter in self.parameters:
            if id(parameter) in seen:
                msg = (
                    "the same tensor appears twice in the parameter list, so it would be "
                    "updated twice per step and move at double the learning rate. "
                    "Module.parameters() already deduplicates; combine parameter lists with "
                    "a dict keyed by id if you are building one by hand."
                )
                raise OptimizerError(msg)
            if not parameter.is_leaf:
                msg = (
                    "an optimiser can only update leaf tensors; one of the parameters was "
                    f"produced by {parameter.node.op if parameter.node else '?'}"
                )
                raise OptimizerError(msg)
            seen.add(id(parameter))
        self.lr = _positive("lr", lr)
        self.step_count = 0
        self.state: list[dict[str, Array]] = [{} for _ in self.parameters]

    # -- the step ----------------------------------------------------------- #

    def update(self, parameter: Tensor, gradient: Array, state: dict[str, Array]) -> Array:
        """Return the new values for one parameter. Subclasses implement this."""
        msg = f"{type(self).__name__} does not implement update()"
        raise NotImplementedError(msg)

    def step(self) -> None:
        """Apply one update to every parameter that has a gradient."""
        self.step_count += 1
        for parameter, state in zip(self.parameters, self.state, strict=True):
            gradient = parameter.grad
            if gradient is None:
                continue
            parameter.assign_(self.update(parameter, gradient, state))

    def zero_grad(self) -> None:
        """Drop every parameter's accumulated gradient.

        Gradients accumulate (see `nabla.grad.backward`), so a loop that forgets
        this sums the gradients of every step it has ever taken. It is offered
        here as well as on `Module` because a training loop holds an optimiser
        and may not hold the model.
        """
        for parameter in self.parameters:
            parameter.zero_grad()

    def set_lr(self, lr: float) -> None:
        """Replace the learning rate, as a schedule does between steps."""
        self.lr = _positive("lr", lr, allow_zero=True)

    # -- serialisation ------------------------------------------------------ #

    def hyperparameters(self) -> dict[str, Any]:
        """The scalar configuration, for `state_dict` and `__repr__`."""
        return {"lr": self.lr}

    def state_dict(self) -> dict[str, Any]:
        """The optimiser's full state: hyperparameters, step count, per-parameter arrays.

        Arrays are copied. The moments are read and written every step, and a
        checkpoint that aliased them would change after it was taken.
        """
        return {
            "type": type(self).__name__,
            "hyperparameters": dict(self.hyperparameters()),
            "step_count": self.step_count,
            "state": [{k: np.array(v, copy=True) for k, v in s.items()} for s in self.state],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from `state_dict()`, refusing anything that does not match.

        The type name and the number of parameters are both checked. Loading an
        Adam state into an SGD would leave `exp_avg` sitting unused in a
        dictionary, and loading a state for a different model would pair the
        wrong moments with the wrong weights - neither raises on its own, and
        both produce a run that trains slightly wrong for the rest of its life.
        """
        expected = type(self).__name__
        if state.get("type") != expected:
            msg = f"this is a {expected} but the state dict was written by a {state.get('type')!r}"
            raise OptimizerError(msg)
        entries = state["state"]
        if len(entries) != len(self.parameters):
            msg = (
                f"the state dict describes {len(entries)} parameters but this optimiser holds "
                f"{len(self.parameters)}; the model it was saved from had a different shape"
            )
            raise OptimizerError(msg)
        for name, value in state["hyperparameters"].items():
            setattr(self, name, value)
        self.step_count = int(state["step_count"])
        self.state = [{k: np.array(v, copy=True) for k, v in entry.items()} for entry in entries]

    # -- display ------------------------------------------------------------ #

    def __repr__(self) -> str:
        settings = ", ".join(f"{k}={v}" for k, v in self.hyperparameters().items())
        return f"{type(self).__name__}({settings}, parameters={len(self.parameters)})"


def _positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Validate a hyperparameter that has to be positive.

    Checked at construction rather than at the first step, because a negative
    learning rate produces gradient *ascent*: the loss goes up smoothly and
    every other diagnostic looks normal.
    """
    if value < 0 or (value == 0 and not allow_zero):
        msg = f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value}"
        raise OptimizerError(msg)
    return float(value)


def _unit_interval(name: str, value: float) -> float:
    """Validate a decay coefficient, which must lie in `[0, 1)`."""
    if not 0.0 <= value < 1.0:
        msg = f"{name} must be in [0, 1), got {value}"
        raise OptimizerError(msg)
    return float(value)
