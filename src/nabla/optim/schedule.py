"""Learning-rate schedules.

# Schedules are pure functions of the step

A schedule here is `Callable[[int], float]` and nothing else. It does not own
the optimiser, it does not hold a step counter, and calling it has no effect. A
training loop reads

    optimizer.set_lr(schedule(step))

which is one line, and the schedule can be plotted, tested and compared without
constructing a model. The stateful alternative - a scheduler object that wraps
the optimiser and is stepped alongside it - has to answer whether `step()` means
"before" or "after" the optimiser's, and gets it wrong often enough that the
question is a recurring bug report in every framework that offers it.

The cost is that the loop has to pass the step number, which it has anyway.

# Why warmup exists

Adam's second moment `v` is estimated from very few samples at the start of a
run, so its variance is high and `1/sqrt(v)` can be far too large for a
parameter that happened to get a small first gradient. The result is a few
enormous steps before anything has been learned, which for a transformer is
usually fatal rather than merely wasteful. Warmup ramps the learning rate from
(near) zero so those early steps are small while the estimate settles. It is a
patch on the optimiser rather than a property of the problem, which is worth
knowing: with a better second-moment estimator it would not be needed.

# Why cosine rather than a step decay

A step decay has to be told when to drop. A cosine has to be told only how long
the run is, spends most of its time at a high learning rate, and finishes
smoothly at the floor, which is what the final-epoch loss actually cares about.
Step decay is kept because it is what a lot of published recipes specify.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeAlias

__all__ = [
    "Schedule",
    "constant",
    "cosine_with_warmup",
    "linear_warmup_decay",
    "step_decay",
]

#: A schedule maps a zero-based step index to a learning rate.
Schedule: TypeAlias = Callable[[int], float]


def constant(lr: float) -> Schedule:
    """The same learning rate forever.

    Exists so that a training loop can be written one way. A loop with
    `if schedule is not None` in it has two code paths, one of which is only
    exercised by the runs that do not use a schedule.

    >>> from nabla.optim import schedule
    >>> [schedule.constant(0.1)(t) for t in (0, 500)]
    [0.1, 0.1]
    """
    return lambda _step: lr


def step_decay(lr: float, *, step_size: int, gamma: float = 0.1) -> Schedule:
    """Multiply by `gamma` every `step_size` steps.

    The classic image-classification recipe. Discontinuous by construction: the
    loss visibly drops at each boundary, which is a real effect and not an
    artefact - a smaller step lets the optimiser settle into a minimum it was
    previously bouncing around.

    >>> from nabla.optim import schedule
    >>> s = schedule.step_decay(1.0, step_size=10, gamma=0.5)
    >>> [s(t) for t in (0, 9, 10, 25)]
    [1.0, 1.0, 0.5, 0.25]
    """
    if step_size <= 0:
        msg = f"step_size must be positive, got {step_size}"
        raise ValueError(msg)
    return lambda step: lr * gamma ** (step // step_size)


def cosine_with_warmup(
    lr: float,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> Schedule:
    """Linear warmup, then a half cosine down to `min_lr`.

    The warmup is linear from 0 to `lr` over `warmup_steps`; the decay follows
    `min_lr + (lr - min_lr) * (1 + cos(pi * progress)) / 2`, which starts flat,
    is steepest halfway, and arrives at `min_lr` with zero slope.

    Beyond `total_steps` the value stays at `min_lr` rather than continuing the
    cosine back up, which is what the unclamped formula would do and which has
    surprised somebody in every codebase that forgot the clamp.

    >>> from nabla.optim import schedule
    >>> s = schedule.cosine_with_warmup(1.0, warmup_steps=10, total_steps=110)
    >>> [round(s(t), 4) for t in (0, 5, 10, 60, 110, 200)]
    [0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
    """
    _check_horizon(warmup_steps, total_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0 or step >= total_steps:
            return min_lr
        progress = (step - warmup_steps) / decay_steps
        return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def linear_warmup_decay(
    lr: float,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> Schedule:
    """Linear warmup, then a straight line down to `min_lr`.

    The schedule the original transformer papers used once the `1/sqrt(t)` form
    fell out of favour, and the one to reach for when a run may be stopped
    early: at any point before the end, the learning rate is higher than the
    cosine's would be, so a truncated run has made more progress.

    >>> from nabla.optim import schedule
    >>> s = schedule.linear_warmup_decay(1.0, warmup_steps=10, total_steps=110)
    >>> [round(s(t), 4) for t in (0, 5, 10, 60, 110, 200)]
    [0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
    """
    _check_horizon(warmup_steps, total_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return lr * step / warmup_steps
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0 or step >= total_steps:
            return min_lr
        progress = (step - warmup_steps) / decay_steps
        return lr + (min_lr - lr) * progress

    return schedule


def _check_horizon(warmup_steps: int, total_steps: int) -> None:
    """Refuse a warmup that never ends or a horizon that ends before it starts."""
    if warmup_steps < 0:
        msg = f"warmup_steps must be non-negative, got {warmup_steps}"
        raise ValueError(msg)
    if total_steps < warmup_steps:
        msg = f"total_steps {total_steps} is shorter than warmup_steps {warmup_steps}"
        raise ValueError(msg)
