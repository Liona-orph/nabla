"""Optimisers and learning-rate schedules.

Four optimisers, one base class, and four schedules that are plain functions of
the step index. The base class owns the parameter list, the step counter and
serialisation; a subclass writes one method, `update`, which is handed a
parameter, its gradient and its state and returns the new values. That split is
the reason the four can be compared: the differences between them are exactly
the differences between four short methods.

>>> import nabla
>>> from nabla import nn, optim
>>> g = nabla.random.Generator(seed=0)
>>> model = nn.Linear(3, 1, generator=g)
>>> opt = optim.AdamW(model.parameters(), lr=0.01)
>>> schedule = optim.schedule.cosine_with_warmup(0.01, warmup_steps=10, total_steps=100)
>>> for step in range(20):
...     opt.set_lr(schedule(step))
...     opt.zero_grad()
...     nn.loss.mse(model(nabla.ones(4, 3)), nabla.zeros(4, 1)).backward()
...     opt.step()
>>> opt.step_count
20
"""

from __future__ import annotations

from . import schedule
from .adam import Adam, AdamW
from .optimizer import Optimizer
from .rmsprop import RMSProp
from .schedule import Schedule, constant, cosine_with_warmup, linear_warmup_decay, step_decay
from .sgd import SGD

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
    "Optimizer",
    "RMSProp",
    "Schedule",
    "constant",
    "cosine_with_warmup",
    "linear_warmup_decay",
    "schedule",
    "step_decay",
]
