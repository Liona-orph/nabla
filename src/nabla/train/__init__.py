"""The training loop and the checkpoint format.

`Trainer` is a convenience, not a requirement: a training step in this package
is `zero_grad`, forward, `backward`, optional clip, `step`, and a loop that does
only that is five lines. What `Trainer` adds is the four things that get bolted
on afterwards - clipping, schedules, checkpointing, early stopping - written
once instead of once per project.

`save_checkpoint` and `load_checkpoint` are usable on their own, and are worth
knowing about independently: they write a plain `.npz` with a JSON manifest and
read it back with `allow_pickle=False`, so a checkpoint from an untrusted source
cannot execute anything.
"""

from __future__ import annotations

from .checkpoint import load_checkpoint, save_checkpoint
from .trainer import EarlyStopping, History, Trainer, accuracy

__all__ = [
    "EarlyStopping",
    "History",
    "Trainer",
    "accuracy",
    "load_checkpoint",
    "save_checkpoint",
]
