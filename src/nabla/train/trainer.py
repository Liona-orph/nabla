"""A training loop with the four things that are always added back later.

Gradient clipping, checkpointing, metric logging, early stopping. Each of them
is five lines that everybody writes by hand, gets slightly wrong, and then
copies between projects; this module writes them once.

# What the loop assumes

A batch is a tuple whose last element is the target and whose earlier elements
are the model's positional inputs. That covers `(x, y)` and `(a, b, y)` and does
not cover a model with two outputs, which would need a loss function taking a
tuple - and the loss function is a parameter, so a caller who needs that can
pass one and stop using `Trainer`.

# The order of operations in a step

    zero_grad -> forward -> backward -> clip -> step

Clipping goes after the backward pass and before the optimiser, which is the
only place it can go, and the reason it is worth saying is that clipping *the
update* instead of the gradient is a different algorithm - it interacts with
Adam's denominator, and it does not bound the thing that exploded.

# What early stopping actually measures

The validation metric, compared against the best seen so far with a `min_delta`
that a change must exceed to count as an improvement. Without `min_delta` a run
whose validation loss is flat but noisy never stops, because a fresh minimum by
1e-9 arrives every few epochs. The counter resets on improvement, so `patience`
is "epochs since the last real improvement", not "epochs since the start".
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import NablaError
from ..grad import clip_grad_norm_
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..optim.schedule import Schedule
from ..tensor import Tensor, no_grad
from .checkpoint import load_checkpoint, save_checkpoint

__all__ = ["EarlyStopping", "History", "Trainer", "accuracy"]

#: A loss takes the model's output and the batch target, and returns a scalar.
LossFn = Callable[[Tensor, Tensor], Tensor]

#: A metric takes the same two arguments and returns a plain float, so that
#: nothing in the history keeps a tape alive.
MetricFn = Callable[[Tensor, Tensor], float]


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Fraction of examples whose arg-max logit is the target class.

    A plain float, computed outside the tape. A metric that returned a tensor
    would hold the whole graph of the batch it was computed on alive for as long
    as the history existed, which for a hundred-epoch run is the entire training
    set.
    """
    predicted = np.argmax(logits.numpy(copy=False), axis=-1)
    truth = targets.numpy(copy=False)
    return float(np.mean(predicted == truth))


@dataclass(frozen=True)
class EarlyStopping:
    """When to give up on a run that has stopped improving.

    `mode="min"` for a loss, `"max"` for an accuracy. `min_delta` is the amount
    by which the metric must beat the best seen so far to count; the default of
    zero means any improvement counts, which is usually too generous for a noisy
    validation set.
    """

    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    monitor: str = "val_loss"

    def __post_init__(self) -> None:
        """Validate the configuration at construction, not at the first epoch."""
        if self.patience < 1:
            msg = f"patience must be at least 1, got {self.patience}"
            raise ValueError(msg)
        if self.mode not in ("min", "max"):
            msg = f"mode must be 'min' or 'max', got {self.mode!r}"
            raise ValueError(msg)

    def is_improvement(self, value: float, best: float) -> bool:
        """True if `value` beats `best` by more than `min_delta`."""
        if self.mode == "min":
            return value < best - self.min_delta
        return value > best + self.min_delta

    @property
    def worst(self) -> float:
        """The starting value for `best`: whichever infinity loses to everything."""
        return math.inf if self.mode == "min" else -math.inf


@dataclass
class History:
    """Per-epoch metrics, and which epoch was best.

    A list of flat dictionaries rather than a dictionary of lists, because an
    epoch that recorded a metric the others did not - a validation pass that ran
    every fifth epoch - stays representable.
    """

    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_value: float = math.nan
    stopped_early: bool = False

    def append(self, record: Mapping[str, float]) -> None:
        """Record one epoch."""
        self.epochs.append(dict(record))

    def series(self, name: str) -> list[float]:
        """One metric across the epochs that recorded it."""
        return [epoch[name] for epoch in self.epochs if name in epoch]

    def last(self) -> dict[str, float]:
        """The most recent epoch's record."""
        if not self.epochs:
            msg = "no epochs have been recorded"
            raise NablaError(msg)
        return self.epochs[-1]


class Trainer:
    """Fit a model, with clipping, schedules, checkpointing and early stopping.

    Everything is optional except the model, the optimiser and the loss. A
    `Trainer` with none of the options on is a plain loop, and reading its
    `train_epoch` is the fastest way to see what the framework requires of a
    training step.

    >>> import nabla
    >>> from nabla import data, nn, optim, train
    >>> g = nabla.random.Generator(seed=0)
    >>> x, y = data.make_moons(128, g)
    >>> loader = data.DataLoader(data.ArrayDataset(x, y), batch_size=32, shuffle=True, generator=g)
    >>> model = nn.Sequential(nn.Linear(2, 8, generator=g), nn.Tanh(), nn.Linear(8, 2, generator=g))
    >>> trainer = train.Trainer(model, optim.Adam(model.parameters(), lr=0.05), nn.cross_entropy)
    >>> history = trainer.fit(loader, epochs=5, verbose=False)
    >>> len(history.epochs)
    5
    >>> bool(history.series("train_loss")[-1] < history.series("train_loss")[0])
    True
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
        *,
        schedule: Schedule | None = None,
        grad_clip: float | None = None,
        metrics: Mapping[str, MetricFn] | None = None,
        early_stopping: EarlyStopping | None = None,
        checkpoint_path: str | Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.schedule = schedule
        if grad_clip is not None and grad_clip <= 0:
            msg = f"grad_clip must be positive, got {grad_clip}"
            raise ValueError(msg)
        self.grad_clip = grad_clip
        self.metrics: dict[str, MetricFn] = dict(metrics or {})
        self.early_stopping = early_stopping
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.log = log
        #: Global step count, which is what a schedule is a function of. Counted
        #: in optimiser steps rather than epochs so that a schedule does not
        #: have to know the size of the training set.
        self.global_step = 0

    # -- one epoch ---------------------------------------------------------- #

    def train_epoch(self, loader: Iterable[Sequence[Tensor]]) -> dict[str, float]:
        """Run one pass over `loader` in training mode. Returns averaged metrics.

        Averaged by example count, not by batch count. With a final short batch
        the two differ, and averaging by batch quietly over-weights the last few
        examples of the epoch.
        """
        self.model.train()
        totals: dict[str, float] = {}
        seen = 0
        for batch in loader:
            inputs, target = _unpack(batch)
            count = len(target) if target.ndim else 1

            self.optimizer.zero_grad()
            output = self.model(*inputs)
            loss = self.loss_fn(output, target)
            loss.backward()
            if self.grad_clip is not None:
                norm = clip_grad_norm_(self.optimizer.parameters, self.grad_clip)
                _accumulate(totals, "grad_norm", norm, count)
            if self.schedule is not None:
                self.optimizer.set_lr(self.schedule(self.global_step))
            self.optimizer.step()
            self.global_step += 1

            _accumulate(totals, "loss", loss.item(), count)
            for name, metric in self.metrics.items():
                _accumulate(totals, name, metric(output, target), count)
            seen += count
        return {name: total / seen for name, total in totals.items()} if seen else {}

    def evaluate(self, loader: Iterable[Sequence[Tensor]]) -> dict[str, float]:
        """Run one pass over `loader` in evaluation mode, under `no_grad`.

        `no_grad` matters here for memory rather than for correctness: without
        it the validation pass builds a full tape that nobody calls `backward()`
        on and holds every activation until it is garbage collected.
        """
        self.model.eval()
        totals: dict[str, float] = {}
        seen = 0
        with no_grad():
            for batch in loader:
                inputs, target = _unpack(batch)
                count = len(target) if target.ndim else 1
                output = self.model(*inputs)
                _accumulate(totals, "loss", self.loss_fn(output, target).item(), count)
                for name, metric in self.metrics.items():
                    _accumulate(totals, name, metric(output, target), count)
                seen += count
        return {name: total / seen for name, total in totals.items()} if seen else {}

    # -- the run ------------------------------------------------------------ #

    def fit(
        self,
        train_loader: Iterable[Sequence[Tensor]],
        *,
        epochs: int,
        val_loader: Iterable[Sequence[Tensor]] | None = None,
        verbose: bool = True,
    ) -> History:
        """Train for `epochs`, evaluating and checkpointing along the way."""
        if epochs < 1:
            msg = f"epochs must be at least 1, got {epochs}"
            raise ValueError(msg)
        stopping = self.early_stopping
        history = History(best_value=stopping.worst if stopping else math.inf)
        best = history.best_value
        since_improvement = 0

        for epoch in range(epochs):
            started = time.perf_counter()
            record: dict[str, float] = {"epoch": float(epoch)}
            record.update({f"train_{k}": v for k, v in self.train_epoch(train_loader).items()})
            if val_loader is not None:
                record.update({f"val_{k}": v for k, v in self.evaluate(val_loader).items()})
            record["lr"] = self.optimizer.lr
            record["seconds"] = time.perf_counter() - started
            history.append(record)
            if verbose:
                self.log(_format(record))

            monitored = stopping.monitor if stopping else "val_loss"
            value = record.get(monitored, record.get("train_loss"))
            if value is None:
                continue
            improved = stopping.is_improvement(value, best) if stopping else value < best
            if improved:
                best = value
                history.best_epoch = epoch
                history.best_value = value
                since_improvement = 0
                if self.checkpoint_path is not None:
                    self.save(self.checkpoint_path, epoch=epoch, metric=value)
            else:
                since_improvement += 1
                if stopping is not None and since_improvement >= stopping.patience:
                    history.stopped_early = True
                    if verbose:
                        self.log(
                            f"early stop after epoch {epoch}: no improvement in {monitored} "
                            f"for {stopping.patience} epoch(s); best {best:.6f} at epoch "
                            f"{history.best_epoch}"
                        )
                    break
        return history

    # -- checkpoints -------------------------------------------------------- #

    def save(self, path: str | Path, **extra: Any) -> Path:
        """Write model, optimiser and `extra` to `path`.

        The optimiser goes in with the model. A checkpoint that restores the
        weights but not Adam's moments restarts the second-moment estimate from
        zero, so the first few steps after a resume are the same oversized ones
        that bias correction exists to prevent - which looks like a loss spike
        at exactly the point where somebody restarted the run.
        """
        return save_checkpoint(
            path,
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "extra": dict(extra),
            },
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimiser and step count. Returns the `extra` mapping."""
        payload = load_checkpoint(path)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        loaded: dict[str, Any] = payload.get("extra", {})
        return loaded


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _unpack(batch: Sequence[Tensor]) -> tuple[tuple[Tensor, ...], Tensor]:
    """Split a batch into the model's inputs and the target."""
    if len(batch) < 2:
        msg = (
            f"a batch needs at least an input and a target, got {len(batch)} field(s). "
            "An unsupervised objective can pass the input twice."
        )
        raise NablaError(msg)
    return tuple(batch[:-1]), batch[-1]


def _accumulate(totals: dict[str, float], name: str, value: float, weight: int) -> None:
    """Add a weighted contribution to a running total."""
    totals[name] = totals.get(name, 0.0) + float(value) * weight


def _format(record: Mapping[str, float]) -> str:
    """One epoch as a line of text, integers as integers."""
    parts: list[str] = []
    for name, value in record.items():
        if name == "epoch":
            parts.append(f"epoch {int(value):>4d}")
        elif name == "seconds":
            parts.append(f"{value:6.2f}s")
        else:
            parts.append(f"{name} {value:.4f}")
    return "  ".join(parts)
