"""What the tape costs, measured against a hand-differentiated NumPy baseline.

Nabla's forward and backward passes are the same NumPy calls a careful person
would write by hand, plus a `Node` per operation, a closure per operation, a
topological sort per backward pass, and a dictionary of cotangents. This program
measures the difference by writing both.

The baseline is a three-layer MLP whose gradients are derived on paper and coded
directly - no tape, no closures, no `Tensor`. The nabla version is the same
model built out of `nn.Linear` and `nn.ReLU`. Both are given the same weights,
and the program asserts that their losses and gradients agree to float32
precision before it times anything: a speed comparison against an
implementation that computes something else is not a comparison.

Two sizes are reported, because the ratio is not a constant. The overhead per
operation is roughly fixed, so it is a large fraction of a small matmul and a
negligible one of a large one. Quoting a single number for "the cost of the
framework" without saying at what size is how benchmarks mislead.

Run it with:

    python benchmarks/overhead.py
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

import nabla as nb
from nabla import nn
from nabla._typing import Array

SEED = 20240601

#: `ROUNDS` alternating blocks of `BLOCK` steps each, so 200 steps per
#: implementation per configuration. Enough that the minimum has been reached
#: and the whole program still finishes in about seven seconds.
ROUNDS = 5
BLOCK = 40
WARMUP = 20


#: One implementation's training step: inputs, labels, its own state.
Step = Callable[[Array, Array, Any], tuple[float, "list[Array]"]]


class Weights(NamedTuple):
    """The six arrays both implementations share."""

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    w3: Array
    b3: Array


class Result(NamedTuple):
    """One timing, in seconds per training step."""

    label: str
    seconds: float


def make_weights(inputs: int, hidden: int, classes: int, generator: nb.random.Generator) -> Weights:
    """Kaiming-initialised weights, shared by both implementations."""
    return Weights(
        w1=nn.init.kaiming_normal((inputs, hidden), generator).numpy(),
        b1=np.zeros(hidden, dtype=np.float32),
        w2=nn.init.kaiming_normal((hidden, hidden), generator).numpy(),
        b2=np.zeros(hidden, dtype=np.float32),
        w3=nn.init.kaiming_normal((hidden, classes), generator).numpy(),
        b3=np.zeros(classes, dtype=np.float32),
    )


def baseline_step(x: Array, y: Array, w: Any) -> tuple[float, list[Array]]:
    """Forward and backward of the MLP, differentiated by hand.

    The gradients are the textbook ones: `dW = a^T dz` and `da = dz W^T` for each
    affine layer, `dz = da * (z > 0)` for each ReLU, and `softmax(z) - onehot`
    for the cross-entropy - which is the whole reason a classifier's last layer
    and its loss are usually fused.
    """
    batch = x.shape[0]
    z1 = x @ w.w1 + w.b1
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ w.w2 + w.b2
    a2 = np.maximum(z2, 0.0)
    z3 = a2 @ w.w3 + w.b3

    shifted = z3 - z3.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=-1, keepdims=True)
    loss = float(-np.log(probabilities[np.arange(batch), y]).mean())

    dz3 = probabilities.copy()
    dz3[np.arange(batch), y] -= 1.0
    dz3 /= batch
    dw3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ w.w3.T
    dz2 = da2 * (z2 > 0)
    dw2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ w.w2.T
    dz1 = da1 * (z1 > 0)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, [dw1, db1, dw2, db2, dw3, db3]


def build_nabla_model(w: Weights, generator: nb.random.Generator) -> nn.Module:
    """The same MLP as modules, with the baseline's weights loaded into it."""
    inputs, hidden = w.w1.shape
    classes = w.w3.shape[1]
    model = nn.Sequential(
        nn.Linear(inputs, hidden, generator=generator),
        nn.ReLU(),
        nn.Linear(hidden, hidden, generator=generator),
        nn.ReLU(),
        nn.Linear(hidden, classes, generator=generator),
    )
    model.load_state_dict(
        {
            "0.weight": w.w1,
            "0.bias": w.b1,
            "2.weight": w.w2,
            "2.bias": w.b2,
            "4.weight": w.w3,
            "4.bias": w.b3,
        }
    )
    return model


def nabla_step(x: Array, y: Array, model: Any) -> tuple[float, list[Array]]:
    """Forward and backward of the same MLP, through the tape."""
    model.zero_grad()
    loss = nn.cross_entropy(model(nb.Tensor(x)), y)
    loss.backward()
    gradients = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        gradients.append(parameter.grad)
    return loss.item(), gradients


def time_pair(
    x: Array,
    y: Array,
    *,
    first: Step,
    first_state: Any,
    second: Step,
    second_state: Any,
) -> tuple[float, float]:
    """Time two implementations against each other and return the best per-step cost of each.

    Three decisions here, and all of them are about measuring on a shared
    machine rather than about Nabla.

    **Blocks, not single calls.** Each timing covers `BLOCK` consecutive steps
    and is divided by `BLOCK`. At a tenth of a millisecond a step the cost of
    `perf_counter` itself is a visible fraction of the measurement, and a block
    amortises it away. It also keeps each implementation's allocation pattern
    warm, which is the state it is in during a real training loop.

    **Alternating blocks.** A block of the first, then a block of the second,
    `ROUNDS` times over. Timing all of one and then all of the other on a
    container whose neighbours come and go measures the neighbours; alternating
    means a drift in load lands on both.

    **Minimum, not mean or median.** Noise in a microbenchmark is one-sided: a
    preemption, a page fault or a cache eviction can only make a run slower.
    The minimum is therefore the best estimate of the cost of the work itself,
    and it is the statistic that does not move with the load. On the machine the
    README's numbers came from, the median of this benchmark moved by 15%
    between consecutive runs and the minimum by under 2%.
    """
    for _ in range(WARMUP):
        first(x, y, first_state)
        second(x, y, second_state)

    def block(step: Step, state: Any) -> float:
        started = time.perf_counter()
        for _ in range(BLOCK):
            step(x, y, state)
        return (time.perf_counter() - started) / BLOCK

    best_first = math.inf
    best_second = math.inf
    for _ in range(ROUNDS):
        best_first = min(best_first, block(first, first_state))
        best_second = min(best_second, block(second, second_state))
    return best_first, best_second


def compare(batch: int, inputs: int, hidden: int, classes: int) -> tuple[Result, Result]:
    """Check the two implementations agree, then time both."""
    generator = nb.random.Generator(seed=SEED)
    weights = make_weights(inputs, hidden, classes, generator)
    x = generator.normal((batch, inputs)).numpy()
    y = generator.integers(0, classes, batch)

    model = build_nabla_model(weights, generator)
    baseline_loss, baseline_grads = baseline_step(x, y, weights)
    taped_loss, taped_grads = nabla_step(x, y, model)

    assert abs(baseline_loss - taped_loss) < 1e-5, (baseline_loss, taped_loss)
    for hand, taped in zip(baseline_grads, taped_grads, strict=True):
        np.testing.assert_allclose(hand, taped, rtol=2e-4, atol=2e-6)

    by_hand, taped_seconds = time_pair(
        x,
        y,
        first=baseline_step,
        first_state=weights,
        second=nabla_step,
        second_state=model,
    )
    return Result("numpy by hand", by_hand), Result("nabla", taped_seconds)


def main() -> None:
    """Print the comparison at two sizes."""
    sizes = [
        ("small  (batch 32, 784-128-128-10)", 32, 784, 128, 10),
        ("large  (batch 512, 784-1024-1024-10)", 512, 784, 1024, 10),
    ]
    print(f"{'configuration':<40}{'numpy':>12}{'nabla':>12}{'ratio':>10}")
    for label, batch, inputs, hidden, classes in sizes:
        hand, taped = compare(batch, inputs, hidden, classes)
        print(
            f"{label:<40}{hand.seconds * 1e3:>10.3f}ms{taped.seconds * 1e3:>10.3f}ms"
            f"{taped.seconds / hand.seconds:>9.2f}x"
        )
    print()
    print(
        f"fastest of {ROUNDS} alternating blocks of {BLOCK} steps, after {WARMUP} warmup "
        "steps; gradients asserted equal first"
    )


if __name__ == "__main__":
    main()
