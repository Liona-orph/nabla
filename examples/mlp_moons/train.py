"""A two-layer MLP on the two-moons problem.

The smallest end-to-end example: build a dataset, build a model, train it, and
report a number. It exists to be read in one sitting, so every line that is not
about training has been left out.

Two moons is deliberately not linearly separable - a least-squares linear
classifier tops out around 88% on it, and `tests/test_data.py` asserts that -
so an MLP reaching the high nineties has demonstrated that the hidden layer is
doing something and that the gradients through it are right.

Run it with:

    python examples/mlp_moons/train.py
"""

from __future__ import annotations

import time

import numpy as np

import nabla as nb
from nabla import data, nn, optim, train

SEED = 20240601
SAMPLES = 2000
NOISE = 0.20
HIDDEN = 32
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.02


class MLP(nn.Module):
    """Two hidden layers of `tanh`, then a two-way logit head.

    `tanh` rather than ReLU, and it is worth saying why for a problem this
    small: the decision boundary of two moons is curved, a ReLU network's is
    piecewise linear, and with only 32 units the corners are visible in the
    error rate. `tanh` costs one transcendental per unit and produces a smooth
    boundary.
    """

    def __init__(self, generator: nb.random.Generator, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.body = self.register_module(
            "body",
            nn.Sequential(
                nn.Linear(2, hidden, generator=generator),
                nn.Tanh(),
                nn.Linear(hidden, hidden, generator=generator),
                nn.Tanh(),
            ),
        )
        # Two logits rather than one, so the loss is the same `cross_entropy`
        # every other example uses. One logit with a binary cross-entropy is
        # exactly equivalent and one parameter smaller; the uniformity is worth
        # more than the parameter.
        self.head = self.register_module("head", nn.Linear(hidden, 2, generator=generator))

    def forward(self, x: nb.Tensor) -> nb.Tensor:
        """Map a batch of 2-d points to a batch of 2 logits."""
        return self.head(self.body(x))


def decision_boundary(model: nn.Module, extent: float = 2.5, width: int = 48) -> str:
    """Render the classifier's decision regions as text.

    Included because the accuracy alone does not say *what* the model learned,
    and a picture of two interlocking regions does. Evaluated under `no_grad`:
    it is a forward pass over 1,100 points that nobody differentiates.
    """
    height = width // 2
    xs = np.linspace(-extent, extent, width)
    ys = np.linspace(extent, -extent, height)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)
    with nb.no_grad():
        predicted = np.argmax(model(nb.Tensor(grid)).numpy(), axis=-1)
    cells = np.where(predicted.reshape(height, width) == 0, ".", "#")
    return "\n".join("".join(row) for row in cells)


def main() -> None:
    """Train the model and print what happened."""
    generator = nb.random.Generator(seed=SEED)
    features, labels = data.make_moons(SAMPLES, generator, noise=NOISE)

    # 80/20, split before anything is fitted. `make_moons` shuffles its output,
    # so a prefix split is already class-balanced.
    cut = int(0.8 * SAMPLES)
    train_set = data.ArrayDataset(features[:cut], labels[:cut])
    test_set = data.ArrayDataset(features[cut:], labels[cut:])

    loader_generator, model_generator = generator.spawn(2)
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=loader_generator
    )
    test_loader = data.DataLoader(test_set, batch_size=256)

    model = MLP(model_generator)
    print(model)
    print()

    trainer = train.Trainer(
        model,
        optim.Adam(model.parameters(), lr=LEARNING_RATE),
        nn.cross_entropy,
        metrics={"accuracy": train.accuracy},
    )

    started = time.perf_counter()
    history = trainer.fit(train_loader, epochs=EPOCHS, val_loader=test_loader, verbose=False)
    elapsed = time.perf_counter() - started

    print(f"{'epoch':>6}  {'train loss':>10}  {'train acc':>9}  {'test loss':>9}  {'test acc':>8}")
    for record in history.epochs:
        if int(record["epoch"]) % 5 == 0 or int(record["epoch"]) == EPOCHS - 1:
            print(
                f"{int(record['epoch']):>6}  {record['train_loss']:>10.4f}  "
                f"{record['train_accuracy']:>9.4f}  {record['val_loss']:>9.4f}  "
                f"{record['val_accuracy']:>8.4f}"
            )

    final = history.last()
    print()
    print(f"trained {model.parameter_count()} parameters for {EPOCHS} epochs in {elapsed:.2f}s")
    print(f"final test accuracy: {final['val_accuracy']:.4f}")
    print()
    print("decision regions (. = class 0, # = class 1):")
    print(decision_boundary(model))


if __name__ == "__main__":
    main()
