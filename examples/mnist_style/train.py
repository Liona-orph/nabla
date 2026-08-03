"""A ten-class classifier on generated 8x8 digit-like images.

Nothing is downloaded. `nabla.data.make_digits` renders ten hand-drawn 8x8
glyphs, translates each one by up to two pixels in each direction, scales its
intensity and adds Gaussian noise. The whole dataset is a pure function of a
seed, so the accuracy printed below is reproducible on any machine.

What this is *not*: MNIST. It is a jittered bitmap font, and the honest way to
read the accuracy is "the framework's gradients, initialisers and optimiser all
work on a ten-class problem", not "this framework does computer vision". The
translation is what stops it being trivial - without it every class is a single
fixed vector - and `tests/test_data.py` asserts that the classes are not
separable by total ink, so the model has to look at the shapes.

There is no convolution in this package, so the images are flattened and a plain
MLP does the work. That costs the translation invariance a convolution would
have given for free, which is exactly why the shifted digits are harder.

Run it with:

    python examples/mnist_style/train.py
"""

from __future__ import annotations

import time

import numpy as np

import nabla as nb
from nabla import data, nn, optim, train

SEED = 20240601
TRAIN_SAMPLES = 6000
TEST_SAMPLES = 2000
HIDDEN = 96
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

#: How hard the problem is. At `NOISE = 0.15` and `SHIFT = 1` a plain MLP gets
#: every test example right, which is a number that says nothing; at these
#: settings it does not, so the accuracy below is a measurement rather than a
#: formality. Both are arguments to `make_digits`, so the difficulty is visible
#: here rather than buried in the generator.
NOISE = 0.25
SHIFT = 2


def build_model(generator: nb.random.Generator) -> nn.Module:
    """64 -> 96 -> 96 -> 10, with ReLU, dropout and Kaiming initialisation.

    Kaiming rather than the `Linear` default of Xavier, because these layers are
    followed by ReLU and the factor of `sqrt(2)` is exactly the correction for
    it. On a network this shallow the difference is small; the point of writing
    it out is that the choice is a decision and not a default.
    """

    def kaiming(shape: tuple[int, ...], g: nb.random.Generator) -> nb.Tensor:
        return nn.init.kaiming_normal(shape, g, nonlinearity="relu")

    return nn.Sequential(
        nn.Linear(64, HIDDEN, generator=generator, initialiser=kaiming),
        nn.ReLU(),
        nn.Dropout(DROPOUT, generator=generator),
        nn.Linear(HIDDEN, HIDDEN, generator=generator, initialiser=kaiming),
        nn.ReLU(),
        nn.Dropout(DROPOUT, generator=generator),
        nn.Linear(HIDDEN, 10, generator=generator),
    )


def confusion(model: nn.Module, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The 10x10 confusion matrix, rows true and columns predicted."""
    with nb.no_grad():
        predicted = np.argmax(model(nb.Tensor(features)).numpy(), axis=-1)
    matrix = np.zeros((10, 10), dtype=np.int64)
    for truth, guess in zip(labels, predicted, strict=True):
        matrix[int(truth), int(guess)] += 1
    return matrix


def render(image: np.ndarray) -> list[str]:
    """One 64-vector as eight rows of eight characters, for the eye."""
    shades = " .:-=+*#%@"
    grid = image.reshape(8, 8)
    return ["".join(shades[min(9, int(v * 10))] for v in row) for row in grid]


def main() -> None:
    """Train the classifier and print the accuracy and a confusion matrix."""
    generator = nb.random.Generator(seed=SEED)
    data_generator, loader_generator, model_generator = generator.spawn(3)

    train_x, train_y = data.make_digits(TRAIN_SAMPLES, data_generator, noise=NOISE, shift=SHIFT)
    test_x, test_y = data.make_digits(TEST_SAMPLES, data_generator, noise=NOISE, shift=SHIFT)

    train_loader = data.DataLoader(
        data.ArrayDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )
    test_loader = data.DataLoader(data.ArrayDataset(test_x, test_y), batch_size=512)

    print("three training examples:")
    rows = [render(train_x[i]) for i in range(3)]
    for line in zip(*rows, strict=True):
        print("   ".join(line))
    print(f"labels: {train_y[:3].tolist()}")
    print()

    model = build_model(model_generator)
    print(model)
    print()

    total_steps = EPOCHS * len(train_loader)
    trainer = train.Trainer(
        model,
        optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        nn.cross_entropy,
        schedule=optim.schedule.cosine_with_warmup(
            LEARNING_RATE, warmup_steps=total_steps // 20, total_steps=total_steps
        ),
        grad_clip=1.0,
        metrics={"accuracy": train.accuracy},
    )

    started = time.perf_counter()
    history = trainer.fit(train_loader, epochs=EPOCHS, val_loader=test_loader, verbose=False)
    elapsed = time.perf_counter() - started

    print(f"{'epoch':>6}  {'train loss':>10}  {'train acc':>9}  {'test loss':>9}  {'test acc':>8}")
    for record in history.epochs:
        epoch = int(record["epoch"])
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(
                f"{epoch:>6}  {record['train_loss']:>10.4f}  {record['train_accuracy']:>9.4f}  "
                f"{record['val_loss']:>9.4f}  {record['val_accuracy']:>8.4f}"
            )

    final = history.last()
    print()
    print(f"trained {model.parameter_count()} parameters for {EPOCHS} epochs in {elapsed:.1f}s")
    print(f"final test accuracy: {final['val_accuracy']:.4f}")
    print()

    model.eval()
    matrix = confusion(model, test_x, test_y)
    print("confusion matrix (rows true, columns predicted):")
    print("     " + "".join(f"{d:>5}" for d in range(10)))
    for digit in range(10):
        print(f"{digit:>4} " + "".join(f"{count:>5}" for count in matrix[digit]))


if __name__ == "__main__":
    main()
