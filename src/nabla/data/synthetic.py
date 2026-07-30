"""Datasets generated from a seed, so that no example ever downloads anything.

An example that fetches MNIST is an example that fails on an aeroplane, fails
behind a corporate proxy, and eventually fails because a URL moved. Both
generators here take a `nabla.random.Generator` and are a pure function of it,
so the data is byte-identical on every machine and the accuracy numbers in the
READMEs are reproducible rather than approximately reproducible.

The trade-off is honest and worth stating: these are *synthetic* problems. Two
moons is not a proxy for anything real, and the digit glyphs below are a
hand-drawn 8x8 font with jitter, not handwriting. A model reaching 97% here has
demonstrated that the framework's gradients are right and its optimiser works,
which is what these examples are for. It has not demonstrated anything about
computer vision.
"""

from __future__ import annotations

import numpy as np

from .._typing import Array
from ..random import Generator

__all__ = ["DIGIT_GLYPHS", "make_digits", "make_moons"]

#: Ten 8x8 glyphs, one per class, drawn by hand. `#` is ink, a space is paper.
#:
#: A bitmap font rather than a procedural generator: the shapes have to be
#: consistently distinguishable for the accuracy number to mean anything, and
#: procedurally generated digit-like shapes are either trivially separable (all
#: the variation lands in one statistic) or not separable at all.
DIGIT_GLYPHS: tuple[tuple[str, ...], ...] = (
    (
        "  ####  ",
        " ##  ## ",
        "##    ##",
        "##    ##",
        "##    ##",
        " ##  ## ",
        "  ####  ",
        "        ",
    ),
    (
        "   ##   ",
        "  ###   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        " ###### ",
        "        ",
    ),
    (
        " ####   ",
        "##  ##  ",
        "    ##  ",
        "   ##   ",
        "  ##    ",
        " ##     ",
        "####### ",
        "        ",
    ),
    (
        " #####  ",
        "##   ## ",
        "     ## ",
        "  ####  ",
        "     ## ",
        "##   ## ",
        " #####  ",
        "        ",
    ),
    (
        "    ##  ",
        "   ###  ",
        "  # ##  ",
        " #  ##  ",
        "####### ",
        "    ##  ",
        "    ##  ",
        "        ",
    ),
    (
        "####### ",
        "##      ",
        "######  ",
        "     ## ",
        "     ## ",
        "##   ## ",
        " #####  ",
        "        ",
    ),
    (
        "  ####  ",
        " ##     ",
        "##      ",
        "######  ",
        "##   ## ",
        "##   ## ",
        " #####  ",
        "        ",
    ),
    (
        "####### ",
        "##   ## ",
        "     ## ",
        "    ##  ",
        "   ##   ",
        "   ##   ",
        "   ##   ",
        "        ",
    ),
    (
        " #####  ",
        "##   ## ",
        "##   ## ",
        " #####  ",
        "##   ## ",
        "##   ## ",
        " #####  ",
        "        ",
    ),
    (
        " #####  ",
        "##   ## ",
        "##   ## ",
        " ###### ",
        "     ## ",
        "    ##  ",
        " ####   ",
        "        ",
    ),
)


def _glyph_array() -> Array:
    """The glyphs as a `(10, 8, 8)` float array of 0 and 1."""
    grid = np.zeros((10, 8, 8), dtype=np.float32)
    for digit, glyph in enumerate(DIGIT_GLYPHS):
        if len(glyph) != 8 or any(len(row) != 8 for row in glyph):
            msg = f"glyph {digit} is not 8 rows of 8 characters"
            raise ValueError(msg)
        for r, row in enumerate(glyph):
            for c, cell in enumerate(row):
                grid[digit, r, c] = 1.0 if cell == "#" else 0.0
    return grid


def make_moons(
    n_samples: int,
    generator: Generator,
    *,
    noise: float = 0.2,
) -> tuple[Array, Array]:
    """Two interleaving half-circles: the standard two-class toy problem.

    Not linearly separable, which is the whole point - a linear model tops out
    around 87% on it, so an MLP reaching 97% has demonstrated that the hidden
    layer is doing something. Small enough that a full training run takes a
    second.

    Returns `(features, labels)` with features of shape `(n, 2)` in float32 and
    labels of shape `(n,)` in int64. Float32 because that is what the model
    wants; int64 because labels are indices, and an index array that is float
    cannot be used as one.

    >>> import nabla
    >>> from nabla.data import make_moons
    >>> x, y = make_moons(200, nabla.random.Generator(seed=0))
    >>> x.shape, y.shape, sorted(set(y.tolist()))
    ((200, 2), (200,), [0, 1])
    """
    if n_samples < 2:
        msg = f"make_moons needs at least 2 samples, got {n_samples}"
        raise ValueError(msg)
    first = n_samples // 2
    second = n_samples - first
    rng = generator.numpy_generator

    outer = np.linspace(0.0, np.pi, first)
    inner = np.linspace(0.0, np.pi, second)
    points = np.concatenate(
        [
            np.stack([np.cos(outer), np.sin(outer)], axis=1),
            np.stack([1.0 - np.cos(inner), 0.5 - np.sin(inner)], axis=1),
        ]
    )
    labels = np.concatenate([np.zeros(first, dtype=np.int64), np.ones(second, dtype=np.int64)])
    points = points + rng.normal(0.0, noise, points.shape)

    # Shuffled here rather than left in class order. A caller who takes the
    # first 80% as a training split would otherwise get one class only, and the
    # resulting 50% accuracy looks like a broken model rather than a broken
    # split.
    order = rng.permutation(n_samples)
    return points[order].astype(np.float32), labels[order]


def make_digits(
    n_samples: int,
    generator: Generator,
    *,
    noise: float = 0.15,
    shift: int = 1,
) -> tuple[Array, Array]:
    """Jittered 8x8 renderings of ten hand-drawn digit glyphs.

    Each example is one glyph, translated by up to `shift` pixels in each
    direction, scaled by a random intensity, and corrupted with Gaussian noise.
    Translation is what makes it non-trivial: without it every class is a single
    fixed vector and a nearest-neighbour classifier is perfect, so the task
    would measure nothing.

    Returns `(features, labels)` with features of shape `(n, 64)` in float32,
    flattened because there is no convolution in this package, and labels of
    shape `(n,)` in int64.

    >>> import nabla
    >>> from nabla.data import make_digits
    >>> x, y = make_digits(50, nabla.random.Generator(seed=0))
    >>> x.shape, y.shape, float(x.min()) >= 0.0, float(x.max()) <= 1.0
    ((50, 64), (50,), True, True)
    """
    if n_samples < 1:
        msg = f"make_digits needs at least 1 sample, got {n_samples}"
        raise ValueError(msg)
    if not 0 <= shift <= 2:
        msg = f"shift must be between 0 and 2 pixels, got {shift}"
        raise ValueError(msg)
    rng = generator.numpy_generator
    glyphs = _glyph_array()

    labels = rng.integers(0, 10, n_samples)
    images = glyphs[labels].copy()

    if shift:
        offsets = rng.integers(-shift, shift + 1, (n_samples, 2))
        for i in range(n_samples):
            images[i] = _translate(images[i], int(offsets[i, 0]), int(offsets[i, 1]))

    intensity = rng.uniform(0.6, 1.0, (n_samples, 1, 1))
    images = images * intensity + rng.normal(0.0, noise, images.shape)
    np.clip(images, 0.0, 1.0, out=images)
    return images.reshape(n_samples, 64).astype(np.float32), labels.astype(np.int64)


def _translate(image: Array, dr: int, dc: int) -> Array:
    """Shift an image, filling the vacated edge with zeros.

    `np.roll` would wrap, which turns a digit shifted off the bottom into one
    with a stripe across the top - a distinctive artefact that a classifier
    would happily learn instead of the digit.
    """
    out = np.zeros_like(image)
    rows, cols = image.shape
    r0, r1 = max(0, dr), min(rows, rows + dr)
    c0, c1 = max(0, dc), min(cols, cols + dc)
    out[r0:r1, c0:c1] = image[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out
