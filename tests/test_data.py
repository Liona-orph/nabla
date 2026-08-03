"""Datasets, batching, determinism, and the synthetic problems."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nabla as nb
from nabla import data
from nabla.errors import ShapeError

FEATURES = np.arange(20.0).reshape(10, 2)
LABELS = np.arange(10)


def loader(seed: int, **kwargs: object) -> data.DataLoader:
    return data.DataLoader(
        data.ArrayDataset(FEATURES, LABELS),
        batch_size=3,
        shuffle=True,
        generator=nb.random.Generator(seed=seed),
        **kwargs,  # type: ignore[arg-type]
    )


def order_of(source: data.DataLoader, epochs: int) -> list[list[int]]:
    return [[int(v) for v in batch[1].numpy()] for _ in range(epochs) for batch in source]


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


def test_array_dataset_indexes_along_the_first_axis() -> None:
    ds = data.ArrayDataset(FEATURES, LABELS)
    assert len(ds) == 10
    np.testing.assert_allclose(ds[4][0], FEATURES[4])
    assert int(ds[4][1]) == 4


def test_array_dataset_refuses_mismatched_lengths() -> None:
    with pytest.raises(ShapeError, match="disagree about the number"):
        data.ArrayDataset(FEATURES, LABELS[:3])


def test_array_dataset_needs_at_least_one_array() -> None:
    with pytest.raises(ShapeError, match="at least one array"):
        data.ArrayDataset()


def test_random_split_covers_the_dataset_exactly() -> None:
    parts = data.random_split(
        data.ArrayDataset(FEATURES, LABELS), [0.6, 0.3, 0.1], nb.random.Generator(seed=0)
    )
    assert [len(p) for p in parts] == [6, 3, 1]
    seen = sorted(int(p[i][1]) for p in parts for i in range(len(p)))
    assert seen == list(range(10))


def test_random_split_is_reproducible() -> None:
    first = data.random_split(data.ArrayDataset(LABELS), [0.5, 0.5], nb.random.Generator(seed=3))
    second = data.random_split(data.ArrayDataset(LABELS), [0.5, 0.5], nb.random.Generator(seed=3))
    np.testing.assert_array_equal(first[0].indices, second[0].indices)


def test_random_split_refuses_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        data.random_split(data.ArrayDataset(LABELS), [0.5, 0.4], nb.random.Generator(seed=0))


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #


def test_batches_are_tensors_of_the_right_shapes() -> None:
    batch = next(iter(loader(0)))
    assert [t.shape for t in batch] == [(3, 2), (3,)]


def test_integer_labels_stay_integers() -> None:
    """A label cast to float cannot be used as a class index."""
    batch = next(iter(loader(0)))
    assert batch[1].dtype.kind in "iu"


def test_the_same_seed_gives_the_same_batches_across_several_epochs() -> None:
    assert order_of(loader(11), 3) == order_of(loader(11), 3)


def test_a_different_seed_gives_different_batches() -> None:
    assert order_of(loader(11), 2) != order_of(loader(12), 2)


def test_successive_epochs_are_shuffled_differently() -> None:
    epochs = order_of(loader(11), 2)
    assert epochs[:4] != epochs[4:]


def test_every_example_appears_exactly_once_per_epoch() -> None:
    seen = [label for batch in order_of(loader(5), 1) for label in batch]
    assert sorted(seen) == list(range(10))


def test_unshuffled_iteration_is_in_order() -> None:
    plain = data.DataLoader(data.ArrayDataset(FEATURES, LABELS), batch_size=4)
    assert order_of(plain, 1) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_drop_last_removes_the_short_batch() -> None:
    dropped = data.DataLoader(data.ArrayDataset(FEATURES, LABELS), batch_size=4, drop_last=True)
    assert len(dropped) == 2
    assert [len(batch[1]) for batch in dropped] == [4, 4]


def test_length_counts_the_short_batch_by_default() -> None:
    assert len(data.DataLoader(data.ArrayDataset(LABELS), batch_size=4)) == 3


def test_shuffling_without_a_generator_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a generator"):
        data.DataLoader(data.ArrayDataset(LABELS), batch_size=2, shuffle=True)


def test_a_dataset_returning_a_bare_array_is_refused() -> None:
    class Bare:
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> np.ndarray:
            return np.zeros(2)

    with pytest.raises(TypeError, match="must return a tuple"):
        data.DataLoader(Bare(), batch_size=2)  # type: ignore[arg-type]


def test_a_zero_batch_size_is_refused() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        data.DataLoader(data.ArrayDataset(LABELS), batch_size=0)


# --------------------------------------------------------------------------- #
# synthetic problems
# --------------------------------------------------------------------------- #


def test_make_moons_is_a_pure_function_of_the_seed() -> None:
    a = data.make_moons(64, nb.random.Generator(seed=2))
    b = data.make_moons(64, nb.random.Generator(seed=2))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_make_moons_is_balanced_and_typed() -> None:
    x, y = data.make_moons(500, nb.random.Generator(seed=1))
    assert x.dtype == np.float32
    assert y.dtype == np.int64
    assert abs(int(y.sum()) - 250) <= 1


def test_make_moons_is_shuffled_so_a_prefix_split_sees_both_classes() -> None:
    _, y = data.make_moons(200, nb.random.Generator(seed=1))
    assert 0 in y[:20]
    assert 1 in y[:20]


def test_make_moons_is_not_linearly_separable() -> None:
    """The property that makes it worth using as an example."""
    x, y = data.make_moons(1000, nb.random.Generator(seed=4), noise=0.1)
    design = np.hstack([x, np.ones((len(x), 1))])
    weights = np.linalg.lstsq(design, 2.0 * y - 1.0, rcond=None)[0]
    predicted = (design @ weights > 0).astype(np.int64)
    assert float(np.mean(predicted == y)) < 0.92


def test_make_digits_is_a_pure_function_of_the_seed() -> None:
    a = data.make_digits(40, nb.random.Generator(seed=6))
    b = data.make_digits(40, nb.random.Generator(seed=6))
    np.testing.assert_array_equal(a[0], b[0])


def test_make_digits_shapes_and_range() -> None:
    x, y = data.make_digits(200, nb.random.Generator(seed=7))
    assert x.shape == (200, 64)
    assert x.dtype == np.float32
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0
    assert set(y.tolist()) <= set(range(10))


def test_every_digit_glyph_is_eight_by_eight_and_distinct() -> None:
    assert len(data.DIGIT_GLYPHS) == 10
    for glyph in data.DIGIT_GLYPHS:
        assert len(glyph) == 8
        assert all(len(row) == 8 for row in glyph)
    assert len({"".join(glyph) for glyph in data.DIGIT_GLYPHS}) == 10


def test_digit_translation_does_not_wrap() -> None:
    """A wrapped shift is a stripe, and a stripe is a feature a classifier learns."""
    x, _ = data.make_digits(400, nb.random.Generator(seed=8), noise=0.0, shift=1)
    images = x.reshape(-1, 8, 8)
    # Row 7 is blank in every glyph, so after a shift of at most one pixel the
    # top row can only contain ink that was already near the top.
    assert float(images[:, 7, :].max()) <= 1.0


def test_digits_are_not_separable_by_total_ink() -> None:
    """Otherwise the task would be measuring one statistic, not ten shapes."""
    x, y = data.make_digits(1000, nb.random.Generator(seed=9))
    ink = x.sum(-1)
    means = [float(ink[y == digit].mean()) for digit in range(10)]
    assert max(means) / min(means) < 2.0


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #


def test_char_vocabulary_is_sorted_and_bijective() -> None:
    vocab = data.CharVocabulary("the quick brown fox")
    assert list(vocab.characters) == sorted(set("the quick brown fox"))
    assert vocab.decode(vocab.encode("fox")) == "fox"
    assert len(vocab) == vocab.size


def test_char_vocabulary_refuses_an_unseen_character() -> None:
    vocab = data.CharVocabulary("abc")
    with pytest.raises(KeyError, match="not in this vocabulary"):
        vocab.encode("z")


def test_char_vocabulary_refuses_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="empty text"):
        data.CharVocabulary("")


def test_next_char_dataset_is_the_window_and_its_shift() -> None:
    ds = data.NextCharDataset(np.arange(8), length=3)
    assert len(ds) == 5
    inputs, targets = ds[2]
    np.testing.assert_array_equal(inputs, [2, 3, 4])
    np.testing.assert_array_equal(targets, [3, 4, 5])


def test_next_char_dataset_refuses_a_corpus_shorter_than_a_window() -> None:
    with pytest.raises(ValueError, match="too short"):
        data.NextCharDataset(np.arange(4), length=8)


def test_the_checked_in_corpus_loads_and_has_the_documented_size() -> None:
    corpus = Path(__file__).resolve().parent.parent / "testdata" / "sonnets.txt"
    text = corpus.read_text(encoding="ascii")
    assert len(text) == 96210
    assert data.CharVocabulary(text).size == 62
