"""Seeded generation, and the global state it refuses to touch."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb


def test_the_same_seed_gives_the_same_values() -> None:
    first = nb.random.Generator(seed=7).normal((3, 4))
    second = nb.random.Generator(seed=7).normal((3, 4))
    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_different_seeds_give_different_values() -> None:
    first = nb.random.Generator(seed=7).normal((3, 4))
    second = nb.random.Generator(seed=8).normal((3, 4))
    assert not np.array_equal(first.numpy(), second.numpy())


def test_nothing_here_touches_numpys_global_stream() -> None:
    """The point of the module.

    A draw from a Nabla generator must not move anyone else's stream, including
    NumPy's legacy global one.
    """
    np.random.seed(0)  # noqa: NPY002 - deliberately exercising the global state
    before = np.random.get_state()  # noqa: NPY002
    generator = nb.random.Generator(seed=1)
    generator.normal((10,))
    generator.uniform((10,))
    generator.permutation(10)
    after = np.random.get_state()  # noqa: NPY002
    assert before[1].tolist() == after[1].tolist()  # type: ignore[index]
    assert before[2:] == after[2:]


def test_a_generator_advances_its_own_stream() -> None:
    generator = nb.random.Generator(seed=3)
    first = generator.normal((5,)).numpy()
    second = generator.normal((5,)).numpy()
    assert not np.array_equal(first, second)


def test_spawned_children_are_independent_of_each_other_and_of_the_parent() -> None:
    parent = nb.random.Generator(seed=11)
    left, right = parent.spawn(2)
    left_values = left.normal((100,)).numpy()
    right_values = right.normal((100,)).numpy()
    assert not np.array_equal(left_values, right_values)
    # A child's values do not depend on how much anyone else drew first.
    again = nb.random.Generator(seed=11)
    again.normal((1000,))  # consume from the parent
    left_again, _ = again.spawn(2)
    np.testing.assert_array_equal(left_again.normal((100,)).numpy(), left_values)


def test_shapes_dtypes_and_grad_flags() -> None:
    generator = nb.random.Generator(seed=5)
    assert generator.normal((2, 3)).shape == (2, 3)
    assert generator.normal(4).shape == (4,)
    assert generator.normal().shape == ()
    assert generator.uniform((2,), low=-1.0, high=1.0).dtype == np.float32
    assert generator.normal((2,), dtype=np.float64).dtype == np.float64
    assert generator.normal((2,), requires_grad=True).requires_grad
    assert not generator.bernoulli((2,)).requires_grad


def test_normal_and_uniform_respect_their_parameters() -> None:
    generator = nb.random.Generator(seed=13)
    values = generator.normal((20000,), mean=5.0, std=2.0, dtype=np.float64).numpy()
    assert abs(float(values.mean()) - 5.0) < 0.05
    assert abs(float(values.std()) - 2.0) < 0.05

    bounded = generator.uniform((10000,), low=-3.0, high=-1.0).numpy()
    assert bounded.min() >= -3.0
    assert bounded.max() < -1.0


def test_bernoulli_is_a_zero_one_mask_with_roughly_the_right_rate() -> None:
    mask = nb.random.Generator(seed=17).bernoulli((20000,), p=0.25).numpy()
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}
    assert abs(float(mask.mean()) - 0.25) < 0.02


def test_bernoulli_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="probability"):
        nb.random.Generator(seed=0).bernoulli((2,), p=1.5)


def test_integer_helpers_return_arrays_not_tensors() -> None:
    generator = nb.random.Generator(seed=19)
    permutation = generator.permutation(6)
    assert isinstance(permutation, np.ndarray)
    assert sorted(permutation.tolist()) == list(range(6))
    integers = generator.integers(0, 5, (10,))
    assert isinstance(integers, np.ndarray)
    assert integers.min() >= 0
    assert integers.max() < 5


def test_a_non_floating_dtype_is_refused() -> None:
    with pytest.raises(TypeError, match="floating dtype"):
        nb.random.Generator(seed=0).normal((2,), dtype=np.int32)


def test_a_generator_can_be_rebuilt_from_a_seed_sequence() -> None:
    sequence = np.random.SeedSequence(23)
    first = nb.random.Generator(sequence).normal((3,)).numpy()
    second = nb.random.Generator(np.random.SeedSequence(23)).normal((3,)).numpy()
    np.testing.assert_array_equal(first, second)
    assert "23" in repr(nb.random.Generator(23))


def test_generated_tensors_own_a_frozen_buffer() -> None:
    x = nb.random.Generator(seed=29).normal((3,))
    with pytest.raises(ValueError, match="read-only"):
        x.data[0] = 1.0
