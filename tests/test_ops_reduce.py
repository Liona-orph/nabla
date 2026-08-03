"""Reductions: every axis spelling, forwards and backwards."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import nabla as nb
from conftest import assert_close, check, sample
from nabla._typing import Axis
from nabla.errors import ShapeError

X = sample(2, 3, 4, seed=11)
AXES: list[Axis] = [None, 0, 1, 2, -1, (0, 1), (0, 2), (1, 2), (0, 1, 2)]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_and_mean_forward(axis: Axis, keepdims: bool) -> None:
    assert_close(nb.sum(nb.Tensor(X), axis, keepdims=keepdims), np.sum(X, axis, keepdims=keepdims))
    assert_close(
        nb.mean(nb.Tensor(X), axis, keepdims=keepdims), np.mean(X, axis, keepdims=keepdims)
    )


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_and_mean_backward(axis: Axis, keepdims: bool) -> None:
    check(lambda x: nb.sum(x, axis, keepdims=keepdims), X, name="sum")
    check(lambda x: nb.mean(x, axis, keepdims=keepdims), X, name="mean")


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
def test_max_and_min_forward_and_backward(axis: Axis, keepdims: bool) -> None:
    assert_close(nb.max(nb.Tensor(X), axis, keepdims=keepdims), np.max(X, axis, keepdims=keepdims))
    assert_close(nb.min(nb.Tensor(X), axis, keepdims=keepdims), np.min(X, axis, keepdims=keepdims))
    # X is continuous random data, so no group has a tie and the extremum is
    # differentiable everywhere it is evaluated.
    check(lambda x: nb.max(x, axis, keepdims=keepdims), X, name="max")
    check(lambda x: nb.min(x, axis, keepdims=keepdims), X, name="min")


def test_max_routes_the_whole_gradient_to_one_element_per_group() -> None:
    data = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 4.0]])
    x = nb.Tensor(data, requires_grad=True)
    nb.max(x, axis=1).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_max_gives_a_tie_to_the_first_element_like_argmax() -> None:
    x = nb.Tensor(np.array([2.0, 2.0, 1.0]), requires_grad=True)
    nb.max(x).backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [1.0, 0.0, 0.0])


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("ddof", [0, 1])
def test_var_and_std_match_numpy_and_check_numerically(axis: Axis, ddof: int) -> None:
    assert_close(nb.var(nb.Tensor(X), axis, ddof=ddof), np.var(X, axis, ddof=ddof))
    assert_close(nb.std(nb.Tensor(X), axis, ddof=ddof), np.std(X, axis, ddof=ddof))
    check(lambda x: nb.var(x, axis, ddof=ddof), X, name="var")
    check(lambda x: nb.std(x, axis, ddof=ddof), X, name="std")


def test_var_refuses_to_divide_by_a_non_positive_count() -> None:
    with pytest.raises(ShapeError, match="degrees of freedom"):
        nb.var(nb.Tensor(np.zeros(1)), ddof=1)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("keepdims", [False, True])
def test_logsumexp_forward_and_backward(axis: Axis, keepdims: bool) -> None:
    reference = np.log(np.sum(np.exp(X), axis=axis, keepdims=keepdims))
    assert_close(nb.logsumexp(nb.Tensor(X), axis, keepdims=keepdims), reference)
    check(lambda x: nb.logsumexp(x, axis, keepdims=keepdims), X, name="logsumexp")


def test_logsumexp_gradient_is_a_probability_distribution() -> None:
    x = nb.Tensor(X, requires_grad=True)
    nb.logsumexp(x, axis=-1).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.sum(axis=-1), np.ones((2, 3)))


def test_reductions_reject_an_axis_outside_the_rank() -> None:
    with pytest.raises(ShapeError, match="out of range"):
        nb.sum(nb.Tensor(X), axis=3)
    with pytest.raises(ShapeError, match="out of range"):
        nb.sum(nb.Tensor(np.float64(1.0)), axis=0)


def test_reductions_reject_a_repeated_axis() -> None:
    with pytest.raises(ShapeError, match="appears twice"):
        nb.sum(nb.Tensor(X), axis=(0, 0))


def test_a_reduction_of_a_broadcast_input_still_reduces_correctly() -> None:
    """A reduction stacked on a broadcast.

    Both the broadcast reduction and the reduction's own scatter have to be
    right for the gradient shapes to survive to the leaves.
    """
    a = nb.Tensor(np.ones((4, 1, 3)), requires_grad=True)
    b = nb.Tensor(np.ones((2, 3)), requires_grad=True)
    nb.sum(a * b, axis=(0, 2)).sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (4, 1, 3)
    assert b.grad.shape == (2, 3)
    np.testing.assert_allclose(a.grad, np.full((4, 1, 3), 2.0))
    np.testing.assert_allclose(b.grad, np.full((2, 3), 4.0))


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    rank=st.integers(min_value=1, max_value=3),
    keepdims=st.booleans(),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_reductions_over_random_ranks(rank: int, keepdims: bool, seed: int) -> None:
    shape = tuple(np.random.default_rng(seed).integers(1, 4, size=rank))
    data = np.random.default_rng(seed + 1).standard_normal(shape)
    for axis in [None, *range(rank)]:
        for op in (nb.sum, nb.mean, nb.max, nb.logsumexp):
            result = op(nb.Tensor(data), axis, keepdims=keepdims)
            expected_shape = np.sum(data, axis=axis, keepdims=keepdims).shape
            assert result.shape == expected_shape
