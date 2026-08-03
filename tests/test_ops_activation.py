"""Activations, with particular attention to the inputs that overflow."""

from __future__ import annotations

import math

import numpy as np
import pytest

import nabla as nb
from conftest import assert_close, check, sample

X = sample(4, 5, seed=31)
# Far enough from zero that relu's kink is never probed by the difference
# quotient: check_grad perturbs by 1e-6, and every |x| here exceeds 1e-3.
AWAY_FROM_ZERO = np.where(np.abs(X) < 1e-3, 0.5, X)


def test_relu_forward_and_backward() -> None:
    assert_close(nb.relu(nb.Tensor(X)), np.maximum(X, 0.0))
    check(nb.relu, AWAY_FROM_ZERO, name="relu")


def test_relu_is_zero_at_zero_in_both_directions() -> None:
    x = nb.Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
    nb.relu(x).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])


def test_sigmoid_forward_and_backward() -> None:
    assert_close(nb.sigmoid(nb.Tensor(X)), 1.0 / (1.0 + np.exp(-X)))
    check(nb.sigmoid, X, name="sigmoid")


def test_sigmoid_does_not_overflow_on_extreme_inputs() -> None:
    extreme = np.array([-800.0, -100.0, 0.0, 100.0, 800.0])
    with np.errstate(over="raise"):
        values = nb.sigmoid(nb.Tensor(extreme)).numpy()
    np.testing.assert_allclose(values, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-12)
    # The naive form is what this is avoiding; assert that it really does fail.
    with np.errstate(over="raise"), pytest.raises(FloatingPointError):
        _ = 1.0 / (1.0 + np.exp(-extreme))


def test_tanh_forward_and_backward() -> None:
    assert_close(nb.tanh(nb.Tensor(X)), np.tanh(X))
    check(nb.tanh, X, name="tanh")


def test_gelu_stays_close_to_the_exact_definition() -> None:
    grid = np.linspace(-6.0, 6.0, 401)
    exact = grid * 0.5 * (1.0 + np.vectorize(math.erf)(grid / math.sqrt(2.0)))
    approximate = nb.gelu(nb.Tensor(grid)).numpy()
    assert np.max(np.abs(approximate - exact)) < 5e-4


def test_gelu_backward_is_the_exact_derivative_of_what_it_computes() -> None:
    check(nb.gelu, X, name="gelu")


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_softmax_forward_and_backward(axis: int) -> None:
    reference = np.exp(X) / np.sum(np.exp(X), axis=axis, keepdims=True)
    assert_close(nb.softmax(nb.Tensor(X), axis=axis), reference)
    # Checked unreduced: softmax's rows sum to 1, so the gradient of a plain
    # sum is exactly zero and would hide almost any error in the vjp.
    # check_grad's random output weighting is what makes this informative.
    check(lambda x: nb.softmax(x, axis=axis), X, name="softmax")


def test_softmax_survives_logits_that_overflow_the_naive_form() -> None:
    logits = np.array([1000.0, 1000.0, 1000.0, 999.0])
    with np.errstate(over="raise"):
        probabilities = nb.softmax(nb.Tensor(logits)).numpy()
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    expected = np.exp(logits - 1000.0) / np.exp(logits - 1000.0).sum()
    np.testing.assert_allclose(probabilities, expected)

    # Without the max subtraction this is inf/inf.
    with np.errstate(over="raise"), pytest.raises(FloatingPointError):
        _ = np.exp(logits) / np.exp(logits).sum()


def test_softmax_of_large_logits_still_has_a_finite_gradient() -> None:
    x = nb.Tensor(np.array([1000.0, 1001.0, 999.0]), requires_grad=True)
    weights = np.array([1.0, 2.0, 3.0])
    (nb.softmax(x) * nb.Tensor(weights)).sum().backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))


@pytest.mark.parametrize("axis", [-1, 0])
def test_log_softmax_forward_and_backward(axis: int) -> None:
    reference = X - np.log(np.sum(np.exp(X), axis=axis, keepdims=True))
    assert_close(nb.log_softmax(nb.Tensor(X), axis=axis), reference)
    check(lambda x: nb.log_softmax(x, axis=axis), X, name="log_softmax")


def test_log_softmax_is_finite_where_log_of_softmax_is_not() -> None:
    # -800 underflows exp to exactly zero in float64, so the naive form takes
    # the log of 0 and loses the value entirely.
    logits = np.array([0.0, -200.0, -800.0])
    result = nb.log_softmax(nb.Tensor(logits)).numpy()
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[1], -200.0, atol=1e-12)
    np.testing.assert_allclose(result[2], -800.0, atol=1e-12)

    with np.errstate(divide="ignore"):
        naive = np.log(np.exp(logits) / np.exp(logits).sum())
    assert np.isneginf(naive[2])


def test_activations_preserve_float32() -> None:
    x = nb.tensor(X.astype(np.float32))
    for op in (nb.relu, nb.sigmoid, nb.tanh, nb.gelu, nb.softmax, nb.log_softmax):
        assert op(x).dtype == np.float32
