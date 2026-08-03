"""Losses, and the overflow each one is written to avoid.

The stability tests here do not merely assert that the stable form is finite.
They compute the naive form alongside it on the same inputs and assert that the
naive form is *not* finite, so the test would fail if the inputs were quietly
made tame enough for both to work. A stability test that no unstable
implementation would fail is not testing stability.
"""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from conftest import assert_close, check, sample
from nabla import nn
from nabla.errors import ShapeError
from nabla.nn import loss

LOGITS = sample(6, 4, seed=21, scale=2.0)
TARGETS = np.array([0, 3, 1, 1, 2, 0])


# --------------------------------------------------------------------------- #
# mse
# --------------------------------------------------------------------------- #


def test_mse_forward_and_reductions() -> None:
    prediction = sample(4, 3, seed=22)
    target = sample(4, 3, seed=23)
    squared = (prediction - target) ** 2
    assert_close(loss.mse(nb.Tensor(prediction), target), np.asarray(squared.mean()))
    assert_close(
        loss.mse(nb.Tensor(prediction), target, reduction="sum"), np.asarray(squared.sum())
    )
    assert_close(loss.mse(nb.Tensor(prediction), target, reduction="none"), squared)


def test_mse_gradients() -> None:
    target = sample(4, 3, seed=23)
    check(lambda p: loss.mse(p, target, reduction="none"), sample(4, 3, seed=22), name="mse")


def test_mse_does_not_differentiate_the_target() -> None:
    target = nb.Tensor(sample(4, 3, seed=23), requires_grad=True)
    loss.mse(nb.Tensor(sample(4, 3, seed=22), requires_grad=True), target).backward()
    assert target.grad is None


def test_mse_refuses_a_shape_mismatch() -> None:
    with pytest.raises(ShapeError, match="but target has"):
        loss.mse(nb.zeros(4, 3), np.zeros((4, 2)))


def test_reduction_must_be_one_of_three_words() -> None:
    with pytest.raises(ValueError, match="'mean', 'sum' or 'none'"):
        loss.mse(nb.zeros(2), np.zeros(2), reduction="average")


# --------------------------------------------------------------------------- #
# cross entropy
# --------------------------------------------------------------------------- #


def test_cross_entropy_matches_the_definition_on_tame_inputs() -> None:
    shifted = LOGITS - LOGITS.max(-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
    expected = -np.log(probabilities[np.arange(len(TARGETS)), TARGETS]).mean()
    assert_close(loss.cross_entropy(nb.Tensor(LOGITS), TARGETS), np.asarray(expected))


def test_cross_entropy_is_nll_of_log_softmax() -> None:
    """The definition, asserted, so the two can never drift apart."""
    logits = nb.Tensor(LOGITS)
    assert_close(
        loss.cross_entropy(logits, TARGETS),
        loss.nll(nb.log_softmax(logits, axis=-1), TARGETS).numpy(),
    )


def test_cross_entropy_survives_logits_that_overflow_the_naive_form() -> None:
    huge = np.array([[1e4, -1e4, 0.0, 0.0], [0.0, 3e4, 0.0, 0.0]])
    targets = np.array([1, 0])

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        naive_probabilities = np.exp(huge) / np.exp(huge).sum(-1, keepdims=True)
        naive = -np.log(naive_probabilities[np.arange(2), targets])
    assert not np.all(np.isfinite(naive)), "the naive form must actually break on these inputs"

    stable = loss.cross_entropy(nb.Tensor(huge, dtype="float64"), targets)
    assert np.isfinite(stable.item())
    # logsumexp(z) - z[target]: 1e4 - (-1e4) for the first row, 3e4 for the second.
    assert stable.item() == pytest.approx((2e4 + 3e4) / 2, rel=1e-9)


def test_cross_entropy_gradients_stay_bounded_at_extreme_logits() -> None:
    """The gradient is softmax - onehot, so it cannot leave [-1, 1]."""
    logits = nb.Tensor(np.array([[1e4, -1e4, 0.0]]), dtype="float64", requires_grad=True)
    loss.cross_entropy(logits, np.array([1])).backward()
    assert logits.grad is not None
    assert np.all(np.isfinite(logits.grad))
    assert float(np.abs(logits.grad).max()) <= 1.0


def test_cross_entropy_gradients() -> None:
    check(
        lambda z: loss.cross_entropy(z, TARGETS, reduction="none"),
        LOGITS,
        name="cross_entropy",
    )


def test_cross_entropy_handles_a_sequence_shaped_batch() -> None:
    logits = sample(2, 5, 7, seed=24)
    targets = np.arange(10).reshape(2, 5) % 7
    per_position = loss.cross_entropy(nb.Tensor(logits), targets, reduction="none")
    assert per_position.shape == (2, 5)
    assert_close(
        loss.cross_entropy(nb.Tensor(logits), targets), np.asarray(per_position.numpy().mean())
    )


def test_cross_entropy_refuses_float_targets() -> None:
    with pytest.raises(ShapeError, match="must be integers"):
        loss.cross_entropy(nb.Tensor(LOGITS), TARGETS.astype(float))


def test_cross_entropy_refuses_an_out_of_range_class() -> None:
    with pytest.raises(ShapeError, match="out of range"):
        loss.cross_entropy(nb.Tensor(LOGITS), np.array([0, 1, 2, 3, 4, 9]))


def test_cross_entropy_refuses_the_wrong_number_of_targets() -> None:
    with pytest.raises(ShapeError, match="one class index per example"):
        loss.cross_entropy(nb.Tensor(LOGITS), np.array([0, 1]))


# --------------------------------------------------------------------------- #
# binary cross entropy
# --------------------------------------------------------------------------- #


def test_binary_cross_entropy_matches_the_definition_on_tame_inputs() -> None:
    logits = sample(5, seed=25, scale=2.0)
    labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    expected = -(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    assert_close(loss.binary_cross_entropy(nb.Tensor(logits), labels, reduction="none"), expected)


def test_binary_cross_entropy_survives_logits_that_saturate_the_sigmoid() -> None:
    logits = np.array([-800.0, 800.0])
    labels = np.array([1.0, 0.0])

    with np.errstate(over="ignore", divide="ignore"):
        naive_probabilities = 1.0 / (1.0 + np.exp(-logits))
        naive = -(labels * np.log(naive_probabilities))
    assert not np.all(np.isfinite(naive)), "the naive form must actually break on these inputs"

    stable = loss.binary_cross_entropy(nb.Tensor(logits, dtype="float64"), labels, reduction="none")
    np.testing.assert_allclose(stable.numpy(), [800.0, 800.0])


def test_binary_cross_entropy_gradients() -> None:
    labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    check(
        lambda z: loss.binary_cross_entropy(z, labels, reduction="none"),
        sample(5, seed=25, scale=2.0),
        name="binary_cross_entropy",
    )


def test_binary_cross_entropy_is_minimised_at_the_label() -> None:
    labels = np.array([1.0, 0.0])
    confident = loss.binary_cross_entropy(nb.Tensor(np.array([10.0, -10.0])), labels).item()
    wrong = loss.binary_cross_entropy(nb.Tensor(np.array([-10.0, 10.0])), labels).item()
    assert confident < 1e-3 < wrong


def test_binary_cross_entropy_refuses_a_shape_mismatch() -> None:
    with pytest.raises(ShapeError, match="logits have shape"):
        loss.binary_cross_entropy(nb.zeros(4), np.zeros(3))


# --------------------------------------------------------------------------- #
# nll
# --------------------------------------------------------------------------- #


def test_nll_picks_the_target_log_probability() -> None:
    log_probabilities = nb.log_softmax(nb.Tensor(LOGITS), axis=-1)
    picked = log_probabilities.numpy()[np.arange(len(TARGETS)), TARGETS]
    assert_close(loss.nll(log_probabilities, TARGETS, reduction="none"), -picked)


def test_nll_gradients() -> None:
    check(
        lambda z: loss.nll(nb.log_softmax(z, axis=-1), TARGETS, reduction="none"),
        LOGITS,
        name="nll",
    )


def test_losses_are_reachable_from_the_nn_namespace() -> None:
    assert nn.cross_entropy is loss.cross_entropy
    assert nn.mse is loss.mse
    assert nn.binary_cross_entropy is loss.binary_cross_entropy
    assert nn.nll is loss.nll
