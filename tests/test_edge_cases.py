"""The inputs that are not in the middle of the domain.

Each test here asserts what Nabla actually does at a boundary rather than that
it avoids the boundary. Several of the answers are `inf` or `nan`; those are
recorded deliberately, because a framework that quietly substitutes a finite
number for an infinite derivative is differentiating a function the caller never
wrote.
"""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from nabla.errors import ShapeError

# --------------------------------------------------------------------------- #
# zeros and domain boundaries
# --------------------------------------------------------------------------- #


def test_multiplying_by_zero_still_carries_a_gradient() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    (x * 0.0).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [0.0, 0.0])


def test_a_zero_tensor_is_not_a_special_case_for_any_smooth_op() -> None:
    zeros = np.zeros(4)
    for op in (nb.exp, nb.tanh, nb.sigmoid, nb.gelu, nb.relu):
        nb.check_grad(op, zeros + 1e-3, name=op.__name__)
    np.testing.assert_allclose(nb.exp(nb.Tensor(zeros)).numpy(), np.ones(4))


def test_log_at_and_below_zero_gives_the_ieee_answer() -> None:
    x = nb.Tensor(np.array([0.0, -1.0, 1.0]), requires_grad=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = nb.log(x)
        y.sum().backward()
    values = y.numpy()
    assert np.isneginf(values[0])
    assert np.isnan(values[1])
    assert x.grad is not None
    assert np.isinf(x.grad[0])  # d/dx log(x) -> +inf as x -> 0
    np.testing.assert_allclose(x.grad[2], 1.0)


def test_sqrt_at_zero_has_an_infinite_derivative() -> None:
    x = nb.Tensor(np.array([0.0, 4.0]), requires_grad=True)
    with np.errstate(divide="ignore"):
        nb.sqrt(x).sum().backward()
    assert x.grad is not None
    assert np.isinf(x.grad[0])
    np.testing.assert_allclose(x.grad[1], 0.25)


def test_sqrt_of_a_negative_number_is_nan_rather_than_an_exception() -> None:
    with np.errstate(invalid="ignore"):
        result = nb.sqrt(nb.Tensor(np.array([-1.0]))).numpy()
    assert np.isnan(result[0])


def test_division_by_a_small_number_amplifies_the_gradient_rather_than_clipping_it() -> None:
    x = nb.Tensor(np.array([1.0]), requires_grad=True)
    small = nb.Tensor(np.array([1e-8]), requires_grad=True)
    (x / small).backward()
    assert x.grad is not None
    assert small.grad is not None
    np.testing.assert_allclose(x.grad, [1e8])
    np.testing.assert_allclose(small.grad, [-1e16])


def test_division_by_zero_propagates_infinity_in_both_directions() -> None:
    x = nb.Tensor(np.array([1.0]), requires_grad=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x / nb.Tensor(np.array([0.0]))
        y.backward()
    assert np.isinf(y.numpy()[0])
    assert x.grad is not None
    assert np.isinf(x.grad[0])


def test_pow_with_a_fractional_exponent_and_a_negative_base_is_nan() -> None:
    with np.errstate(invalid="ignore"):
        result = nb.pow(nb.Tensor(np.array([-4.0])), 0.5).numpy()
    assert np.isnan(result[0])


def test_zero_to_the_zero_follows_numpy() -> None:
    assert nb.pow(nb.Tensor(np.array([0.0])), 0.0).numpy()[0] == 1.0


def test_pow_of_zero_base_with_a_negative_exponent_gradient_is_infinite() -> None:
    x = nb.Tensor(np.array([0.0]), requires_grad=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        nb.pow(x, -1.0).backward()
    assert x.grad is not None
    assert np.isinf(x.grad[0])


# --------------------------------------------------------------------------- #
# overflow
# --------------------------------------------------------------------------- #


def test_logsumexp_of_values_that_overflow_exp() -> None:
    logits = np.array([1000.0, 1000.0])
    with np.errstate(over="raise"):
        result = nb.logsumexp(nb.Tensor(logits)).item()
    assert result == pytest.approx(1000.0 + np.log(2.0))

    with np.errstate(over="raise"), pytest.raises(FloatingPointError):
        _ = np.log(np.sum(np.exp(logits)))


def test_logsumexp_of_all_negative_infinity_is_negative_infinity() -> None:
    x = nb.Tensor(np.array([-np.inf, -np.inf]))
    # log(0) is the IEEE route to -inf; the shift is left at zero for this group
    # precisely so the arithmetic gets there instead of producing -inf - -inf.
    with np.errstate(divide="ignore"):
        assert np.isneginf(nb.logsumexp(x).item())


def test_exp_overflows_to_infinity_rather_than_wrapping() -> None:
    with np.errstate(over="ignore"):
        assert np.isinf(nb.exp(nb.Tensor(np.array([1000.0]))).item())


def test_float32_softmax_is_stable_where_float32_exp_is_not() -> None:
    logits = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    with np.errstate(over="raise"):
        probabilities = nb.softmax(nb.tensor(logits)).numpy()
    np.testing.assert_allclose(probabilities, np.full(3, 1 / 3), rtol=1e-6)
    with np.errstate(over="raise"), pytest.raises(FloatingPointError):
        _ = np.exp(logits)


# --------------------------------------------------------------------------- #
# degenerate shapes
# --------------------------------------------------------------------------- #


def test_zero_dimensional_tensors_flow_through_the_tape() -> None:
    x = nb.Tensor(np.float64(3.0), requires_grad=True)
    assert x.shape == ()
    assert x.ndim == 0
    y = x * x
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == ()
    np.testing.assert_allclose(x.grad, 6.0)


def test_a_zero_dimensional_tensor_broadcasts_against_anything() -> None:
    scalar = nb.Tensor(np.float64(2.0), requires_grad=True)
    matrix = nb.Tensor(np.ones((3, 4)), requires_grad=True)
    (scalar * matrix).sum().backward()
    assert scalar.grad is not None
    assert scalar.grad.shape == ()
    np.testing.assert_allclose(scalar.grad, 12.0)


def test_check_grad_handles_a_zero_dimensional_input() -> None:
    report = nb.check_grad(lambda x: x**3, np.float64(2.0))
    assert report.probes == 1
    assert report.index == ()


def test_empty_tensors_keep_their_shape_through_a_backward_pass() -> None:
    x = nb.Tensor(np.zeros((0, 3)), requires_grad=True)
    y = (x * 2.0).sum()
    assert y.item() == 0.0
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == (0, 3)


def test_an_empty_axis_broadcasts_against_a_length_one_axis() -> None:
    """An empty axis against a size-1 axis.

    NumPy broadcasts 1 against 0 to 0, so the size-1 operand's gradient is the
    sum of no contributions - zero, not an error.
    """
    empty = nb.Tensor(np.zeros((0,)), requires_grad=True)
    single = nb.Tensor(np.ones((1,)), requires_grad=True)
    (empty * single).sum().backward()
    assert single.grad is not None
    assert single.grad.shape == (1,)
    np.testing.assert_allclose(single.grad, [0.0])


def test_the_mean_of_an_empty_tensor_is_nan_and_says_so() -> None:
    # NumPy raises its own warning here rather than setting an IEEE flag, and
    # Nabla passes it through instead of swallowing it. The errstate suppresses
    # the 0/0 that follows it, which is the same event reported twice.
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning, match="Mean of empty"):
        result = nb.mean(nb.Tensor(np.zeros((0,)))).item()
    assert np.isnan(result)


def test_a_size_one_tensor_can_seed_backward_without_being_zero_dimensional() -> None:
    x = nb.tensor([[[2.0]]], requires_grad=True)
    (x * 3.0).backward()
    assert x.grad is not None
    assert x.grad.shape == (1, 1, 1)


def test_reshaping_to_an_incompatible_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="reshape"):
        nb.reshape(nb.tensor([1.0, 2.0, 3.0]), 2, 2)


def test_a_single_element_reduction_over_every_axis_gives_a_scalar() -> None:
    x = nb.Tensor(np.ones((1, 1, 1)), requires_grad=True)
    total = nb.sum(x)
    assert total.shape == ()
    total.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, np.ones((1, 1, 1)))


def test_concat_of_an_empty_piece_contributes_an_empty_gradient() -> None:
    empty = nb.Tensor(np.zeros((0, 3)), requires_grad=True)
    full = nb.Tensor(np.ones((2, 3)), requires_grad=True)
    nb.concat([empty, full]).sum().backward()
    assert empty.grad is not None
    assert empty.grad.shape == (0, 3)
    assert full.grad is not None
    np.testing.assert_allclose(full.grad, np.ones((2, 3)))


def test_item_on_an_empty_tensor_is_an_error_not_a_nan() -> None:
    with pytest.raises(ShapeError, match="exactly one element"):
        nb.Tensor(np.zeros((0,))).item()
