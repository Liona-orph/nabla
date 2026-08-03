"""Elementwise arithmetic: forward against NumPy, backward against a difference quotient."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import nabla as nb
from conftest import assert_close, check, sample
from nabla._typing import Array

# Inputs are offset away from zero so that `div`, `log` and `sqrt` are evaluated
# where they are smooth. Checking them at their singularities is the job of
# test_edge_cases.py, which asserts what happens there rather than pretending it
# is differentiable.
A = sample(3, 4, seed=1)
B = sample(3, 4, seed=2)
POSITIVE = np.abs(sample(3, 4, seed=3)) + 0.5

BINARY = [
    pytest.param(nb.add, np.add, id="add"),
    pytest.param(nb.sub, np.subtract, id="sub"),
    pytest.param(nb.mul, np.multiply, id="mul"),
    pytest.param(nb.div, np.divide, id="div"),
    pytest.param(nb.maximum, np.maximum, id="maximum"),
    pytest.param(nb.minimum, np.minimum, id="minimum"),
]

UNARY = [
    pytest.param(nb.neg, np.negative, A, id="neg"),
    pytest.param(nb.exp, np.exp, A, id="exp"),
    pytest.param(nb.log, np.log, POSITIVE, id="log"),
    pytest.param(nb.sqrt, np.sqrt, POSITIVE, id="sqrt"),
    pytest.param(nb.abs, np.abs, A, id="abs"),
]


@pytest.mark.parametrize(("op", "reference"), BINARY)
def test_binary_forward(
    op: Callable[..., nb.Tensor],
    reference: Callable[..., Array],
) -> None:
    b = B + 2.0  # keeps the divisor away from zero
    assert_close(op(nb.Tensor(A), nb.Tensor(b)), reference(A, b))


@pytest.mark.parametrize(("op", "reference"), BINARY)
def test_binary_backward(
    op: Callable[..., nb.Tensor],
    reference: Callable[..., Array],
) -> None:
    # `maximum` and `minimum` are checked on inputs with no ties: at a tie the
    # function is not differentiable and the difference quotient returns the
    # average of the two one-sided derivatives, which no convention matches.
    b = B + 2.0
    assert_close(op(nb.Tensor(A), nb.Tensor(b)), reference(A, b))
    check(op, A, b, name=op.__name__)


@pytest.mark.parametrize(("op", "reference", "data"), UNARY)
def test_unary_forward_and_backward(
    op: Callable[[nb.Tensor], nb.Tensor],
    reference: Callable[[Array], Array],
    data: Array,
) -> None:
    assert_close(op(nb.Tensor(data)), reference(data))
    check(op, data, name=op.__name__)


def test_pow_with_a_constant_exponent_handles_negative_bases() -> None:
    x = nb.Tensor(np.array([-2.0, -0.5, 3.0]), requires_grad=True)
    (x**3).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, 3.0 * x.numpy() ** 2)


def test_pow_differentiates_the_exponent_only_when_asked() -> None:
    base = nb.Tensor(np.array([-2.0, 3.0]), requires_grad=True)
    exponent = nb.Tensor(np.array([2.0, 2.0]))  # no gradient wanted
    # A negative base would make log(base) invalid; the vjp must not evaluate it
    # at all, which under `filterwarnings = error` is what this asserts.
    (nb.pow(base, exponent) ** 1).sum().backward()
    assert base.grad is not None
    assert exponent.grad is None


def test_pow_reports_nan_for_a_non_positive_base_when_the_exponent_needs_a_gradient() -> None:
    base = nb.Tensor(np.array([-2.0, 3.0]))
    exponent = nb.Tensor(np.array([2.0, 2.0]), requires_grad=True)
    nb.pow(base, exponent).sum().backward()
    assert exponent.grad is not None
    assert np.isnan(exponent.grad[0])
    assert np.isfinite(exponent.grad[1])


def test_pow_gradients_check_numerically_on_a_positive_base() -> None:
    check(nb.pow, POSITIVE, B, name="pow")


def test_maximum_sends_a_tie_to_the_left_operand() -> None:
    x = nb.Tensor(np.array([1.0]), requires_grad=True)
    y = nb.Tensor(np.array([1.0]), requires_grad=True)
    nb.maximum(x, y).sum().backward()
    assert x.grad is not None
    assert y.grad is not None
    np.testing.assert_allclose(x.grad, [1.0])
    np.testing.assert_allclose(y.grad, [0.0])


def test_abs_uses_zero_as_the_subgradient_at_zero() -> None:
    x = nb.Tensor(np.array([-2.0, 0.0, 3.0]), requires_grad=True)
    nb.abs(x).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [-1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("expression", "reference"),
    [
        (lambda x: x + 2.0, lambda a: a + 2.0),
        (lambda x: 2.0 + x, lambda a: 2.0 + a),
        (lambda x: x - 2.0, lambda a: a - 2.0),
        (lambda x: 2.0 - x, lambda a: 2.0 - a),
        (lambda x: x * 2.0, lambda a: a * 2.0),
        (lambda x: 2.0 * x, lambda a: 2.0 * a),
        (lambda x: x / 2.0, lambda a: a / 2.0),
        (lambda x: 2.0 / x, lambda a: 2.0 / a),
        (lambda x: x**2.0, lambda a: a**2.0),
        (lambda x: 2.0**x, lambda a: 2.0**a),
        (lambda x: -x, lambda a: -a),
        (lambda x: +x, lambda a: +a),
        (abs, abs),
    ],
)
def test_operators_agree_with_numpy_and_carry_gradients(
    expression: Callable[[nb.Tensor], nb.Tensor],
    reference: Callable[[Array], Array],
) -> None:
    assert_close(expression(nb.Tensor(POSITIVE)), reference(POSITIVE))
    check(expression, POSITIVE)


def test_an_ndarray_on_the_left_does_not_silently_drop_the_tape() -> None:
    """`ndarray * tensor` must return a Tensor, not an ndarray.

    NumPy would otherwise win the dispatch, convert through `__array__` and
    return a plain array - losing every gradient, and only when the operands are
    written in that order.
    """
    x = nb.Tensor(np.ones(3), requires_grad=True)
    result = np.full(3, 2.0) * x
    assert isinstance(result, nb.Tensor)
    result.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [2.0, 2.0, 2.0])


def test_python_scalars_do_not_widen_a_float32_tensor() -> None:
    x = nb.tensor([1.0, 2.0])
    assert x.dtype == np.float32
    assert (x * 2.0).dtype == np.float32
    assert (x + 1).dtype == np.float32
