"""Broadcasting and its gradient reduction.

This file gets more attention than any other because the defect it looks for is
invisible from the forward pass: an engine that forgets to sum a broadcast
gradient back down still trains, just with the wrong gradient magnitude on every
broadcast parameter - which is to say on every bias in the model.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import nabla as nb
from conftest import broadcast_pairs, slow_unbroadcast
from nabla._typing import Shape
from nabla.errors import ShapeError


@pytest.mark.parametrize(("left", "right"), broadcast_pairs())
def test_unbroadcast_matches_the_definition(left: Shape, right: Shape) -> None:
    result = np.broadcast_shapes(left, right)
    g = np.arange(1, int(np.prod(result)) + 1, dtype=np.float64).reshape(result)
    np.testing.assert_array_equal(nb.unbroadcast(g, left), slow_unbroadcast(g, left))
    np.testing.assert_array_equal(nb.unbroadcast(g, right), slow_unbroadcast(g, right))


@pytest.mark.parametrize(("left", "right"), broadcast_pairs())
def test_every_binary_op_reduces_its_gradient(left: Shape, right: Shape) -> None:
    """Forward shape, backward shape and backward value, for each broadcast pair."""
    a = nb.Tensor(np.full(left, 2.0), requires_grad=True)
    b = nb.Tensor(np.full(right, 3.0), requires_grad=True)
    product = a * b
    assert product.shape == np.broadcast_shapes(left, right)

    product.sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == left
    assert b.grad.shape == right

    ones = np.ones(product.shape)
    np.testing.assert_allclose(a.grad, slow_unbroadcast(ones * 3.0, left))
    np.testing.assert_allclose(b.grad, slow_unbroadcast(ones * 2.0, right))


@pytest.mark.parametrize(("left", "right"), broadcast_pairs())
def test_broadcast_gradients_pass_a_numerical_check(left: Shape, right: Shape) -> None:
    if 0 in np.broadcast_shapes(left, right):
        pytest.skip("an empty result has no elements to perturb")
    rng = np.random.default_rng(7)
    a = rng.standard_normal(left)
    b = rng.standard_normal(right) + 3.0  # away from zero, so div stays smooth
    nb.check_grad(lambda x, y: (x * y).sum(), a, b, name="mul")
    nb.check_grad(lambda x, y: (x / y).sum(), a, b, name="div")
    nb.check_grad(lambda x, y: (x - y).sum(), a, b, name="sub")


def test_unbroadcast_is_a_no_op_when_the_shapes_agree() -> None:
    g = np.ones((2, 3))
    assert nb.unbroadcast(g, (2, 3)) is g


def test_unbroadcast_refuses_to_invent_axes() -> None:
    with pytest.raises(ShapeError, match="more axes than the gradient"):
        nb.unbroadcast(np.ones((3,)), (2, 3))


def test_unbroadcast_refuses_a_shape_that_never_broadcast() -> None:
    with pytest.raises(ShapeError, match="did not broadcast"):
        nb.unbroadcast(np.ones((2, 3)), (2, 5))


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(
    shapes=hnp.mutually_broadcastable_shapes(num_shapes=2, min_dims=0, max_dims=4, max_side=4),
    swap=st.booleans(),
)
def test_broadcast_reduction_over_random_shape_pairs(
    shapes: hnp.BroadcastableShapes,
    swap: bool,
) -> None:
    left, right = shapes.input_shapes
    if swap:
        left, right = right, left
    a = nb.Tensor(np.full(left, 1.5), requires_grad=True)
    b = nb.Tensor(np.full(right, -0.5), requires_grad=True)
    (a * b).sum().backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == left
    assert b.grad.shape == right
    ones = np.ones(np.broadcast_shapes(left, right))
    np.testing.assert_allclose(a.grad, slow_unbroadcast(ones * -0.5, left))
    np.testing.assert_allclose(b.grad, slow_unbroadcast(ones * 1.5, right))


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(shapes=hnp.mutually_broadcastable_shapes(num_shapes=2, min_dims=0, max_dims=3, max_side=3))
def test_unbroadcast_agrees_with_the_slow_definition_everywhere(
    shapes: hnp.BroadcastableShapes,
) -> None:
    result = shapes.result_shape
    g = np.linspace(1.0, 2.0, int(np.prod(result)) or 1)[: int(np.prod(result))].reshape(result)
    for shape in shapes.input_shapes:
        np.testing.assert_allclose(nb.unbroadcast(g, shape), slow_unbroadcast(g, shape))


def test_a_broadcast_bias_gets_the_gradient_of_every_row() -> None:
    """The concrete case the whole file exists for: a bias added to a batch."""
    batch = nb.Tensor(np.zeros((8, 4)), requires_grad=True)
    bias = nb.Tensor(np.zeros(4), requires_grad=True)
    (batch + bias).sum().backward()
    assert bias.grad is not None
    np.testing.assert_allclose(bias.grad, np.full(4, 8.0))
