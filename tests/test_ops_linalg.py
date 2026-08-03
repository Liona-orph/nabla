"""Matrix products, including the rank combinations NumPy treats specially."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from conftest import assert_close, check, sample
from nabla.errors import ShapeError

# Every combination that np.matmul handles differently: matrix/matrix,
# vector/matrix, matrix/vector, vector/vector, and a batch against an
# unbatched operand (which broadcasts, and therefore needs a reduction).
CASES = [
    pytest.param((3, 4), (4, 5), id="matrix-matrix"),
    pytest.param((4,), (4, 5), id="vector-matrix"),
    pytest.param((3, 4), (4,), id="matrix-vector"),
    pytest.param((4,), (4,), id="vector-vector"),
    pytest.param((2, 3, 4), (4, 5), id="batch-matrix"),
    pytest.param((3, 4), (2, 4, 5), id="matrix-batch"),
    pytest.param((2, 3, 4), (2, 4, 5), id="batch-batch"),
    pytest.param((2, 1, 3, 4), (5, 4, 6), id="broadcast-batch"),
    pytest.param((2, 3, 4), (4,), id="batch-vector"),
    pytest.param((4,), (2, 4, 5), id="vector-batch"),
]


@pytest.mark.parametrize(("left", "right"), CASES)
def test_matmul_forward(left: tuple[int, ...], right: tuple[int, ...]) -> None:
    a, b = sample(*left, seed=41), sample(*right, seed=42)
    assert_close(nb.Tensor(a) @ nb.Tensor(b), np.matmul(a, b))


@pytest.mark.parametrize(("left", "right"), CASES)
def test_matmul_backward(left: tuple[int, ...], right: tuple[int, ...]) -> None:
    a, b = sample(*left, seed=41), sample(*right, seed=42)
    # Unreduced: with a plain sum, a transposed gradient for a square result can
    # still add up to the right total. check_grad weights the output randomly.
    check(lambda x, y: x @ y, a, b, name=f"matmul{left}x{right}")


@pytest.mark.parametrize(("left", "right"), CASES)
def test_matmul_gradient_shapes_survive_broadcasting(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> None:
    a = nb.Tensor(sample(*left, seed=41), requires_grad=True)
    b = nb.Tensor(sample(*right, seed=42), requires_grad=True)
    (a @ b).sum().backward()
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == left
    assert b.grad.shape == right


def test_matmul_gradient_is_the_textbook_pair() -> None:
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)
    ta = nb.Tensor(a, requires_grad=True)
    tb = nb.Tensor(b, requires_grad=True)
    weights = np.arange(8.0).reshape(2, 4)
    ((ta @ tb) * nb.Tensor(weights)).sum().backward()
    assert ta.grad is not None
    assert tb.grad is not None
    np.testing.assert_allclose(ta.grad, weights @ b.T)
    np.testing.assert_allclose(tb.grad, a.T @ weights)


def test_matmul_rejects_scalars_and_mismatched_inner_dimensions() -> None:
    with pytest.raises(ShapeError, match="rank 1 or more"):
        _ = nb.Tensor(np.float64(2.0)) @ nb.Tensor(np.ones(3))
    with pytest.raises(ShapeError, match="cannot matmul"):
        _ = nb.Tensor(np.ones((3, 4))) @ nb.Tensor(np.ones((5, 6)))


def test_dot_and_outer() -> None:
    a, b = sample(4, seed=44), sample(4, seed=45)
    assert nb.dot(nb.Tensor(a), nb.Tensor(b)).shape == ()
    assert_close(nb.dot(nb.Tensor(a), nb.Tensor(b)), np.dot(a, b))
    assert_close(nb.outer(nb.Tensor(a), nb.Tensor(b)), np.outer(a, b))
    check(nb.dot, a, b, name="dot")
    check(nb.outer, a, b, name="outer")


def test_dot_and_outer_reject_non_vectors() -> None:
    with pytest.raises(ShapeError, match="dot is for vectors"):
        nb.dot(nb.Tensor(np.ones((2, 2))), nb.Tensor(np.ones(2)))
    with pytest.raises(ShapeError, match="outer is for vectors"):
        nb.outer(nb.Tensor(np.ones((2, 2))), nb.Tensor(np.ones(2)))


def test_trace_forward_and_backward() -> None:
    a = sample(3, 3, seed=47)
    assert_close(nb.trace(nb.Tensor(a)), np.trace(a))
    check(nb.trace, a, name="trace")

    x = nb.Tensor(a, requires_grad=True)
    nb.trace(x).backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, np.eye(3))


def test_trace_rejects_non_matrices() -> None:
    with pytest.raises(ShapeError, match="needs a matrix"):
        nb.trace(nb.Tensor(np.ones(3)))


def test_a_two_layer_network_gradient_checks_end_to_end() -> None:
    """The composition every later layer will build on.

    Two matmuls, a bias, an activation and a mean. If the pieces disagree about
    shapes, this is where it shows.
    """
    x = sample(6, 3, seed=48)
    w1 = sample(3, 5, seed=49)
    b1 = sample(5, seed=50)
    w2 = sample(5, 2, seed=51)

    def loss(*params: nb.Tensor) -> nb.Tensor:
        weight1, bias1, weight2 = params
        hidden = nb.tanh(nb.Tensor(x) @ weight1 + bias1)
        return nb.mean((hidden @ weight2) ** 2)

    check(loss, w1, b1, w2, name="two-layer network")
