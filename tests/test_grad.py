"""The backward pass, the functional gradient, gradient checking and clipping."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from conftest import sample
from nabla._typing import Array
from nabla.errors import AutogradError, DTypeError, GradientCheckError, ShapeError

# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #


def test_backward_needs_a_scalar_or_an_explicit_cotangent() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(AutogradError, match="needs an explicit"):
        y.backward()
    y.backward(np.array([1.0, 10.0]))
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [2.0, 20.0])


def test_backward_rejects_a_cotangent_of_the_wrong_shape() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ShapeError, match="gradient has shape"):
        (x * 2.0).backward(np.ones(3))


def test_backward_refuses_a_tensor_with_no_gradient_path() -> None:
    x = nb.tensor([1.0, 2.0])
    with pytest.raises(AutogradError, match="does not require grad"):
        (x * 2.0).sum().backward()


def test_a_detached_branch_contributes_nothing() -> None:
    x = nb.tensor([2.0], requires_grad=True)
    y = nb.tensor([3.0], requires_grad=True)
    loss = (x * y.detach()).sum()
    loss.backward()
    assert x.grad is not None
    assert y.grad is None


def test_the_backward_pass_names_the_op_when_a_vjp_returns_the_wrong_shape() -> None:
    """The invariant that catches a missing `unbroadcast` at its source."""
    x = nb.tensor([1.0, 2.0], requires_grad=True)

    def broken(t: nb.Tensor) -> nb.Tensor:
        return nb.record(np.sum(t.data), "wrong_shape_op", (t,), lambda _g: (np.zeros(5),))

    with pytest.raises(AutogradError, match="wrong_shape_op"):
        broken(x).backward()


def test_the_backward_pass_rejects_a_vjp_with_the_wrong_number_of_cotangents() -> None:
    x = nb.tensor([1.0], requires_grad=True)

    def broken(t: nb.Tensor) -> nb.Tensor:
        return nb.record(t.data * 2.0, "too_few", (t, t), lambda g: (g,))

    with pytest.raises(AutogradError, match="too_few"):
        broken(x).sum().backward()


# --------------------------------------------------------------------------- #
# topological order
# --------------------------------------------------------------------------- #


def test_a_diamond_is_visited_consumers_first() -> None:
    x = nb.tensor([1.0], requires_grad=True)
    left = x * 2.0
    right = x * 3.0
    top = left + right
    order = nb.topological_order(top)
    assert order[0] is top
    # The property that matters: x is visited only after both of its consumers,
    # so its two contributions are already accumulated when its turn comes.
    assert order.index(left) < order.index(x)
    assert order.index(right) < order.index(x)
    assert len(order) == 6  # top, left, right, x, and the two scalar constants


def test_a_deep_chain_neither_overflows_the_stack_nor_loses_a_factor() -> None:
    """A chain long enough to break the two easy implementations.

    2000 links is past CPython's recursion limit, so a recursive topological
    sort would raise instead of answering, and a sort that visited a node before
    all of its consumers would drop part of the product rule.
    """
    depth = 2000
    x = nb.tensor([1.0], dtype=np.float64, requires_grad=True)
    y = x
    for _ in range(depth):
        y = y * 1.001
    assert len(nb.topological_order(y)) == 2 * depth + 1
    y.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [1.001**depth], rtol=1e-9)


def test_a_wide_reuse_of_one_tensor_accumulates_every_contribution() -> None:
    x = nb.tensor([2.0], dtype=np.float64, requires_grad=True)
    total = nb.tensor([0.0], dtype=np.float64, requires_grad=True)
    for i in range(50):
        total = total + x * float(i)
    total.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [sum(range(50))])


# --------------------------------------------------------------------------- #
# functional grad
# --------------------------------------------------------------------------- #


def test_grad_returns_one_tensor_for_an_int_argnums() -> None:
    result = nb.grad(lambda x: (x**2).sum())(nb.tensor([3.0, 4.0]))
    assert isinstance(result, nb.Tensor)
    np.testing.assert_allclose(result.numpy(), [6.0, 8.0])


def test_grad_returns_a_tuple_for_a_sequence_of_argnums() -> None:
    gradients = nb.grad(lambda x, y: (x * y).sum(), argnums=(0, 1))(
        nb.tensor([2.0]), nb.tensor([5.0])
    )
    assert isinstance(gradients, tuple)
    np.testing.assert_allclose(gradients[0].numpy(), [5.0])
    np.testing.assert_allclose(gradients[1].numpy(), [2.0])


def test_grad_leaves_the_callers_tensors_untouched() -> None:
    x = nb.tensor([3.0], requires_grad=True)
    nb.grad(lambda a: (a**2).sum())(x)
    assert x.grad is None


def test_grad_gives_an_unused_argument_an_explicit_zero() -> None:
    gradients = nb.grad(lambda x, _y: (x * 2.0).sum(), argnums=(0, 1))(
        nb.tensor([1.0, 2.0]), nb.tensor([3.0, 4.0])
    )
    assert isinstance(gradients, tuple)
    np.testing.assert_allclose(gradients[1].numpy(), [0.0, 0.0])


def test_grad_requires_a_scalar_output_and_a_valid_argnums() -> None:
    with pytest.raises(AutogradError, match="scalar-valued"):
        nb.grad(lambda x: x * 2.0)(nb.tensor([1.0, 2.0]))
    with pytest.raises(AutogradError, match="out of range"):
        nb.grad(lambda x: x.sum(), argnums=3)(nb.tensor([1.0]))


def test_the_gradient_function_is_not_itself_differentiable() -> None:
    """The boundary of what the package claims.

    Nabla has no higher-order differentiation: a vjp is written in NumPy, not in
    tensor operations, so the backward pass builds no tape of its own. The second
    derivative has to be taken numerically.
    """
    first = nb.grad(lambda x: (x**3).sum())

    def at(value: float) -> float:
        return float(first(nb.tensor([value], dtype=np.float64)).item())

    step = 1e-5
    numerical_second = (at(2.0 + step) - at(2.0 - step)) / (2 * step)
    assert abs(numerical_second - 12.0) < 1e-6


# --------------------------------------------------------------------------- #
# check_grad
# --------------------------------------------------------------------------- #


def test_check_grad_passes_on_a_correct_gradient_with_room_to_spare() -> None:
    report = nb.check_grad(lambda x: (nb.tanh(x) ** 2).sum(), sample(3, 4, seed=61))
    assert report.ok
    assert report.worst_relative_error < 1e-8
    assert report.probes == 12
    assert "matches" in str(report)


def test_check_grad_names_the_op_the_shape_and_the_index_when_it_fails() -> None:
    def broken(x: nb.Tensor) -> nb.Tensor:
        # The forward pass doubles; the vjp claims it tripled.
        return nb.record(x.data * 2.0, "broken_double", (x,), lambda g: (g * 3.0,))

    with pytest.raises(GradientCheckError) as failure:
        nb.check_grad(lambda x: broken(x).sum(), np.ones((2, 3)), name="broken_double")

    message = str(failure.value)
    assert "broken_double" in message
    assert "(2, 3)" in message
    assert "index (" in message
    assert "MISMATCH" in message


def test_check_grad_can_report_instead_of_raising() -> None:
    def broken(x: nb.Tensor) -> nb.Tensor:
        return nb.record(x.data * 2.0, "broken", (x,), lambda g: (g * 3.0,))

    report = nb.check_grad(
        lambda x: broken(x).sum(), np.ones(3), raise_on_failure=False, name="broken"
    )
    assert not report.ok
    assert report.analytic == pytest.approx(3.0 * report.numeric / 2.0)
    assert report.argument == 0
    assert report.shape == (3,)


def test_check_grad_catches_a_missing_broadcast_reduction() -> None:
    """The defect this whole package is organised around, injected on purpose."""

    def broken_add(a: nb.Tensor, b: nb.Tensor) -> nb.Tensor:
        data = a.data + b.data

        def vjp(g: Array) -> tuple[Array | None, ...]:
            # Right shape, wrong value: it takes the first broadcast copy of the
            # cotangent instead of summing all four of them.
            return g, g[0]

        return nb.record(data, "broken_broadcast_add", (a, b), vjp)

    with pytest.raises(GradientCheckError, match="broken_broadcast_add"):
        nb.check_grad(
            broken_add,
            np.ones((4, 3)),
            np.ones((3,)),
            name="broken_broadcast_add",
        )


def test_check_grad_uses_a_random_projection_not_a_plain_sum() -> None:
    """A transposed gradient survives `.sum()` and must not survive check_grad."""

    def broken_transpose(x: nb.Tensor) -> nb.Tensor:
        return nb.record(x.data.T.copy(), "broken_transpose", (x,), lambda g: (g.copy(),))

    square = sample(3, 3, seed=62)
    # Reduced to a scalar by the caller, the defect is invisible: a sum is
    # symmetric under transposition, so the analytic and numerical gradients
    # agree to full precision on a gradient that is wrong everywhere off the
    # diagonal.
    hidden = nb.check_grad(
        lambda x: broken_transpose(x).sum(), square, raise_on_failure=False, name="pre-reduced"
    )
    assert hidden.ok
    # Left unreduced, check_grad's own random weighting finds it.
    with pytest.raises(GradientCheckError, match="broken_transpose"):
        nb.check_grad(broken_transpose, square, name="broken_transpose")


def test_check_grad_samples_when_asked_and_stays_reproducible() -> None:
    data = sample(6, 6, seed=63)
    first = nb.check_grad(lambda x: nb.sigmoid(x).sum(), data, max_probes=5, seed=99)
    second = nb.check_grad(lambda x: nb.sigmoid(x).sum(), data, max_probes=5, seed=99)
    assert first.probes == 5
    assert first.index == second.index
    assert first.worst_relative_error == second.worst_relative_error


def test_check_grad_refuses_integer_inputs_and_empty_calls() -> None:
    with pytest.raises(DTypeError, match="only floating-point inputs"):
        nb.check_grad(lambda x: x.sum(), np.arange(3))
    with pytest.raises(ValueError, match="at least one input"):
        nb.check_grad(lambda x: x.sum())


def test_check_grad_promotes_to_float64_without_touching_the_input() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    nb.check_grad(lambda a: (a**2).sum(), x)
    assert x.dtype == np.float32
    assert x.grad is None


def test_check_grad_notices_an_output_shape_that_moves() -> None:
    def unstable(x: nb.Tensor) -> nb.Tensor:
        return x if float(x.data[0]) < 1.0 else x[:1]

    with pytest.raises(ShapeError, match="output shape changed"):
        nb.check_grad(unstable, np.array([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# clipping
# --------------------------------------------------------------------------- #


def test_clip_grad_norm_scales_the_whole_set_by_one_factor() -> None:
    a = nb.tensor([3.0, 0.0], requires_grad=True)
    b = nb.tensor([0.0, 4.0], requires_grad=True)
    (a.sum() * 1.0 + b.sum()).backward()
    a.grad = np.array([3.0, 0.0])
    b.grad = np.array([0.0, 4.0])

    total = nb.clip_grad_norm_([a, b], max_norm=1.0)
    assert total == pytest.approx(5.0)
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad, [0.6, 0.0])
    np.testing.assert_allclose(b.grad, [0.0, 0.8])
    assert np.hypot(*[np.linalg.norm(t.grad) for t in (a, b)]) == pytest.approx(1.0)


def test_clip_grad_norm_leaves_a_small_gradient_exactly_alone() -> None:
    a = nb.tensor([0.3, 0.4], requires_grad=True)
    a.grad = np.array([0.3, 0.4])
    total = nb.clip_grad_norm_([a], max_norm=10.0)
    assert total == pytest.approx(0.5)
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [0.3, 0.4])


def test_clip_grad_norm_supports_other_norms_and_missing_gradients() -> None:
    a = nb.tensor([1.0, 1.0], requires_grad=True)
    a.grad = np.array([3.0, -4.0])
    assert nb.clip_grad_norm_([a], 100.0, norm_type=1.0) == pytest.approx(7.0)
    assert nb.clip_grad_norm_([a], 100.0, norm_type=np.inf) == pytest.approx(4.0)
    assert nb.clip_grad_norm_([nb.tensor([1.0])], 1.0) == 0.0
    with pytest.raises(ValueError, match="max_norm must be positive"):
        nb.clip_grad_norm_([a], 0.0)


def test_clip_grad_value_bounds_each_element() -> None:
    a = nb.tensor([1.0, 2.0, 3.0], requires_grad=True)
    a.grad = np.array([-5.0, 0.5, 5.0])
    nb.clip_grad_value_([a], 1.0)
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [-1.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="clip_value must be positive"):
        nb.clip_grad_value_([a], -1.0)
