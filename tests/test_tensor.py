"""The tensor itself: dtypes, buffers, the in-place policy, and the tape."""

from __future__ import annotations

import operator

import numpy as np
import pytest

import nabla as nb
from nabla.errors import DTypeError, InPlaceOperationError, ShapeError

BOOM = "boom"


# --------------------------------------------------------------------------- #
# dtype policy
# --------------------------------------------------------------------------- #


def test_python_data_defaults_to_float32() -> None:
    assert nb.tensor(1.0).dtype == np.float32
    assert nb.tensor([1.0, 2.0]).dtype == np.float32
    assert nb.tensor([[1.0], [2.0]]).dtype == np.float32


def test_an_ndarray_keeps_the_dtype_it_was_given() -> None:
    assert nb.tensor(np.zeros(3, np.float64)).dtype == np.float64
    assert nb.tensor(np.zeros(3, np.float32)).dtype == np.float32
    assert nb.tensor(np.zeros(3, np.int32)).dtype == np.int32


def test_an_explicit_dtype_always_wins() -> None:
    assert nb.tensor([1.0, 2.0], dtype=np.float64).dtype == np.float64
    assert nb.tensor(np.zeros(3, np.float64), dtype=np.float32).dtype == np.float32


def test_python_integers_stay_integers() -> None:
    assert nb.tensor([1, 2, 3]).dtype == np.int64


def test_integer_tensors_cannot_require_grad() -> None:
    with pytest.raises(DTypeError, match="only floating-point tensors"):
        nb.tensor([1, 2, 3], requires_grad=True)
    x = nb.tensor(np.zeros(3, np.int64))
    with pytest.raises(DTypeError, match="cannot require grad"):
        x.requires_grad = True


def test_an_integer_result_never_gets_a_tape_node() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    index = nb.tensor([0, 1])
    combined = index + 1
    assert combined.node is None
    assert not combined.requires_grad
    assert x.requires_grad  # untouched


def test_non_numeric_data_is_refused() -> None:
    with pytest.raises(DTypeError, match="expected a numeric array"):
        nb.tensor(["a", "b"])


def test_mixed_precision_follows_numpy_promotion() -> None:
    single = nb.tensor(np.ones(3, np.float32))
    double = nb.tensor(np.ones(3, np.float64))
    assert (single + double).dtype == np.float64
    assert (single + 1.0).dtype == np.float32


# --------------------------------------------------------------------------- #
# buffers and the in-place policy
# --------------------------------------------------------------------------- #


def test_the_underlying_buffer_is_read_only() -> None:
    x = nb.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="read-only"):
        x.data[0] = 5.0


def test_constructing_from_an_array_does_not_freeze_the_callers_array() -> None:
    source = np.ones(3)
    nb.tensor(source)
    source[0] = 5.0  # must still work
    assert source[0] == 5.0


def test_mutating_the_source_array_cannot_change_the_tensor() -> None:
    source = np.ones(3)
    x = nb.tensor(source)
    source[0] = 5.0
    np.testing.assert_allclose(x.numpy(), np.ones(3))


@pytest.mark.parametrize(
    ("symbol", "apply"),
    [
        ("+=", operator.iadd),
        ("-=", operator.isub),
        ("*=", operator.imul),
        ("/=", operator.itruediv),
        ("**=", operator.ipow),
        ("@=", operator.imatmul),
    ],
)
def test_in_place_operators_are_refused(symbol: str, apply: object) -> None:
    x = nb.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    other = x if symbol == "@=" else 1.0
    with pytest.raises(InPlaceOperationError, match="in-place"):
        apply(x, other)  # type: ignore[operator]


def test_item_assignment_is_refused() -> None:
    x = nb.tensor([1.0, 2.0])
    with pytest.raises(InPlaceOperationError, match="item assignment"):
        x[0] = 3.0


def test_assign_replaces_a_leaf_without_disturbing_the_tape() -> None:
    """The optimiser's update path, and the reason it rebinds instead of writing.

    The product below captured `p`'s buffer for its backward pass. After the
    update, that captured buffer must still hold the value the forward pass
    used, or the gradient would describe a computation that never happened.
    """
    p = nb.tensor([2.0, 3.0], requires_grad=True)
    y = p * 10.0
    p.assign_(p.data - 1.0)
    np.testing.assert_allclose(p.numpy(), [1.0, 2.0])
    y.sum().backward()
    assert p.grad is not None
    np.testing.assert_allclose(p.grad, [10.0, 10.0])


def test_assign_refuses_non_leaves_and_shape_changes() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ShapeError, match="expects shape"):
        x.assign_(np.ones(3))
    y = x * 2
    with pytest.raises(InPlaceOperationError, match="non-leaf"):
        y.assign_(np.ones(2))


def test_assign_casts_to_the_tensors_dtype() -> None:
    x = nb.tensor([1.0, 2.0])
    x.assign_(np.array([3.0, 4.0], dtype=np.float64))
    assert x.dtype == np.float32


# --------------------------------------------------------------------------- #
# the tape
# --------------------------------------------------------------------------- #


def test_no_grad_builds_no_tape_at_all() -> None:
    """Asserted on the tape, not on the outcome.

    Checking that `.grad` came out empty would also pass if the graph were
    built and simply never traversed, which is the failure mode - the memory
    stays alive - that `no_grad` exists to prevent.
    """
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    with nb.no_grad():
        y = x * 3.0 + 1.0
        z = nb.softmax(y)
    assert y.node is None
    assert z.node is None
    assert not y.requires_grad
    assert not z.requires_grad
    assert nb.topological_order(z) == [z]


def test_no_grad_is_reentrant() -> None:
    assert nb.is_grad_enabled()
    with nb.no_grad():
        assert not nb.is_grad_enabled()
        with nb.no_grad():
            assert not nb.is_grad_enabled()
        assert not nb.is_grad_enabled()
    assert nb.is_grad_enabled()


def test_grad_mode_is_restored_even_if_the_block_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"), nb.no_grad():
        raise RuntimeError(BOOM)
    assert nb.is_grad_enabled()


def test_a_tensor_used_twice_accumulates_rather_than_overwrites() -> None:
    x = nb.tensor([3.0], requires_grad=True)
    y = x * x  # two paths to the same leaf
    y.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [6.0])


def test_accumulation_across_several_backward_calls() -> None:
    x = nb.tensor([2.0], requires_grad=True)
    (x * 5.0).sum().backward()
    (x * 5.0).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [10.0])
    x.zero_grad()
    assert x.grad is None
    (x * 5.0).sum().backward()
    np.testing.assert_allclose(x.grad, [5.0])


def test_a_shared_subexpression_contributes_through_every_path() -> None:
    x = nb.tensor([2.0], requires_grad=True)
    shared = x * 3.0
    shared.retain_grad()
    (shared + shared * x).backward()
    assert x.grad is not None
    # d/dx [3x + 3x^2] = 3 + 6x = 15
    np.testing.assert_allclose(x.grad, [15.0])
    assert shared.grad is not None
    np.testing.assert_allclose(shared.grad, [3.0])


def test_interior_tensors_keep_no_gradient_unless_asked() -> None:
    x = nb.tensor([2.0], requires_grad=True)
    interior = x * 3.0
    assert not interior.retains_grad
    (interior * 2.0).backward()
    assert interior.grad is None
    assert x.grad is not None


def test_detach_shares_the_buffer_and_cuts_the_tape() -> None:
    x = nb.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    cut = y.detach()
    assert cut.node is None
    assert not cut.requires_grad
    assert cut.data is y.data  # no copy: safe because the buffer is read-only
    (cut * 3.0).sum()
    assert x.grad is None


def test_requires_grad_cannot_be_changed_on_a_non_leaf() -> None:
    x = nb.tensor([1.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(InPlaceOperationError, match="non-leaf"):
        y.requires_grad = False


def test_the_node_names_the_operation_that_produced_the_tensor() -> None:
    x = nb.tensor([1.0], requires_grad=True)
    y = nb.relu(x @ nb.tensor([[1.0]], requires_grad=True))
    assert y.node is not None
    assert y.node.op == "relu"
    assert "relu" in repr(y.node)


# --------------------------------------------------------------------------- #
# plain surface area
# --------------------------------------------------------------------------- #


def test_shape_dtype_ndim_size_and_transpose() -> None:
    x = nb.tensor(np.zeros((2, 3, 4)))
    assert x.shape == (2, 3, 4)
    assert x.ndim == 3
    assert x.size == 24
    assert x.T.shape == (4, 3, 2)
    assert len(x) == 2


def test_item_float_int_and_bool() -> None:
    scalar = nb.tensor([[2.5]])
    assert scalar.item() == 2.5
    assert float(scalar) == 2.5
    assert int(scalar) == 2
    assert bool(scalar)
    assert not bool(nb.tensor(0.0))


def test_item_and_bool_refuse_multi_element_tensors() -> None:
    x = nb.tensor([1.0, 2.0])
    with pytest.raises(ShapeError, match="exactly one element"):
        x.item()
    with pytest.raises(ShapeError, match="ambiguous"):
        bool(x)
    with pytest.raises(ShapeError, match="0-d"):
        len(nb.tensor(1.0))


def test_numpy_returns_a_writable_copy_by_default() -> None:
    x = nb.tensor([1.0, 2.0])
    copied = x.numpy()
    copied[0] = 9.0
    np.testing.assert_allclose(x.numpy(), [1.0, 2.0])
    assert x.numpy(copy=False) is x.data


def test_numpy_conversion_protocol() -> None:
    x = nb.tensor([1.0, 2.0])
    np.testing.assert_allclose(np.asarray(x), [1.0, 2.0])
    assert np.asarray(x, dtype=np.float64).dtype == np.float64


def test_repr_shows_the_values_the_dtype_and_the_provenance() -> None:
    x = nb.tensor([1.0], requires_grad=True)
    assert "requires_grad=True" in repr(x)
    assert "float32" in repr(x)
    assert "node=mul" in repr(x * 2.0)


def test_tensors_hash_by_identity() -> None:
    """The tape keys tensors by identity; an elementwise `__eq__` would break it."""
    x = nb.tensor([1.0, 2.0])
    y = nb.tensor([1.0, 2.0])
    assert len({x, y}) == 2
    assert x != y
    assert x == x  # noqa: PLR0124 - identity equality is the property under test


def test_creation_helpers() -> None:
    assert nb.zeros(2, 3).shape == (2, 3)
    assert nb.zeros((2, 3)).shape == (2, 3)
    assert nb.ones(2).numpy().tolist() == [1.0, 1.0]
    assert nb.full((2,), 7.0).numpy().tolist() == [7.0, 7.0]
    assert nb.eye(2).numpy().tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert nb.arange(3).numpy().tolist() == [0.0, 1.0, 2.0]
    assert nb.arange(1, 4).numpy().tolist() == [1.0, 2.0, 3.0]
    assert nb.linspace(0.0, 1.0, 3).numpy().tolist() == [0.0, 0.5, 1.0]
    assert nb.zeros_like(nb.ones(3)).numpy().tolist() == [0.0, 0.0, 0.0]
    assert nb.ones_like(nb.zeros(3)).numpy().tolist() == [1.0, 1.0, 1.0]
    assert nb.zeros(2, requires_grad=True).requires_grad
    assert nb.from_numpy(np.ones(2, np.float64)).dtype == np.float64


def test_method_forms_match_the_functions() -> None:
    x = nb.tensor(np.arange(6.0).reshape(2, 3))
    np.testing.assert_allclose(x.sum().numpy(), nb.sum(x).numpy())
    np.testing.assert_allclose(x.mean(axis=0).numpy(), nb.mean(x, axis=0).numpy())
    assert x.reshape(3, 2).shape == (3, 2)
    assert x.transpose().shape == (3, 2)
