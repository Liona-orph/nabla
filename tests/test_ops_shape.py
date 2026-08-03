"""Shape ops: the rearrangement, and the rearrangement back."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from conftest import assert_close, check, sample
from nabla.errors import ShapeError

X = sample(2, 3, 4, seed=21)
M = sample(3, 4, seed=22)


def test_reshape_accepts_both_calling_conventions() -> None:
    assert nb.reshape(nb.Tensor(X), 6, 4).shape == (6, 4)
    assert nb.reshape(nb.Tensor(X), (6, 4)).shape == (6, 4)
    assert nb.reshape(nb.Tensor(X), -1).shape == (24,)
    check(lambda x: nb.reshape(x, 4, 6), X, name="reshape")


def test_transpose_reverses_every_axis_and_inverts_it_in_the_backward_pass() -> None:
    assert_close(nb.transpose(nb.Tensor(X)), X.T)
    assert nb.Tensor(X).T.shape == (4, 3, 2)
    # Checked unreduced. A vjp that applied the forward permutation instead of
    # its inverse produces the same total under a plain sum; only check_grad's
    # random weighting of the output can tell the two apart.
    check(nb.transpose, X, name="transpose")


def test_permute_and_swapaxes_agree_with_numpy() -> None:
    assert_close(nb.permute(nb.Tensor(X), 2, 0, 1), np.transpose(X, (2, 0, 1)))
    assert_close(nb.permute(nb.Tensor(X), (2, 0, 1)), np.transpose(X, (2, 0, 1)))
    assert_close(nb.swapaxes(nb.Tensor(X), 0, 2), np.swapaxes(X, 0, 2))
    check(lambda x: nb.permute(x, 2, 0, 1), X, name="permute")


def test_transpose_rejects_the_wrong_number_of_axes() -> None:
    with pytest.raises(ShapeError, match="needs 3 axes"):
        nb.permute(nb.Tensor(X), 1, 0)


def test_swapaxes_rejects_an_axis_outside_the_rank() -> None:
    with pytest.raises(ShapeError, match="cannot swap axes"):
        nb.swapaxes(nb.Tensor(X), 0, 5)


def test_expand_dims_and_squeeze_are_inverses() -> None:
    x = nb.Tensor(M)
    assert nb.expand_dims(x, 1).shape == (3, 1, 4)
    assert nb.squeeze(nb.expand_dims(x, 1), 1).shape == (3, 4)
    check(lambda a: nb.squeeze(nb.expand_dims(a, 0)), M, name="expand_dims")


def test_broadcast_to_forward_and_backward() -> None:
    assert_close(nb.broadcast_to(nb.Tensor(M), (2, 3, 4)), np.broadcast_to(M, (2, 3, 4)))
    check(lambda x: nb.broadcast_to(x, (2, 3, 4)), M, name="broadcast_to")

    x = nb.Tensor(M, requires_grad=True)
    nb.broadcast_to(x, (5, 3, 4)).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, np.full((3, 4), 5.0))


def test_broadcast_to_rejects_an_incompatible_shape() -> None:
    with pytest.raises(ShapeError, match="cannot broadcast"):
        nb.broadcast_to(nb.Tensor(M), (2, 5))


def test_concat_forward_and_backward() -> None:
    other = sample(1, 4, seed=23)
    assert_close(nb.concat([nb.Tensor(M), nb.Tensor(other)]), np.concatenate([M, other]))
    # Unreduced, so a vjp that sliced the cotangent back in the wrong order is
    # visible rather than cancelling out in the total.
    check(lambda a, b: nb.concat([a, b]), M, other, name="concat")


def test_stack_forward_and_backward() -> None:
    other = sample(3, 4, seed=24)
    assert_close(nb.stack([nb.Tensor(M), nb.Tensor(other)], axis=1), np.stack([M, other], axis=1))
    check(lambda a, b: nb.stack([a, b], axis=1), M, other, name="stack")


def test_concat_and_stack_reject_mismatched_shapes() -> None:
    with pytest.raises(ShapeError, match="cannot concatenate"):
        nb.concat([nb.Tensor(M), nb.Tensor(sample(3, 5, seed=25))])
    with pytest.raises(ShapeError, match="cannot stack"):
        nb.stack([nb.Tensor(M), nb.Tensor(sample(3, 5, seed=25))])
    with pytest.raises(ShapeError, match="at least one tensor"):
        nb.concat([])


def test_split_partitions_and_reassembles() -> None:
    pieces = nb.split(nb.Tensor(M), 2, axis=1)
    assert [p.shape for p in pieces] == [(3, 2), (3, 2)]
    assert_close(nb.concat(list(pieces), axis=1), M)

    uneven = nb.split(nb.Tensor(M), [1, 3], axis=1)
    assert [p.shape for p in uneven] == [(3, 1), (3, 3)]

    # Both pieces concatenated back, so each one's scatter is checked against a
    # different set of random output weights.
    check(lambda x: nb.concat(list(nb.split(x, 2, axis=1)), axis=1), M, name="split")


def test_split_rejects_sizes_that_do_not_partition_the_axis() -> None:
    with pytest.raises(ShapeError, match="equal parts"):
        nb.split(nb.Tensor(M), 3, axis=1)
    with pytest.raises(ShapeError, match="do not add up"):
        nb.split(nb.Tensor(M), [1, 1], axis=1)


@pytest.mark.parametrize(
    "index",
    [
        0,
        -1,
        (1, 2),
        slice(1, 3),
        (slice(None), 2),
        (Ellipsis, 1),
        ([0, 2, 2],),
        (np.array([0, 1]), np.array([2, 3])),
    ],
)
def test_getitem_forward_matches_numpy(index: object) -> None:
    assert_close(nb.Tensor(M)[index], np.asarray(M[index]))


def test_getitem_backward_accumulates_repeated_reads() -> None:
    x = nb.Tensor(np.arange(4.0), requires_grad=True)
    x[[0, 0, 3]].sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 0.0, 1.0])
    check(lambda a: a[[0, 0, 3]], np.arange(4.0), name="getitem")


def test_getitem_with_a_boolean_mask() -> None:
    mask = M > 0
    assert_close(nb.Tensor(M)[mask], M[mask])
    check(lambda x: x[mask], M, name="getitem-mask")


def test_getitem_accepts_an_index_tensor() -> None:
    index = nb.tensor([0, 2], dtype=np.int64)
    assert_close(nb.Tensor(M)[index], M[[0, 2]])


def test_getitem_reports_a_bad_index_with_the_shape() -> None:
    with pytest.raises(ShapeError, match=r"shape \(3, 4\)"):
        _ = nb.Tensor(M)[9]


def test_indexing_a_single_element_gives_a_0d_tensor() -> None:
    element = nb.Tensor(M, requires_grad=True)[1, 2]
    assert element.shape == ()
    element.backward()


def test_iterating_yields_differentiable_slices() -> None:
    x = nb.Tensor(M, requires_grad=True)
    rows = list(x)
    assert len(rows) == 3
    rows[1].sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [[0.0] * 4, [1.0] * 4, [0.0] * 4])


@pytest.mark.parametrize(
    "widths",
    [1, (1, 2), ((1, 0), (2, 3))],
)
def test_pad_forward_and_backward(widths: object) -> None:
    assert_close(nb.pad(nb.Tensor(M), widths, 0.5), np.pad(M, widths, constant_values=0.5))
    check(lambda x: nb.pad(x, widths, 0.5), M, name="pad")


def test_pad_gradient_ignores_the_padding() -> None:
    x = nb.Tensor(np.ones((2, 2)), requires_grad=True)
    weights = np.arange(16.0).reshape(4, 4)
    (nb.pad(x, 1, 3.0) * weights).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, weights[1:3, 1:3])


def test_pad_rejects_malformed_widths() -> None:
    with pytest.raises(ShapeError, match="non-negative"):
        nb.pad(nb.Tensor(M), -1)
    with pytest.raises(ShapeError, match="entries for a rank-2"):
        nb.pad(nb.Tensor(M), [(1, 1), (1, 1), (1, 1)])
