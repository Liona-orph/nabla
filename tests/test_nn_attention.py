"""Attention: the mask, the shapes, and the gradients through both.

The load-bearing test in this file is
`test_a_later_token_cannot_change_an_earlier_output`. Every other property of
causal attention can hold while the mask is off by one row, and a model with an
off-by-one mask trains beautifully and generates nonsense, because at training
time it could see the answer.
"""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from conftest import check, check_module, sample
from nabla import nn
from nabla.errors import ShapeError

F64 = "float64"


@pytest.fixture
def g() -> nb.random.Generator:
    return nb.random.Generator(seed=808)


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #


def test_causal_mask_is_lower_triangular_and_infinite_above() -> None:
    mask = nn.causal_mask(4)
    assert np.all(mask[np.tril_indices(4)] == 0.0)
    assert np.all(np.isneginf(mask[np.triu_indices(4, k=1)]))


def test_attention_weights_are_zero_above_the_diagonal(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=6, dtype=F64)
    x = nb.Tensor(sample(2, 6, 8, seed=1))
    q, k, _ = nb.split(attn.qkv(x), 3, axis=2)
    shape = (2, 6, 2, 4)
    qh = nb.permute(nb.reshape(q, shape), 0, 2, 1, 3)
    kh = nb.permute(nb.reshape(k, shape), 0, 2, 1, 3)
    scores = (qh @ nb.permute(kh, 0, 1, 3, 2)) * attn.scale + attn.mask[:6, :6]
    weights = nb.softmax(scores, axis=-1).numpy()

    upper = np.triu_indices(6, k=1)
    assert np.all(weights[:, :, upper[0], upper[1]] == 0.0)
    np.testing.assert_allclose(weights.sum(-1), np.ones((2, 2, 6)), rtol=1e-12)


def test_a_later_token_cannot_change_an_earlier_output(g: nb.random.Generator) -> None:
    """Causality, asserted by perturbing the future and looking at the past."""
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=5, dtype=F64)
    x = sample(1, 5, 8, seed=2)
    baseline = attn(nb.Tensor(x)).numpy()

    disturbed = x.copy()
    disturbed[0, 3:] += 100.0
    after = attn(nb.Tensor(disturbed)).numpy()

    np.testing.assert_allclose(after[:, :3], baseline[:, :3], rtol=1e-10)
    assert not np.allclose(after[:, 3:], baseline[:, 3:])


def test_without_the_mask_the_future_does_leak(g: nb.random.Generator) -> None:
    """The same probe on a non-causal layer, so the test above cannot pass by accident."""
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=5, causal=False, dtype=F64)
    x = sample(1, 5, 8, seed=2)
    baseline = attn(nb.Tensor(x)).numpy()
    disturbed = x.copy()
    disturbed[0, 3:] += 100.0
    assert not np.allclose(attn(nb.Tensor(disturbed)).numpy()[:, :3], baseline[:, :3])


def test_a_non_causal_layer_registers_no_mask(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=5, causal=False)
    assert attn.mask is None
    assert list(attn.named_buffers()) == []


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #


def test_attention_preserves_the_input_shape(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(12, 3, generator=g, max_length=7)
    assert attn(nb.zeros(4, 7, 12)).shape == (4, 7, 12)


def test_attention_accepts_a_shorter_sequence_than_the_mask(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=16)
    assert attn(nb.zeros(2, 3, 8)).shape == (2, 3, 8)


def test_attention_refuses_a_sequence_longer_than_the_mask(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=4)
    with pytest.raises(ShapeError, match="exceeds max_length"):
        attn(nb.zeros(1, 5, 8))


def test_attention_refuses_an_indivisible_width(g: nb.random.Generator) -> None:
    with pytest.raises(ShapeError, match="not divisible"):
        nn.MultiHeadAttention(10, 3, generator=g, max_length=4)


def test_attention_refuses_a_rank_two_input(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=4)
    with pytest.raises(ShapeError, match=r"expects \(batch, time, 8\)"):
        attn(nb.zeros(4, 8))


def test_the_scale_is_one_over_root_head_dim(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(64, 4, generator=g, max_length=4)
    assert attn.head_dim == 16
    assert attn.scale == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #


def test_attention_gradients(g: nb.random.Generator) -> None:
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=4, dtype=F64)
    x = sample(2, 4, 8, seed=3)
    check(attn, x, name="MultiHeadAttention/input")
    check_module(attn, lambda m: m(nb.Tensor(x)), name="MultiHeadAttention/parameters")


def test_masked_positions_receive_no_gradient(g: nb.random.Generator) -> None:
    """`-inf` in the forward pass must not become `nan` in the backward one."""
    attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=4, dtype=F64)
    x = nb.Tensor(sample(2, 4, 8, seed=4), requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))
    for parameter in attn.parameters():
        assert parameter.grad is not None
        assert np.all(np.isfinite(parameter.grad))


# --------------------------------------------------------------------------- #
# the block
# --------------------------------------------------------------------------- #


def test_block_preserves_the_input_shape(g: nb.random.Generator) -> None:
    block = nn.TransformerBlock(16, 4, generator=g, max_length=8)
    assert block(nb.zeros(3, 8, 16)).shape == (3, 8, 16)


def test_block_gradients(g: nb.random.Generator) -> None:
    block = nn.TransformerBlock(8, 2, generator=g, max_length=4, dtype=F64)
    x = sample(2, 4, 8, seed=5)
    check(block, x, name="TransformerBlock/input", tol=1e-5)
    check_module(block, lambda m: m(nb.Tensor(x)), name="TransformerBlock/parameters", tol=1e-5)


def test_block_is_causal(g: nb.random.Generator) -> None:
    block = nn.TransformerBlock(8, 2, generator=g, max_length=5, dtype=F64)
    x = sample(1, 5, 8, seed=6)
    baseline = block(nb.Tensor(x)).numpy()
    disturbed = x.copy()
    disturbed[0, 2:] += 50.0
    np.testing.assert_allclose(
        block(nb.Tensor(disturbed)).numpy()[:, :2], baseline[:, :2], rtol=1e-9
    )


def test_two_thirds_of_a_block_is_the_feed_forward_network(g: nb.random.Generator) -> None:
    """The claim in the docstring, as an assertion rather than a sentence."""
    d = 64
    block = nn.TransformerBlock(d, 4, generator=g, max_length=8)
    attention = block.attention.parameter_count()
    mlp = block.mlp.parameter_count()
    assert attention == 4 * d * d + 4 * d  # 3d^2 fused qkv + d^2 projection, plus biases
    assert mlp == 8 * d * d + 5 * d  # two 4d^2 matrices, plus their biases
    assert mlp / attention == pytest.approx(2.0, rel=0.05)


def test_pre_norm_leaves_the_residual_path_unnormalised(g: nb.random.Generator) -> None:
    """A zeroed block output must return the input exactly, which post-norm would not."""
    block = nn.TransformerBlock(8, 2, generator=g, max_length=4, dtype=F64)
    for name, parameter in block.named_parameters():
        if name.endswith("weight") and ("projection" in name or name.startswith("mlp.2")):
            parameter.assign_(np.zeros(parameter.shape))
        if name.endswith("bias") and ("projection" in name or name.startswith("mlp.2")):
            parameter.assign_(np.zeros(parameter.shape))
    x = sample(2, 4, 8, seed=7)
    np.testing.assert_allclose(block(nb.Tensor(x)).numpy(), x, rtol=1e-12)
