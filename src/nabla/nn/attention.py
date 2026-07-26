"""Multi-head self-attention and the block that wraps it.

# What attention costs here

Nabla has no fused attention kernel, so the `(batch, heads, time, time)` score
tensor is materialised in full, and so is its cotangent. At the sizes
`examples/char_transformer` uses - batch 32, 4 heads, 64 timesteps - that is
`32 * 4 * 64 * 64 * 4` bytes, or 2 MiB per block per direction, which is
nothing. The cost is quadratic in the sequence length: at 1,024 timesteps the
same expression is 512 MiB, per block, per direction. That is why every serious
implementation of this operation is a fused kernel that never writes the matrix
down, and it is the limit to know about before scaling anything here up.

# The masking convention

The causal mask is *added* to the scores as an array of 0 and `-inf` before the
softmax, rather than applied afterwards. Afterwards is wrong twice over: the
probabilities would no longer sum to 1, and the masked entries would still have
contributed to the denominator, so a token's attention over its own past would
depend on how many future tokens happened to be in the window.

`-inf` rather than a large negative constant. `exp(-inf)` is exactly 0 with no
warning, whereas a constant like -1e9 relies on underflow and leaves a residue
that is only zero because float32 ran out of exponent. The one hazard with
`-inf` is a row that is masked everywhere, which would give `nan`; a causal mask
always leaves the diagonal open, so no such row exists here.
"""

from __future__ import annotations

import math

import numpy as np

from .._typing import Array, DTypeLike
from ..errors import ShapeError
from ..ops.activation import softmax
from ..ops.shape import permute, reshape, split
from ..random import Generator
from ..tensor import Tensor
from .layers import GELU, Dropout, LayerNorm, Linear
from .module import Module, Sequential

__all__ = ["MultiHeadAttention", "TransformerBlock", "causal_mask"]


def causal_mask(length: int, dtype: DTypeLike | None = None) -> Array:
    """A `(length, length)` additive mask: 0 where attention is allowed, `-inf` above.

    Built once per model rather than per forward pass and kept as a buffer. It
    is a constant, and rebuilding it every step would allocate a `T^2` array on
    the hot path to hold values that never change.

    >>> from nabla.nn.attention import causal_mask
    >>> causal_mask(3).tolist()
    [[0.0, -inf, -inf], [0.0, 0.0, -inf], [0.0, 0.0, 0.0]]
    """
    resolved = np.dtype("float32" if dtype is None else dtype)
    allowed = np.tril(np.ones((length, length), dtype=bool))
    mask: Array = np.where(allowed, 0.0, -np.inf).astype(resolved)
    return mask


class MultiHeadAttention(Module):
    """Scaled dot-product attention over `n_heads` independent subspaces.

    Self-attention only: queries, keys and values all come from the same input.
    Cross-attention would need a second input and a second projection, and
    nothing in this package uses it.

    # Why heads at all

    One head of width `d_model` computes a single similarity between two
    positions. Splitting the same width into `h` subspaces gives `h`
    similarities for the same arithmetic - the projections are the same total
    size, and the score matmul is `h` products of width `d_model/h` instead of
    one of width `d_model`. The cost is that each head sees a narrower subspace,
    which is why `d_model` per head below about 32 stops helping.

    # Why one fused qkv projection

    Queries, keys and values are produced by one `Linear` of width `3 * d_model`
    and then split, rather than by three separate layers. Mathematically
    identical; one matmul of three times the width is measurably faster than
    three matmuls in NumPy because it amortises the call overhead and gives BLAS
    a larger block. The cost is that `split` puts three tape nodes where a
    three-layer version would have none, and that the parameter is one tensor
    whose thirds have to be documented rather than three named ones.

    >>> import nabla
    >>> from nabla import nn
    >>> g = nabla.random.Generator(seed=0)
    >>> attn = nn.MultiHeadAttention(8, 2, generator=g, max_length=4)
    >>> attn(nabla.zeros(2, 4, 8)).shape
    (2, 4, 8)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        generator: Generator,
        max_length: int,
        causal: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model {d_model} is not divisible by n_heads {n_heads}"
            raise ShapeError(msg)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.max_length = max_length
        # 1/sqrt(head_dim). Without it the dot product of two vectors of
        # independent unit-variance components has variance head_dim, so the
        # softmax saturates as soon as the width is large and the gradient
        # through it vanishes before training starts.
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = self.register_module(
            "qkv", Linear(d_model, 3 * d_model, generator=generator, bias=bias, dtype=dtype)
        )
        self.projection = self.register_module(
            "projection", Linear(d_model, d_model, generator=generator, bias=bias, dtype=dtype)
        )
        self.dropout = self.register_module("dropout", Dropout(dropout, generator=generator))
        self.mask: Tensor | None = None
        if causal:
            self.mask = self.register_buffer("mask", Tensor(causal_mask(max_length, dtype)))

    def extra_repr(self) -> str:
        """Width, head count and whether the mask is applied."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Attend over the time axis of `x`, of shape `(batch, time, d_model)`."""
        if x.ndim != 3 or x.shape[2] != self.d_model:
            msg = f"MultiHeadAttention expects (batch, time, {self.d_model}), got shape {x.shape}"
            raise ShapeError(msg)
        batch, time, _ = x.shape
        if time > self.max_length:
            msg = f"sequence of length {time} exceeds max_length {self.max_length}"
            raise ShapeError(msg)

        q, k, v = split(self.qkv(x), 3, axis=2)
        # (batch, time, d_model) -> (batch, heads, time, head_dim). The permute
        # is what puts the head axis where matmul's batch broadcasting expects
        # it; doing the reshape without it would interleave heads and positions.
        shape = (batch, time, self.n_heads, self.head_dim)
        qh = permute(reshape(q, shape), 0, 2, 1, 3)
        kh = permute(reshape(k, shape), 0, 2, 1, 3)
        vh = permute(reshape(v, shape), 0, 2, 1, 3)

        scores = (qh @ permute(kh, 0, 1, 3, 2)) * self.scale
        if self.mask is not None:
            scores = scores + self.mask[:time, :time]
        weights: Tensor = self.dropout(softmax(scores, axis=-1))

        attended = permute(weights @ vh, 0, 2, 1, 3)
        merged = reshape(attended, (batch, time, self.d_model))
        out: Tensor = self.projection(merged)
        return out


class TransformerBlock(Module):
    """Pre-norm attention and feed-forward, each behind a residual connection.

    # Pre-norm, not post-norm

    The normalisation goes *before* each sub-layer and outside the residual:

        x = x + attn(norm(x))
        x = x + mlp(norm(x))

    rather than `x = norm(x + attn(x))`. The difference is the identity path:
    with pre-norm there is an unbroken sum from the input to the output, so the
    gradient reaching layer 1 is the gradient at the output plus corrections,
    and it cannot vanish with depth. Post-norm puts a normalisation on that
    path, whose Jacobian rescales the gradient at every layer; the product of
    those rescalings is why the original transformer needed a learning-rate
    warmup to train at all, and why every implementation since has quietly
    switched.

    The cost is that the output of a deep pre-norm stack has a growing variance
    - each block adds to it - which is why a final normalisation before the
    output head is not optional.

    # The feed-forward width

    Four times `d_model`, which is the ratio the original paper used and nobody
    has since found a reason to change. It is also where most of the parameters
    are, and by more than people expect: attention holds `3 d^2` in the fused
    qkv projection and `d^2` in the output projection, for `4 d^2` in total,
    while the MLP holds `4 d^2` in each of its two matrices. Two thirds of a
    block's parameters are in the feed-forward network, not in the attention -
    which is why parameter-efficient fine-tuning methods that only touch the
    attention projections are touching a third of the model.

    >>> import nabla
    >>> from nabla import nn
    >>> g = nabla.random.Generator(seed=0)
    >>> block = nn.TransformerBlock(8, 2, generator=g, max_length=4)
    >>> block(nabla.zeros(2, 4, 8)).shape
    (2, 4, 8)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        generator: Generator,
        max_length: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        causal: bool = True,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm1 = self.register_module("norm1", LayerNorm(d_model, dtype=dtype))
        self.attention = self.register_module(
            "attention",
            MultiHeadAttention(
                d_model,
                n_heads,
                generator=generator,
                max_length=max_length,
                causal=causal,
                dropout=dropout,
                dtype=dtype,
            ),
        )
        self.norm2 = self.register_module("norm2", LayerNorm(d_model, dtype=dtype))
        self.mlp = self.register_module(
            "mlp",
            Sequential(
                Linear(d_model, mlp_ratio * d_model, generator=generator, dtype=dtype),
                GELU(),
                Linear(mlp_ratio * d_model, d_model, generator=generator, dtype=dtype),
                Dropout(dropout, generator=generator),
            ),
        )

    def extra_repr(self) -> str:
        """The model width."""
        return f"d_model={self.d_model}"

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention and the feed-forward network, each with a residual."""
        attended: Tensor = x + self.attention(self.norm1(x))
        mixed: Tensor = attended + self.mlp(self.norm2(attended))
        return mixed
