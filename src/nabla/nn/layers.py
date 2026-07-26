"""The layers.

Every layer here is a `forward` written in terms of ops that already have
gradient-checked vjps. None of them registers a new primitive, which is a
deliberate constraint: a layer that needed one would be a layer whose gradient
had never been compared against a central difference, and the whole argument for
trusting this package is that no such code exists in it.

The cost of that constraint is tape nodes. `LayerNorm` is six nodes where a
fused implementation would be one, and each node holds the buffers its closure
captured. On a transformer block that is roughly 30% more peak memory than a
fused version would need, and it is the price of every gradient in the file
being verified rather than argued.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._typing import Array, DTypeLike, Shape
from ..errors import ShapeError
from ..ops.activation import gelu, relu, sigmoid, tanh
from ..ops.basic import sqrt
from ..ops.reduce import var
from ..ops.shape import getitem
from ..random import Generator
from ..tensor import Tensor, ones, zeros
from . import init
from .module import Module

__all__ = [
    "GELU",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "ReLU",
    "Sigmoid",
    "Tanh",
]

#: The signature every initialiser in `nabla.nn.init` satisfies once its keyword
#: arguments are bound. A layer takes one of these rather than a string naming a
#: scheme, so a caller can pass their own without the layer needing a registry.
Initialiser = Callable[[Shape, Generator], Tensor]


class Linear(Module):
    """Affine map: `x @ w + b`.

    The weight is stored as `(in_features, out_features)` and used without a
    transpose. Most frameworks store `(out, in)` and transpose in the forward
    pass, which is a legacy of the row-major convention in the original BLAS
    call sites; here the forward pass is `np.matmul` on contiguous data and the
    only thing the choice costs is that a checkpoint from another framework has
    to be transposed on the way in. `nabla.nn.init.fan_in_out` is the one place
    that has to agree with this layout.

    The bias starts at zero. Symmetry breaking is the weights' job - two units
    with identical weights compute the same function and receive the same
    gradient forever, whatever their biases are - so a random bias adds an
    arbitrary prior on the output and buys nothing.

    >>> import nabla
    >>> from nabla import nn
    >>> layer = nn.Linear(4, 3, generator=nabla.random.Generator(seed=0))
    >>> layer(nabla.zeros(8, 4)).shape
    (8, 3)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: Generator,
        bias: bool = True,
        initialiser: Initialiser | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            msg = f"Linear needs positive feature counts, got {in_features} and {out_features}"
            raise ShapeError(msg)
        self.in_features = in_features
        self.out_features = out_features
        # Xavier by default because a Linear does not know what comes after it.
        # Kaiming's factor of sqrt(2) is a correction for ReLU specifically, and
        # applying it to a layer feeding a softmax makes the initial logits
        # larger than they should be.
        draw = initialiser or _default_initialiser(dtype)
        self.weight = self.register_parameter(
            "weight", draw((in_features, out_features), generator)
        )
        self.bias: Tensor | None = None
        if bias:
            self.bias = self.register_parameter(
                "bias", zeros(out_features, dtype=dtype, requires_grad=True)
            )

    def extra_repr(self) -> str:
        """Feature counts and whether there is a bias."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the affine map to the last axis of `x`."""
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


def _default_initialiser(dtype: DTypeLike | None) -> Initialiser:
    def draw(shape: Shape, generator: Generator) -> Tensor:
        return init.xavier_uniform(shape, generator, dtype=dtype)

    return draw


class Embedding(Module):
    """A lookup table: integer ids in, rows of a weight matrix out.

    The forward pass is `weight[ids]`, which is `nabla.getitem`, so the backward
    pass is that op's `np.add.at` scatter. That matters for a token that appears
    twice in a batch: it must receive both cotangents, and the plain assignment
    form of the scatter would keep only the last. It is the reason `getitem`
    pays for the unbuffered `add.at` rather than the fast assignment, and an
    embedding table is where the difference shows up.

    Initialised from a truncated normal with a small standard deviation. The
    fan-in argument does not apply - a lookup performs no sum, so there is no
    variance to preserve - and what matters instead is that the embedding starts
    small enough that the first layer's pre-activations are in range. 0.02 is
    the value the GPT-2 code used and it has become the convention.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        generator: Generator,
        std: float = 0.02,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            msg = f"Embedding needs positive sizes, got {num_embeddings} and {embedding_dim}"
            raise ShapeError(msg)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = self.register_parameter(
            "weight",
            init.truncated_normal((num_embeddings, embedding_dim), generator, std=std, dtype=dtype),
        )

    def extra_repr(self) -> str:
        """Table size."""
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

    def forward(self, ids: Tensor | Array) -> Tensor:
        """Look up `ids`, of any shape, returning `ids.shape + (embedding_dim,)`."""
        index = ids.data if isinstance(ids, Tensor) else np.asarray(ids)
        if index.dtype.kind not in "iu":
            msg = f"Embedding ids must be integers, got dtype {index.dtype}"
            raise ShapeError(msg)
        return getitem(self.weight, index)


class LayerNorm(Module):
    """Normalise the last axis to zero mean and unit variance, then rescale.

    Per *example*, not per batch, which is the whole point: the statistics do
    not depend on the other elements of the batch, so a model behaves the same
    at batch size 1 as at 256, and there is no separate inference path and no
    running average to keep. Batch normalisation buys slightly better statistics
    and pays for them with all three of those.

    `eps` sits inside the square root, not outside. Outside, it stops a division
    by zero but leaves the derivative of `sqrt` at zero infinite, so a constant
    input produces an infinite gradient rather than a bounded one.

    Built from `var`, `sqrt` and arithmetic rather than as one primitive: six
    tape nodes instead of one, in exchange for a gradient that is composed
    entirely of vjps `check_grad` has already verified. The variance op
    recomputes the mean that the centring also needs, which is one extra
    reduction over the last axis per call.
    """

    def __init__(
        self,
        normalised_shape: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        if normalised_shape <= 0:
            msg = f"LayerNorm needs a positive width, got {normalised_shape}"
            raise ShapeError(msg)
        self.normalised_shape = normalised_shape
        self.eps = eps
        self.weight: Tensor | None = None
        self.bias: Tensor | None = None
        if affine:
            # Scale starts at one and shift at zero, so a freshly built LayerNorm
            # is exactly the normalisation and nothing else. Any other start
            # would mean the layer's initial behaviour depended on a constant
            # nobody chose deliberately.
            self.weight = self.register_parameter("weight", _ones(normalised_shape, dtype=dtype))
            self.bias = self.register_parameter(
                "bias", zeros(normalised_shape, dtype=dtype, requires_grad=True)
            )

    def extra_repr(self) -> str:
        """Width, epsilon and whether the affine parameters exist."""
        return (
            f"normalised_shape={self.normalised_shape}, eps={self.eps}, "
            f"affine={self.weight is not None}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Normalise the last axis of `x`."""
        _check_last_axis(x, self.normalised_shape, "LayerNorm")
        centred = x - x.mean(-1, keepdims=True)
        normed = centred / sqrt(var(x, -1, keepdims=True) + self.eps)
        if self.weight is not None and self.bias is not None:
            return normed * self.weight + self.bias
        return normed


class RMSNorm(Module):
    """Divide the last axis by its root mean square, then rescale.

    LayerNorm without the re-centring. The claim behind it (Zhang and Sennrich,
    2019) is that subtracting the mean is not what makes LayerNorm work - the
    rescaling is - and dropping it removes one reduction, one broadcast
    subtraction and their two tape nodes from every normalisation in the model.
    In a transformer with pre-norm that is two normalisations per block, so the
    saving is real and the reported quality difference is not.

    There is no shift parameter, for the same reason: a shift is what a
    re-centring layer needs to undo its own centring.
    """

    def __init__(
        self,
        normalised_shape: int,
        *,
        eps: float = 1e-6,
        affine: bool = True,
        dtype: DTypeLike | None = None,
    ) -> None:
        super().__init__()
        if normalised_shape <= 0:
            msg = f"RMSNorm needs a positive width, got {normalised_shape}"
            raise ShapeError(msg)
        self.normalised_shape = normalised_shape
        self.eps = eps
        self.weight: Tensor | None = None
        if affine:
            self.weight = self.register_parameter("weight", _ones(normalised_shape, dtype=dtype))

    def extra_repr(self) -> str:
        """Width, epsilon and whether the scale parameter exists."""
        return (
            f"normalised_shape={self.normalised_shape}, eps={self.eps}, "
            f"affine={self.weight is not None}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Scale the last axis of `x` to unit root mean square."""
        _check_last_axis(x, self.normalised_shape, "RMSNorm")
        normed = x / sqrt((x * x).mean(-1, keepdims=True) + self.eps)
        if self.weight is not None:
            return normed * self.weight
        return normed


class Dropout(Module):
    """Zero a random fraction `p` of the activations during training.

    # Why inverted

    Dropout has to leave the expected activation unchanged, or every layer after
    it sees a different input scale in training than in inference. There are two
    places to put the correction:

    * Scale at *test* time - multiply by `1 - p` when not training. This is the
      form in the original paper, and it means the deployed model carries a
      constant that exists only because of how it was trained. Every export
      path, every reimplementation for serving, and every person reading the
      inference code has to know about `p`.
    * Scale at *train* time - divide the surviving activations by `1 - p`. Then
      `E[output] = input` during training, and inference is the identity: the
      layer disappears completely in eval mode.

    The second is "inverted dropout" and it is what this class does. It costs
    one extra division per element during training, on the array that was going
    to be multiplied by the mask anyway, and it buys an inference path with no
    dropout-shaped hole in it.

    # The generator

    Dropout needs randomness, and it takes an explicit `Generator` for the
    reason `nabla.random` exists: a global stream makes the initialisation of
    every layer built after a dropout layer depend on how many forward passes
    happened first. With an explicit generator, a run is reproducible from its
    seeds and nothing else.

    `p = 0` short-circuits entirely - no draw, no mask, no tape node - so a model
    written with dropout everywhere costs nothing when it is switched off.
    """

    def __init__(self, p: float, *, generator: Generator) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            msg = f"dropout probability must be in [0, 1), got {p}"
            raise ValueError(msg)
        self.p = p
        self._generator = generator

    def extra_repr(self) -> str:
        """The drop probability."""
        return f"p={self.p}"

    def forward(self, x: Tensor) -> Tensor:
        """Drop and rescale if training; return `x` unchanged otherwise."""
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = self._generator.bernoulli(x.shape, p=keep, dtype=x.dtype)
        return x * mask / keep


# --------------------------------------------------------------------------- #
# activations as modules
# --------------------------------------------------------------------------- #
#
# These hold no parameters and exist only so that `Sequential` can contain them.
# Calling `nabla.relu` directly in a hand-written `forward` is the same thing and
# one object cheaper.


class ReLU(Module):
    """`max(x, 0)` as a module. See `nabla.relu`."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU elementwise."""
        return relu(x)


class GELU(Module):
    """GELU as a module, in the tanh approximation. See `nabla.gelu`."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU elementwise."""
        return gelu(x)


class Tanh(Module):
    """`tanh` as a module. See `nabla.tanh`."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh elementwise."""
        return tanh(x)


class Sigmoid(Module):
    """The logistic function as a module. See `nabla.sigmoid`."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply the logistic function elementwise."""
        return sigmoid(x)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ones(width: int, *, dtype: DTypeLike | None) -> Tensor:
    """A trainable vector of ones, for a normalisation scale."""
    return ones(width, dtype=dtype, requires_grad=True)


def _check_last_axis(x: Tensor, width: int, layer: str) -> None:
    """Refuse an input whose last axis is not the width the layer normalises.

    Without this the broadcast in the affine step would silently succeed for a
    width of 1 and produce a shape nobody asked for; a normalisation applied
    over the wrong axis is a bug that trains, slowly and worse.
    """
    if not x.shape or x.shape[-1] != width:
        msg = f"{layer} normalises the last axis over {width} elements, but got shape {x.shape}"
        raise ShapeError(msg)
