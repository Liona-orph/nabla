"""Layers, initialisers and losses.

The module system is deliberately small: a `Module` owns three dictionaries -
parameters, buffers, submodules - and everything else in this package is derived
from them. Registration into those dictionaries is an explicit call rather than
an intercepted attribute assignment; `nabla.nn.module` explains why at length,
and the short version is that the magic version fails silently when a parameter
is stored somewhere the interceptor does not look.

Layout:

* `nabla.nn.module` - `Module`, `Sequential`, `ModuleList`, `ModuleDict`.
* `nabla.nn.init` - initialisers and the fan-in/fan-out reasoning.
* `nabla.nn.layers` - `Linear`, `Embedding`, the normalisations, `Dropout`, and
  the activations wrapped as modules.
* `nabla.nn.attention` - `MultiHeadAttention` and `TransformerBlock`.
* `nabla.nn.loss` - `mse`, `cross_entropy`, `binary_cross_entropy`, `nll`.

>>> import nabla
>>> from nabla import nn
>>> g = nabla.random.Generator(seed=0)
>>> model = nn.Sequential(nn.Linear(2, 8, generator=g), nn.Tanh(), nn.Linear(8, 1, generator=g))
>>> model.parameter_count()
33
>>> loss = nn.loss.mse(model(nabla.zeros(4, 2)), nabla.zeros(4, 1))
>>> loss.backward()
>>> all(p.grad is not None for p in model.parameters())
True
"""

from __future__ import annotations

from . import init, loss
from .attention import MultiHeadAttention, TransformerBlock, causal_mask
from .layers import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    ReLU,
    RMSNorm,
    Sigmoid,
    Tanh,
)
from .loss import binary_cross_entropy, cross_entropy, mse, nll
from .module import Module, ModuleDict, ModuleList, Sequential

__all__ = [
    "GELU",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Linear",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MultiHeadAttention",
    "RMSNorm",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "TransformerBlock",
    "binary_cross_entropy",
    "causal_mask",
    "cross_entropy",
    "init",
    "loss",
    "mse",
    "nll",
]
