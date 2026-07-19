"""The operation set.

An operation is a forward NumPy expression plus a closure that maps the output
cotangent to input cotangents, handed to `nabla.tensor.record`. That is the
whole protocol: there is no registry, no dispatch table and no base class, so
adding an op is one function in one module and it is immediately usable from
`Tensor` if it deserves an operator.

The modules are split by what the vjp *is*, not by what the forward pass
computes, because the vjp is the part that has to be reviewed:

* `basic` - diagonal Jacobians; the backward pass is a multiply and an
  `unbroadcast`.
* `reduce` - many-to-few; the backward pass scatters.
* `shape` - rearrangements; the backward pass rearranges back, or adds where the
  forward pass duplicated.
* `linalg` - contractions; the backward pass contracts with the other operand.
* `activation` - elementwise, but each one carries a numerical stability
  argument that belongs next to it.
"""

from __future__ import annotations

from .activation import gelu, log_softmax, relu, sigmoid, softmax, tanh
from .basic import abs, add, div, exp, log, maximum, minimum, mul, neg, pow, sqrt, sub
from .linalg import dot, matmul, outer, trace
from .reduce import logsumexp, max, mean, min, std, sum, var
from .shape import (
    broadcast_to,
    concat,
    expand_dims,
    getitem,
    pad,
    permute,
    reshape,
    split,
    squeeze,
    stack,
    swapaxes,
    transpose,
)

__all__ = [
    "abs",
    "add",
    "broadcast_to",
    "concat",
    "div",
    "dot",
    "exp",
    "expand_dims",
    "gelu",
    "getitem",
    "log",
    "log_softmax",
    "logsumexp",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "mul",
    "neg",
    "outer",
    "pad",
    "permute",
    "pow",
    "relu",
    "reshape",
    "sigmoid",
    "softmax",
    "split",
    "sqrt",
    "squeeze",
    "stack",
    "std",
    "sub",
    "sum",
    "swapaxes",
    "tanh",
    "trace",
    "transpose",
    "var",
]
