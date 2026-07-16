"""Nabla: reverse-mode automatic differentiation over NumPy arrays.

A small, auditable training stack. It is not trying to be fast - there is no
fusion, no threading and no device support, and every operation is a NumPy call
with a Python closure attached. It is trying to be correct and legible, on the
premise that when a training run produces the wrong number, a framework you can
read end to end in an afternoon is worth more than one you cannot.

What that buys, concretely: every gradient in this package is verified against a
central-difference numerical gradient in float64 (`nabla.check_grad`), the
broadcast reduction that hand-rolled engines usually get wrong is implemented
once and tested exhaustively, and the backward pass refuses to continue if any
vjp returns a cotangent whose shape does not match its input.

    >>> import nabla
    >>> g = nabla.random.Generator(seed=0)
    >>> w = g.normal((3, 2), std=0.1, requires_grad=True)
    >>> b = nabla.zeros(2, requires_grad=True)
    >>> x = g.normal((8, 3))
    >>> y = nabla.relu(x @ w + b)
    >>> loss = (y * y).mean()
    >>> loss.backward()
    >>> w.grad.shape, b.grad.shape
    ((3, 2), (2,))

Layout:

* `nabla.tensor` - the `Tensor` type, the tape, `no_grad`, and the broadcast
  reduction helper.
* `nabla.ops` - forward implementations and their vector-Jacobian products.
* `nabla.grad` - the backward pass, functional `grad`, `check_grad`, clipping.
* `nabla.random` - explicit seeded generators; nothing here touches NumPy's
  global random state.
* `nabla.errors` - the exception hierarchy.

On top of that core, four packages that are ordinary users of it and contain no
autodiff of their own:

* `nabla.nn` - `Module`, layers, initialisers, losses.
* `nabla.optim` - SGD, Adam, AdamW, RMSProp, and learning-rate schedules.
* `nabla.data` - datasets, batching, and the synthetic problems the examples use.
* `nabla.train` - a training loop and a pickle-free checkpoint format.
"""

from __future__ import annotations

# The dependency runs one way: `nn`, `optim`, `data` and `train` are ordinary
# users of `tensor`, `ops` and `grad`, and no part of the core imports them. They
# are listed here so that `import nabla` gives the whole stack; the cost is a few
# milliseconds of import time for a caller who only wanted tensors.
from . import data, errors, nn, ops, optim, random, train
from .errors import (
    AutogradError,
    DTypeError,
    GradientCheckError,
    InPlaceOperationError,
    ModuleError,
    NablaError,
    OptimizerError,
    ShapeError,
)
from .grad import (
    GradCheckReport,
    backward,
    check_grad,
    clip_grad_norm_,
    clip_grad_value_,
    grad,
    topological_order,
)
from .ops import (
    abs,
    add,
    broadcast_to,
    concat,
    div,
    dot,
    exp,
    expand_dims,
    gelu,
    getitem,
    log,
    log_softmax,
    logsumexp,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    mul,
    neg,
    outer,
    pad,
    permute,
    pow,
    relu,
    reshape,
    sigmoid,
    softmax,
    split,
    sqrt,
    squeeze,
    stack,
    std,
    sub,
    sum,
    swapaxes,
    tanh,
    trace,
    transpose,
    var,
)
from .tensor import (
    Node,
    Tensor,
    arange,
    eye,
    from_numpy,
    full,
    is_grad_enabled,
    linspace,
    no_grad,
    ones,
    ones_like,
    record,
    tensor,
    unbroadcast,
    zeros,
    zeros_like,
)

__version__ = "0.1.0"

__all__ = [
    "AutogradError",
    "DTypeError",
    "GradCheckReport",
    "GradientCheckError",
    "InPlaceOperationError",
    "ModuleError",
    "NablaError",
    "Node",
    "OptimizerError",
    "ShapeError",
    "Tensor",
    "__version__",
    "abs",
    "add",
    "arange",
    "backward",
    "broadcast_to",
    "check_grad",
    "clip_grad_norm_",
    "clip_grad_value_",
    "concat",
    "data",
    "div",
    "dot",
    "errors",
    "exp",
    "expand_dims",
    "eye",
    "from_numpy",
    "full",
    "gelu",
    "getitem",
    "grad",
    "is_grad_enabled",
    "linspace",
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
    "nn",
    "no_grad",
    "ones",
    "ones_like",
    "ops",
    "optim",
    "outer",
    "pad",
    "permute",
    "pow",
    "random",
    "record",
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
    "tensor",
    "topological_order",
    "trace",
    "train",
    "transpose",
    "unbroadcast",
    "var",
    "zeros",
    "zeros_like",
]
