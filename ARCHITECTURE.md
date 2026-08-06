# Architecture

How Nabla is put together, what an operation has to provide, and how to add one.
Read `src/nabla/tensor.py` and `src/nabla/grad.py` alongside this; the code is
the specification and this document is the map.

## Contents

1. [The layers of the package](#the-layers-of-the-package)
2. [The tape](#the-tape)
3. [The backward pass](#the-backward-pass)
4. [The vjp protocol](#the-vjp-protocol)
5. [Broadcast reduction](#broadcast-reduction)
6. [The dtype policy](#the-dtype-policy)
7. [Immutability](#immutability)
8. [How to add an operation](#how-to-add-an-operation)
9. [How to add a layer](#how-to-add-a-layer)
10. [Invariants a change must not break](#invariants-a-change-must-not-break)

## The layers of the package

The dependency graph is acyclic and one-directional. Nothing in the core knows
that `nn`, `optim`, `data` or `train` exist.

```
        numpy
          |
     _typing, errors
          |
       tensor  <-------- the Tensor type, Node, no_grad, unbroadcast, record
        /   \
      ops    grad  <---- vjps                backward, grad, check_grad, clipping
        \   /
       random
          |
    +-----+-----+-----+
    |     |     |     |
    nn  optim  data  train
```

`train` depends on `nn` and `optim`; nothing else crosses sideways. The one
cycle in the whole package is deliberate and confined: `Tensor.__add__` needs
`nabla.ops.basic.add`, and `nabla.ops.basic` needs `Tensor`. It is broken by
importing the op *inside* the operator method, which costs one `sys.modules`
lookup per operator call and keeps `nabla.ops` importable on its own.

## The tape

There is no tape object. A tensor that was produced by an operation holds a
`Node`:

```python
class Node:
    __slots__ = ("backward_fn", "op", "parents")
```

- `op` is a string, used only in error messages and `repr`. Nothing dispatches
  on it.
- `parents` is the tuple of input tensors, in the order the operation received
  them.
- `backward_fn` is a closure: it takes the output's cotangent as a NumPy array
  and returns one cotangent per parent, in the same order.

The graph is therefore the tensors themselves. Consequences:

- It is freed by reference counting the moment the loss goes out of scope.
- There is no global state to reset between iterations, and two models training
  in one process cannot interfere.
- Anything that keeps a tensor alive keeps its whole history alive. Appending
  `loss` to a list for logging retains every activation of that step; appending
  `loss.item()` retains a float. This is the most common memory leak in code
  built on any framework with this design.

A leaf is a tensor with `node is None`: a parameter, an input, a constant, or
anything produced under `no_grad()`.

## The backward pass

`nabla.grad.backward(output, gradient=None)` does four things.

**1. Seed.** The cotangent of the output with respect to itself. For a scalar
loss that is `1`; for a non-scalar output there is no canonical choice, so an
explicit `gradient` of the same shape is required and the error says so.

**2. Order.** `topological_order(root)` is an iterative reverse post-order
depth-first search, returning every tensor the root depends on with consumers
before producers. Iterative rather than recursive because CPython's default
recursion limit is 1,000 and 1,000 operations is a small model. Ordered
topologically rather than by creation because creation order happens to be a
valid topological order for straight-line code, and relying on that is the
traditional way for a hand-rolled engine to be subtly wrong the first time a
tensor is reused across a branch.

**3. Walk.** For each tensor in order, pop its accumulated cotangent from a
working dictionary. If the tensor is a leaf or has called `retain_grad()`, add
the cotangent into `.grad`. If it has a node, call `backward_fn` and add each
returned cotangent into the pending entry for the corresponding parent.

Interior cotangents are dropped as soon as they are consumed, so peak memory is
proportional to the *width* of the graph, not its size.

**4. Check.** Every returned cotangent's shape must equal its parent's shape.
This is checked at the operation that produced it, not at the leaf, because by
the leaf the shapes have already been summed into each other and the culprit is
unidentifiable. Almost every failure of this check is a missing `unbroadcast`.

Gradients *accumulate*: `.grad` is added to, never replaced. That is the only
behaviour that makes a tensor used twice come out right, and it is what makes
gradient accumulation across micro-batches work with no extra machinery. The
cost is the canonical beginner bug - a loop that forgets `zero_grad()` sums the
gradients of every step it has ever taken - and the alternative, overwriting, is
worse, because it turns a tensor used twice into a wrong answer with no error at
all.

## The vjp protocol

This is the entire extension point.

```python
def record(
    data: Array,  # the forward result, freshly allocated
    op: str,  # a name, for error messages
    parents: tuple[Tensor, ...],  # the inputs, in order
    backward_fn: BackwardFn,  # cotangent -> tuple of cotangents
) -> Tensor: ...
```

where

```python
BackwardFn = Callable[[Array], tuple[Array | None, ...]]
```

An operation must satisfy five rules.

1. **Return one cotangent per parent, in order.** The backward pass checks the
   count and raises `AutogradError` if it disagrees.
2. **Each cotangent must have exactly its parent's shape.** If the forward pass
   broadcast, the vjp must call `unbroadcast`. This is checked.
3. **`None` is allowed** for a parent whose gradient the operation chose not to
   compute. It means "skipped", not "zero". `nabla.ops.basic.pow` uses it: the
   derivative with respect to the exponent is `nan` for a non-positive base, and
   computing it unconditionally would flood a perfectly well-defined backward
   pass - `(-x) ** 2` - with NumPy invalid-value warnings for a value that is
   then discarded.
4. **Close over what you need, and no more.** Whatever the closure captures stays
   alive as long as the output tensor does. `exp` captures its output rather
   than its input, because `d/dx exp(x) = exp(x)`; `sigmoid` does the same.
   `mul` has to capture both inputs, which is reverse mode's memory cost in its
   smallest form.
5. **Do not mutate anything you captured.** Buffers are read-only, so NumPy will
   stop you, but the rule is worth knowing before you meet the exception.

`record` attaches no node - and the operation is therefore free - when recording
is off (`no_grad()`), when no parent requires a gradient, or when the result is
not floating point. The third case covers comparisons and integer arithmetic,
where a gradient is meaningless.

## Broadcast reduction

`nabla.tensor.unbroadcast(grad, shape)` sums a cotangent back down to an input's
shape. It exists once, and every elementwise vjp and `matmul` calls it.

Broadcasting is a *copy*: `x` of shape `(3,)` used against `(4, 3)` behaves as
four copies of `x`. Four copies means four contributions to the loss, so the
gradient of the original `x` is the sum of the four rows.

Two cases, applied in order:

1. `grad` has more axes than `shape`. Those leading axes did not exist in the
   input at all, so they are summed away entirely.
2. `shape` has a `1` where `grad` has more. That axis was stretched, so it is
   summed with `keepdims=True` to leave the `1` in place.

```
grad (4, 3) -> shape (3,)     sum over axis 0            -> (3,)
grad (4, 3) -> shape (1, 3)   sum over axis 0, keepdims  -> (1, 3)
grad (2, 3) -> shape ()       sum over both axes         -> ()
grad (3,)   -> shape (3,)     no copy at all
```

A dimension of size 0 needs no special case: NumPy broadcasts 1 against 0 to 0,
and summing an empty axis with `keepdims=True` gives the 0.0 that the empty sum
should be.

Getting this wrong is the single most common defect in hand-written autodiff
engines, because the forward pass keeps working and only the gradient is wrong -
by a factor equal to the broadcast width, which looks exactly like a badly tuned
learning rate. `tests/test_broadcasting.py` compares it against an independent
element-by-element implementation over every interesting shape pair up to rank 3
and over Hypothesis-generated shapes.

## The dtype policy

Four rules, all in `nabla.tensor._coerce` and `as_operands`.

1. **A Python scalar or nested list becomes float32.** It carries no dtype
   intent; NumPy would make it float64, which doubles every buffer to buy
   precision that stochastic gradient descent does not use.
2. **An ndarray keeps its dtype.** `np.zeros(3, np.float64)` stays float64.
   Silently halving the precision of an array a caller explicitly widened is the
   kind of surprise this package exists to avoid.
3. **Integers and booleans are preserved and are never differentiable.** They
   are legal tensor contents - indices, masks, labels - but `requires_grad` on
   an integer tensor raises `DTypeError`. An integer gradient is not a rounding
   problem, it is a category error.
4. **A bare Python number stays weak.** `float32_tensor + 1` is float32, not
   float64, because the scalar adopts the other operand's dtype. This
   reproduces NumPy's NEP 50 behaviour deliberately rather than by accident. A
   scalar that cannot be represented in the other operand's dtype - a float
   alongside an integer tensor - falls back to the ordinary rules, so nothing is
   silently truncated.

`check_grad` promotes everything to float64 before it starts, because in float32
a central difference cannot resolve better than about 1e-3 relative. The check
validates the *mathematics* of the vjp, not the float32 rounding of the forward
pass.

## Immutability

`Tensor.data` is a read-only NumPy array. Every constructor copies; every
operation result is a freshly allocated array that `Tensor._owning` freezes.

The reason is the vjp closure. An operation saves the forward values its backward
pass needs. If a caller mutated one of those buffers afterwards, `backward()`
would compute the exact gradient of a function that was never evaluated, and it
would do it silently, with no error and no obvious symptom. Marking buffers
read-only turns that into a NumPy exception at the point of the mutation.

The cost is one array copy when a tensor is built from a caller's array. In
exchange, `detach()` and `numpy(copy=False)` are free - nothing can write
through them - and the following are all errors with messages that name the
alternative: `x += y`, `x[0] = 1`, `x.data[0] = 1`.

The one sanctioned mutation is `Tensor.assign_`, which an optimiser needs. It
*rebinds* the buffer rather than writing into it, so any vjp closure holding the
old values still sees what it was given. That costs one allocation per call. It
is legal only on a leaf: an interior tensor's value is a claim about a
computation that already happened.

Gradient buffers are exempt. `clip_grad_norm_` writes to `.grad` in place, which
is safe precisely because no vjp closure ever captures a gradient.

## How to add an operation

A worked example: `softplus(x) = log(1 + exp(x))`, the smooth relu.

### 1. Write it

In `src/nabla/ops/activation.py`, because its interesting property is numerical
stability:

```python
def softplus(a: TensorLike) -> Tensor:
    """`log(1 + exp(x))`, computed without overflowing.

    The naive form overflows for `x` above about 88 in float32, where the true
    answer is simply `x` to well within float32 resolution. The stable identity
    is `max(x, 0) + log(1 + exp(-|x|))`: the exponent is never positive, so
    nothing overflows, and the `max` term carries the magnitude.

    The derivative is `sigmoid(x)`, which the vjp computes branch-wise for the
    same reason.
    """
    ta = as_tensor(a)
    x = ta.data
    out: Array = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

    def vjp(g: Array) -> tuple[Array | None, ...]:
        positive = x >= 0
        exp_neg_abs = np.exp(-np.abs(x))
        derivative = np.where(
            positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs)
        )
        return (g * derivative,)

    return record(out, "softplus", (ta,), vjp)
```

Note what the shape of this is: a forward NumPy expression, a closure, and a
call to `record`. There is no registry, no base class and no dispatch table.

### 2. Export it

Add `softplus` to `__all__` and the import list in `src/nabla/ops/__init__.py`,
and to `__all__` and the `from .ops import (...)` block in
`src/nabla/__init__.py`. Both lists are kept sorted; `ruff check` enforces it.

### 3. Test it

In `tests/test_ops_activation.py`. Two assertions, and they must be independent
of each other:

```python
def test_softplus_forward_and_backward() -> None:
    x = sample(4, 5, seed=77)
    # Forward, against a NumPy expression written from the definition -
    # never against the implementation's own intermediates.
    assert_close(nb.softplus(nb.Tensor(x)), np.log1p(np.exp(x)))
    # Backward, through check_grad, which knows nothing about the vjp.
    check(nb.softplus, x, name="softplus")


def test_softplus_survives_the_input_the_naive_form_cannot() -> None:
    x = np.array([-800.0, 800.0])
    with np.errstate(over="ignore"):
        naive = np.log(1.0 + np.exp(x))
    assert not np.all(np.isfinite(naive))
    np.testing.assert_allclose(nb.softplus(nb.Tensor(x)).numpy(), [0.0, 800.0])
```

Pass `fn` to `check` **unreduced**. `check_grad` contracts the output with a
fixed random vector of its own, and reducing first throws away the asymmetry
that catches a transposed or mis-scattered gradient. `tests/test_grad.py` has a
test that demonstrates exactly that, with a deliberately broken vjp that a
pre-summed check declares correct.

### 4. Add a method or an operator, if it earns one

Only if it reads better that way. `Tensor` mirrors `sum`, `mean`, `reshape` and
`transpose` and nothing else; duplicating the whole op set would double the
surface that has to stay consistent.

### 5. Run the checks

```bash
make ci
```

`ruff check`, `ruff format --check`, `mypy --strict src`, and `pytest -q` - which
runs the doctests in your new docstring as well.

## How to add a layer

Layers live in `nabla.nn` and are ordinary users of the op set. A layer that
needed a new primitive would be a layer whose gradient had never been compared
against a central difference, so the rule is: build it out of existing ops, and
if you cannot, add the op first and gradient-check it there.

```python
class Scale(nn.Module):
    """Multiply by a learned per-feature scale."""

    def __init__(self, width: int) -> None:
        super().__init__()  # first, always
        self.width = width
        self.weight = self.register_parameter(  # explicit; returns the tensor
            "weight", nabla.ones(width, requires_grad=True)
        )

    def extra_repr(self) -> str:
        """One line of configuration for the tree repr."""
        return f"width={self.width}"

    def forward(self, x: nabla.Tensor) -> nabla.Tensor:
        """Scale the last axis of `x`."""
        return x * self.weight
```

Then gradient-check it, with respect to the input and with respect to every
parameter, using the helper in `tests/conftest.py`:

```python
def test_scale_gradients() -> None:
    layer = Scale(5)
    x = sample(4, 5, seed=1)
    check(layer, x, name="Scale/input")
    check_module(layer, lambda m: m(nb.Tensor(x)), name="Scale/parameters")
```

`check_module` turns the module's captured parameters into arguments so
`check_grad` can perturb them. Build the layer in float64 for the check: a
float32 weight inside a float64 comparison puts a 1e-7 rounding error into a
difference quotient that is being judged at 1e-6.

## Invariants a change must not break

A pull request that violates any of these needs an ADR, not a patch.

1. **NumPy is the only runtime dependency.** `pytest`, `hypothesis`, `mypy` and
   `ruff` are development extras.
2. **Every vjp is verified by `check_grad`.** Not by inspection, not by "the
   loss went down".
3. **Buffers are read-only.** `assign_` on a leaf is the only mutation.
4. **There is no global random state.** Every source of randomness takes an
   explicit `Generator`.
5. **Parameter registration is explicit.** No `__setattr__` interception.
6. **`mypy --strict src` passes** and every public symbol has a docstring that
   explains *why*, not what.
7. **No number in the documentation was not measured.**
