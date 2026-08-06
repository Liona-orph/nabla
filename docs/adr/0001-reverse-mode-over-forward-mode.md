# 1. Reverse mode over forward mode

Date: 2026-01-14

## Status

Accepted.

## Context

A derivative can be propagated in either direction through a program.

**Forward mode** carries one input's perturbation through the computation and
reads off its effect on every output. One pass gives one column of the Jacobian,
so the cost is one pass per *input*. It needs no extra memory: the tangent
travels alongside the value and nothing has to be stored.

**Reverse mode** carries one output's sensitivity backwards and reads off the
effect of every input. One pass gives one row of the Jacobian, so the cost is
one pass per *output*. It needs the intermediate values of the forward pass to
still be available when the backward pass runs, because each operation's
vector-Jacobian product is a function of the values it saw.

Training a neural network is the extreme case of the second shape: millions of
parameters, one scalar loss. A Jacobian of shape `(1, n)`.

There is a third option: implement both. JAX does, and the two modes compose
into `jvp` of `vjp` for Hessian products.

## Decision

Implement reverse mode only.

## Consequences

**What it buys.** One backward pass computes the gradient of a scalar loss with
respect to every parameter, at a cost of roughly twice the forward pass. Forward
mode would need one pass per parameter; for the 420,926-parameter transformer in
`examples/char_transformer/`, that is 420,926 forward passes per training step
instead of one backward pass. The comparison is not close and no amount of
engineering changes it.

**What it costs, first: memory.** Every intermediate value that some backward
function needs must survive until the backward pass runs. This is why a training
step's peak memory is dominated by activations rather than by weights, and it is
a property of the algorithm, not of this implementation. Nabla pays it
explicitly: each operation stores exactly the arrays its vjp closure captured,
and those arrays stay alive as long as the output tensor does. `exp` and
`sigmoid` capture their outputs instead of their inputs where the derivative
allows it; `mul` has to capture both operands.

**What it costs, second: no higher-order derivatives.** Each vjp is written in
NumPy, not in Nabla operations, so the backward pass is not itself on the tape
and is not differentiable. There is no `grad(grad(f))`, no Hessian-vector
product, and no meta-learning. Making the vjps differentiable would mean writing
every one of them in terms of Nabla ops, which roughly doubles the code and
makes each backward pass allocate tape nodes of its own - a real cost paid on
every training step for a capability that this package's intended use does not
need. `nabla.grad.grad` exists as a functional interface but is a wrapper around
the same single-level machinery.

**What it costs, third: no efficient Jacobians of wide-output functions.** A
function with `m` outputs needs `m` backward passes to get its full Jacobian.
`nabla.grad.grad` therefore requires a scalar-valued function and says so in its
error message.

**When to revisit.** If someone needs second-order optimisation or
forward-over-reverse Hessian products. That is a rewrite of every vjp, not an
addition, so it would supersede this record rather than amend it.
