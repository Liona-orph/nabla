# 3. Read-only tensor buffers

Date: 2026-01-15

## Status

Accepted.

## Context

Reverse mode requires that the values a vjp closure captured are still the
values the forward pass computed with. If `mul` saves `a` and `b` and something
writes into `a` before `backward()` runs, the gradient that comes out is the
exact gradient of a function nobody evaluated.

The failure is silent. There is no exception, no warning, and no shape mismatch.
The loss keeps going down, slightly wrong, forever. PyTorch handles this with
version counters: every tensor carries a version number, in-place operations
bump it, and the backward pass raises if a saved tensor's version changed. That
works, and it costs a counter on every tensor, a check on every save, and a
class of confusing errors that arrive one step after the mutation that caused
them.

The alternative is to make the mutation impossible.

## Decision

`Tensor.data` is a NumPy array with `WRITEABLE` cleared. Every constructor
copies its input; every operation result is a freshly allocated array frozen by
`Tensor._owning`. In-place operators (`+=`, `*=`, `x[i] = v`, ...) raise
`InPlaceOperationError` with a message naming the alternative.

One mutation is sanctioned: `Tensor.assign_`, which an optimiser needs. It
*rebinds* the tensor's buffer to a new array rather than writing into the old
one, so any closure holding the old values still sees them. It is legal only on
a leaf.

## Consequences

**The bug is impossible rather than detected.** `x.data[0] = 1.0` raises
`ValueError: assignment destination is read-only` from NumPy, at the assignment,
with a traceback pointing at the line responsible. No version counters, no
deferred errors, no rule to learn.

**`detach()` and `numpy(copy=False)` are free.** They hand out the tensor's own
buffer, which is safe precisely because nothing can write through it. In a
framework with mutable buffers both would have to copy or would have to document
an aliasing hazard.

**One copy per constructed tensor.** `nabla.Tensor(some_array)` copies. For a
100 MB feature matrix that is 100 MB and a memcpy. `from_numpy` does the same
thing for the same reason - freezing a caller's array in place would be a worse
surprise than copying it.

**One allocation per optimiser step per parameter.** `assign_` allocates a new
buffer rather than writing into the existing one. For a 420,926-parameter model
that is 1.7 MB of float32 per step, immediately garbage. This is the price, it
is measurable, and it is paid deliberately: writing in place is the bug this
record exists to prevent.

**Gradient buffers are exempt.** `clip_grad_norm_` and `clip_grad_value_` write
to `.grad` in place, and their trailing underscores say so. That is not an
exception to the rule but a consequence of it: no vjp closure ever captures a
gradient buffer, so nothing downstream can be invalidated by writing to one.

**No in-place activation memory saving.** Frameworks offer `relu(x,
inplace=True)` to avoid one allocation per activation, which on a large model is
a meaningful fraction of peak memory. Nabla cannot, and would not want to: the
in-place version is only safe when the input is provably dead, and "provably" is
doing a lot of work in that sentence.
