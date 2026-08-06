# 2. A tape over a static graph

Date: 2026-01-14

## Status

Accepted.

## Context

Reverse mode needs to know what the forward pass did. There are two ways to find
out.

**A static graph.** The user describes the computation as data - a graph of
operations with placeholders for inputs - and the framework compiles it before
running it. TensorFlow 1.x worked this way, and so does every framework with a
serious compiler behind it. The graph can be optimised, fused, serialised,
shipped to another process and run without Python.

**A tape.** The framework records operations as they execute. What is recorded
is what actually ran, including whatever the Python control flow decided. This
is what PyTorch, Autograd and JAX's `grad` (over traced functions) do.

There is a sub-decision inside the tape option: whether the tape is a global
list that operations append to, or whether each output tensor holds its own
node.

## Decision

Record a tape, and make the tape *be* the tensors: each non-leaf tensor holds a
`Node` with the operation's name, its input tensors, and a closure computing the
vector-Jacobian product. There is no tape object and no global list.

## Consequences

**Python control flow works.** An `if` on a data-dependent condition, a loop
whose length varies per batch, a recursive model - all of them are differentiated
correctly, because what is recorded is what ran. A static graph needs
`tf.cond` and `tf.while_loop` for the same thing, and every one of those is a
place where the user has to learn a second language.

**Debugging is ordinary debugging.** A shape error is raised at the line that
caused it, with a Python traceback, mid-forward-pass. There is no separate
compile step whose error messages describe a graph the user never wrote.

**Memory is managed by reference counting.** Because the graph is the tensors,
dropping the loss drops the whole graph. There is no `reset()` to forget, no
session to close, and two models training in the same process cannot see each
other's tape. This is the main reason for making the tape be the tensors rather
than a module-level list: a global list is global mutable state, and global
mutable state in a framework is where thread-safety bugs and
forgot-to-clear-the-tape bugs live.

**A retained tensor retains its history.** The flip side. Appending `loss` to a
list for logging keeps every activation of that step alive; appending
`loss.item()` keeps a float. This is the most common memory leak in code built
on any framework with this design, and no amount of framework cleverness
prevents it.

**No graph-level optimisation.** Nothing can fuse `x * w + b` into one kernel,
because by the time anything could look at it the multiply has already happened.
Every operation allocates its output. `benchmarks/overhead.py` measures what the
per-operation bookkeeping costs against hand-written NumPy: 2.22x at batch 32
with 128 hidden units, 1.19x at batch 512 with 1,024.

**No serialisation of the computation.** A Nabla model is Python code. It cannot
be exported to ONNX, run without Python, or shipped to a device. `state_dict()`
serialises the *weights*; the architecture is the source file.

**The order still has to be computed.** Recording does not by itself give a
correct backward order. `topological_order` does a reverse post-order DFS rather
than trusting creation order, because creation order happens to be a valid
topological order for straight-line code and is wrong the first time a tensor
computed earlier is reused across a branch. That is a real bug that a
creation-order tape would have, and it would only appear in models with skip
connections - which is to say, all of them.
