# 6. NumPy as the only runtime dependency

Date: 2026-01-14

## Status

Accepted.

## Context

The obvious things to reach for while building this were:

- **SciPy**, for `erf` (the exact GELU), `erfinv` (truncated normal by inverse
  CDF), and `logsumexp`.
- **numba** or **Cython**, to make the elementwise operations and the tape
  bookkeeping faster.
- **PyTorch**, as a reference implementation to test gradients against.
- **A plotting library**, so the examples could produce a loss curve as an image.

Each is a small, reasonable addition. Together they are the difference between a
package a reader can audit in an afternoon and one they cannot.

## Decision

`numpy>=1.24` is the only runtime dependency. `pytest`, `hypothesis`, `mypy` and
`ruff` are development extras and are not imported by anything under
`src/nabla`.

## Consequences

**The audit surface is the source tree.** A reader who wants to know what
happens to their gradient reads `src/nabla`, all of it, and is done. That is the
entire premise of this project, and every dependency is a subtree the reader has
to take on trust instead.

**Installation is `pip install numpy`.** No build step, no compiler, no wheels
for six platforms, no version matrix between a JIT compiler and a Python
release. The CI matrix is three Python versions and nothing else.

**GELU is the tanh approximation.** NumPy has no vectorised `erf`, and
`math.erf` is scalar, so the exact `x * Phi(x)` would need a Python-level loop
over every element - two orders of magnitude slower than the rest of the
package, which would make GELU the most expensive operation in any model that
used it. The tanh form stays within about 3e-4 of the exact function everywhere,
which is below float32 resolution at the magnitudes where activations live. It
is the only place in Nabla where a *forward* value is knowingly inexact, and it
is documented at the function.

**Truncated normal is rejection sampling.** The inverse-CDF method needs
`erfinv`. Rejection at a limit of 2 standard deviations accepts 95% of draws, so
the expected number of rounds is small; the loop is bounded at 64 so a
pathological limit is an error rather than a hang.

**Gradients are tested against a central difference, not against another
framework.** This is ADR 0005, and it is partly a consequence of this one. It is
also better: the numerical check tests every operation, including the ones no
other framework has, and it cannot be fooled by a shared misunderstanding.

**Examples print text.** Loss curves are tables, the two-moons decision boundary
is ASCII, and the digit samples are rendered with `" .:-=+*#%@"`. This turns out
to be a feature in a repository whose output is meant to be pasted into a README
and diffed.

**It is slower than it could be.** Measured in `benchmarks/overhead.py` against
hand-written NumPy: 2.22x at batch 32 with 128 hidden units, 1.19x at batch 512
with 1,024. A JIT would close most of that gap and would also mean that the code
a reader audits is not the code that runs.

**When to revisit.** If someone needs to train something large enough that the
constant factor matters, the right answer is a different framework, not a
dependency here.
