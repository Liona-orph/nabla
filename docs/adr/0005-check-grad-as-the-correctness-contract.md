# 5. `check_grad` as the correctness contract

Date: 2026-01-16

## Status

Accepted.

## Context

A vector-Jacobian product is a *claim* about a derivative. Nothing about the
code enforces the claim: a vjp that transposes the wrong matrix, forgets a
broadcast reduction, or is off by a factor of `n` compiles, runs, produces
correctly shaped arrays, and drives a training run whose loss goes down for a
while. "The loss decreased" is not evidence of anything; a wrong gradient is
still usually a descent direction for the first few thousand steps.

Three ways to establish that a vjp is right:

1. **Review.** Someone checks the algebra. This is how most of these bugs are
   introduced in the first place.
2. **Comparison against another framework.** Accurate, and it makes the other
   framework a dependency of the test suite, which contradicts ADR 0006. It also
   only tests the operations the other framework happens to have.
3. **Comparison against the function's own behaviour under a perturbation.** A
   central difference. It needs nothing but the forward pass, which is the part
   that is independently checkable against a NumPy expression.

## Decision

`nabla.check_grad` is the correctness contract. Every operation has a test that
calls it, every layer has a test that calls it, and no gradient enters this
repository without one.

Four details of the implementation are load-bearing.

**A random output projection.** `check_grad` differentiates `sum(w * fn(x))`
for a fixed random `w`, not `sum(fn(x))`. Equal weights are a symmetry, and a
vjp that scrambles the correspondence between outputs and inputs - a transposed
matrix gradient, a permutation applied forwards instead of backwards - passes an
equally-weighted check while being wrong everywhere off the diagonal. The seed is
fixed so failures reproduce. `tests/test_grad.py` contains a deliberately broken
vjp that a pre-summed check declares correct and the projected check catches.

**float64.** Inputs are promoted before anything happens. In float32 a central
difference cannot resolve better than about 1e-3 relative, because subtracting
two nearly equal numbers loses most of the mantissa, and at that resolution a
correct gradient and a subtly wrong one are indistinguishable. Promoting means
the check validates the *mathematics* of the vjp rather than the float32
rounding of the forward pass - which is the right division of labour, since the
rounding is NumPy's problem and the mathematics is ours.

**eps = 1e-6, tol = 1e-6.** The error of a central difference is a truncation
term growing like `eps^2` plus a rounding term growing like `1/eps`. In float64
those cross around `1e-5`; `1e-6` sits near the minimum while staying above the
point where cancellation dominates. A correct gradient then has three or four
orders of magnitude of margin, while a factor of two, a missing reduction or a
transposition misses by O(1).

**A floor of 1 in the relative-error denominator.** Error is measured against
`max(|analytic|, |numerical|, 1)`, which makes the comparison effectively
absolute for small gradients. The difference quotient has an absolute noise
floor of roughly `|s| * 2^-52 / eps`, and demanding a *relative* match on a
gradient of 1e-12 fails on noise rather than on error.

## Consequences

**Correctness is established by something that does not read the code.**
`check_grad` knows nothing about how a vjp was written and cannot be fooled by
the same misunderstanding that produced it.

**Tests are slow in the way that does not matter.** Two forward passes per
probed element, each rebuilding the input tensors. The whole suite - 655 tests
including every operation's and every layer's gradients - runs in about two
seconds, because the test tensors are small on purpose. Anything large uses
`max_probes` to sample.

**Non-differentiable points must be avoided by the caller.** `relu` at 0, `max`
at a tie and `abs` at 0 have a different one-sided derivative on each side, and
the difference quotient returns their average, which is not what any subgradient
convention returns. Tests choose inputs away from the kinks, and this is a
property of the mathematics rather than a limitation of the implementation.
`tests/test_ops_activation.py` says so where it moves its sample away from zero.

**`GradientCheckError` derives from `AssertionError`**, so pytest reports it as a
failed assertion rather than an unexpected error. When the whole test suite is
gradient checks, that distinction is the difference between a readable failure
report and a wall of tracebacks.

**A report is returned even on success.** A test can assert on how much margin is
left, and a gradient that agrees to 1e-9 today and 1e-7 tomorrow is telling you
something before it starts failing.
