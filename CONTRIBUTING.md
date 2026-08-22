# Contributing to Nabla

Thank you for considering it. This document says what a change needs in order to
be merged, and why, so that a review is about the substance rather than about
the checklist.

## The short version

```bash
git clone https://github.com/Liona-orph/nabla
cd nabla
python -m venv .venv && source .venv/bin/activate
make install
make ci
```

`make ci` runs exactly what the CI workflow runs, in the same order. If it is
green locally it will be green on the pull request.

## What a change needs

### 1. It must be green

`ruff check .`, `ruff format --check .`, `mypy --strict src`, `pytest -q`. All
four, on Python 3.10, 3.11 and 3.12. `make ci` runs all of them.

Note that `pytest` collects `src/nabla` as well as `tests/`, with
`--doctest-modules`. Every runnable example in a docstring is executed on every
run, so a stale docstring is a build failure. That is deliberate: documentation
that is not tested is documentation that is wrong.

`filterwarnings = ["error"]` is set. A NumPy overflow, underflow-to-invalid or
divide-by-zero warning inside a test is a finding, not noise. A test that
deliberately exercises a domain boundary opts out locally with `np.errstate`.

### 2. Every gradient must be checked numerically

This is not negotiable and it is the whole reason this repository is worth
reading. A new operation needs a test that calls `nabla.check_grad` (via the
`check` helper in `tests/conftest.py`); a new layer needs both `check` on its
input and `check_module` on its parameters.

Pass the function to `check` **unreduced**. `check_grad` contracts the output
with a fixed random vector of its own, and reducing to a scalar first throws
away the asymmetry that catches a transposed or mis-scattered gradient.
`tests/test_grad.py` has a test with a deliberately broken vjp that a pre-summed
check declares correct.

`ARCHITECTURE.md` has a worked example of adding an operation, end to end.

### 3. Forward results must be checked against an independent expression

Assert against a NumPy expression written from the *definition*, never against
the implementation's own intermediate values. Agreement between an
implementation and itself is not evidence.

### 4. Docstrings must explain why

Every public symbol has one, and `ruff`'s pydocstyle rules enforce that it
exists. What they cannot enforce is the part that matters: a Nabla docstring
says why the code is the way it is, what the alternative was, and what the
choice cost. "Computes the softmax of the input" restates the name and is worse
than nothing, because it occupies the space where the reason should be.

Concretely, a good docstring in this repository usually contains at least one
of:

- the failure mode being avoided ("the naive form overflows above 88 in
  float32"),
- the trade-off being made ("six tape nodes instead of one, in exchange for a
  gradient composed of vjps `check_grad` has already verified"),
- the cost, as a quantity ("one allocation per call", "two `exp` calls over the
  whole array instead of one").

No marketing, and no emoji.

### 5. Every number must be measured

If a docstring, a README or a comment states a number - a speed ratio, an
accuracy, a memory figure - it must come from a run you did. Re-run the examples
and paste the real output. A number nobody measured is worse than no number,
because it will be believed.

### 6. Do not weaken a test to make something pass

If a change makes a test fail, either the change is wrong or the test was.
Deciding it was the test requires an argument in the pull request, not a looser
tolerance.

## Invariants

These are recorded in `docs/adr/` and a change that breaks one needs a new ADR
that supersedes the old, not a patch.

1. NumPy is the only runtime dependency.
2. Every vjp is verified by `check_grad`.
3. Tensor buffers are read-only; `assign_` on a leaf is the only mutation.
4. There is no global random state; every source of randomness takes a
   `Generator`.
5. Parameter registration is explicit.
6. `mypy --strict src` passes.

## Adding an architecture decision record

Copy the format of the existing ones - context, decision, consequences, in
Michael Nygard's structure - number it sequentially, and add a row to
`docs/adr/README.md`. Records are immutable once accepted; a decision that turns
out to be wrong gets a new record that supersedes it, and the old one stays,
because the reasoning that led to it is the useful part.

## Style

`ruff format` decides layout; do not argue with it. Line length is 100, which is
set in `pyproject.toml` with a comment explaining why (88 columns fights prose,
and the prose is most of the value here).

Naming follows the domain: `a`, `b`, `x`, `g` for tensors and cotangents in
operation implementations, because that is what the mathematics calls them.
Elsewhere, full words.

British spelling in prose ("normalisation", "initialiser"), because the existing
text uses it and consistency is worth more than either preference. Public API
names follow the field's conventions, which are American where they differ
(`normalised_shape` is a parameter of `LayerNorm`; `nn.LayerNorm` is not
`nn.LayerNormalisation`).

## Reporting a bug

The most useful bug report for this project is a failing `check_grad`. If you
have one, paste the `GradCheckReport` - it names the argument, the index, the
analytic value and the numerical one.

For anything else: what you ran, what happened, what you expected, and the
output of `python -c "import numpy, nabla; print(numpy.__version__,
nabla.__version__)"`.

## Security

See `SECURITY.md`. In short: the checkpoint format is the only part of this
package that reads untrusted input, and it is deliberately incapable of
executing anything.

## Code of conduct

This project follows the Contributor Covenant 2.1; see `CODE_OF_CONDUCT.md`.
