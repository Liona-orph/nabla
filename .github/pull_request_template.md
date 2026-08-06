## What this changes

<!-- One or two sentences. The diff says what; this says why. -->

## Why

<!-- The problem, and why this is the right shape of fix. If it supersedes a
decision in docs/adr/, say which one. -->

## What it cost

<!-- Every change costs something: memory, an allocation per step, a tape node,
a dependency, an invariant. Name it. "Nothing" is a valid answer and is
sometimes even true. -->

## Checklist

- [ ] `make ci` is green locally (`ruff check`, `ruff format --check`,
      `mypy --strict src`, `pytest -q`).
- [ ] New gradients are verified with `check_grad` - `check` for an operation or
      a layer's input, `check_module` for a layer's parameters - and the
      function is passed **unreduced**.
- [ ] New forward results are asserted against an independent NumPy expression
      written from the definition, not against the implementation's own
      intermediates.
- [ ] New public symbols have docstrings that explain *why*, name the
      alternative, and state the cost.
- [ ] Every number added to a docstring, README or comment came from a run I
      did, and the run is reproducible.
- [ ] No existing test was weakened. (If one was, the argument is above.)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`, if the change is visible to a
      user.

## Measurements

<!-- If this changes a documented number - an example's accuracy, a benchmark
ratio, a runtime - paste the new output here and update the README and the
example's README to match. -->
