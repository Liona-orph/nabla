# Architecture decision records

Six decisions that shape everything else in this repository, in
[Michael Nygard's format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):
context, decision, consequences. Each one records what was actually decided and
what it cost, not what would sound best.

A record is immutable once accepted. A decision that turns out to be wrong gets
a new record that supersedes the old one, and the old one stays, because the
reasoning that led to it is the useful part.

| ADR | Title | Status |
| --- | ----- | ------ |
| [0001](0001-reverse-mode-over-forward-mode.md) | Reverse mode over forward mode | Accepted |
| [0002](0002-a-tape-over-a-static-graph.md) | A tape over a static graph | Accepted |
| [0003](0003-read-only-buffers.md) | Read-only tensor buffers | Accepted |
| [0004](0004-explicit-parameter-registration.md) | Explicit parameter registration | Accepted |
| [0005](0005-check-grad-as-the-correctness-contract.md) | `check_grad` as the correctness contract | Accepted |
| [0006](0006-numpy-as-the-only-dependency.md) | NumPy as the only runtime dependency | Accepted |
