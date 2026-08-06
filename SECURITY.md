# Security policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

Nabla is pre-1.0. Only the latest release receives fixes.

## Reporting a vulnerability

Please report privately, through GitHub's security advisory form on this
repository ("Security" -> "Report a vulnerability"), rather than by opening a
public issue.

Include what you ran, what happened, and the versions of Python and NumPy. If
you have a proof of concept, a minimal one is more useful than a complete one.

Expect an acknowledgement within seven days and an assessment within thirty. If
the report is valid, the fix and the advisory are published together.

## What the threat model actually is

Nabla is a library, not a service. It has no network code, no subprocess
execution, no file parsing beyond the checkpoint format below, and one runtime
dependency. Most of the categories a security policy usually covers do not
apply. Two things genuinely do.

### Checkpoints

`nabla.train.save_checkpoint` and `load_checkpoint` are the only part of this
package that reads a file a user did not write. The format is deliberately
constrained:

- A checkpoint is a plain `.npz` archive: NumPy arrays plus a single JSON
  manifest describing the structure.
- It is read with `allow_pickle=False`. Nothing in it can execute.
- The payload may only contain mappings, sequences, arrays and JSON scalars.
  `save_checkpoint` refuses anything else with a `TypeError` rather than
  reaching for `pickle`.

This is the reason the format exists at all rather than using `np.save` with
pickling: a model checkpoint is exactly the kind of file that gets downloaded
from a stranger, and `pickle.load` on a hostile file is arbitrary code
execution. `tests/test_train.py::test_checkpoints_are_loaded_without_pickle`
pins the property.

A malicious checkpoint can still exhaust memory by declaring enormous arrays,
and can still contain weights that make a model behave badly. Neither is
something a file format can prevent.

### Resource exhaustion

Shapes come from user data, and a `matmul` of two large tensors allocates a
large result. Nabla does not impose limits. A service that runs untrusted model
code should run it in a process with its own memory limit.

## What is not a vulnerability

- A gradient that disagrees with another framework's. That is a correctness bug
  and belongs in an issue with a `check_grad` report attached.
- Numerical overflow producing `inf` or `nan`. The operations that can are
  documented as such, and clamping them silently would be worse.
- `Tensor(huge_shape)` raising `MemoryError`.
