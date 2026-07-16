"""Exception hierarchy.

Every exception raised deliberately by Nabla derives from `NablaError`, so a
caller can catch the whole package with one clause. Each one *also* derives from
the builtin exception a reader would have guessed (`ValueError`, `TypeError`,
`RuntimeError`), because code that wraps Nabla usually predates it and already
has `except ValueError` in the right places. The double inheritance costs
nothing at runtime and removes the need for callers to learn a new hierarchy
before they can handle a shape mismatch.
"""

from __future__ import annotations


class NablaError(Exception):
    """Base class for every error this package raises on purpose."""


class ShapeError(NablaError, ValueError):
    """A shape is inconsistent.

    Operands that will not broadcast, a reshape that changes the number of
    elements, an axis outside the rank of a tensor.
    """


class DTypeError(NablaError, TypeError):
    """A dtype is not usable for the requested operation.

    Raised when `requires_grad` is asked for on an integer or boolean tensor, and
    when a tensor is built from data NumPy cannot represent as a numeric array.
    """


class AutogradError(NablaError, RuntimeError):
    """The tape cannot answer the question that was asked.

    Covers `backward()` on a tensor that carries no gradient, a non-scalar output
    with no explicit seed cotangent, and internal invariant violations such as a
    vjp returning a cotangent whose shape does not match its input.
    """


class InPlaceOperationError(NablaError, RuntimeError):
    """An attempt was made to mutate a tensor's buffer.

    Tensor buffers are immutable because the tape holds references to them: an
    operation that saved `x` for its backward pass would silently compute a
    gradient of a function nobody evaluated if `x` changed afterwards. The error
    names the sanctioned alternative, `Tensor.assign_`.
    """


class ModuleError(NablaError, RuntimeError):
    """A module was built, traversed or loaded in a way that cannot be honoured.

    Registering a name twice, registering a computed tensor as a parameter,
    skipping `super().__init__()`, or loading a state dict whose keys do not
    match the model. All of them are construction-time or load-time mistakes,
    which is why they are one class: none of them can happen mid-training.
    """


class OptimizerError(NablaError, RuntimeError):
    """An optimiser was given something it cannot step.

    A parameter list that is empty or contains duplicates, a hyperparameter
    outside its valid range, or a state dict describing a different number of
    parameters than the optimiser holds.
    """


class GradientCheckError(NablaError, AssertionError):
    """A numerical gradient check failed.

    Derives from `AssertionError` so that pytest reports it as a failed
    assertion rather than an unexpected error; the distinction matters when the
    whole test suite is gradient checks.
    """
