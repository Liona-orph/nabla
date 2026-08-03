"""Shared test helpers.

The whole suite rests on two ideas. Forward correctness is asserted against a
direct NumPy expression written independently of the implementation - never
against the implementation's own intermediate values. Backward correctness is
asserted with `nabla.check_grad`, which knows nothing about how the vjp was
written and can only compare it to the function's own behaviour.

Test data is float64 and drawn from a fixed seed. float64 because the gradient
check needs the resolution (see `check_grad`); fixed seed because a gradient
check that fails one run in fifty is worse than one that never runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

import nabla as nb
from nabla._typing import Array, Shape

SEED = 20240601


@pytest.fixture
def gen() -> nb.random.Generator:
    """A generator seeded identically for every test that asks for one."""
    return nb.random.Generator(seed=SEED)


def sample(*shape: int, seed: int = SEED, scale: float = 1.0, offset: float = 0.0) -> Array:
    """Deterministic float64 test data.

    `offset` moves the data away from zero, which matters for `log`, `sqrt` and
    `div`, where a sample that lands near the singularity turns a correct
    gradient into a failed check.
    """
    values: Array = np.random.default_rng(seed).standard_normal(shape)
    return values * scale + offset


def tensor(*shape: int, seed: int = SEED, scale: float = 1.0, offset: float = 0.0) -> nb.Tensor:
    """A float64 leaf tensor of random values, requiring grad."""
    return nb.Tensor(sample(*shape, seed=seed, scale=scale, offset=offset), requires_grad=True)


def check(
    fn: Callable[..., nb.Tensor],
    *inputs: nb.Tensor | Array,
    name: str | None = None,
    tol: float = 1e-6,
    eps: float = 1e-6,
) -> nb.GradCheckReport:
    """Gradient-check `fn` and return the report.

    Pass `fn` *unreduced* - `lambda x: nb.softmax(x)`, not
    `lambda x: nb.softmax(x).sum()`. `check_grad` contracts the output with a
    fixed random vector of its own, and reducing first throws away the asymmetry
    that catches a transposed or mis-scattered gradient; `test_grad.py` has a
    test that demonstrates exactly that, with a deliberately broken vjp that a
    pre-summed check declares correct.

    `check_grad` raises on failure by itself; the explicit assertion is here so
    a reader of the test can see that the check is the point of it.
    """
    report = nb.check_grad(fn, *inputs, name=name, tol=tol, eps=eps, raise_on_failure=False)
    assert report.ok, str(report)
    return report


def check_module(
    module: nb.nn.Module,
    forward: Callable[[nb.nn.Module], nb.Tensor],
    *,
    name: str | None = None,
    tol: float = 1e-6,
) -> nb.GradCheckReport:
    """Gradient-check a module with respect to every one of its parameters.

    `check_grad` differentiates a function of its arguments, and a module's
    parameters are not arguments - they are captured state. This turns them into
    arguments: it takes the current parameter values, hands them to `check_grad`
    as inputs, and rebinds them into the module before each forward pass.

    Rebinding writes both the registry entry and the attribute of the same name,
    because a layer's `forward` reads the attribute. That is a property of the
    explicit-registration design worth exercising: the two are the same object,
    and a helper that updated only one of them would silently check the gradient
    of a stale weight.

    `forward` is a callable taking the module, so the caller decides what the
    input is - which is what lets an `Embedding`, whose input is an integer
    array with no gradient, be checked the same way as a `Linear`.
    """
    paths = [path for path, _ in module.named_parameters()]
    values = [parameter.numpy() for _, parameter in module.named_parameters()]

    def fn(*tensors: nb.Tensor) -> nb.Tensor:
        for path, tensor in zip(paths, tensors, strict=True):
            _rebind_parameter(module, path, tensor)
        return forward(module)

    return check(fn, *values, name=name or type(module).__name__, tol=tol)


def _rebind_parameter(module: nb.nn.Module, path: str, value: nb.Tensor) -> None:
    """Replace one parameter of `module`, by dotted path, in registry and attribute."""
    parts = path.split(".")
    target = module
    for part in parts[:-1]:
        target = target._modules[part]
    target._parameters[parts[-1]] = value
    setattr(target, parts[-1], value)


def assert_close(actual: nb.Tensor | Array, expected: Array, *, tol: float = 1e-6) -> None:
    """Compare a forward result against a NumPy reference."""
    values = actual.numpy() if isinstance(actual, nb.Tensor) else actual
    np.testing.assert_allclose(values, expected, rtol=tol, atol=tol)


def slow_unbroadcast(g: Array, shape: Shape) -> Array:
    """Reduce a broadcast gradient by walking every element, one at a time.

    Deliberately naive and deliberately independent of `nabla.unbroadcast`: it
    reproduces the definition (each source element receives the sum of every
    broadcast copy of itself) rather than the implementation, so agreement
    between the two is evidence rather than tautology.
    """
    out = np.zeros(shape, dtype=g.dtype)
    offset = g.ndim - len(shape)
    for index in np.ndindex(*g.shape):
        source = tuple(
            0 if shape[axis] == 1 else index[axis + offset] for axis in range(len(shape))
        )
        out[source] += g[index]
    return out


def broadcast_pairs() -> Sequence[tuple[Shape, Shape]]:
    """Every interesting broadcasting pair up to rank 3, enumerated by hand.

    The hypothesis tests cover the space randomly; this list pins the cases that
    are known to break naive implementations - a scalar against everything, a
    trailing 1 against a real axis, ranks differing by two, and a zero-length
    axis - so that a failure names a specific shape instead of a random draw.
    """
    return [
        ((), ()),
        ((), (3,)),
        ((3,), ()),
        ((1,), (3,)),
        ((3,), (1,)),
        ((3,), (3,)),
        ((2, 3), (3,)),
        ((3,), (2, 3)),
        ((2, 1), (1, 3)),
        ((1, 3), (2, 1)),
        ((2, 3), (1, 3)),
        ((2, 3), (2, 1)),
        ((4, 1, 3), (2, 3)),
        ((4, 2, 3), (1, 1, 3)),
        ((1, 1, 1), (2, 3, 4)),
        ((5, 1, 1), (1, 4, 3)),
        ((0,), (1,)),
        ((0, 3), (1, 3)),
        ((2, 0), (2, 1)),
    ]
