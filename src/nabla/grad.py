"""The backward pass, and the numerical check that keeps it honest.

`check_grad` is the most important function in this package. Every operation's
test calls it, and it is the reason the rest of the code can be trusted: a vjp
is a claim about a derivative, and the only cheap way to falsify that claim is
to compare it against the function's own behaviour under a small perturbation.
An analytic gradient that is wrong by a transposition, a missing broadcast
reduction or a factor of `n` still produces a training run that goes down for a
while, which is why "the loss decreased" is not evidence of anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._typing import Array, Shape, TensorLike
from .errors import AutogradError, DTypeError, GradientCheckError, ShapeError
from .tensor import Tensor, as_tensor, no_grad

__all__ = [
    "GradCheckReport",
    "backward",
    "check_grad",
    "clip_grad_norm_",
    "clip_grad_value_",
    "grad",
    "topological_order",
]


# --------------------------------------------------------------------------- #
# the backward pass
# --------------------------------------------------------------------------- #


def topological_order(root: Tensor) -> list[Tensor]:
    """Every tensor `root` depends on, ordered so consumers come before producers.

    Reverse post-order depth-first search. A node may be reached by several
    paths - that is what makes the graph a DAG rather than a tree - and the
    ordering guarantees that when the walk in `backward` reaches a tensor, every
    operation that consumed it has already contributed its share. Accumulating
    in creation order instead would work for straight-line code and quietly
    produce a partial gradient the first time a tensor is reused across a branch.

    The search is iterative. A recursive one is shorter but dies at CPython's
    default recursion limit of 1000, and 1000 operations is a small model, not a
    large one.

    Cost: O(V + E) time, O(V) memory for the explicit stack and the visited set.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        tensor, expanded = stack.pop()
        if expanded:
            order.append(tensor)
            continue
        if id(tensor) in visited:
            continue
        visited.add(id(tensor))
        stack.append((tensor, True))
        node = tensor.node
        if node is not None:
            stack.extend((parent, False) for parent in node.parents if id(parent) not in visited)
    order.reverse()
    return order


def _seed_cotangent(output: Tensor, gradient: TensorLike | None) -> Array:
    """The cotangent the backward pass starts from."""
    if gradient is None:
        if output.size != 1:
            msg = (
                f"backward() on a tensor of shape {output.shape} needs an explicit "
                "gradient: there is no canonical cotangent for a non-scalar output. "
                "Reduce to a scalar loss, or pass gradient=<same shape>."
            )
            raise AutogradError(msg)
        return np.ones_like(output.data)
    seed: Array = np.asarray(as_tensor(gradient).data, dtype=output.dtype)
    if seed.shape != output.shape:
        msg = f"gradient has shape {seed.shape}, but the output has shape {output.shape}"
        raise ShapeError(msg)
    return seed


def backward(output: Tensor, gradient: TensorLike | None = None) -> None:
    """Accumulate gradients from `output` into every leaf it depends on.

    Gradients *accumulate*: `.grad` is added to, not replaced. That is the only
    behaviour that makes a tensor used twice come out right, and it is also what
    makes gradient accumulation across micro-batches work without any extra
    machinery. The consequence is that a training loop that forgets `zero_grad`
    silently sums the gradients of every step it has ever taken, which is the
    canonical beginner bug in every framework that made this choice - and the
    alternative, overwriting, is worse: it turns a tensor used twice into a
    wrong answer with no error at all.

    `.grad` is populated for leaves and for tensors that asked with
    `retain_grad()`. Interior cotangents are held in a working dictionary and
    dropped as soon as they have been consumed, so peak memory is proportional
    to the width of the graph rather than its size.

    The graph is not freed. Calling `backward()` twice traverses the same tape
    again and accumulates twice, which is occasionally what you want and is
    always what the arithmetic says should happen; the cost is that
    intermediates stay alive until the output tensor is dropped.

    >>> import nabla
    >>> x = nabla.tensor([2.0], requires_grad=True)
    >>> y = x * x + x  # x is used three times
    >>> y.backward()
    >>> x.grad.tolist()  # 2x + 1
    [5.0]
    """
    if not output.requires_grad:
        msg = (
            "backward() called on a tensor that does not require grad. Either no "
            "input required a gradient, or the computation ran under no_grad()."
        )
        raise AutogradError(msg)

    pending: dict[int, Array] = {id(output): _seed_cotangent(output, gradient)}
    for tensor in topological_order(output):
        cotangent = pending.pop(id(tensor), None)
        if cotangent is None:
            # Reachable from the root but not on any path that carries a
            # gradient - a branch whose inputs were all detached, for instance.
            continue

        if tensor.retains_grad:
            contribution: Array = np.asarray(cotangent, dtype=tensor.dtype)
            tensor.grad = (
                np.array(contribution, copy=True)
                if tensor.grad is None
                else tensor.grad + contribution
            )

        node = tensor.node
        if node is None:
            continue
        cotangents = node.backward_fn(cotangent)
        if len(cotangents) != len(node.parents):
            msg = (
                f"the vjp of {node.op} returned {len(cotangents)} cotangents for "
                f"{len(node.parents)} inputs"
            )
            raise AutogradError(msg)
        for parent, parent_cotangent in zip(node.parents, cotangents, strict=True):
            if parent_cotangent is None or not parent.requires_grad:
                continue
            if parent_cotangent.shape != parent.shape:
                # Almost always a missing `unbroadcast`. Caught here rather than
                # at the leaf, where the shapes would already have been summed
                # into each other and the op responsible would be unidentifiable.
                msg = (
                    f"the vjp of {node.op} returned a cotangent of shape "
                    f"{parent_cotangent.shape} for an input of shape {parent.shape}"
                )
                raise AutogradError(msg)
            previous = pending.get(id(parent))
            pending[id(parent)] = (
                parent_cotangent if previous is None else previous + parent_cotangent
            )


def grad(
    fn: Callable[..., Tensor],
    argnums: int | Sequence[int] = 0,
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    """Turn a scalar-valued function into its gradient function.

    The functional counterpart to `backward()`: nothing is stored on the inputs,
    so there is no `.grad` to zero and no way for two calls to interfere. The
    arguments being differentiated are copied into fresh tensors and the others
    are detached, which means `grad(f)(x)` never touches the caller's `x.grad` -
    at the cost of one copy per differentiated argument.

    An argument the function ignores gets a zero gradient rather than `None`. A
    function whose return type is "a tensor, or sometimes nothing" is a function
    every caller has to write a branch for.

    >>> import nabla
    >>> f = lambda x, y: (x * y).sum()
    >>> dx, dy = nabla.grad(f, argnums=(0, 1))(nabla.tensor([3.0]), nabla.tensor([5.0]))
    >>> dx.item(), dy.item()
    (5.0, 3.0)
    """
    indices = (argnums,) if isinstance(argnums, int) else tuple(int(i) for i in argnums)

    def gradient_fn(*args: TensorLike) -> Tensor | tuple[Tensor, ...]:
        for index in indices:
            if not 0 <= index < len(args):
                msg = f"argnums {index} is out of range for {len(args)} argument(s)"
                raise AutogradError(msg)
        prepared: list[Tensor] = [
            Tensor(arg, requires_grad=True) if i in indices else as_tensor(arg).detach()
            for i, arg in enumerate(args)
        ]
        output = fn(*prepared)
        if output.size != 1:
            msg = (
                f"grad() needs a scalar-valued function; got shape {output.shape}. "
                "Reverse mode computes one gradient per output, so a vector-valued "
                "function needs one backward pass per component."
            )
            raise AutogradError(msg)
        backward(output)
        results = tuple(_collected_grad(prepared[i]) for i in indices)
        return results[0] if isinstance(argnums, int) else results

    return gradient_fn


def _collected_grad(x: Tensor) -> Tensor:
    """The gradient accumulated into `x`, or an explicit zero if none arrived."""
    accumulated = x.grad
    return Tensor(accumulated if accumulated is not None else np.zeros_like(x.data))


# --------------------------------------------------------------------------- #
# gradient checking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GradCheckReport:
    """The outcome of a `check_grad` call, worst disagreement first.

    Returned even when the check passes, so a test can assert on how much margin
    is left - a gradient that agrees to 1e-9 today and 1e-7 tomorrow is telling
    you something before it starts failing.
    """

    name: str
    ok: bool
    worst_relative_error: float
    argument: int
    index: tuple[int, ...]
    shape: Shape
    analytic: float
    numeric: float
    eps: float
    tol: float
    probes: int

    def __str__(self) -> str:
        verdict = "matches" if self.ok else "MISMATCH"
        return (
            f"check_grad({self.name}): {verdict} within {self.worst_relative_error:.3e} "
            f"(tol {self.tol:.3e}, {self.probes} probe(s), eps {self.eps:.1e}); worst at "
            f"argument {self.argument} of shape {self.shape}, index {self.index}: "
            f"analytic {self.analytic:.12g}, numerical {self.numeric:.12g}"
        )


def check_grad(
    fn: Callable[..., Tensor],
    *inputs: TensorLike,
    name: str | None = None,
    eps: float = 1e-6,
    tol: float = 1e-6,
    max_probes: int | None = None,
    seed: int = 20240601,
    raise_on_failure: bool = True,
) -> GradCheckReport:
    r"""Compare every analytic gradient of `fn` against a central difference.

    ## What it does

    For a random fixed weighting `w` of the outputs, it differentiates
    `s(x) = sum(w * fn(x))` twice: once with the tape, once with

        ds/dx_i ~= (s(x + eps e_i) - s(x - eps e_i)) / (2 eps)

    and reports the largest disagreement.

    The random projection matters, and it is the reason `fn` should be passed
    *unreduced*. Checking `fn(x).sum()` - the obvious choice - weights every
    output equally, and a vjp that scrambles the correspondence between outputs
    and inputs (a transposed matrix gradient, a permutation applied forwards
    instead of backwards) passes it while being wrong everywhere off the
    diagonal. A random `w` breaks that symmetry, and a fixed seed keeps the
    failure reproducible. Reducing to a scalar before calling throws the
    protection away; `fn` may return any shape.

    ## Why float64

    Inputs are promoted to float64 before anything happens. In float32 the
    central difference cannot resolve better than about 1e-3 relative, because
    the subtraction of two nearly equal numbers loses most of the mantissa; at
    that resolution a correct gradient and a subtly wrong one are
    indistinguishable. Promoting means the check validates the *mathematics* of
    the vjp, not the float32 rounding of the forward pass - which is the right
    division of labour, since the rounding is NumPy's problem and the
    mathematics is ours.

    ## Choosing eps and tol

    The error of a central difference is the sum of a truncation term that grows
    like `eps^2` and a rounding term that grows like `1/eps`. In float64 the two
    cross at about `1e-5`, and `eps = 1e-6` sits close enough to the minimum
    while staying comfortably above the point where cancellation dominates. The
    default `tol = 1e-6` then leaves three or four orders of magnitude of margin
    for a correct gradient, while any of the usual mistakes - a factor of two, a
    missing reduction, a transposition - misses by O(1).

    Relative error is measured against `max(|analytic|, |numerical|, 1)`. The
    floor at 1 makes the comparison effectively absolute for small gradients,
    deliberately: the difference quotient has an absolute noise floor of roughly
    `|s| * 2^-52 / eps`, and demanding a *relative* match on a gradient of 1e-12
    would fail on noise rather than on error.

    ## What it cannot do

    It cannot check a point where the function is not differentiable. `relu` at
    0, `max` at a tie and `abs` at 0 all have a one-sided derivative on each
    side and the difference quotient returns their average, which is not what
    any subgradient convention returns. Tests must choose inputs away from the
    kinks; that is a property of the mathematics, not a limitation of this
    implementation.

    Cost: two forward passes per probed element, each rebuilding the input
    tensors. Use `max_probes` to sample a subset for anything large.

    >>> import nabla
    >>> report = nabla.check_grad(lambda x: (x**3).sum(), nabla.tensor([1.0, 2.0]))
    >>> report.ok
    True
    >>> bool(report.worst_relative_error < 1e-8)
    True
    """
    if not inputs:
        msg = "check_grad needs at least one input to differentiate with respect to"
        raise ValueError(msg)
    label = name or getattr(fn, "__name__", None) or "function"

    # float64 copies, so the caller's tensors keep their dtype and their .grad.
    xs: list[Tensor] = []
    for position, value in enumerate(inputs):
        array = np.asarray(as_tensor(value).data)
        if array.dtype.kind not in "fc":
            msg = (
                f"check_grad({label}): argument {position} has dtype {array.dtype}; "
                "only floating-point inputs have gradients to check"
            )
            raise DTypeError(msg)
        xs.append(Tensor(array.astype(np.float64), requires_grad=True))

    base_output = fn(*xs)
    rng = np.random.default_rng(seed)
    weights: Array = rng.standard_normal(base_output.shape)
    projected = (base_output * Tensor(weights)).sum()
    backward(projected)

    def evaluate(values: Sequence[Array]) -> float:
        with no_grad():
            out = fn(*[Tensor(v) for v in values])
        if out.shape != base_output.shape:
            msg = (
                f"check_grad({label}): the function's output shape changed from "
                f"{base_output.shape} to {out.shape} under perturbation"
            )
            raise ShapeError(msg)
        return float(np.sum(out.data * weights))

    worst = GradCheckReport(
        name=label,
        ok=True,
        worst_relative_error=0.0,
        argument=0,
        index=(),
        shape=xs[0].shape,
        analytic=0.0,
        numeric=0.0,
        eps=eps,
        tol=tol,
        probes=0,
    )
    buffers: list[Array] = [np.array(x.data, copy=True) for x in xs]
    probe_count = 0

    for position, x in enumerate(xs):
        analytic_all = x.grad if x.grad is not None else np.zeros(x.shape)
        buffer = buffers[position]
        for index in _probe_indices(x.shape, max_probes, rng):
            original = float(buffer[index])
            buffer[index] = original + eps
            plus = evaluate(buffers)
            buffer[index] = original - eps
            minus = evaluate(buffers)
            buffer[index] = original
            probe_count += 1

            analytic = float(analytic_all[index])
            numeric = (plus - minus) / (2.0 * eps)
            error = math.fabs(analytic - numeric) / max(
                math.fabs(analytic), math.fabs(numeric), 1.0
            )
            if error > worst.worst_relative_error:
                worst = GradCheckReport(
                    name=label,
                    ok=error <= tol,
                    worst_relative_error=error,
                    argument=position,
                    index=index,
                    shape=x.shape,
                    analytic=analytic,
                    numeric=numeric,
                    eps=eps,
                    tol=tol,
                    probes=probe_count,
                )

    report = GradCheckReport(
        name=worst.name,
        ok=worst.worst_relative_error <= tol,
        worst_relative_error=worst.worst_relative_error,
        argument=worst.argument,
        index=worst.index,
        shape=worst.shape,
        analytic=worst.analytic,
        numeric=worst.numeric,
        eps=eps,
        tol=tol,
        probes=probe_count,
    )
    if raise_on_failure and not report.ok:
        raise GradientCheckError(str(report))
    return report


def _probe_indices(
    shape: Shape,
    max_probes: int | None,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Which elements to perturb: all of them, or a random sample.

    Sampling uses the same generator as the output projection, so a check is
    fully determined by its seed and a failure can be reproduced exactly.
    """
    everything: list[tuple[int, ...]] = [tuple(int(i) for i in idx) for idx in np.ndindex(*shape)]
    if max_probes is None or len(everything) <= max_probes:
        return everything
    chosen = rng.choice(len(everything), size=max_probes, replace=False)
    return [everything[int(i)] for i in chosen]


# --------------------------------------------------------------------------- #
# clipping
# --------------------------------------------------------------------------- #


def clip_grad_norm_(
    tensors: Iterable[Tensor],
    max_norm: float,
    *,
    norm_type: float = 2.0,
) -> float:
    """Rescale gradients so their combined norm is at most `max_norm`.

    The norm is taken over all the tensors at once, not one at a time: the
    direction of the update is what clipping is meant to preserve, and
    normalising each parameter separately would change it.

    Returns the norm *before* clipping, which is the number worth logging - it
    says whether clipping is doing anything, and a norm that spikes by three
    orders of magnitude is the earliest visible sign of a diverging run.

    Unlike some implementations, the scale factor here has no epsilon added to
    its denominator. An epsilon makes the clipped norm come out slightly below
    `max_norm`, which is invisible in practice and wrong on paper; the zero-norm
    case that the epsilon exists to guard is handled by not scaling at all.

    Trailing underscore: this modifies `.grad` in place. That is not a violation
    of the package's in-place policy, which is about tensor *values* - no vjp
    closure captures a gradient buffer, so nothing downstream can be
    invalidated by writing to one.
    """
    if max_norm <= 0:
        msg = f"max_norm must be positive, got {max_norm}"
        raise ValueError(msg)
    with_grads = [t for t in tensors if t.grad is not None]
    if not with_grads:
        return 0.0

    if math.isinf(norm_type):
        total = max(float(np.abs(t.grad).max()) for t in with_grads if t.grad is not None)
    else:
        accumulated = 0.0
        for t in with_grads:
            if t.grad is not None:
                accumulated += float(np.sum(np.abs(t.grad) ** norm_type))
        total = accumulated ** (1.0 / norm_type)

    if total > max_norm and total > 0.0:
        scale = max_norm / total
        for t in with_grads:
            if t.grad is not None:
                t.grad = t.grad * scale
    return total


def clip_grad_value_(tensors: Iterable[Tensor], clip_value: float) -> None:
    """Clamp every gradient element to `[-clip_value, clip_value]`.

    Cruder than norm clipping - it changes the direction of the update, not just
    its length - but it is the only one of the two that bounds a single
    exploding coordinate without letting it dominate the rescaling of everything
    else.
    """
    if clip_value <= 0:
        msg = f"clip_value must be positive, got {clip_value}"
        raise ValueError(msg)
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad = np.clip(tensor.grad, -clip_value, clip_value)
