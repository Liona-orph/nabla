"""Type aliases shared across the package.

These exist so that a signature reads as domain vocabulary (`Shape`, `Axis`,
`TensorLike`) instead of a nest of unions repeated in forty places. They are
private: the public API mentions them only in annotations, and nothing here is
re-exported from `nabla`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - import exists only for the annotations
    from nabla.tensor import Tensor

#: Any numeric ndarray. Parameterising on the scalar type would be more precise,
#: but a tensor's dtype is a runtime property here (float32 by default, float64
#: on request), so the static type cannot say anything useful about it.
Array: TypeAlias = npt.NDArray[Any]

#: A tensor shape. `()` is a 0-d tensor, not an error.
Shape: TypeAlias = tuple[int, ...]

#: Reduction axes. `None` means "every axis", matching NumPy.
Axis: TypeAlias = int | tuple[int, ...] | None

#: Anything `np.dtype()` accepts.
DTypeLike: TypeAlias = Any

#: Anything that can be turned into a tensor. The forward reference must use
#: `Union[...]` rather than `|`: the right-hand side of a type alias is
#: evaluated at runtime, and `"Tensor" | int` is a TypeError.
TensorLike: TypeAlias = Union["Tensor", Array, float, int, bool, Sequence[Any]]

#: A NumPy index expression: an int, a slice, an ellipsis, `None`, an index or
#: mask array, or a tuple of those. NumPy's own indexing type is `Any` in
#: practice, and reproducing the real grammar here would not make any caller's
#: code safer.
IndexLike: TypeAlias = Any

#: The vector-Jacobian product of a recorded operation.
#:
#: Takes the cotangent of the operation's output and returns one cotangent per
#: input, in the order the inputs were recorded. `None` means "this input does
#: not need a gradient and computing one was skipped"; see `nabla.ops.basic.pow`
#: for why an op would want to skip rather than compute and discard.
BackwardFn: TypeAlias = Callable[[Array], "tuple[Array | None, ...]"]

#: The dtype every tensor gets unless told otherwise. See `nabla.tensor` for the
#: reasoning behind float32 rather than NumPy's float64.
DEFAULT_FLOAT_DTYPE: np.dtype[Any] = np.dtype(np.float32)
