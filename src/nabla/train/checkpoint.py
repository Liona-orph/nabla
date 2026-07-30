"""Checkpoints: one `.npz` file, arrays plus a JSON manifest.

# Why not pickle

`np.save` with `allow_pickle=True` and `pickle` itself both serialise arbitrary
Python objects, which means loading a checkpoint executes whatever the file says
to execute. A model checkpoint is exactly the kind of file that gets downloaded
from a stranger, so this package will not write one and will not read one.

# The format

A checkpoint is a `.npz` archive. Every array in the payload becomes one entry,
keyed by its path through the nested structure (`model/layers/0/weight`).
Everything that is not an array - integers, floats, strings, booleans, the shape
of the nesting itself - goes into a single JSON document stored as a
zero-dimensional string array under `__manifest__`, with each array leaf
replaced by a reference to its key.

The result is inspectable with `unzip -l`, loadable by any NumPy, and incapable
of executing anything. The cost is that a payload may only contain mappings,
sequences, arrays and JSON scalars; a custom object has to be reduced to those
by whoever is saving it, which is a restriction and is meant to be.

# What round-trips

`load_checkpoint(save_checkpoint(payload))` returns an equal payload, with two
documented exceptions that fall out of JSON: a tuple comes back as a list, and
a mapping's keys come back as strings. `tests/test_train.py` asserts the round
trip for a real model-and-optimiser payload.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._typing import Array

__all__ = ["load_checkpoint", "save_checkpoint"]

#: The manifest key. Chosen to be illegal as a path component so it cannot
#: collide with a payload key that happens to be called "manifest".
_MANIFEST = "__manifest__"

#: The marker a manifest uses in place of an array.
_ARRAY_REF = "__array__"


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write `payload` to `path` as an `.npz` archive. Returns the path written.

    The file is written to a temporary name in the same directory and then
    renamed. A checkpoint written in place is a checkpoint that is corrupt if
    the process dies during the write, and the moment a process is most likely
    to die is when it has just run out of memory allocating the buffers it was
    about to save.
    """
    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Array] = {}
    manifest = _describe(payload, "", arrays)
    entries: dict[str, Array] = dict(arrays)
    entries[_MANIFEST] = np.array(json.dumps(manifest))

    temporary = target.with_name(target.name + ".partial")
    with temporary.open("wb") as handle:
        # NumPy's stub for `savez` declares its second positional parameter as
        # `allow_pickle: bool`, so a `**dict[str, ndarray]` splat is a type error
        # against the stub and a correct call against the implementation. The
        # keys here contain "/" and so could not be written as keyword arguments
        # in source anyway.
        np.savez(handle, **entries)  # type: ignore[arg-type]
    temporary.replace(target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint written by `save_checkpoint`.

    Loaded with `allow_pickle=False`, which is NumPy's default and is restated
    here because it is the whole security argument for this format.
    """
    source = Path(path)
    if source.suffix != ".npz":
        source = source.with_suffix(".npz")
    with np.load(source, allow_pickle=False) as archive:
        if _MANIFEST not in archive.files:
            msg = f"{source} has no {_MANIFEST}; it was not written by nabla.train.save_checkpoint"
            raise ValueError(msg)
        manifest = json.loads(str(archive[_MANIFEST]))
        arrays = {name: archive[name] for name in archive.files if name != _MANIFEST}
    restored = _rebuild(manifest, arrays)
    if not isinstance(restored, dict):
        msg = f"{source} does not contain a mapping at the top level"
        raise TypeError(msg)
    return restored


def _describe(value: Any, prefix: str, arrays: dict[str, Array]) -> Any:
    """Split `value` into a JSON-able skeleton and a flat dictionary of arrays."""
    if isinstance(value, np.ndarray):
        key = prefix or "__root__"
        arrays[key] = value
        return {_ARRAY_REF: key}
    if isinstance(value, Mapping):
        return {
            str(name): _describe(item, f"{prefix}/{name}" if prefix else str(name), arrays)
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _describe(item, f"{prefix}/{index}" if prefix else str(index), arrays)
            for index, item in enumerate(value)
        ]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, np.generic):
        # A NumPy scalar is not a Python one and is not JSON-able; it is also
        # what `float(array.mean())` returns if somebody forgets the `float`.
        return value.item()
    msg = (
        f"cannot checkpoint a value of type {type(value).__name__} at {prefix or '<root>'!r}: "
        "a payload may contain mappings, sequences, arrays, and JSON scalars"
    )
    raise TypeError(msg)


def _rebuild(node: Any, arrays: Mapping[str, Array]) -> Any:
    """Put the arrays back where the manifest says they came from."""
    if isinstance(node, Mapping):
        if set(node) == {_ARRAY_REF}:
            return arrays[str(node[_ARRAY_REF])]
        return {name: _rebuild(item, arrays) for name, item in node.items()}
    if isinstance(node, Sequence) and not isinstance(node, str):
        return [_rebuild(item, arrays) for item in node]
    return node
