"""The module base class: what owns parameters, and how it says so.

# Why registration is explicit

The usual design intercepts attribute assignment. A framework overrides
`__setattr__`, checks whether the value being assigned is a parameter or a
module, and quietly files it in a hidden dictionary. `self.weight = ...` then
has two meanings depending on the runtime type of the right-hand side, and
which one you got is invisible at the call site.

That has three costs, and Nabla declines to pay them:

1. A parameter stored anywhere the interceptor does not look - a plain list, a
   dict, a tuple, a closure - is silently not a parameter. It gets no gradient
   update, `state_dict()` never sees it, and the model trains anyway, slightly
   worse, with nothing to read in a traceback. This is the single most common
   way a hand-written model silently under-trains, and it is caused entirely by
   the magic that was supposed to be convenient.
2. Every attribute assignment on every module - including `self.eps = 1e-5` -
   pays for an isinstance check against two or three types.
3. Renaming an attribute silently renames a checkpoint key. With an explicit
   name argument, the key is a string you can search for.

So a module registers what it owns by calling `register_parameter`, which
*returns* the tensor. The assignment still reads the way it should:

    self.weight = self.register_parameter("weight", init.kaiming_normal(...))

and the answer to "which tensors does this module own" is obtained by reading
its constructor, not by knowing the framework's dispatch rules. The cost is one
extra call and one repeated name per parameter, paid once at construction.

# Why there is no Parameter type

A parameter is a leaf `Tensor` with `requires_grad=True`; nothing else
distinguishes it. Making it a subclass would mean every operation has to decide
whether to preserve the subclass through arithmetic (it cannot - `w * 2` is not
a parameter), and the type would end up meaning "was a parameter once", which
is not a useful thing for a type to mean. Membership of the registry is the
definition, and the registry is a dictionary you can print.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeVar

import numpy as np

from .._typing import Array
from ..errors import ModuleError, ShapeError
from ..tensor import Tensor

__all__ = ["Module", "ModuleDict", "ModuleList", "Sequential"]

#: The type of a registered child, preserved through `register_module`.
ChildT = TypeVar("ChildT", bound="Module")


class Module:
    """Base class for anything that owns parameters.

    Subclasses call `super().__init__()` first, then register what they own.
    Everything else on this class - `parameters()`, `state_dict()`, `train()`,
    the tree `__repr__` - is derived from those three registries and needs no
    cooperation from the subclass.

    >>> import nabla
    >>> from nabla import nn
    >>> layer = nn.Linear(3, 2, generator=nabla.random.Generator(seed=0))
    >>> sorted(name for name, _ in layer.named_parameters())
    ['bias', 'weight']
    >>> layer.parameter_count()
    8
    """

    #: Set by `train()` and `eval()`. Layers whose forward pass differs between
    #: the two - `Dropout` is the only one here - read it. It is a plain public
    #: attribute rather than a property because reading it is the hot path and
    #: because a layer that needs to lie about it (a frozen sub-model kept in
    #: eval mode) should be able to set it directly.
    training: bool

    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}
        self._buffers: dict[str, Tensor] = {}
        self._modules: dict[str, Module] = {}
        self.training = True

    # -- registration ------------------------------------------------------- #

    def _registries(self) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Module]]:
        """The three registries, with a readable error if `__init__` was skipped.

        Forgetting `super().__init__()` is the one mistake this design makes
        easy, so it gets a message that names the fix instead of an
        `AttributeError` on a private name.
        """
        try:
            return self._parameters, self._buffers, self._modules
        except AttributeError as exc:
            msg = (
                f"{type(self).__name__} has no registries: its __init__ did not call "
                "super().__init__() before registering anything"
            )
            raise ModuleError(msg) from exc

    def _claim(self, name: str) -> None:
        """Reserve `name` across all three registries, refusing a collision."""
        if not name or "." in name:
            msg = f"{name!r} is not a usable registry name: it must be non-empty and contain no '.'"
            raise ModuleError(msg)
        parameters, buffers, modules = self._registries()
        for kind, registry in (("parameter", parameters), ("buffer", buffers)):
            if name in registry:
                msg = f"{type(self).__name__} already has a {kind} named {name!r}"
                raise ModuleError(msg)
        if name in modules:
            msg = f"{type(self).__name__} already has a submodule named {name!r}"
            raise ModuleError(msg)

    def register_parameter(self, name: str, value: Tensor) -> Tensor:
        """Record `value` as a trainable parameter and return it.

        Returning the tensor is what keeps the call site readable: the caller
        writes `self.weight = self.register_parameter("weight", ...)`, so the
        attribute and the registry entry are the same object and cannot drift
        apart. They stay the same object for the whole run because an optimiser
        updates a parameter with `Tensor.assign_`, which rebinds the buffer
        inside the tensor rather than replacing the tensor.

        The parameter must be a leaf that requires grad. A non-leaf would mean
        the module owns a value computed from something else, so an optimiser
        stepping it would be writing to the wrong end of the graph; that is
        caught here, where the module is being built, rather than at the first
        `assign_` several hundred steps later.
        """
        self._claim(name)
        if not value.is_leaf:
            msg = (
                f"cannot register {name!r} as a parameter: it was produced by "
                f"{value.node.op if value.node else '?'}, so it is a computed value, not a "
                "variable. Call detach() if an independent copy is what you meant."
            )
            raise ModuleError(msg)
        if not value.requires_grad:
            msg = (
                f"cannot register {name!r} as a parameter: it does not require grad, so no "
                "gradient would ever reach it. Use register_buffer for a constant."
            )
            raise ModuleError(msg)
        self._parameters[name] = value
        return value

    def register_buffer(self, name: str, value: Tensor) -> Tensor:
        """Record `value` as a non-trainable tensor and return it.

        Buffers are part of the model's state - they are saved and loaded - but
        no optimiser touches them. Running statistics and constant masks belong
        here. A buffer must *not* require grad, for the same reason a parameter
        must: the two categories are distinguished by what happens to them, and
        a tensor that is in the wrong one produces a model that trains a
        constant or fails to train a variable.
        """
        self._claim(name)
        if value.requires_grad:
            msg = (
                f"cannot register {name!r} as a buffer: it requires grad. A tensor that "
                "receives gradients is a parameter; use register_parameter."
            )
            raise ModuleError(msg)
        self._buffers[name] = value
        return value

    def register_module(self, name: str, value: ChildT) -> ChildT:
        """Record a child module and return it, keeping its concrete type.

        Registered children are what make `parameters()` and `state_dict()`
        recursive. A module held in a plain attribute is invisible to both,
        which is exactly the failure mode this class refuses to hide - hence the
        explicit call, and hence `ModuleList` and `ModuleDict` for the cases
        where the children live in a container.

        Generic in the child's type so that
        `self.head = self.register_module("head", Linear(...))` leaves `head`
        typed as a `Linear` rather than as a `Module`. Without that the return
        of every registration would have to be re-annotated at the call site,
        and a type checker would have nothing to say about `self.head.weight`.
        """
        self._claim(name)
        self._modules[name] = value
        return value

    # -- traversal ---------------------------------------------------------- #

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Immediate submodules, in registration order."""
        yield from self._registries()[2].items()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        """This module and every descendant, depth first, parents before children."""
        yield prefix, self
        for name, child in self.named_children():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Every parameter in the tree with its dotted path.

        A tensor shared between two places in the tree is yielded once, under
        the first path that reaches it. Yielding it twice would make an
        optimiser step it twice per iteration - a tied embedding would move at
        double the learning rate, which looks like an unstable model rather than
        like a bug in the traversal.
        """
        seen: set[int] = set()
        for module_name, module in self.named_modules(prefix):
            for name, parameter in module._registries()[0].items():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                yield (f"{module_name}.{name}" if module_name else name), parameter

    def parameters(self) -> Iterator[Tensor]:
        """Every parameter in the tree, deduplicated. What an optimiser is given."""
        for _, parameter in self.named_parameters():
            yield parameter

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Every buffer in the tree with its dotted path, deduplicated."""
        seen: set[int] = set()
        for module_name, module in self.named_modules(prefix):
            for name, buffer in module._registries()[1].items():
                if id(buffer) in seen:
                    continue
                seen.add(id(buffer))
                yield (f"{module_name}.{name}" if module_name else name), buffer

    def buffers(self) -> Iterator[Tensor]:
        """Every buffer in the tree, deduplicated."""
        for _, buffer in self.named_buffers():
            yield buffer

    def parameter_count(self) -> int:
        """Total number of scalar parameters in the tree."""
        return sum(p.size for p in self.parameters())

    # -- modes -------------------------------------------------------------- #

    def train(self, mode: bool = True) -> Module:
        """Put the whole tree into training mode. Returns self, so it chains."""
        for _, module in self.named_modules():
            module.training = mode
        return self

    def eval(self) -> Module:
        """Put the whole tree into evaluation mode. `train(False)`, named for the reader."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Drop every accumulated gradient in the tree.

        Sets `.grad` to `None` rather than to zeros, for the reason given in
        `Tensor.zero_grad`: a parameter that never received a gradient stays
        distinguishable from one whose gradient happened to be zero.
        """
        for parameter in self.parameters():
            parameter.zero_grad()

    # -- state -------------------------------------------------------------- #

    def state_dict(self) -> dict[str, Array]:
        """Parameters and buffers as plain NumPy arrays, keyed by dotted path.

        Arrays rather than tensors, and copies rather than the live buffers: a
        checkpoint is a snapshot, and one that aliased the model would change
        under it as training continued. The copy costs one buffer per parameter
        at save time, which is the same order as the optimiser state that will
        be saved next to it.

        Buffers are included. They are part of what the model needs to reproduce
        its output, and a checkpoint that restores the weights but not the
        running statistics is a checkpoint that evaluates differently from the
        model it was taken from.
        """
        state: dict[str, Array] = {}
        for name, parameter in self.named_parameters():
            state[name] = parameter.numpy()
        for name, buffer in self.named_buffers():
            state[name] = buffer.numpy()
        return state

    def load_state_dict(self, state: Mapping[str, Array], *, strict: bool = True) -> None:
        """Copy values from `state` into this model's parameters and buffers.

        Loads with `Tensor.assign_`, so every tensor object in the tree survives
        the load. That matters because an optimiser built before the load holds
        references to those objects and keys its state by their position: a
        loader that replaced the tensors would leave the optimiser updating
        detached copies, and the model would simply stop learning.

        `strict=True` refuses a state dict whose key set differs from the
        model's. The non-strict form is for deliberate partial loads - a
        pretrained trunk under a fresh head - and reports nothing, so anything
        that could be a typo should be loaded strictly.
        """
        targets: dict[str, Tensor] = dict(self.named_parameters())
        targets.update(dict(self.named_buffers()))
        if strict:
            missing = sorted(set(targets) - set(state))
            unexpected = sorted(set(state) - set(targets))
            if missing or unexpected:
                msg = (
                    f"state dict does not match {type(self).__name__}: "
                    f"missing {missing}, unexpected {unexpected}"
                )
                raise ModuleError(msg)
        for name, target in targets.items():
            if name not in state:
                continue
            value = np.asarray(state[name])
            if value.shape != target.shape:
                msg = (
                    f"state dict entry {name!r} has shape {value.shape}, "
                    f"but the model's is {target.shape}"
                )
                raise ShapeError(msg)
            target.assign_(value)

    # -- calling ------------------------------------------------------------ #

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute this module's output. Subclasses override this."""
        msg = f"{type(self).__name__} does not implement forward()"
        raise NotImplementedError(msg)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call `forward`.

        A thin indirection, kept because it is the one hook a wrapper needs -
        profiling, shape logging, activation checkpointing - and because
        `model(x)` reads better than `model.forward(x)` at every call site.
        """
        return self.forward(*args, **kwargs)

    # -- display ------------------------------------------------------------ #

    def extra_repr(self) -> str:
        """One line of configuration for `__repr__`. Subclasses override this."""
        return ""

    def __repr__(self) -> str:
        return "\n".join(self._repr_lines())

    def _repr_lines(self) -> list[str]:
        """The tree, one line per module, each annotated with its parameter count.

        The count is what makes this worth printing. A model's architecture is
        usually obvious from the code that built it; where the parameters
        actually are is not, and "the embedding is 80% of this model" is the
        kind of thing you want to see before you spend an afternoon optimising
        the attention.

        Each line reports the parameters *below* that node, so the root's number
        is the model's total and the children's numbers add up to it - shared
        parameters excepted, which are counted once under the first path that
        reaches them, exactly as `named_parameters` yields them.

        Indentation rather than brackets. A bracketed form has to be balanced,
        which means the interesting number ends up on a line by itself several
        rows away from the module it describes.
        """
        lines = [f"{type(self).__name__}({self.extra_repr()})  [{self.parameter_count():,} params]"]
        for name, child in self.named_children():
            child_lines = child._repr_lines()
            lines.append(f"  ({name}): {child_lines[0]}")
            lines.extend(f"  {line}" for line in child_lines[1:])
        return lines


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #


class Sequential(Module):
    """Apply a chain of modules in order.

    Only usable when every step takes one tensor and returns one tensor, which
    is why it is a container and not the way models are written here. Anything
    with a residual connection, two inputs, or a value used twice needs a
    `forward` you can read; see `TransformerBlock`.

    >>> import nabla
    >>> from nabla import nn
    >>> g = nabla.random.Generator(seed=0)
    >>> net = nn.Sequential(nn.Linear(4, 8, generator=g), nn.ReLU(), nn.Linear(8, 2, generator=g))
    >>> net(nabla.zeros(3, 4)).shape
    (3, 2)
    """

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self.register_module(str(index), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return list(self._modules.values())[index]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, x: Tensor) -> Tensor:
        """Apply each module to the output of the previous one."""
        out = x
        for module in self._modules.values():
            out = module(out)
        return out


class ModuleList(Module):
    """A list of modules that is visible to `parameters()`.

    A plain Python list held in an attribute is not: nothing registers its
    contents, so the layers exist, run, and never learn. This class exists
    solely to close that hole, which is why it registers on `append` rather than
    storing a list of its own.
    """

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        for module in modules or []:
            self.append(module)

    def append(self, module: Module) -> ModuleList:
        """Add a module at the end. Returns self, so construction chains."""
        self.register_module(str(len(self._modules)), module)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return list(self._modules.values())[index]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleDict(Module):
    """A string-keyed collection of modules, visible to `parameters()`.

    Insertion order is preserved and is what `state_dict()` and `__repr__`
    follow, so a model built from a dict has reproducible checkpoint key order.
    """

    def __init__(self, modules: Mapping[str, Module] | None = None) -> None:
        super().__init__()
        for name, module in (modules or {}).items():
            self.register_module(name, module)

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.register_module(key, module)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self) -> Iterator[str]:
        """The keys, in insertion order."""
        return iter(self._modules)

    def items(self) -> Iterator[tuple[str, Module]]:
        """The (key, module) pairs, in insertion order."""
        return iter(self._modules.items())

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)
