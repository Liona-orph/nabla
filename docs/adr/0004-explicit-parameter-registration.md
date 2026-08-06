# 4. Explicit parameter registration

Date: 2026-01-20

## Status

Accepted.

## Context

A module has to know which of its attributes are trainable parameters, so that
`parameters()` can hand them to an optimiser and `state_dict()` can save them.
There are two ways to find out.

**Interception.** Override `__setattr__`. When the value being assigned is a
`Parameter` or a `Module`, file it in a hidden dictionary. This is what PyTorch
does, and it makes `self.weight = Parameter(...)` do two things at once.

**Declaration.** Make the module say what it owns.

The interception approach has a specific, common, silent failure: a parameter
stored anywhere the interceptor does not look. A list of layers, a dict of
heads, a tuple, a closure. The layers exist, the forward pass uses them, and
they never appear in `parameters()`, so the optimiser never updates them and
`state_dict()` never saves them. The model trains - slightly worse - and nothing
in any traceback says why. PyTorch's answer is `nn.ModuleList` and
`nn.ModuleDict`, which exist entirely to work around a problem the interception
created.

## Decision

`Module.register_parameter(name, tensor)`, `register_buffer` and
`register_module` are explicit calls. Each returns the value it registered, so
the assignment still reads naturally:

```python
self.weight = self.register_parameter("weight", init.xavier_uniform(shape, g))
```

There is no `Parameter` subclass. A parameter is a leaf `Tensor` with
`requires_grad=True`; membership of the registry is what makes it a parameter.

## Consequences

**"What does this module own" is answered by reading its constructor.** Not by
knowing the framework's dispatch rules and then auditing every assignment for
whether its right-hand side happened to be the intercepted type.

**Registration validates.** `register_parameter` refuses a non-leaf tensor
(which would mean the module owns a computed value, so stepping it writes to the
wrong end of the graph) and refuses one that does not require grad (which would
never receive a gradient). `register_buffer` refuses one that *does*. All three
are caught where the module is built, not several hundred steps into training.

**A checkpoint key is a string you can search for.** With interception, renaming
an attribute silently renames a checkpoint key. Here the name is an argument.

**No isinstance check on every attribute assignment.** `self.eps = 1e-5` is a
plain assignment.

**The cost: one extra call and one repeated name per parameter.** Paid once, at
construction. `self.weight = self.register_parameter("weight", ...)` says
"weight" twice, which is genuinely redundant and is the price of the name being
data rather than metadata.

**`ModuleList` and `ModuleDict` still exist**, because a list of blocks is a
reasonable thing to want. They are three lines each here rather than a
workaround: they register on `append` and `__setitem__`, and that is all they
do.

**Forgetting `super().__init__()` is the one mistake this makes easy**, so it
gets a message that names the fix rather than an `AttributeError` on a private
name. `tests/test_nn_module.py::test_forgetting_super_init_says_so` pins it, and
`test_a_module_in_a_plain_list_is_invisible` pins the failure this whole record
is about - asserting that the hidden version really does report zero parameters,
so the difference is visible in the test suite rather than only in prose.
