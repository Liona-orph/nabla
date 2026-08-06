# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-02-09

First release. A complete training stack: autodiff, layers, optimisers, data
loading, a training loop, and three runnable examples.

### Added

**Autodiff core**

- `Tensor` with reverse-mode automatic differentiation on a per-tensor tape.
  Each non-leaf tensor holds a `Node` with its operation's name, inputs and
  vector-Jacobian product; there is no global tape and no mutable global state.
- `backward()` with a topological walk (reverse post-order DFS) rather than
  creation order, so a tensor reused across a branch accumulates correctly.
- `unbroadcast`, one implementation of broadcast gradient reduction, used by
  every elementwise vjp and by `matmul`.
- Read-only tensor buffers, with `Tensor.assign_` as the single sanctioned
  mutation; every in-place operator raises with a message naming the
  alternative.
- `no_grad()`, thread-local and re-entrant. `retain_grad()` for interior
  tensors.
- Functional `grad(fn, argnums=...)`.
- `clip_grad_norm_` and `clip_grad_value_`.
- 41 differentiable operations over 36 recorded primitives: arithmetic,
  reductions, shape manipulation, matrix products and activations.
- `check_grad`: central-difference verification in float64 against a fixed
  random output projection, returning a `GradCheckReport` even on success.
- Explicit seeded `random.Generator` with `spawn`; nothing in the package
  touches NumPy's global random state.
- An exception hierarchy where every error is both a `NablaError` and the
  builtin a caller would have guessed.

**Modules (`nabla.nn`)**

- `Module` with explicit `register_parameter`, `register_buffer` and
  `register_module`; `parameters()`, `named_parameters()` with deduplication of
  tied weights, `state_dict()`/`load_state_dict()`, `train()`/`eval()`,
  `zero_grad()`, and a tree `__repr__` annotated with parameter counts.
- `Sequential`, `ModuleList`, `ModuleDict`.
- `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout` (inverted, train/eval
  aware), and `ReLU`/`GELU`/`Tanh`/`Sigmoid` as modules.
- `MultiHeadAttention` with additive `-inf` causal masking, and a pre-norm
  `TransformerBlock`.
- Initialisers: Xavier/Glorot and Kaiming/He in normal and uniform forms, and a
  truncated normal that rescales so the standard deviation it returns is the one
  that was asked for.
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy` and `nll`, all taking
  logits, all in the form that cannot overflow.

**Optimisers (`nabla.optim`)**

- `SGD` with momentum, dampening, Nesterov and weight decay.
- `Adam`, `AdamW` with decoupled decay, and `RMSProp` with optional centring and
  momentum.
- `state_dict`/`load_state_dict` for all four, keyed by parameter position.
- Schedules as pure functions of the step: `constant`, `step_decay`,
  `cosine_with_warmup`, `linear_warmup_decay`.

**Data (`nabla.data`)**

- `Dataset` protocol, `ArrayDataset`, `Subset`, `random_split`.
- `DataLoader` with seeded shuffling and deterministic iteration order.
- `make_moons` and `make_digits`, generated from a seed so no example downloads
  anything.
- `CharVocabulary` and `NextCharDataset` for character-level language modelling.

**Training (`nabla.train`)**

- `Trainer` with gradient clipping, learning-rate schedules, per-example metric
  averaging, early stopping and checkpointing.
- `save_checkpoint`/`load_checkpoint`: a plain `.npz` with a JSON manifest, read
  with `allow_pickle=False`, so a checkpoint cannot execute anything.

**Examples**

- `examples/mlp_moons` - 96.25% test accuracy in 0.44 s.
- `examples/mnist_style` - 95.60% test accuracy on ten classes in 1.2 s.
- `examples/char_transformer` - a 420,926-parameter transformer reaching 1.6631
  nats per character in 91.4 s on one CPU core.
- `benchmarks/overhead.py` - the tape's cost against hand-differentiated NumPy:
  2.22x at batch 32 with 128 hidden units, 1.19x at batch 512 with 1,024.

**Documentation**

- `ARCHITECTURE.md`, six architecture decision records in `docs/adr/`, and a
  README whose every number was measured.

### Known limitations

No GPU, no fused kernels, no higher-order derivatives, no `einsum`, no
convolutions or recurrence. The graph is not freed by `backward()`. GELU is the
tanh approximation. See the "What this is not" section of the README.

[Unreleased]: https://github.com/nabla-authors/nabla/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nabla-authors/nabla/releases/tag/v0.1.0
