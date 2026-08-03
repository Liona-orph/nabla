"""The module system: registration, traversal, modes and state."""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from nabla import nn
from nabla.errors import ModuleError, ShapeError


class Tiny(nn.Module):
    def __init__(self, generator: nb.random.Generator) -> None:
        super().__init__()
        self.first = self.register_module("first", nn.Linear(3, 4, generator=generator))
        self.second = self.register_module("second", nn.Linear(4, 2, generator=generator))
        self.register_buffer("scale", nb.ones(4))

    def forward(self, x: nb.Tensor) -> nb.Tensor:
        return self.second(nb.relu(self.first(x)) * self._buffers["scale"])


@pytest.fixture
def tiny(gen: nb.random.Generator) -> Tiny:
    return Tiny(gen)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_register_parameter_returns_the_same_object_it_stores(gen: nb.random.Generator) -> None:
    layer = nn.Linear(2, 3, generator=gen)
    assert layer.weight is dict(layer.named_parameters())["weight"]


def test_registration_refuses_a_duplicate_name(gen: nb.random.Generator) -> None:
    layer = nn.Linear(2, 3, generator=gen)
    with pytest.raises(ModuleError, match="already has a parameter"):
        layer.register_parameter("weight", gen.normal((2, 3), requires_grad=True))


def test_registration_refuses_a_computed_tensor_as_a_parameter(gen: nb.random.Generator) -> None:
    module = nn.Module()
    computed = gen.normal((2,), requires_grad=True) * 2.0
    with pytest.raises(ModuleError, match="computed value"):
        module.register_parameter("w", computed)


def test_registration_refuses_a_parameter_that_does_not_require_grad() -> None:
    module = nn.Module()
    with pytest.raises(ModuleError, match="does not require grad"):
        module.register_parameter("w", nb.zeros(3))


def test_registration_refuses_a_buffer_that_requires_grad() -> None:
    module = nn.Module()
    with pytest.raises(ModuleError, match="it requires grad"):
        module.register_buffer("b", nb.zeros(3, requires_grad=True))


def test_registration_refuses_a_name_with_a_dot() -> None:
    module = nn.Module()
    with pytest.raises(ModuleError, match="not a usable registry name"):
        module.register_buffer("a.b", nb.zeros(3))


def test_forgetting_super_init_says_so() -> None:
    class Broken(nn.Module):
        def __init__(self) -> None:  # deliberately does not call super().__init__()
            self.register_buffer("b", nb.zeros(1))

    with pytest.raises(ModuleError, match="did not call super"):
        Broken()


# --------------------------------------------------------------------------- #
# traversal
# --------------------------------------------------------------------------- #


def test_named_parameters_uses_dotted_paths(tiny: Tiny) -> None:
    assert [name for name, _ in tiny.named_parameters()] == [
        "first.weight",
        "first.bias",
        "second.weight",
        "second.bias",
    ]


def test_parameter_count_matches_the_arithmetic(tiny: Tiny) -> None:
    assert tiny.parameter_count() == 3 * 4 + 4 + 4 * 2 + 2


def test_a_tied_parameter_is_yielded_once(gen: nb.random.Generator) -> None:
    shared = nn.Linear(3, 3, generator=gen)

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_module("a", shared)
            self.register_module("b", shared)

    model = Tied()
    assert [name for name, _ in model.named_parameters()] == ["a.weight", "a.bias"]
    assert model.parameter_count() == 12


def test_buffers_are_traversed_but_are_not_parameters(tiny: Tiny) -> None:
    assert [name for name, _ in tiny.named_buffers()] == ["scale"]
    assert "scale" not in dict(tiny.named_parameters())


def test_a_module_in_a_plain_list_is_invisible(gen: nb.random.Generator) -> None:
    """The failure explicit registration exists to make impossible to miss."""

    class Hidden(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [nn.Linear(2, 2, generator=gen)]

    class Visible(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = self.register_module(
                "layers", nn.ModuleList([nn.Linear(2, 2, generator=gen)])
            )

    assert Hidden().parameter_count() == 0
    assert Visible().parameter_count() == 6


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


def test_train_and_eval_reach_the_whole_tree(tiny: Tiny) -> None:
    tiny.eval()
    assert all(not module.training for _, module in tiny.named_modules())
    tiny.train()
    assert all(module.training for _, module in tiny.named_modules())


def test_zero_grad_clears_to_none(tiny: Tiny, gen: nb.random.Generator) -> None:
    tiny(gen.normal((5, 3))).sum().backward()
    assert all(p.grad is not None for p in tiny.parameters())
    tiny.zero_grad()
    assert all(p.grad is None for p in tiny.parameters())


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


def test_state_dict_round_trips_through_a_fresh_model(gen: nb.random.Generator) -> None:
    source = Tiny(nb.random.Generator(seed=1))
    destination = Tiny(nb.random.Generator(seed=2))
    x = gen.normal((4, 3))
    assert not np.allclose(source(x).numpy(), destination(x).numpy())

    destination.load_state_dict(source.state_dict())
    np.testing.assert_allclose(source(x).numpy(), destination(x).numpy())


def test_state_dict_is_a_snapshot_not_a_view(tiny: Tiny) -> None:
    state = tiny.state_dict()
    before = state["first.weight"].copy()
    tiny.first.weight.assign_(tiny.first.weight.data + 1.0)
    np.testing.assert_allclose(state["first.weight"], before)


def test_load_state_dict_keeps_the_tensor_objects(tiny: Tiny) -> None:
    """An optimiser built before the load must still be pointing at the weights."""
    weight = tiny.first.weight
    tiny.load_state_dict(tiny.state_dict())
    assert tiny.first.weight is weight


def test_strict_load_reports_both_directions(tiny: Tiny) -> None:
    state = tiny.state_dict()
    del state["first.bias"]
    state["nonsense"] = np.zeros(3)
    with pytest.raises(ModuleError, match=r"missing \['first.bias'\].*unexpected \['nonsense'\]"):
        tiny.load_state_dict(state)


def test_non_strict_load_accepts_a_partial_state(tiny: Tiny) -> None:
    tiny.load_state_dict({"first.bias": np.arange(4.0)}, strict=False)
    np.testing.assert_allclose(tiny.first.bias.numpy(), np.arange(4.0))


def test_load_state_dict_refuses_a_shape_change(tiny: Tiny) -> None:
    state = tiny.state_dict()
    state["first.weight"] = np.zeros((5, 5))
    with pytest.raises(ShapeError, match="has shape"):
        tiny.load_state_dict(state)


# --------------------------------------------------------------------------- #
# containers and display
# --------------------------------------------------------------------------- #


def test_sequential_applies_in_order(gen: nb.random.Generator) -> None:
    net = nn.Sequential(nn.Linear(3, 4, generator=gen), nn.ReLU(), nn.Linear(4, 2, generator=gen))
    assert net(nb.zeros(6, 3)).shape == (6, 2)
    assert len(net) == 3
    assert isinstance(net[1], nn.ReLU)


def test_module_list_and_dict_register_their_contents(gen: nb.random.Generator) -> None:
    listed = nn.ModuleList([nn.Linear(2, 2, generator=gen) for _ in range(3)])
    mapped = nn.ModuleDict({"a": nn.Linear(2, 2, generator=gen)})
    assert listed.parameter_count() == 18
    assert mapped.parameter_count() == 6
    assert list(mapped.keys()) == ["a"]
    assert "a" in mapped


def test_repr_shows_the_tree_and_the_counts(tiny: Tiny) -> None:
    text = repr(tiny)
    lines = text.splitlines()
    assert lines[0].startswith("Tiny()")
    assert "[26 params]" in lines[0]
    assert "  (first): Linear(in_features=3, out_features=4, bias=True)  [16 params]" in lines
    assert "  (second): Linear(in_features=4, out_features=2, bias=True)  [10 params]" in lines


def test_calling_an_unimplemented_forward_names_the_class() -> None:
    with pytest.raises(NotImplementedError, match="Module does not implement forward"):
        nn.Module()(nb.zeros(1))
