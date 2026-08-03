"""Every layer's forward result and every layer's gradients.

Two assertions per layer. The forward pass is compared against a NumPy
expression written from the definition, not from the implementation. The
backward pass goes through `check_grad`, twice: once with respect to the input,
and once with respect to every parameter, using `conftest.check_module` to turn
captured state into arguments.

All the layers here are built in float64. `check_grad` promotes its *arguments*
to float64, but a module's parameters are captured, and a float32 weight inside
a float64 check puts a 1e-7 rounding error into a difference quotient that is
being compared at 1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb
from conftest import assert_close, check, check_module, sample
from nabla import nn
from nabla.errors import ShapeError

F64 = "float64"


@pytest.fixture
def g() -> nb.random.Generator:
    return nb.random.Generator(seed=404)


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #


def test_linear_forward_is_the_affine_map(g: nb.random.Generator) -> None:
    layer = nn.Linear(4, 3, generator=g, dtype=F64)
    x = sample(5, 4, seed=1)
    expected = x @ layer.weight.numpy() + layer.bias.numpy()
    assert_close(layer(nb.Tensor(x)), expected)


def test_linear_weight_is_in_by_out(g: nb.random.Generator) -> None:
    """The layout every initialiser and every checkpoint depends on."""
    assert nn.Linear(7, 3, generator=g).weight.shape == (7, 3)


def test_linear_gradients(g: nb.random.Generator) -> None:
    layer = nn.Linear(4, 3, generator=g, dtype=F64)
    x = sample(5, 4, seed=2)
    check(layer, x, name="Linear/input")
    check_module(layer, lambda m: m(nb.Tensor(x)), name="Linear/parameters")


def test_linear_without_bias_registers_none(g: nb.random.Generator) -> None:
    layer = nn.Linear(4, 3, generator=g, bias=False)
    assert layer.bias is None
    assert [name for name, _ in layer.named_parameters()] == ["weight"]


def test_linear_broadcasts_over_leading_axes(g: nb.random.Generator) -> None:
    layer = nn.Linear(4, 3, generator=g)
    assert layer(nb.zeros(2, 6, 4)).shape == (2, 6, 3)


def test_linear_refuses_a_zero_width(g: nb.random.Generator) -> None:
    with pytest.raises(ShapeError, match="positive feature counts"):
        nn.Linear(0, 3, generator=g)


def test_linear_accepts_a_custom_initialiser(g: nb.random.Generator) -> None:
    def zeros(shape: tuple[int, ...], _generator: nb.random.Generator) -> nb.Tensor:
        return nb.zeros(shape, requires_grad=True)

    layer = nn.Linear(3, 2, generator=g, initialiser=zeros)
    np.testing.assert_allclose(layer.weight.numpy(), np.zeros((3, 2)))


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def test_embedding_forward_is_a_lookup(g: nb.random.Generator) -> None:
    table = nn.Embedding(6, 4, generator=g, dtype=F64)
    ids = np.array([[0, 3], [5, 3]])
    assert_close(table(ids), table.weight.numpy()[ids])


def test_embedding_gradients_accumulate_for_a_repeated_token(g: nb.random.Generator) -> None:
    table = nn.Embedding(4, 3, generator=g, dtype=F64)
    ids = np.array([1, 1, 2])
    table(ids).sum().backward()
    assert table.weight.grad is not None
    np.testing.assert_allclose(table.weight.grad[1], np.full(3, 2.0))
    np.testing.assert_allclose(table.weight.grad[0], np.zeros(3))


def test_embedding_gradients(g: nb.random.Generator) -> None:
    table = nn.Embedding(6, 4, generator=g, dtype=F64)
    ids = np.array([[0, 3, 3], [5, 1, 0]])
    check_module(table, lambda m: m(ids), name="Embedding")


def test_embedding_refuses_float_ids(g: nb.random.Generator) -> None:
    table = nn.Embedding(4, 2, generator=g)
    with pytest.raises(ShapeError, match="must be integers"):
        table(np.array([0.0, 1.0]))


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


def test_layer_norm_forward_matches_the_definition() -> None:
    layer = nn.LayerNorm(5, dtype=F64)
    x = sample(4, 5, seed=3, scale=3.0, offset=2.0)
    expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    assert_close(layer(nb.Tensor(x)), expected)


def test_layer_norm_output_has_zero_mean_and_unit_variance() -> None:
    layer = nn.LayerNorm(64, eps=0.0, dtype=F64)
    out = layer(nb.Tensor(sample(8, 64, seed=4, scale=7.0, offset=-3.0))).numpy()
    np.testing.assert_allclose(out.mean(-1), np.zeros(8), atol=1e-12)
    np.testing.assert_allclose(out.var(-1), np.ones(8), rtol=1e-10)


def test_layer_norm_is_per_example_not_per_batch() -> None:
    """The property that makes batch size irrelevant to the forward pass."""
    layer = nn.LayerNorm(5, dtype=F64)
    x = sample(4, 5, seed=5)
    whole = layer(nb.Tensor(x)).numpy()
    alone = layer(nb.Tensor(x[2:3])).numpy()
    np.testing.assert_allclose(whole[2:3], alone, rtol=1e-12)


def test_layer_norm_gradients() -> None:
    layer = nn.LayerNorm(5, dtype=F64)
    x = sample(4, 5, seed=6)
    check(layer, x, name="LayerNorm/input")
    check_module(layer, lambda m: m(nb.Tensor(x)), name="LayerNorm/parameters")


def test_layer_norm_starts_as_the_identity_transform() -> None:
    layer = nn.LayerNorm(3, dtype=F64)
    np.testing.assert_allclose(layer.weight.numpy(), np.ones(3))
    np.testing.assert_allclose(layer.bias.numpy(), np.zeros(3))


def test_rms_norm_forward_matches_the_definition() -> None:
    layer = nn.RMSNorm(5, dtype=F64)
    x = sample(4, 5, seed=7, offset=1.0)
    expected = x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6)
    assert_close(layer(nb.Tensor(x)), expected)


def test_rms_norm_does_not_centre() -> None:
    """The whole difference from LayerNorm, stated as an assertion."""
    x = sample(4, 5, seed=8) + 10.0
    out = nn.RMSNorm(5, dtype=F64)(nb.Tensor(x)).numpy()
    assert np.all(np.abs(out.mean(-1)) > 0.5)


def test_rms_norm_gradients() -> None:
    layer = nn.RMSNorm(5, dtype=F64)
    x = sample(4, 5, seed=9, offset=0.5)
    check(layer, x, name="RMSNorm/input")
    check_module(layer, lambda m: m(nb.Tensor(x)), name="RMSNorm/parameters")


def test_normalisations_refuse_the_wrong_width() -> None:
    for layer in (nn.LayerNorm(5), nn.RMSNorm(5)):
        with pytest.raises(ShapeError, match="last axis over 5"):
            layer(nb.zeros(4, 6))


def test_normalisations_can_drop_their_affine_parameters() -> None:
    assert nn.LayerNorm(5, affine=False).parameter_count() == 0
    assert nn.RMSNorm(5, affine=False).parameter_count() == 0


# --------------------------------------------------------------------------- #
# Dropout
# --------------------------------------------------------------------------- #


def test_dropout_is_the_identity_in_eval_mode(g: nb.random.Generator) -> None:
    layer = nn.Dropout(0.5, generator=g).eval()
    x = nb.Tensor(sample(4, 8, seed=10))
    out = layer(x)
    assert out is x


def test_dropout_zeroes_about_p_of_the_entries(g: nb.random.Generator) -> None:
    layer = nn.Dropout(0.25, generator=g)
    out = layer(nb.ones(400, 400)).numpy()
    assert float(np.mean(out == 0.0)) == pytest.approx(0.25, abs=0.01)


def test_dropout_is_inverted_so_the_expectation_is_preserved(g: nb.random.Generator) -> None:
    """The property that makes the eval-mode path the identity.

    E[output] = input, so the mean of a large dropped batch is the mean of the
    input. A test-time-scaling implementation would come out at 1 - p instead.
    """
    layer = nn.Dropout(0.3, generator=g)
    out = layer(nb.ones(500, 500)).numpy()
    assert float(out.mean()) == pytest.approx(1.0, abs=0.01)
    assert float(out.max()) == pytest.approx(1.0 / 0.7, rel=1e-6)


def test_dropout_at_zero_allocates_nothing(g: nb.random.Generator) -> None:
    layer = nn.Dropout(0.0, generator=g)
    x = nb.ones(3, 3)
    assert layer(x) is x


def test_dropout_gradient_follows_the_mask(g: nb.random.Generator) -> None:
    layer = nn.Dropout(0.5, generator=g)
    x = nb.Tensor(np.ones((200, 20)), requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    survived = out.numpy() != 0.0
    np.testing.assert_allclose(x.grad[~survived], 0.0)
    np.testing.assert_allclose(x.grad[survived], 2.0)


def test_dropout_refuses_a_probability_of_one(g: nb.random.Generator) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        nn.Dropout(1.0, generator=g)


# --------------------------------------------------------------------------- #
# activation modules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("module", "reference"),
    [
        (nn.ReLU(), nb.relu),
        (nn.GELU(), nb.gelu),
        (nn.Tanh(), nb.tanh),
        (nn.Sigmoid(), nb.sigmoid),
    ],
)
def test_activation_modules_wrap_their_function(module: nn.Module, reference: object) -> None:
    x = nb.Tensor(sample(3, 4, seed=11))
    assert_close(module(x), reference(x).numpy())  # type: ignore[operator]
