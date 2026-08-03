"""Initialisers, checked against the variance each one promises.

An initialiser's entire contract is a number: the variance of what it returns.
These tests measure that number on a large enough sample that the sampling error
is well below the difference between the schemes, and compare it to the closed
form. A test that only asserted "the values are small" would pass for an
initialiser off by the factor of `sqrt(2)` that distinguishes Kaiming from
Xavier, which is precisely the mistake worth catching.

Sample size and tolerance: 512 x 512 is 262,144 draws, so the relative standard
error of the *variance* is `sqrt(2/n)` = 0.28%. A tolerance of 3% is therefore
about ten sigma - loose enough never to flake, tight enough that a factor of 1.4
fails by a mile.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import nabla as nb
from nabla.errors import ShapeError
from nabla.nn import init

SHAPE = (512, 512)
RECTANGULAR = (1024, 256)
TOLERANCE = 0.03


def variance(tensor: nb.Tensor) -> float:
    return float(np.var(tensor.numpy()))


# --------------------------------------------------------------------------- #
# fans
# --------------------------------------------------------------------------- #


def test_fan_in_out_follows_nablas_weight_layout() -> None:
    assert init.fan_in_out((784, 128)) == (784, 128)
    assert init.fan_in_out((64,)) == (64, 64)


def test_fan_in_out_refuses_a_rank_it_cannot_interpret() -> None:
    with pytest.raises(ShapeError, match="rank-3"):
        init.fan_in_out((2, 3, 4))


def test_recommended_gains_are_the_published_values() -> None:
    assert init.recommended_gain("linear") == 1.0
    assert init.recommended_gain("relu") == pytest.approx(math.sqrt(2.0))
    assert init.recommended_gain("tanh") == pytest.approx(5.0 / 3.0)
    with pytest.raises(ValueError, match="no recommended gain"):
        init.recommended_gain("swish")


# --------------------------------------------------------------------------- #
# the promised variances
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("draw", [init.xavier_normal, init.xavier_uniform])
def test_xavier_delivers_two_over_fan_in_plus_fan_out(draw: object) -> None:
    fan_in, fan_out = RECTANGULAR
    weights = draw(RECTANGULAR, nb.random.Generator(seed=7))  # type: ignore[operator]
    assert variance(weights) == pytest.approx(2.0 / (fan_in + fan_out), rel=TOLERANCE)


@pytest.mark.parametrize("draw", [init.kaiming_normal, init.kaiming_uniform])
def test_kaiming_fan_in_delivers_two_over_fan_in(draw: object) -> None:
    fan_in, _ = RECTANGULAR
    weights = draw(RECTANGULAR, nb.random.Generator(seed=7))  # type: ignore[operator]
    assert variance(weights) == pytest.approx(2.0 / fan_in, rel=TOLERANCE)


def test_kaiming_fan_out_uses_the_other_fan() -> None:
    _, fan_out = RECTANGULAR
    weights = init.kaiming_normal(RECTANGULAR, nb.random.Generator(seed=7), mode="fan_out")
    assert variance(weights) == pytest.approx(2.0 / fan_out, rel=TOLERANCE)


def test_kaiming_and_xavier_differ_by_the_relu_correction() -> None:
    """The factor this whole file exists to be able to see."""
    square = (256, 256)
    g = nb.random.Generator(seed=11)
    kaiming = variance(init.kaiming_normal(square, g))
    xavier = variance(init.xavier_normal(square, g))
    # Kaiming(fan_in) is 2/n, Xavier on a square layer is 2/(2n) = 1/n.
    assert kaiming / xavier == pytest.approx(2.0, rel=TOLERANCE)


def test_gain_scales_the_standard_deviation_not_the_variance() -> None:
    g = nb.random.Generator(seed=3)
    plain = variance(init.xavier_normal(SHAPE, g, gain=1.0))
    gained = variance(init.xavier_normal(SHAPE, g, gain=3.0))
    assert gained / plain == pytest.approx(9.0, rel=TOLERANCE)


def test_mode_must_be_one_of_two_strings() -> None:
    with pytest.raises(ValueError, match="fan_in"):
        init.kaiming_normal(SHAPE, nb.random.Generator(seed=0), mode="fan_sideways")


# --------------------------------------------------------------------------- #
# truncated normal
# --------------------------------------------------------------------------- #


def test_truncation_scale_matches_the_closed_form() -> None:
    for limit in (1.0, 2.0, 3.0):
        density = math.exp(-0.5 * limit**2) / math.sqrt(2.0 * math.pi)
        mass = math.erf(limit / math.sqrt(2.0))
        expected = math.sqrt(1.0 - 2.0 * limit * density / mass)
        assert init.truncation_scale(limit) == pytest.approx(expected)


def test_truncated_normal_delivers_the_requested_standard_deviation() -> None:
    weights = init.truncated_normal(SHAPE, nb.random.Generator(seed=5), std=0.05)
    assert float(np.std(weights.numpy())) == pytest.approx(0.05, rel=TOLERANCE)


def test_truncated_normal_without_rescaling_would_be_twelve_percent_small() -> None:
    """States the size of the error the rescaling removes."""
    assert init.truncation_scale(2.0) == pytest.approx(0.8796, abs=1e-4)


def test_truncated_normal_has_no_value_beyond_the_limit() -> None:
    limit = 2.0
    std = 0.05
    weights = init.truncated_normal(SHAPE, nb.random.Generator(seed=5), std=std, limit=limit)
    bound = limit * std / init.truncation_scale(limit)
    assert float(np.abs(weights.numpy()).max()) <= bound + 1e-9


def test_truncated_normal_can_be_asked_for_a_constant() -> None:
    weights = init.truncated_normal((4, 4), nb.random.Generator(seed=0), std=0.0)
    np.testing.assert_allclose(weights.numpy(), np.zeros((4, 4)))


def test_truncated_normal_refuses_a_negative_standard_deviation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        init.truncated_normal((2, 2), nb.random.Generator(seed=0), std=-1.0)


# --------------------------------------------------------------------------- #
# the reason any of it matters
# --------------------------------------------------------------------------- #


def test_a_deep_stack_keeps_its_variance_under_kaiming() -> None:
    """The exponential in the depth, and the initialiser that removes it.

    Twenty ReLU layers of width 256. Under Kaiming the activation variance is
    flat; under an initialiser off by a factor of two per layer it would be off
    by 2^20.
    """
    g = nb.random.Generator(seed=13)
    x = g.normal((512, 256))
    scaled = x
    for _ in range(20):
        w = init.kaiming_normal((256, 256), g)
        scaled = nb.relu(scaled @ w)
    ratio = float(np.var(scaled.numpy())) / float(np.var(x.numpy()))
    assert 0.2 < ratio < 5.0


def test_the_same_stack_collapses_under_a_too_small_initialiser() -> None:
    g = nb.random.Generator(seed=13)
    x = g.normal((512, 256))
    scaled = x
    for _ in range(20):
        w = init.xavier_normal((256, 256), g, gain=0.7)
        scaled = nb.relu(scaled @ w)
    assert float(np.var(scaled.numpy())) / float(np.var(x.numpy())) < 1e-6
