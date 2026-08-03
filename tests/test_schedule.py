"""Learning-rate schedules, which are pure functions and therefore easy to pin down."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from nabla.optim import schedule


def test_constant_never_changes() -> None:
    s = schedule.constant(0.3)
    assert [s(t) for t in (0, 1, 10_000)] == [0.3, 0.3, 0.3]


def test_step_decay_drops_at_the_boundaries() -> None:
    s = schedule.step_decay(1.0, step_size=10, gamma=0.5)
    assert [s(t) for t in (0, 9, 10, 19, 20)] == [1.0, 1.0, 0.5, 0.5, 0.25]


def test_step_decay_refuses_a_zero_period() -> None:
    with pytest.raises(ValueError, match="step_size must be positive"):
        schedule.step_decay(1.0, step_size=0)


@pytest.mark.parametrize(
    "make", [schedule.cosine_with_warmup, schedule.linear_warmup_decay], ids=["cosine", "linear"]
)
def test_warmup_is_linear_from_zero_to_the_peak(make: object) -> None:
    s = make(0.5, warmup_steps=100, total_steps=1000)  # type: ignore[operator]
    assert s(0) == 0.0
    assert s(50) == pytest.approx(0.25)
    assert s(100) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "make", [schedule.cosine_with_warmup, schedule.linear_warmup_decay], ids=["cosine", "linear"]
)
def test_the_floor_is_reached_and_held(make: object) -> None:
    s = make(0.5, warmup_steps=10, total_steps=110, min_lr=0.05)  # type: ignore[operator]
    assert s(110) == pytest.approx(0.05)
    assert s(10_000) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "make", [schedule.cosine_with_warmup, schedule.linear_warmup_decay], ids=["cosine", "linear"]
)
def test_the_decay_is_monotone(make: object) -> None:
    s = make(1.0, warmup_steps=10, total_steps=210)  # type: ignore[operator]
    values = [s(t) for t in range(10, 211)]
    assert all(a >= b for a, b in pairwise(values))


def test_cosine_follows_the_half_cosine() -> None:
    s = schedule.cosine_with_warmup(1.0, warmup_steps=0, total_steps=100, min_lr=0.0)
    for step in (0, 25, 50, 75):
        expected = 0.5 * (1.0 + math.cos(math.pi * step / 100))
        assert s(step) == pytest.approx(expected)


def test_cosine_spends_longer_at_a_high_rate_than_the_linear_one() -> None:
    """The reason to prefer one over the other, stated as a measurement."""
    cosine = schedule.cosine_with_warmup(1.0, warmup_steps=0, total_steps=100)
    linear = schedule.linear_warmup_decay(1.0, warmup_steps=0, total_steps=100)
    assert cosine(25) > linear(25)
    assert cosine(75) < linear(75)
    assert cosine(50) == pytest.approx(linear(50))


def test_a_warmup_longer_than_the_run_is_refused() -> None:
    with pytest.raises(ValueError, match="shorter than warmup_steps"):
        schedule.cosine_with_warmup(1.0, warmup_steps=100, total_steps=10)


def test_a_negative_warmup_is_refused() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        schedule.linear_warmup_decay(1.0, warmup_steps=-1, total_steps=10)


def test_a_zero_length_decay_holds_the_floor() -> None:
    s = schedule.cosine_with_warmup(1.0, warmup_steps=10, total_steps=10, min_lr=0.2)
    assert s(5) == pytest.approx(0.5)
    assert s(10) == pytest.approx(0.2)
