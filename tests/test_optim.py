"""Optimisers, checked against a problem whose optimum is known in closed form.

The convergence tests all minimise the same least-squares objective, whose
minimiser `np.linalg.lstsq` computes exactly. That is the point: "the loss went
down" is compatible with an optimiser that has a factor of two in the wrong
place, and "the parameters arrived at the known optimum" is not.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import nabla as nb
from conftest import sample
from nabla import optim
from nabla.errors import OptimizerError

# A well-conditioned overdetermined system: 40 equations, 5 unknowns.
DESIGN = sample(40, 5, seed=91)
TRUTH = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
OBSERVED = DESIGN @ TRUTH + 0.01 * sample(40, seed=92)
OPTIMUM = np.linalg.lstsq(DESIGN, OBSERVED, rcond=None)[0]


def least_squares(w: nb.Tensor) -> nb.Tensor:
    """Half the mean squared residual, as a scalar tensor."""
    residual = nb.Tensor(DESIGN) @ w - nb.Tensor(OBSERVED)
    return (residual * residual).mean()


def run(
    make: Callable[[list[nb.Tensor]], optim.Optimizer],
    steps: int,
    schedule: optim.Schedule | None = None,
) -> np.ndarray:
    """Minimise `least_squares` from the same start and return the final parameters."""
    w = nb.Tensor(np.zeros(5), requires_grad=True)
    opt = make([w])
    for step in range(steps):
        if schedule is not None:
            opt.set_lr(schedule(step))
        opt.zero_grad()
        least_squares(w).backward()
        opt.step()
    return w.numpy()


#: A badly conditioned quadratic: one direction a hundred times shallower than
#: the other. The least-squares problem above is well conditioned, so it cannot
#: show what momentum is for - on it, plain descent is already optimal.
CURVATURE = np.array([1.0, 0.01])


def run_ill_conditioned(make: Callable[[list[nb.Tensor]], optim.Optimizer], steps: int) -> float:
    """Descend `sum(c_i w_i^2)` from `(1, 1)` and return the distance left to the origin."""
    w = nb.Tensor(np.ones(2), requires_grad=True)
    opt = make([w])
    for _ in range(steps):
        opt.zero_grad()
        (nb.Tensor(CURVATURE) * w * w).sum().backward()
        opt.step()
    return float(np.linalg.norm(w.numpy()))


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #


def test_sgd_converges_to_the_known_optimum() -> None:
    final = run(lambda p: optim.SGD(p, lr=0.05), steps=4000)
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-4, atol=1e-4)


def test_momentum_is_worth_orders_of_magnitude_on_a_stiff_problem() -> None:
    """What momentum is actually for: the shallow direction, not the steep one."""
    plain = run_ill_conditioned(lambda p: optim.SGD(p, lr=0.5), steps=200)
    heavy = run_ill_conditioned(lambda p: optim.SGD(p, lr=0.5, momentum=0.9), steps=200)
    assert plain > 1e-2
    assert heavy < 1e-4


def test_nesterov_beats_plain_momentum_on_the_same_problem() -> None:
    heavy = run_ill_conditioned(lambda p: optim.SGD(p, lr=0.5, momentum=0.9), steps=200)
    looked_ahead = run_ill_conditioned(
        lambda p: optim.SGD(p, lr=0.5, momentum=0.9, nesterov=True), steps=200
    )
    assert looked_ahead < heavy


def test_nesterov_converges_to_the_known_optimum() -> None:
    final = run(lambda p: optim.SGD(p, lr=0.02, momentum=0.9, nesterov=True), steps=1500)
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-4, atol=1e-4)


def test_adam_converges_to_the_known_optimum() -> None:
    final = run(lambda p: optim.Adam(p, lr=0.05), steps=3000)
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-4, atol=1e-4)


def test_adamw_with_no_decay_converges_to_the_known_optimum() -> None:
    final = run(lambda p: optim.AdamW(p, lr=0.05, weight_decay=0.0), steps=3000)
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-4, atol=1e-4)


# RMSProp needs a decaying learning rate to *arrive* rather than to orbit. Its
# step is `lr * g / sqrt(<g^2>)`, and near the optimum the numerator and the
# denominator vanish together, so the ratio stays of order 1 and the parameters
# oscillate with an amplitude of about `lr`. That is a property of the algorithm,
# not of this implementation, and the schedule is how every real run handles it.
RMSPROP_DECAY = optim.schedule.step_decay(0.01, step_size=500, gamma=0.5)


def test_rmsprop_converges_to_the_known_optimum() -> None:
    final = run(lambda p: optim.RMSProp(p, lr=0.01), steps=4000, schedule=RMSPROP_DECAY)
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-3, atol=1e-3)


def test_centred_rmsprop_converges_too() -> None:
    final = run(
        lambda p: optim.RMSProp(p, lr=0.01, centred=True), steps=4000, schedule=RMSPROP_DECAY
    )
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-3, atol=1e-3)


def test_rmsprop_with_momentum_converges_too() -> None:
    final = run(
        lambda p: optim.RMSProp(p, lr=0.002, momentum=0.9),
        steps=4000,
        schedule=optim.schedule.step_decay(0.002, step_size=500, gamma=0.5),
    )
    np.testing.assert_allclose(final, OPTIMUM, rtol=1e-3, atol=1e-3)


def test_rmsprop_orbits_without_a_schedule() -> None:
    """The claim above, measured: a constant learning rate leaves an `lr`-sized residue."""
    final = run(lambda p: optim.RMSProp(p, lr=0.01), steps=4000)
    residue = float(np.abs(final - OPTIMUM).max())
    assert 1e-3 < residue < 0.05


# --------------------------------------------------------------------------- #
# the AdamW claim
# --------------------------------------------------------------------------- #
#
# With no data gradient at all, weight decay is the only thing moving the
# parameters, so the difference between the two forms is visible in isolation.


def decay_only(make: Callable[[list[nb.Tensor]], optim.Optimizer], steps: int) -> np.ndarray:
    """Step an optimiser with a zero data gradient, starting from two very different scales."""
    w = nb.Tensor(np.array([1.0, 100.0]), requires_grad=True)
    opt = make([w])
    for _ in range(steps):
        opt.zero_grad()
        (w * 0.0).sum().backward()
        opt.step()
    return w.numpy()


def test_adamw_decays_both_scales_by_the_same_factor() -> None:
    final = decay_only(lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.1), steps=200)
    expected = (1.0 - 0.01 * 0.1) ** 200
    np.testing.assert_allclose(final, np.array([1.0, 100.0]) * expected, rtol=1e-6)
    assert final[1] / final[0] == pytest.approx(100.0, rel=1e-6)


def test_adam_plus_l2_decays_both_scales_by_the_same_absolute_amount() -> None:
    """The failure the AdamW paper describes: the decay stops being proportional.

    Once the decay term is the only gradient, `m` and `v` are both built from it
    and the ratio `m_hat / sqrt(v_hat)` is +-1 whatever the parameter's
    magnitude. Every coordinate therefore moves by about `lr` per step, so after
    50 steps both have travelled about 0.5 - one of them half of its length, the
    other half a percent of it.

    The tolerance is 10% rather than something tighter because the two
    coordinates' gradients shrink at different rates, so their bias-corrected
    ratios differ a little in the transient; the measured travels are 0.537 and
    0.500. What matters is that they are the same *number*, while the fractional
    change it represents differs by a factor of a hundred.
    """
    start = np.array([1.0, 100.0])
    final = decay_only(lambda p: optim.Adam(p, lr=0.01, weight_decay=0.1), steps=50)
    travelled = start - final
    np.testing.assert_allclose(travelled, np.full(2, 0.5), rtol=0.1)
    fractional = travelled / start
    assert fractional[0] / fractional[1] > 50.0
    assert final[1] / final[0] > 150.0  # AdamW would keep this at exactly 100


def test_doubling_the_coefficient_doubles_adamws_decay_rate_and_not_adams() -> None:
    """The coefficient controls the regularisation in one of the two, and not the other."""
    single_w = decay_only(lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.1), steps=100)
    double_w = decay_only(lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.2), steps=100)

    # AdamW's decay is multiplicative, so it is the *rate* - the logarithm - that
    # doubles, not the amount shrunk. 1 - (1 - lr*lambda)^n is not linear in
    # lambda, and a test asserting it was would be asserting the wrong law.
    assert single_w[0] == pytest.approx((1.0 - 0.01 * 0.1) ** 100, rel=1e-9)
    assert double_w[0] == pytest.approx((1.0 - 0.01 * 0.2) ** 100, rel=1e-9)
    assert np.log(double_w[0]) / np.log(single_w[0]) == pytest.approx(2.0, rel=1e-3)

    # Adam's, by contrast, is the same trajectory for both coefficients: the
    # decay cancels out of m / sqrt(v) and only its sign survives. The agreement
    # is to 1e-7 rather than exact because `eps` sits in the denominator and does
    # not cancel; doubling the coefficient changes the answer by less than the
    # epsilon does, which is another way of saying the coefficient does nothing.
    single_a = decay_only(lambda p: optim.Adam(p, lr=0.01, weight_decay=0.1), steps=100)
    double_a = decay_only(lambda p: optim.Adam(p, lr=0.01, weight_decay=0.2), steps=100)
    np.testing.assert_allclose(single_a, double_a, rtol=1e-5)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make",
    [
        lambda p: optim.SGD(p, lr=0.05, momentum=0.9),
        lambda p: optim.Adam(p, lr=0.05),
        lambda p: optim.AdamW(p, lr=0.05),
        lambda p: optim.RMSProp(p, lr=0.01, momentum=0.5, centred=True),
    ],
    ids=["sgd", "adam", "adamw", "rmsprop"],
)
def test_state_dict_round_trip_resumes_the_identical_trajectory(
    make: Callable[[list[nb.Tensor]], optim.Optimizer],
) -> None:
    reference = nb.Tensor(np.zeros(5), requires_grad=True)
    opt = make([reference])
    for _ in range(20):
        opt.zero_grad()
        least_squares(reference).backward()
        opt.step()
    snapshot = opt.state_dict()
    halfway = reference.numpy()
    for _ in range(20):
        opt.zero_grad()
        least_squares(reference).backward()
        opt.step()

    resumed = nb.Tensor(halfway, requires_grad=True)
    fresh = make([resumed])
    fresh.load_state_dict(snapshot)
    assert fresh.step_count == 20
    for _ in range(20):
        fresh.zero_grad()
        least_squares(resumed).backward()
        fresh.step()

    np.testing.assert_allclose(resumed.numpy(), reference.numpy(), rtol=1e-12)


def test_state_dict_is_a_copy_not_a_view() -> None:
    w = nb.Tensor(np.zeros(5), requires_grad=True)
    opt = optim.Adam([w], lr=0.1)
    opt.zero_grad()
    least_squares(w).backward()
    opt.step()
    snapshot = opt.state_dict()
    before = snapshot["state"][0]["exp_avg"].copy()
    opt.step()
    np.testing.assert_allclose(snapshot["state"][0]["exp_avg"], before)


def test_loading_the_wrong_optimisers_state_is_refused() -> None:
    w = nb.Tensor(np.zeros(5), requires_grad=True)
    with pytest.raises(OptimizerError, match="this is a SGD"):
        optim.SGD([w], lr=0.1).load_state_dict(optim.Adam([w], lr=0.1).state_dict())


def test_loading_a_state_for_a_different_model_is_refused() -> None:
    one = nb.Tensor(np.zeros(5), requires_grad=True)
    two = nb.Tensor(np.zeros(5), requires_grad=True)
    saved = optim.Adam([one, two], lr=0.1).state_dict()
    with pytest.raises(OptimizerError, match="describes 2 parameters"):
        optim.Adam([one], lr=0.1).load_state_dict(saved)


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #


def test_a_parameter_with_no_gradient_is_left_alone() -> None:
    used = nb.Tensor(np.ones(3), requires_grad=True)
    frozen = nb.Tensor(np.ones(3), requires_grad=True)
    opt = optim.SGD([used, frozen], lr=0.1, weight_decay=0.5, momentum=0.9)
    used.sum().backward()
    opt.step()
    np.testing.assert_allclose(frozen.numpy(), np.ones(3))
    assert not np.allclose(used.numpy(), np.ones(3))


def test_duplicate_parameters_are_refused() -> None:
    w = nb.Tensor(np.zeros(3), requires_grad=True)
    with pytest.raises(OptimizerError, match="appears twice"):
        optim.SGD([w, w], lr=0.1)


def test_an_empty_parameter_list_is_refused() -> None:
    with pytest.raises(OptimizerError, match="at least one parameter"):
        optim.SGD([], lr=0.1)


def test_a_computed_tensor_cannot_be_optimised() -> None:
    w = nb.Tensor(np.zeros(3), requires_grad=True)
    with pytest.raises(OptimizerError, match="only update leaf tensors"):
        optim.SGD([w * 2.0], lr=0.1)


def test_a_negative_learning_rate_is_refused() -> None:
    w = nb.Tensor(np.zeros(3), requires_grad=True)
    with pytest.raises(OptimizerError, match="lr must be positive"):
        optim.SGD([w], lr=-0.1)


def test_beta_outside_the_unit_interval_is_refused() -> None:
    w = nb.Tensor(np.zeros(3), requires_grad=True)
    with pytest.raises(OptimizerError, match=r"beta2 must be in \[0, 1\)"):
        optim.Adam([w], lr=0.1, beta2=1.0)


def test_nesterov_needs_momentum() -> None:
    w = nb.Tensor(np.zeros(3), requires_grad=True)
    with pytest.raises(OptimizerError, match="nesterov momentum needs"):
        optim.SGD([w], lr=0.1, nesterov=True)


def test_set_lr_accepts_the_zero_a_cosine_schedule_ends_at() -> None:
    w = nb.Tensor(np.ones(3), requires_grad=True)
    opt = optim.Adam([w], lr=0.1)
    opt.set_lr(0.0)
    w.sum().backward()
    opt.step()
    np.testing.assert_allclose(w.numpy(), np.ones(3))


def test_repr_names_the_hyperparameters() -> None:
    w = nb.Tensor(np.zeros(3), requires_grad=True)
    text = repr(optim.AdamW([w], lr=0.001))
    assert "AdamW(" in text
    assert "weight_decay=0.01" in text
    assert "parameters=1" in text


def test_bias_correction_makes_the_first_step_the_full_learning_rate() -> None:
    """Without the correction the first Adam step is about a third of `lr`."""
    w = nb.Tensor(np.array([10.0]), requires_grad=True)
    opt = optim.Adam([w], lr=0.1)
    (w * w).sum().backward()
    opt.step()
    assert 10.0 - float(w.item()) == pytest.approx(0.1, rel=1e-6)
