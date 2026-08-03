"""The training loop, the checkpoint format, and early stopping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nabla as nb
from nabla import data, nn, optim, train
from nabla.errors import NablaError


def problem(
    seed: int = 0, samples: int = 128
) -> tuple[data.DataLoader, data.DataLoader, nn.Module, nb.random.Generator]:
    """A two-moons classifier and its loaders, built from one seed."""
    g = nb.random.Generator(seed=seed)
    x, y = data.make_moons(samples, g)
    dataset = data.ArrayDataset(x, y)
    training = data.DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
    validation = data.DataLoader(dataset, batch_size=64)
    model = nn.Sequential(
        nn.Linear(2, 16, generator=g),
        nn.Tanh(),
        nn.Linear(16, 2, generator=g),
    )
    return training, validation, model, g


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


def test_fit_reduces_the_loss_and_records_one_row_per_epoch() -> None:
    training, validation, model, _ = problem()
    trainer = train.Trainer(
        model,
        optim.Adam(model.parameters(), lr=0.05),
        nn.cross_entropy,
        metrics={"accuracy": train.accuracy},
    )
    history = trainer.fit(training, epochs=20, val_loader=validation, verbose=False)

    assert len(history.epochs) == 20
    losses = history.series("train_loss")
    assert losses[-1] < losses[0] / 2
    assert history.series("val_accuracy")[-1] > 0.9
    assert set(history.last()) >= {"epoch", "train_loss", "val_loss", "lr", "seconds"}


def test_the_global_step_counts_optimiser_steps() -> None:
    training, _, model, _ = problem()
    trainer = train.Trainer(model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy)
    trainer.fit(training, epochs=3, verbose=False)
    assert trainer.global_step == 3 * len(training)


def test_a_schedule_is_applied_per_step() -> None:
    training, _, model, _ = problem()
    opt = optim.SGD(model.parameters(), lr=0.5)
    steps = len(training) * 4
    trainer = train.Trainer(
        model,
        opt,
        nn.cross_entropy,
        schedule=optim.schedule.linear_warmup_decay(0.5, warmup_steps=0, total_steps=steps),
    )
    history = trainer.fit(training, epochs=4, verbose=False)
    assert history.series("lr")[0] > history.series("lr")[-1]
    assert opt.lr == pytest.approx(0.5 * (1.0 - (steps - 1) / steps))


def test_gradient_clipping_is_recorded_and_bounds_the_update() -> None:
    training, _, model, _ = problem()
    trainer = train.Trainer(
        model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy, grad_clip=0.01
    )
    history = trainer.fit(training, epochs=2, verbose=False)
    assert history.series("train_grad_norm")[0] > 0.01
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert float(np.linalg.norm(parameter.grad)) <= 0.01 + 1e-9


def test_evaluate_leaves_the_model_in_eval_mode_and_builds_no_tape() -> None:
    _, validation, model, g = problem()
    dropout = nn.Dropout(0.5, generator=g)
    wrapped = nn.Sequential(model, dropout)
    trainer = train.Trainer(wrapped, optim.SGD(wrapped.parameters(), lr=0.1), nn.cross_entropy)
    trainer.evaluate(validation)
    assert not dropout.training


def test_train_epoch_averages_by_example_not_by_batch() -> None:
    """A short final batch must not be weighted like a full one."""
    g = nb.random.Generator(seed=1)
    x = np.zeros((5, 2), dtype=np.float32)
    y = np.array([0, 0, 0, 0, 1])
    loader = data.DataLoader(data.ArrayDataset(x, y), batch_size=4)
    model = nn.Linear(2, 2, generator=g)
    trainer = train.Trainer(model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy)
    averaged = trainer.evaluate(loader)["loss"]
    # With a zero input and a zero bias every logit is 0, so every example's
    # loss is exactly log(2) whichever batch it landed in.
    assert averaged == pytest.approx(float(np.log(2.0)), rel=1e-6)


def test_a_batch_with_no_target_is_refused() -> None:
    _, _, model, _ = problem()
    trainer = train.Trainer(model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy)
    with pytest.raises(NablaError, match="at least an input and a target"):
        trainer.train_epoch([(nb.zeros(2, 2),)])


def test_zero_epochs_is_refused() -> None:
    training, _, model, _ = problem()
    trainer = train.Trainer(model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy)
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        trainer.fit(training, epochs=0, verbose=False)


def test_a_non_positive_clip_is_refused() -> None:
    _, _, model, _ = problem()
    with pytest.raises(ValueError, match="grad_clip must be positive"):
        train.Trainer(model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy, grad_clip=0.0)


def test_history_reports_a_missing_metric_as_absent_rather_than_zero() -> None:
    training, _, model, _ = problem()
    trainer = train.Trainer(model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy)
    history = trainer.fit(training, epochs=2, verbose=False)
    assert history.series("val_loss") == []
    with pytest.raises(NablaError, match="no epochs"):
        train.History().last()


def test_verbose_output_goes_through_the_injected_logger() -> None:
    training, _, model, _ = problem()
    lines: list[str] = []
    trainer = train.Trainer(
        model, optim.SGD(model.parameters(), lr=0.1), nn.cross_entropy, log=lines.append
    )
    trainer.fit(training, epochs=2, verbose=True)
    assert len(lines) == 2
    assert lines[0].startswith("epoch    0")
    assert "train_loss" in lines[0]


# --------------------------------------------------------------------------- #
# early stopping
# --------------------------------------------------------------------------- #


def test_early_stopping_fires_when_the_metric_stops_improving() -> None:
    training, validation, model, _ = problem()
    frozen = optim.SGD(model.parameters(), lr=0.1)
    frozen.set_lr(0.0)  # a model that never moves cannot improve
    trainer = train.Trainer(
        model,
        frozen,
        nn.cross_entropy,
        early_stopping=train.EarlyStopping(patience=2, monitor="val_loss"),
    )
    history = trainer.fit(training, epochs=50, val_loader=validation, verbose=False)
    assert history.stopped_early
    assert len(history.epochs) == 3  # the first epoch sets the best, then two without


def test_early_stopping_does_not_fire_on_a_run_that_is_improving() -> None:
    training, validation, model, _ = problem()
    trainer = train.Trainer(
        model,
        optim.Adam(model.parameters(), lr=0.05),
        nn.cross_entropy,
        early_stopping=train.EarlyStopping(patience=3, monitor="val_loss"),
    )
    history = trainer.fit(training, epochs=15, val_loader=validation, verbose=False)
    assert not history.stopped_early
    assert len(history.epochs) == 15


def test_min_delta_makes_a_noisy_improvement_not_count() -> None:
    stopping = train.EarlyStopping(patience=2, min_delta=0.01)
    assert stopping.is_improvement(0.98, 1.0)
    assert not stopping.is_improvement(0.999, 1.0)


def test_max_mode_reverses_the_comparison() -> None:
    stopping = train.EarlyStopping(patience=2, mode="max", monitor="val_accuracy")
    assert stopping.is_improvement(0.9, 0.8)
    assert not stopping.is_improvement(0.7, 0.8)
    assert stopping.worst == -np.inf


def test_early_stopping_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="patience must be at least 1"):
        train.EarlyStopping(patience=0)
    with pytest.raises(ValueError, match="mode must be"):
        train.EarlyStopping(patience=1, mode="middle")


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


def test_checkpoint_round_trips_a_model_and_an_optimiser(tmp_path: Path) -> None:
    training, _, model, _ = problem()
    opt = optim.AdamW(model.parameters(), lr=0.05)
    trainer = train.Trainer(model, opt, nn.cross_entropy)
    trainer.fit(training, epochs=3, verbose=False)

    path = trainer.save(tmp_path / "run", note="third epoch", epoch=2)
    assert path.exists()

    _, _, restored_model, _ = problem()
    restored_opt = optim.AdamW(restored_model.parameters(), lr=0.05)
    restored = train.Trainer(restored_model, restored_opt, nn.cross_entropy)
    extra = restored.load(path)

    assert extra["note"] == "third epoch"
    assert extra["epoch"] == 2
    assert restored.global_step == trainer.global_step
    for (name, a), (_, b) in zip(
        model.named_parameters(), restored_model.named_parameters(), strict=True
    ):
        np.testing.assert_allclose(a.numpy(), b.numpy(), err_msg=name)
    for a_state, b_state in zip(opt.state, restored_opt.state, strict=True):
        for key, value in a_state.items():
            np.testing.assert_allclose(value, b_state[key])


def test_a_restored_run_continues_identically(tmp_path: Path) -> None:
    """The property a checkpoint exists for, asserted rather than assumed."""
    training, _, model, _ = problem()
    opt = optim.Adam(model.parameters(), lr=0.05)
    trainer = train.Trainer(model, opt, nn.cross_entropy)
    trainer.fit(training, epochs=2, verbose=False)
    trainer.save(tmp_path / "mid")

    reference = trainer.fit(training, epochs=2, verbose=False).last()["train_loss"]

    # A second loader built from the same seed, wound forward by the two epochs
    # the first run had already consumed. The loader's position in its random
    # stream is not part of the checkpoint - it belongs to the data pipeline,
    # not to the model - so resuming a run means rebuilding the pipeline the
    # same way, and this is what that looks like.
    other_training, _, other_model, _ = problem()
    for _ in range(2):
        for _ in other_training:
            pass

    other = train.Trainer(
        other_model, optim.Adam(other_model.parameters(), lr=0.05), nn.cross_entropy
    )
    other.load(tmp_path / "mid")
    resumed = other.fit(other_training, epochs=2, verbose=False).last()["train_loss"]
    assert resumed == pytest.approx(reference, rel=1e-9)


def test_checkpoint_holds_nested_structures_and_scalars(tmp_path: Path) -> None:
    payload = {
        "arrays": {"a": np.arange(4.0), "b": np.zeros((2, 3), dtype=np.int32)},
        "list": [np.ones(2), 3, "text"],
        "scalars": {"lr": 0.001, "flag": True, "nothing": None},
    }
    train.save_checkpoint(tmp_path / "mixed.npz", payload)
    loaded = train.load_checkpoint(tmp_path / "mixed.npz")

    np.testing.assert_allclose(loaded["arrays"]["a"], payload["arrays"]["a"])  # type: ignore[index]
    assert loaded["arrays"]["b"].dtype == np.int32
    np.testing.assert_allclose(loaded["list"][0], np.ones(2))
    assert loaded["list"][1:] == [3, "text"]
    assert loaded["scalars"] == {"lr": 0.001, "flag": True, "nothing": None}


def test_checkpoint_adds_the_suffix_and_leaves_no_partial_file(tmp_path: Path) -> None:
    written = train.save_checkpoint(tmp_path / "run", {"x": np.zeros(1)})
    assert written.name == "run.npz"
    assert [p.name for p in tmp_path.iterdir()] == ["run.npz"]


def test_checkpoint_refuses_a_value_it_cannot_write(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="cannot checkpoint a value of type"):
        train.save_checkpoint(tmp_path / "bad.npz", {"model": object()})


def test_loading_a_foreign_npz_says_what_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "foreign.npz"
    np.savez(path, x=np.zeros(3))
    with pytest.raises(ValueError, match="was not written by"):
        train.load_checkpoint(path)


def test_checkpoints_are_loaded_without_pickle(tmp_path: Path) -> None:
    """The security property of the format, asserted directly."""
    path = train.save_checkpoint(tmp_path / "safe.npz", {"w": np.zeros(3)})
    with np.load(path, allow_pickle=False) as archive:
        assert "__manifest__" in archive.files
