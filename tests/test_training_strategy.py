from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.train_models import (
    _checkpoint_is_better,
    _classification_metrics,
    _initial_report,
    _make_optimizer,
    _selection_score,
    _set_backbone_trainable,
    _stratified_image_split,
)


def test_classification_metrics_track_macro_performance_not_accuracy_alone() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 0, 1])

    metrics = _classification_metrics(torch, logits, targets, multilabel=False)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] < metrics["accuracy"]
    assert metrics["per_class"]["1"]["recall"] == 0.0


def test_multilabel_metrics_include_per_class_precision_recall_and_f1() -> None:
    logits = torch.tensor(
        [
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    metrics = _classification_metrics(torch, logits, targets, multilabel=True)

    assert metrics["macro_precision"] == pytest.approx(0.75)
    assert metrics["macro_recall"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(5.0 / 6.0)
    assert metrics["per_class"]["0"]["precision"] == pytest.approx(0.5)


def test_checkpoint_selection_prefers_macro_f1_then_validation_loss() -> None:
    assert _selection_score({"macro_f1": 0.8}) == 0.8
    with pytest.raises(ValueError, match="macro_f1"):
        _selection_score({"accuracy": 1.0})

    assert _checkpoint_is_better(0.8, 0.9, 0.7, 0.1)
    assert _checkpoint_is_better(0.8, 0.1, 0.8, 0.2)
    assert not _checkpoint_is_better(0.7, 0.01, 0.8, 0.2)


def test_backbone_warmup_and_optimizer_use_lower_backbone_learning_rate() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(nn.Linear(4, 4))
            self.classifier = nn.Sequential(nn.Linear(4, 2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(inputs))

    model = Model()
    _set_backbone_trainable(model, False)

    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    _set_backbone_trainable(model, True)
    optimizer = _make_optimizer(
        torch,
        model,
        SimpleNamespace(
            learning_rate=3e-4,
            backbone_lr_scale=0.1,
            weight_decay=1e-4,
        ),
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(3e-4)


def test_task_subset_preserves_other_saved_report_entries(tmp_path) -> None:
    report_path = tmp_path / "training_report.json"
    report_path.write_text(
        json.dumps({"models": {"age": {"validation_macro_f1": 0.4}}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(seed=42, pretrained=True)

    report = _initial_report(report_path, args, "cuda")

    assert report["models"] == {"age": {"validation_macro_f1": 0.4}}
    assert report["device"] == "cuda"


def test_image_validation_split_is_stratified_reproducible_and_keeps_train_examples() -> None:
    targets = [0] * 10 + [1] * 5 + [2] * 2

    train, validation = _stratified_image_split(
        targets, ["a", "b", "c"], validation_fraction=0.2, seed=17
    )
    train_again, validation_again = _stratified_image_split(
        targets, ["a", "b", "c"], validation_fraction=0.2, seed=17
    )

    assert train == train_again
    assert validation == validation_again
    assert set(train).isdisjoint(validation)
    assert {targets[index] for index in train} == {0, 1, 2}
    assert {targets[index] for index in validation} == {0, 1, 2}
