#!/usr/bin/env python3
"""Train age, emotion, drowsiness, and distraction models from data/raw."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"
MODELS_ROOT = ROOT / "models"
IMAGE_SIZE = 224
RESIZE_SHORT_EDGE = 256
DEFAULT_TASKS = ("age", "emotion", "drowsiness", "distraction")
IMAGE_TASKS = {
    "age": ("age_detection", "Age band"),
    "emotion": ("emotion_detection", "Emotion"),
}
DMD_TASKS = {
    "drowsiness": ("*_rgb_ann_drowsiness.json", ("eyes_state/", "yawning/", "blinks/")),
    "distraction": ("*_rgb_ann_distraction.json", ("driver_actions/",)),
}


@dataclass(frozen=True)
class DMDFrameSample:
    image_path: Path
    labels: frozenset[str]
    recording: str
    frame_index: int


class DMDFrameDataset:
    def __init__(
        self,
        samples: list[DMDFrameSample],
        class_names: list[str],
        transform: Any,
    ) -> None:
        self.samples = samples
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
        target = torch.zeros(len(self.class_to_index), dtype=torch.float32)
        for label in sample.labels:
            target[self.class_to_index[label]] = 1.0
        return self.transform(image), target


def parse_dmd_frame_labels(document: dict[str, Any], task: str) -> dict[int, set[str]]:
    """Convert OpenLABEL action intervals into frame-indexed labels."""
    if task not in DMD_TASKS:
        raise ValueError(f"Unknown DMD task: {task}")
    try:
        openlabel = document["openlabel"]
        actions = openlabel["actions"]
    except (KeyError, TypeError) as exc:
        raise ValueError("DMD annotation is missing openlabel.actions") from exc

    prefixes = DMD_TASKS[task][1]
    labels_by_frame: dict[int, set[str]] = {}
    for action in actions.values():
        label = action.get("type")
        if not isinstance(label, str) or not label.startswith(prefixes):
            continue
        for interval in action.get("frame_intervals", []):
            try:
                first = int(interval["frame_start"])
                last = int(interval["frame_end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame interval for DMD label {label!r}") from exc
            if first < 0 or last < first:
                raise ValueError(f"Invalid frame interval {first}..{last} for {label!r}")
            for frame_index in range(first, last + 1):
                labels_by_frame.setdefault(frame_index, set()).add(label)
    return labels_by_frame


def temporal_split(
    samples: list[DMDFrameSample], validation_fraction: float = 0.2
) -> tuple[list[DMDFrameSample], list[DMDFrameSample]]:
    """Split each recording chronologically and keep at least one train sample per label."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    by_recording: dict[str, list[DMDFrameSample]] = {}
    for sample in samples:
        by_recording.setdefault(sample.recording, []).append(sample)

    train: list[DMDFrameSample] = []
    validation: list[DMDFrameSample] = []
    for recording_samples in by_recording.values():
        ordered = sorted(recording_samples, key=lambda item: item.frame_index)
        if len(ordered) < 2:
            train.extend(ordered)
            continue
        split_at = min(len(ordered) - 1, max(1, int(len(ordered) * (1 - validation_fraction))))
        train.extend(ordered[:split_at])
        validation.extend(ordered[split_at:])

    train_labels = {label for sample in train for label in sample.labels}
    for sample in validation[:]:
        missing = sample.labels - train_labels
        if missing:
            train.append(sample)
            validation.remove(sample)
            train_labels.update(sample.labels)
    if not train:
        raise ValueError("DMD annotations produced no training frames")
    if not validation:
        raise ValueError(
            "DMD annotations need at least two labeled frames per task for temporal validation"
        )
    return train, validation


def extract_dmd_face_frames(dmd_root: Path, cache_root: Path, task: str) -> list[DMDFrameSample]:
    """Extract every labeled frame from DMD face-camera videos only."""
    if task not in DMD_TASKS:
        raise ValueError(f"Unknown DMD task: {task}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("DMD extraction requires opencv-python") from exc

    annotation_pattern = DMD_TASKS[task][0]
    annotations = sorted(dmd_root.rglob(annotation_pattern))
    if not annotations:
        raise FileNotFoundError(
            f"No DMD {task} annotations matching {annotation_pattern} under {dmd_root}"
        )

    samples: list[DMDFrameSample] = []
    root_resolved = dmd_root.resolve()
    for annotation_path in annotations:
        try:
            document = json.loads(annotation_path.read_text(encoding="utf-8"))
            openlabel = document["openlabel"]
            face_uri = openlabel["streams"]["face_camera"]["uri"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid DMD face-camera annotation: {annotation_path}") from exc

        video_path = (dmd_root / face_uri).resolve()
        try:
            relative_video = video_path.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"DMD face-camera URI escapes dataset root: {face_uri}") from exc
        if not video_path.is_file():
            raise FileNotFoundError(f"DMD face-camera video missing: {video_path}")

        labels_by_frame = parse_dmd_frame_labels(document, task)
        if not labels_by_frame:
            raise ValueError(f"No {task} frame labels in {annotation_path}")
        recording = relative_video.parent.as_posix()
        frame_dir = cache_root / task / relative_video.parent / relative_video.stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to decode DMD face-camera video: {video_path}")
        try:
            frame_index = 0
            last_labeled_frame = max(labels_by_frame)
            while frame_index <= last_labeled_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                labels = labels_by_frame.get(frame_index)
                if labels:
                    output_path = frame_dir / f"frame_{frame_index:07d}.jpg"
                    if not output_path.is_file():
                        height, width = frame.shape[:2]
                        scale = RESIZE_SHORT_EDGE / min(height, width)
                        resized = cv2.resize(
                            frame,
                            (round(width * scale), round(height * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                        if not cv2.imwrite(
                            str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92]
                        ):
                            raise OSError(f"Could not write cached DMD frame: {output_path}")
                    samples.append(
                        DMDFrameSample(
                            image_path=output_path,
                            labels=frozenset(labels),
                            recording=recording,
                            frame_index=frame_index,
                        )
                    )
                frame_index += 1
        finally:
            capture.release()
    if not samples:
        raise ValueError(f"No labeled DMD {task} frames were extracted")
    return samples


def _require_training_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, torchvision, Pillow, and opencv-python. "
            "Install the dependencies listed in training/requirements.txt."
        ) from exc
    return torch, (datasets, models, transforms), Image


def _make_model(models: Any, class_count: int, pretrained: bool) -> Any:
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    network = models.mobilenet_v3_large(weights=weights)
    final_layer = network.classifier[-1]
    network.classifier[-1] = type(final_layer)(final_layer.in_features, class_count)
    return network


def _make_transform(transforms: Any, training: bool) -> Any:
    if training:
        operations: list[Any] = [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        operations = [
            transforms.Resize(RESIZE_SHORT_EDGE),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


def _classification_metrics(
    torch: Any, logits: Any, targets: Any, multilabel: bool
) -> dict[str, Any]:
    if multilabel:
        predicted = logits.sigmoid() >= 0.5
        expected = targets >= 0.5
    else:
        predicted = torch.nn.functional.one_hot(
            logits.argmax(dim=1), num_classes=logits.shape[1]
        ).bool()
        expected = torch.nn.functional.one_hot(targets.long(), num_classes=logits.shape[1]).bool()

    true_positive = (predicted & expected).sum(dim=0).float()
    false_positive = (predicted & ~expected).sum(dim=0).float()
    false_negative = (~predicted & expected).sum(dim=0).float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    metrics: dict[str, Any] = {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "per_class": {
            str(index): {
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
            }
            for index in range(logits.shape[1])
        },
    }
    if not multilabel:
        metrics["accuracy"] = float((logits.argmax(dim=1) == targets).float().mean().item())
    return metrics


def _selection_score(metrics: dict[str, Any]) -> float:
    score = metrics.get("macro_f1")
    if not isinstance(score, (int, float)):
        raise ValueError("Validation metrics must include macro_f1")
    return float(score)


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        if key in metrics
    }


def _make_optimizer(torch: Any, model: Any, args: argparse.Namespace) -> Any:
    head_parameters = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    groups = [
        {"params": backbone_parameters, "lr": args.learning_rate * args.backbone_lr_scale},
        {"params": head_parameters, "lr": args.learning_rate},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _set_backbone_trainable(model: Any, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad_(trainable)


def _checkpoint_is_better(
    score: float,
    validation_loss: float,
    best_score: float,
    best_loss: float,
) -> bool:
    return score > best_score or (score == best_score and validation_loss < best_loss)


def _dataset_targets(dataset: Any) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(target) for _image, target in dataset.samples]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return [int(dataset.dataset.targets[index]) for index in dataset.indices]
    raise TypeError("Classification dataset does not expose sample targets")


def _stratified_image_split(
    targets: list[int],
    class_names: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index, class_name in enumerate(class_names):
        class_indices = [index for index, target in enumerate(targets) if target == class_index]
        if len(class_indices) < 2:
            raise ValueError(f"Class {class_name} needs at least two training examples")
        rng.shuffle(class_indices)
        validation_count = min(
            len(class_indices) - 1,
            max(1, round(len(class_indices) * validation_fraction)),
        )
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])
    return train_indices, validation_indices


def _save_bundle(
    torch: Any,
    model: Any,
    output_path: Path,
    *,
    task: str,
    classes: list[str],
    multilabel: bool,
    validation_score: float,
) -> None:
    bundle = {
        "format_version": 1,
        "task": task,
        "architecture": "mobilenet_v3_large",
        "input_size": IMAGE_SIZE,
        "classes": classes,
        "multilabel": multilabel,
        "validation_macro_f1": validation_score,
        "state_dict": model.state_dict(),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(bundle, temporary_path)
    os.replace(temporary_path, output_path)


def _run_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    multilabel: bool,
    optimizer: Any | None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    if training and not any(parameter.requires_grad for parameter in model.features.parameters()):
        model.features.eval()
    total_loss = 0.0
    total_count = 0
    metric_logits: list[Any] = []
    metric_targets: list[Any] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss_targets = targets.float() if multilabel else targets.long()
            loss = criterion(logits, loss_targets)
            if training:
                loss.backward()
                optimizer.step()
        batch_count = images.shape[0]
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
        metric_logits.append(logits.detach().cpu())
        metric_targets.append(targets.detach().cpu())
    if not total_count:
        raise ValueError("Training/validation split is empty")
    metrics = _classification_metrics(
        torch, torch.cat(metric_logits), torch.cat(metric_targets), multilabel
    )
    return total_loss / total_count, metrics


def _loader(
    torch: Any,
    dataset: Any,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: Any,
) -> Any:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=getattr(device, "type", "cpu") == "cuda",
        persistent_workers=workers > 0,
    )


def _train(
    *,
    task: str,
    classes: list[str],
    train_dataset: Any,
    validation_dataset: Any,
    multilabel: bool,
    args: argparse.Namespace,
    output_path: Path,
    split_description: str,
    test_dataset: Any | None = None,
) -> dict[str, Any]:
    torch, modules, _image = _require_training_dependencies()
    _datasets, models, _transforms = modules
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    initial_bundle = None
    if args.init_from_dir is not None:
        initial_path = args.init_from_dir / f"{task}.pt"
        if not initial_path.is_file():
            raise FileNotFoundError(f"Initial model bundle not found: {initial_path}")
        initial_bundle = torch.load(initial_path, map_location="cpu", weights_only=True)
        if not isinstance(initial_bundle, dict) or (
            initial_bundle.get("format_version") != 1
            or initial_bundle.get("task") != task
            or initial_bundle.get("architecture")
            not in {"mobilenet_v3_large", "mobilenet_v3_small"}
            or initial_bundle.get("classes") != classes
            or bool(initial_bundle.get("multilabel")) != multilabel
            or not isinstance(initial_bundle.get("state_dict"), dict)
        ):
            raise ValueError(f"Initial model bundle is incompatible: {initial_path}")
    model = _make_model(
        models,
        len(classes),
        pretrained=args.pretrained and initial_bundle is None,
    ).to(device)
    if initial_bundle is not None:
        try:
            model.load_state_dict(initial_bundle["state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Initial model weights do not match MobileNetV3-Large: {initial_path}"
            ) from exc
    if multilabel:
        positives = torch.zeros(len(classes), dtype=torch.float32)
        for sample in train_dataset.samples:
            for label in sample.labels:
                positives[classes.index(label)] += 1
        if (positives == 0).any():
            missing = [classes[i] for i, count in enumerate(positives) if count == 0]
            raise ValueError(f"Training split has no positive examples for: {missing}")
        negatives = len(train_dataset) - positives
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=(negatives / positives).clamp(min=1.0, max=50.0).to(device)
        )
    else:
        targets = _dataset_targets(train_dataset)
        counts = torch.bincount(torch.tensor(targets), minlength=len(classes)).float()
        if (counts == 0).any():
            raise ValueError(f"Training split is missing one or more {task} classes")
        class_weights = (counts.sum() / (len(classes) * counts)).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    train_loader = _loader(torch, train_dataset, args.batch_size, args.workers, True, device)
    validation_loader = _loader(
        torch, validation_dataset, args.batch_size, args.workers, False, device
    )

    warmup_epochs = (
        min(args.warmup_epochs, args.epochs - 1)
        if args.pretrained and initial_bundle is None
        else 0
    )
    if warmup_epochs:
        _set_backbone_trainable(model, False)
    optimizer = _make_optimizer(torch, model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.2,
        patience=args.lr_patience,
        min_lr=args.min_learning_rate,
    )
    best_loss = float("inf")
    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    if initial_bundle is not None:
        best_loss, best_metrics = _run_epoch(
            torch, model, validation_loader, criterion, device, multilabel, None
        )
        best_score = _selection_score(best_metrics)
        _save_bundle(
            torch,
            model,
            output_path,
            task=task,
            classes=classes,
            multilabel=multilabel,
            validation_score=best_score,
        )
        print(
            f"[{task}] initial checkpoint validation_loss={best_loss:.4f} "
            f"validation_macro_f1={best_score:.4f}",
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        if warmup_epochs and epoch == warmup_epochs + 1:
            _set_backbone_trainable(model, True)
            optimizer = _make_optimizer(torch, model, args)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.2,
                patience=args.lr_patience,
                min_lr=args.min_learning_rate,
            )
        train_loss, train_metrics = _run_epoch(
            torch, model, train_loader, criterion, device, multilabel, optimizer
        )
        validation_loss, validation_metrics = _run_epoch(
            torch, model, validation_loader, criterion, device, multilabel, None
        )
        validation_score = _selection_score(validation_metrics)
        scheduler.step(validation_score)
        print(
            f"[{task}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} "
            f"validation_macro_f1={validation_score:.4f} "
            f"train={_metric_summary(train_metrics)} "
            f"validation={_metric_summary(validation_metrics)}",
            flush=True,
        )
        if _checkpoint_is_better(validation_score, validation_loss, best_score, best_loss):
            best_score = validation_score
            best_loss = validation_loss
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            _save_bundle(
                torch,
                model,
                output_path,
                task=task,
                classes=classes,
                multilabel=multilabel,
                validation_score=best_score,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"[{task}] stopping early after {epoch} epochs", flush=True)
                break
    test_metrics = None
    if test_dataset is not None:
        best_bundle = torch.load(output_path, map_location=device, weights_only=True)
        model.load_state_dict(best_bundle["state_dict"], strict=True)
        test_loader = _loader(torch, test_dataset, args.batch_size, args.workers, False, device)
        test_loss, test_metrics = _run_epoch(
            torch, model, test_loader, criterion, device, multilabel, None
        )
        print(
            f"[{task}] held-out test_loss={test_loss:.4f} test={_metric_summary(test_metrics)}",
            flush=True,
        )
    return {
        "classes": classes,
        "validation_loss": best_loss,
        "validation_macro_f1": best_score,
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "validation_split": split_description,
        "weights": os.path.relpath(output_path.resolve(), ROOT),
        "training": {
            "architecture": "mobilenet_v3_large",
            "pretrained": args.pretrained,
            "initialization": (
                str(args.init_from_dir / f"{task}.pt")
                if initial_bundle is not None
                else "imagenet"
                if args.pretrained
                else "random"
            ),
            "frozen_backbone_epochs": warmup_epochs,
            "backbone_lr_scale": args.backbone_lr_scale,
            "epochs_run": epoch,
        },
    }


def _image_datasets(
    task: str,
    data_root: Path,
    transforms: Any,
    datasets: Any,
    validation_fraction: float,
    seed: int,
) -> tuple[Any, Any, list[str], Any]:
    directory = data_root / IMAGE_TASKS[task][0]
    train_directory = directory / "train"
    test_directory = directory / "test"
    if not train_directory.is_dir() or not test_directory.is_dir():
        raise FileNotFoundError(f"{task} expects {train_directory} and {test_directory}")
    augmented_dataset = datasets.ImageFolder(
        train_directory, transform=_make_transform(transforms, training=True)
    )
    deterministic_dataset = datasets.ImageFolder(
        train_directory, transform=_make_transform(transforms, training=False)
    )
    test_dataset = datasets.ImageFolder(
        test_directory, transform=_make_transform(transforms, training=False)
    )
    if (
        augmented_dataset.classes != deterministic_dataset.classes
        or augmented_dataset.classes != test_dataset.classes
    ):
        raise ValueError(f"{task} train/test class folders do not match")
    if not augmented_dataset.classes or not test_dataset.samples:
        raise ValueError(f"{task} dataset is empty")
    train_indices, validation_indices = _stratified_image_split(
        augmented_dataset.targets,
        augmented_dataset.classes,
        validation_fraction,
        seed,
    )

    import torch

    train_dataset = torch.utils.data.Subset(augmented_dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(deterministic_dataset, validation_indices)
    return (
        train_dataset,
        validation_dataset,
        list(augmented_dataset.classes),
        test_dataset,
    )


def _dmd_datasets(
    task: str, data_root: Path, cache_root: Path, transforms: Any
) -> tuple[Any, Any, list[str], str]:
    samples = extract_dmd_face_frames(data_root / "dmd", cache_root, task)
    train_samples, validation_samples = temporal_split(samples)
    class_names = sorted({label for sample in samples for label in sample.labels})
    if not class_names:
        raise ValueError(f"DMD {task} did not yield any classes")
    train_dataset = DMDFrameDataset(
        train_samples, class_names, _make_transform(transforms, training=True)
    )
    validation_dataset = DMDFrameDataset(
        validation_samples, class_names, _make_transform(transforms, training=False)
    )
    split_description = (
        "chronological 80/20 holdout per recording; recordings may share drivers, "
        "so this is not subject-independent validation"
    )
    return train_dataset, validation_dataset, class_names, split_description


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data" / "cache" / "dmd")
    parser.add_argument("--models-dir", type=Path, default=MODELS_ROOT / "trained")
    parser.add_argument(
        "--init-from-dir",
        type=Path,
        help="Warm-start compatible models from this directory and retain the best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_TASKS,
        default=list(DEFAULT_TASKS),
        help="Tasks to train; defaults to all four",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Do not initialize MobileNetV3 from ImageNet weights",
    )
    parser.set_defaults(pretrained=True)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("--epochs and --batch-size must be positive; --workers cannot be negative")
    if (
        args.learning_rate <= 0
        or args.backbone_lr_scale <= 0
        or args.weight_decay < 0
        or args.min_learning_rate <= 0
        or args.min_learning_rate > args.learning_rate
    ):
        parser.error(
            "learning rates must be positive, min-learning-rate <= learning-rate, "
            "and weight decay cannot be negative"
        )
    if args.warmup_epochs < 0 or args.early_stopping_patience < 1 or args.lr_patience < 0:
        parser.error("warm-up must be non-negative and scheduler/early-stop patience valid")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks cannot contain duplicates")
    return args


def _initial_report(report_path: Path, args: argparse.Namespace, device: str) -> dict[str, Any]:
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing training report is invalid: {report_path}") from exc
        if not isinstance(report, dict) or not isinstance(report.get("models", {}), dict):
            raise ValueError(f"Existing training report has an invalid shape: {report_path}")
    else:
        report = {"models": {}}
    report["seed"] = args.seed
    report["pretrained"] = args.pretrained
    report["device"] = device
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        torch, modules, Image = _require_training_dependencies()
        datasets, _models, transforms = modules
        args.models_dir.mkdir(parents=True, exist_ok=True)
        selected_device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        report = _initial_report(args.models_dir / "training_report.json", args, selected_device)
        for task in args.tasks:
            if task in IMAGE_TASKS:
                train_ds, validation_ds, class_names, test_ds = _image_datasets(
                    task,
                    args.data_root,
                    transforms,
                    datasets,
                    args.validation_fraction,
                    args.seed,
                )
                split_description = (
                    "stratified image holdout from train/; subject IDs are not supplied, "
                    "so this is not person-independent; test/ is reserved for final evaluation"
                )
                multilabel = False
            else:
                train_ds, validation_ds, class_names, split_description = _dmd_datasets(
                    task, args.data_root, args.cache_root, transforms
                )
                multilabel = True
                test_ds = None
            result = _train(
                task=task,
                classes=class_names,
                train_dataset=train_ds,
                validation_dataset=validation_ds,
                multilabel=multilabel,
                args=args,
                output_path=args.models_dir / f"{task}.pt",
                split_description=split_description,
                test_dataset=test_ds,
            )
            report["models"][task] = result

        report_path = args.models_dir / "training_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Training complete. Model bundle and report saved in {args.models_dir}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
