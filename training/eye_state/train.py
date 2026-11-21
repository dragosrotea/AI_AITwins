from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.cv.eye_crops import preprocess_eye
from backend.cv.settings import ROOT, float_setting, integer_setting, load_cv_settings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class EyeSample:
    path: Path
    subject: str
    closed: int


def parse_mrl_filename(path: Path) -> EyeSample:
    fields = path.stem.split("_")
    if len(fields) < 5 or not fields[0].startswith("s"):
        raise ValueError(f"Unexpected MRL eye filename: {path.name}")
    if fields[4] not in {"0", "1"}:
        raise ValueError(f"Unknown MRL eye-state label in {path.name}")
    return EyeSample(path=path, subject=fields[0], closed=int(fields[4] == "0"))


def discover_mrl_samples(root: str | Path) -> list[EyeSample]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"MRL Eye Dataset directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No MRL eye images found under {directory}")
    samples = [parse_mrl_filename(path) for path in paths]
    return samples


def split_by_subject(
    samples: list[EyeSample], validation_fraction: float, seed: int
) -> tuple[list[EyeSample], list[EyeSample]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    subjects = sorted({sample.subject for sample in samples})
    if len(subjects) < 2:
        raise ValueError("At least two MRL subjects are needed for a person-held-out split")
    random.Random(seed).shuffle(subjects)
    validation_count = min(len(subjects) - 1, max(1, round(len(subjects) * validation_fraction)))
    validation_subjects = set(subjects[:validation_count])
    train = [sample for sample in samples if sample.subject not in validation_subjects]
    validation = [sample for sample in samples if sample.subject in validation_subjects]
    if {sample.closed for sample in train} != {0, 1}:
        raise ValueError("Training subjects must include both open and closed eyes")
    if {sample.closed for sample in validation} != {0, 1}:
        raise ValueError(
            "Held-out subjects must include both open and closed eyes; use a different split seed"
        )
    return train, validation


class MrlEyeDataset:
    def __init__(self, samples: list[EyeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import cv2
        import torch

        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read MRL eye image: {sample.path}")
        processed = preprocess_eye(image)
        return (
            torch.from_numpy(processed).unsqueeze(0),
            torch.tensor(sample.closed, dtype=torch.long),
        )


def make_model() -> Any:
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 2),
    )


def _metrics(labels: list[int], predictions: list[int]) -> tuple[float, float]:
    true_positive = false_positive = false_negative = 0
    for actual, predicted in zip(labels, predictions, strict=True):
        true_positive += actual == 1 and predicted == 1
        false_positive += actual == 0 and predicted == 1
        false_negative += actual == 1 and predicted == 0
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    return precision, recall


def train(args: argparse.Namespace) -> None:
    settings = load_cv_settings(args.config)
    samples = discover_mrl_samples(args.data_dir)
    print(f"MRL root: {args.data_dir}")
    print(f"Image count: {len(samples)}; subjects: {len({item.subject for item in samples})}")
    print("Subject directories:")
    for parent in sorted({sample.path.parent for sample in samples})[:12]:
        print(f"  {parent.relative_to(args.data_dir)}")
    print("Sample files:")
    for sample in samples[:10]:
        print(f"  {sample.path.relative_to(args.data_dir)}")
    print(f"Eye labels: {Counter('closed' if item.closed else 'open' for item in samples)}")

    train_samples, validation_samples = split_by_subject(
        samples,
        args.validation_fraction,
        args.seed,
    )
    train_subjects = {sample.subject for sample in train_samples}
    validation_subjects = {sample.subject for sample in validation_samples}
    if train_subjects & validation_subjects:
        raise RuntimeError("Subject leakage detected between training and validation")

    import cv2
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(
        MrlEyeDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / total_samples:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.zeros(1, 1, 64, 64),
        str(args.output),
        input_names=["eye"],
        output_names=["logits"],
        dynamic_axes={"eye": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Validating the exported model requires onnxruntime; install training/requirements.txt"
        ) from exc
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    threshold = float_setting(settings, "eye_state", "closed_probability_threshold")
    first_image = cv2.imread(
        str(validation_samples[0].path),
        cv2.IMREAD_GRAYSCALE,
    )
    if first_image is None:
        raise OSError(f"Could not read MRL eye image: {validation_samples[0].path}")
    warmup = preprocess_eye(first_image)[np.newaxis, np.newaxis, :, :]
    session.run([output_name], {input_name: warmup})
    labels: list[int] = []
    predictions: list[int] = []
    inference_seconds = 0.0
    for sample in validation_samples:
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read MRL eye image: {sample.path}")
        tensor = preprocess_eye(image)[np.newaxis, np.newaxis, :, :]
        started = time.perf_counter()
        logits = session.run([output_name], {input_name: tensor})[0][0]
        inference_seconds += time.perf_counter() - started
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        labels.append(sample.closed)
        predictions.append(int(float(probabilities[1]) >= threshold))

    precision, recall = _metrics(labels, predictions)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    inference_ms = inference_seconds * 1000 / len(validation_samples)
    model_size_bytes = args.output.stat().st_size
    print(f"held-out subjects: {len(validation_subjects)}")
    print(f"closed-eye precision: {precision:.4f}")
    print(f"closed-eye recall: {recall:.4f}")
    print(f"closed-eye F1: {f1:.4f}")
    print(f"ONNX inference: {inference_ms:.3f} ms/eye")
    print(f"model size: {model_size_bytes} bytes ({args.output})")
    max_model_size = integer_setting(settings, "eye_state", "max_model_size_bytes", minimum=1)
    max_inference_ms = float_setting(settings, "eye_state", "max_inference_ms_per_eye")
    if model_size_bytes >= max_model_size:
        raise RuntimeError(
            f"Eye-state model must be under {max_model_size} bytes, got {model_size_bytes}"
        )
    if inference_ms >= max_inference_ms:
        raise RuntimeError(
            f"Eye-state inference must be under {max_inference_ms} ms/eye, got {inference_ms:.3f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=ROOT / "config" / "thresholds.yaml")
    known, _ = preliminary.parse_known_args(argv)
    settings = load_cv_settings(known.config)
    parser = argparse.ArgumentParser(description="Train an MRL eye-state ONNX classifier")
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mrl_eye")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "eye_state.onnx")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=float_setting(settings, "eye_state", "training_validation_fraction"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=integer_setting(settings, "eye_state", "training_seed"),
    )
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


if __name__ == "__main__":
    train(parse_args())
