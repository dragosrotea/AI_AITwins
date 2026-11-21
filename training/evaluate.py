from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from backend.cv.eye_crops import preprocess_eye
from backend.cv.eye_state import EyeStateModel
from backend.cv.settings import (
    DEFAULT_CONFIG,
    ROOT,
    float_setting,
    integer_setting,
    load_cv_settings,
    resolved_path,
    string_setting,
)
from training.eye_state.train import discover_mrl_samples, split_by_subject


def binary_metrics(labels: list[int], predictions: list[int]) -> tuple[float, float, float]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal lengths")
    if not labels:
        raise ValueError("at least one label is required")
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    data_dir: Path,
    model_path: Path,
    config_path: Path,
    validation_fraction: float | None = None,
    seed: int | None = None,
) -> tuple[float, float, float, float, int]:
    settings = load_cv_settings(config_path)
    samples = discover_mrl_samples(data_dir)
    fraction = (
        validation_fraction
        if validation_fraction is not None
        else float_setting(settings, "eye_state", "training_validation_fraction")
    )
    split_seed = (
        seed if seed is not None else integer_setting(settings, "eye_state", "training_seed")
    )
    _, held_out = split_by_subject(samples, fraction, split_seed)
    held_out_subjects = len({sample.subject for sample in held_out})
    model = EyeStateModel(
        model_path,
        closed_threshold=float_setting(settings, "eye_state", "closed_probability_threshold"),
        max_model_size_bytes=integer_setting(
            settings, "eye_state", "max_model_size_bytes", minimum=1
        ),
    )

    first_image = cv2.imread(str(held_out[0].path), cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise OSError(f"Could not read MRL eye image: {held_out[0].path}")
    model.predict([preprocess_eye(first_image)])

    labels: list[int] = []
    predictions: list[int] = []
    inference_seconds = 0.0
    for sample in held_out:
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read MRL eye image: {sample.path}")
        crop = preprocess_eye(image)
        started = time.perf_counter()
        probability, _ = model.predict([crop])
        inference_seconds += time.perf_counter() - started
        labels.append(sample.closed)
        predictions.append(int(model.is_closed(probability)))

    precision, recall, f1 = binary_metrics(labels, predictions)
    latency_ms = inference_seconds * 1000 / len(held_out)
    return precision, recall, f1, latency_ms, held_out_subjects


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the eye-state ONNX model on held-out MRL subjects"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "mrl_eye")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_cv_settings(args.config)
    args.model = args.model or resolved_path(string_setting(settings, "eye_state", "model_path"))
    if args.validation_fraction is not None and not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    precision, recall, f1, latency_ms, subjects = evaluate(
        args.data_dir,
        args.model,
        args.config,
        args.validation_fraction,
        args.seed,
    )
    print("| Model | Held-out subjects | Precision | Recall | F1 | CPU ms/eye |")
    print("|---|---:|---:|---:|---:|---:|")
    print(
        f"| Eye-state (closed) | {subjects} | {precision:.4f} | {recall:.4f} "
        f"| {f1:.4f} | {latency_ms:.3f} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
