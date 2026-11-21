from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


MODEL_NAMES = ("age", "emotion", "drowsiness", "distraction")
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "trained"
IMAGE_SIZE = 224
RESIZE_SHORT_EDGE = 256


class FourModelPipeline:
    """Run the four trained classifiers on one shared face-camera frame."""

    def __init__(
        self,
        models_dir: str | Path = DEFAULT_MODELS_DIR,
        device: str = "auto",
        multilabel_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= multilabel_threshold <= 1.0:
            raise ValueError("multilabel_threshold must be between 0 and 1")
        try:
            import torch
            from PIL import Image
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError(
                "Inference requires torch, torchvision, Pillow, and opencv-python. "
                "Install training/requirements.txt."
            ) from exc

        self._torch = torch
        self._Image = Image
        self._transforms = transforms
        self._torchvision_models = models
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.multilabel_threshold = multilabel_threshold
        self.models: dict[str, Any] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize(RESIZE_SHORT_EDGE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._load(Path(models_dir))

    def _load(self, models_dir: Path) -> None:
        torch = self._torch
        for task in MODEL_NAMES:
            path = models_dir / f"{task}.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing trained model {path}. Run python training/train_models.py first."
                )
            bundle = torch.load(path, map_location=self.device, weights_only=True)
            if bundle.get("format_version") != 1 or bundle.get("task") != task:
                raise ValueError(f"Unsupported or mismatched model bundle: {path}")
            classes = bundle.get("classes")
            state_dict = bundle.get("state_dict")
            if not isinstance(classes, list) or not classes or not isinstance(state_dict, dict):
                raise ValueError(f"Invalid model bundle: {path}")
            network = self._torchvision_models.mobilenet_v3_large(weights=None)
            final_layer = network.classifier[-1]
            network.classifier[-1] = type(final_layer)(final_layer.in_features, len(classes))
            network.load_state_dict(state_dict)
            network.to(self.device).eval()
            self.models[task] = network
            self.metadata[task] = {
                "classes": classes,
                "multilabel": bool(bundle.get("multilabel")),
            }

    def predict_frame(self, frame_bgr: Any) -> dict[str, Any]:
        """Predict age, emotion, drowsiness, and distraction from the same frame."""
        cv2 = self._load_cv2()
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb_frame)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        predictions: dict[str, Any] = {}
        with self._torch.inference_mode():
            for task in MODEL_NAMES:
                logits = self.models[task](tensor)[0]
                classes = self.metadata[task]["classes"]
                if self.metadata[task]["multilabel"]:
                    scores = logits.sigmoid().tolist()
                    probabilities = {
                        label: float(score) for label, score in zip(classes, scores)
                    }
                    active = [
                        label
                        for label, score in probabilities.items()
                        if score >= self.multilabel_threshold
                    ]
                    predictions[task] = {
                        "probabilities": probabilities,
                        "active": active,
                    }
                else:
                    scores = logits.softmax(dim=0)
                    score, index = scores.max(dim=0)
                    predictions[task] = {
                        "label": classes[int(index.item())],
                        "confidence": float(score.item()),
                    }
        return predictions

    @staticmethod
    def _load_cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video inference requires opencv-python") from exc
        return cv2


def _run_source(pipeline: FourModelPipeline, source: str | int, display: bool) -> int:
    cv2 = pipeline._load_cv2()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {source}")
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            predictions = pipeline.predict_frame(frame)
            event = {
                "topic": "cv.predictions",
                "ts": time.time(),
                "frame": frame_number,
                "src": "face_camera",
                "data": predictions,
            }
            print(json.dumps(event, separators=(",", ":")), flush=True)
            if display:
                preview = frame.copy()
                for row, (task, result) in enumerate(predictions.items(), start=1):
                    if "label" in result:
                        text = f"{task}: {result['label']} ({result['confidence']:.2f})"
                    else:
                        text = f"{task}: {', '.join(result['active']) or 'none'}"
                    cv2.putText(
                        preview,
                        text,
                        (12, 28 * row),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("DriveMate four-model pipeline", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
    if frame_number == 0:
        raise RuntimeError(f"No frames were read from video source: {source}")
    return frame_number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="Face-camera device index")
    source.add_argument("--video", type=Path, help="Face-camera video file")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--display", action="store_true", help="Display input and predictions")
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pipeline = FourModelPipeline(
            args.models_dir,
            device=args.device,
            multilabel_threshold=args.threshold,
        )
        source: str | int = args.camera if args.camera is not None else str(args.video)
        if args.video is not None and not args.video.is_file():
            raise FileNotFoundError(f"Video not found: {args.video}")
        frame_count = _run_source(pipeline, source, args.display)
        print(f"Processed {frame_count} frames", file=sys.stderr)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
