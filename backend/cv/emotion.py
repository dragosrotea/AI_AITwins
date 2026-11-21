from __future__ import annotations

from pathlib import Path
from typing import Any


def build_emotion_network(torch: Any) -> Any:
    class EmotionCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.25), torch.nn.Linear(128, 2)
            )

        def forward(self, image: Any) -> Any:
            return self.classifier(self.features(image))

    return EmotionCNN()


class EmotionModel:
    """FER2013 anger-vs-other inference on a landmark-bounded face crop."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Emotion inference requires torch") from exc
        path = Path(model_path)
        bundle = torch.load(path, map_location=device, weights_only=True)
        if (
            not isinstance(bundle, dict)
            or bundle.get("format_version") != 1
            or bundle.get("task") != "emotion"
            or bundle.get("architecture") != "emotion_cnn"
            or bundle.get("input_size") != 48
            or bundle.get("classes") != ["other", "anger"]
            or not isinstance(bundle.get("state_dict"), dict)
        ):
            raise ValueError(f"Unsupported emotion model bundle: {path}")
        self._torch = torch
        self._device = torch.device(device)
        self._model = build_emotion_network(torch).to(self._device)
        self._model.load_state_dict(bundle["state_dict"], strict=True)
        self._model.eval()

    def predict(self, frame_bgr: Any, landmarks: list[Any]) -> float:
        import cv2
        import numpy as np

        if frame_bgr is None or not landmarks:
            raise ValueError("Emotion prediction requires a face frame and landmarks")
        height, width = frame_bgr.shape[:2]
        points = [
            (float(point.x) * width, float(point.y) * height) for point in landmarks
        ]
        left = min(point[0] for point in points)
        right = max(point[0] for point in points)
        top = min(point[1] for point in points)
        bottom = max(point[1] for point in points)
        pad_x = (right - left) * 0.15
        pad_y = (bottom - top) * 0.15
        x1 = max(0, int(left - pad_x))
        x2 = min(width, int(right + pad_x + 1))
        y1 = max(0, int(top - pad_y))
        y2 = min(height, int(bottom + pad_y + 1))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("Face landmarks produced an empty emotion crop")
        rgb = cv2.cvtColor(cv2.resize(crop, (48, 48)), cv2.COLOR_BGR2RGB)
        tensor = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
        tensor = self._torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            probabilities = self._model(tensor).softmax(dim=1)[0]
        return float(probabilities[1].item())
