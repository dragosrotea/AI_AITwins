from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from backend.cv.geometry import LEFT_EYE, RIGHT_EYE


def preprocess_eye(image: Any, output_size: int = 64) -> np.ndarray:
    """Convert an eye image to the shared grayscale float32 model input."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Eye preprocessing requires opencv-python") from exc

    if image is None or not hasattr(image, "shape") or image.size == 0:
        raise ValueError("Eye image must be a non-empty image array")
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        grayscale = image
    else:
        raise ValueError(f"Expected a grayscale or BGR eye image, got shape {image.shape}")
    resized = cv2.resize(
        grayscale,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32) / 255.0


def crop_eyes(
    frame: Any,
    landmarks: Sequence[Any],
    *,
    padding_fraction: float = 0.25,
    output_size: int = 64,
) -> list[np.ndarray]:
    """Crop both eye regions and apply the same preprocessing used by training."""
    if not 0.0 <= padding_fraction <= 1.0:
        raise ValueError("padding_fraction must be between 0 and 1")
    if frame is None or not hasattr(frame, "shape") or frame.ndim < 2:
        raise ValueError("frame must be a non-empty image array")
    height, width = frame.shape[:2]
    crops: list[np.ndarray] = []
    for indices in (LEFT_EYE, RIGHT_EYE):
        if max(indices) >= len(landmarks):
            raise ValueError(f"Face mesh requires landmark {max(indices)}")
        points = [
            (
                round(float(landmarks[index].x) * width),
                round(float(landmarks[index].y) * height),
            )
            for index in indices
        ]
        left = min(point[0] for point in points)
        right = max(point[0] for point in points)
        top = min(point[1] for point in points)
        bottom = max(point[1] for point in points)
        padding_x = round((right - left) * padding_fraction)
        padding_y = round((bottom - top) * padding_fraction)
        left = max(0, left - padding_x)
        right = min(width, right + padding_x + 1)
        top = max(0, top - padding_y)
        bottom = min(height, bottom + padding_y + 1)
        if right <= left or bottom <= top:
            raise ValueError("Face landmarks produced an empty eye crop")
        crops.append(preprocess_eye(frame[top:bottom, left:right], output_size))
    return crops
