from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)


@dataclass(frozen=True)
class FaceGeometry:
    yaw: float
    pitch: float
    roll: float
    gaze_horizontal: float
    gaze_vertical: float
    eye_aspect_ratio: float
    mouth_aspect_ratio: float


def _point(landmarks: Sequence[Any], index: int) -> tuple[float, float]:
    if index >= len(landmarks):
        raise ValueError(f"Face mesh requires landmark {index}; got {len(landmarks)}")
    point = landmarks[index]
    x, y = float(point.x), float(point.y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"Face mesh landmark {index} contains non-finite coordinates")
    return x, y


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _eye_aspect_ratio(landmarks: Sequence[Any], indices: tuple[int, ...]) -> float:
    points = [_point(landmarks, index) for index in indices]
    horizontal = _distance(points[0], points[3])
    if horizontal <= 1e-8:
        return 0.0
    return (_distance(points[1], points[5]) + _distance(points[2], points[4])) / (2.0 * horizontal)


def _iris_direction(
    landmarks: Sequence[Any], corners: tuple[int, int], iris_indices: tuple[int, ...]
) -> tuple[float, float]:
    outer = _point(landmarks, corners[0])
    inner = _point(landmarks, corners[1])
    iris = [_point(landmarks, index) for index in iris_indices]
    center = (
        sum(point[0] for point in iris) / len(iris),
        sum(point[1] for point in iris) / len(iris),
    )
    width = _distance(outer, inner)
    if width <= 1e-8:
        return 0.0, 0.0
    center_x = (outer[0] + inner[0]) / 2.0
    center_y = (outer[1] + inner[1]) / 2.0
    horizontal_sign = 1.0 if inner[0] >= outer[0] else -1.0
    return (
        horizontal_sign * (center[0] - center_x) / width,
        (center[1] - center_y) / width,
    )


def _head_angles(
    matrix: Sequence[Sequence[float]], pitch_sign: float
) -> tuple[float, float, float]:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Face transformation matrix must be 4x4, got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError("Face transformation matrix contains non-finite values")
    rotation = transform[:3, :3]
    horizontal_norm = math.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal_norm > 1e-8:
        roll_x = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch_y = math.atan2(-rotation[2, 0], horizontal_norm)
        yaw_z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll_x = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch_y = math.atan2(-rotation[2, 0], horizontal_norm)
        yaw_z = 0.0
    return (
        math.degrees(yaw_z),
        math.degrees(pitch_y) * pitch_sign,
        math.degrees(roll_x),
    )


def extract_geometry(
    landmarks: Sequence[Any],
    transformation_matrix: Sequence[Sequence[float]] | None,
    *,
    pitch_sign: float = -1.0,
) -> FaceGeometry:
    if transformation_matrix is None:
        raise ValueError("Face transformation matrix is required for head pose")
    yaw, pitch, roll = _head_angles(transformation_matrix, pitch_sign)

    left_gaze = _iris_direction(landmarks, (33, 133), LEFT_IRIS)
    right_gaze = _iris_direction(landmarks, (263, 362), RIGHT_IRIS)
    gaze_horizontal = (left_gaze[0] + right_gaze[0]) / 2.0
    gaze_vertical = (left_gaze[1] + right_gaze[1]) / 2.0

    left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE)
    mouth_width = _distance(_point(landmarks, 61), _point(landmarks, 291))
    mouth_height = _distance(_point(landmarks, 13), _point(landmarks, 14))
    mouth_aspect = mouth_height / mouth_width if mouth_width > 1e-8 else 0.0
    return FaceGeometry(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        gaze_horizontal=gaze_horizontal,
        gaze_vertical=gaze_vertical,
        eye_aspect_ratio=(left_ear + right_ear) / 2.0,
        mouth_aspect_ratio=mouth_aspect,
    )


class BlinkDetector:
    def __init__(self, closed_threshold: float, open_threshold: float) -> None:
        if closed_threshold <= 0 or open_threshold <= closed_threshold:
            raise ValueError("Blink thresholds must satisfy 0 < closed < open")
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self._closed = False

    def update(self, eye_aspect_ratio: float) -> bool:
        if not math.isfinite(eye_aspect_ratio) or eye_aspect_ratio < 0:
            raise ValueError("eye_aspect_ratio must be finite and non-negative")
        if not self._closed and eye_aspect_ratio <= self.closed_threshold:
            self._closed = True
            return False
        if self._closed and eye_aspect_ratio >= self.open_threshold:
            self._closed = False
            return True
        return False
