from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LandmarkResult:
    face_present: bool
    landmarks: list[Any]
    blendshapes: dict[str, float]
    transformation_matrix: list[list[float]] | None
    timestamp_ms: int


class FaceLandmarker:
    def __init__(
        self,
        model_path: str | Path,
        num_faces: int = 1,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.num_faces = num_faces
        self.detection_confidence = detection_confidence
        self.presence_confidence = presence_confidence
        self.tracking_confidence = tracking_confidence
        self._landmarker: Any | None = None
        self._mediapipe: Any | None = None
        self._cv2: Any | None = None
        self._last_timestamp_ms = 0

    def open(self) -> None:
        if self.num_faces < 1:
            raise ValueError("num_faces must be positive")
        for value in (
            self.detection_confidence,
            self.presence_confidence,
            self.tracking_confidence,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("MediaPipe confidence thresholds must be between 0 and 1")
        if not self.model_path.is_file():
            raise RuntimeError(
                f"Face landmarker model missing: {self.model_path}. "
                "Run data/download_datasets.sh mediapipe."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required: python3 -m pip install mediapipe") from exc
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required: python3 -m pip install opencv-python") from exc

        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.detection_confidence,
            min_face_presence_confidence=self.presence_confidence,
            min_tracking_confidence=self.tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mediapipe = mp
        self._cv2 = cv2

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._mediapipe = None
            self._cv2 = None

    def process(self, bgr_frame: Any) -> LandmarkResult:
        if self._landmarker is None:
            raise RuntimeError("FaceLandmarker is not open")
        if self._mediapipe is None or self._cv2 is None:
            raise RuntimeError("FaceLandmarker dependencies were not initialized")
        mp = self._mediapipe
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(rgb_frame, timestamp_ms)

        if not result.face_landmarks:
            return LandmarkResult(False, [], {}, None, timestamp_ms)

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = float(category.score)
        transformation_matrix = None
        if result.facial_transformation_matrixes:
            matrix = result.facial_transformation_matrixes[0]
            get_data = getattr(matrix, "get_data", None)
            if callable(get_data):
                matrix = get_data()
            if hasattr(matrix, "tolist"):
                matrix = matrix.tolist()
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise ValueError("MediaPipe returned a face transformation matrix that is not 4x4")
            transformation_matrix = [[float(value) for value in row] for row in matrix]
        return LandmarkResult(
            True,
            list(result.face_landmarks[0]),
            blendshapes,
            transformation_matrix,
            timestamp_ms,
        )
