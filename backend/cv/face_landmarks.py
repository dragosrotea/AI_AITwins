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
        self._BaseOptions: Any | None = None
        self._FaceLandmarker: Any | None = None

    def open(self) -> None:
        if not self.model_path.is_file():
            raise RuntimeError(
                f"Face landmarker model missing: {self.model_path}. "
                "Run data/download_datasets.sh mediapipe."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required: python3 -m pip install mediapipe") from exc

        options_module = mp.tasks
        self._BaseOptions = options_module.BaseOptions
        self._FaceLandmarker = options_module.vision.FaceLandmarker
        vision = options_module.vision
        options = vision.FaceLandmarkerOptions(
            base_options=self._BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.detection_confidence,
            min_face_presence_confidence=self.presence_confidence,
            min_tracking_confidence=self.tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = self._FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process(self, bgr_frame: Any) -> LandmarkResult:
        if self._landmarker is None:
            raise RuntimeError("FaceLandmarker is not open")
        mp = __import__("mediapipe")
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=__import__("cv2").cvtColor(bgr_frame, __import__("cv2").COLOR_BGR2RGB),
        )
        timestamp_ms = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(rgb_frame, timestamp_ms)

        if not result.face_landmarks:
            return LandmarkResult(False, [], {}, timestamp_ms)

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = float(category.score)
        return LandmarkResult(True, list(result.face_landmarks[0]), blendshapes, timestamp_ms)
