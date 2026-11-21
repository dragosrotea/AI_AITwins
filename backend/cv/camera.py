from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol


class CameraFrameProtocol(Protocol):
    def read(self) -> tuple[bool, object]: ...
    def set(self, property_id: int, value: float) -> bool: ...
    def isOpened(self) -> bool: ...
    def release(self) -> None: ...


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    process_width: int = 640
    process_height: int = 360

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Camera index must be non-negative")
        if (
            min(self.capture_width, self.capture_height, self.process_width, self.process_height)
            <= 0
        ):
            raise ValueError("Camera capture and processing dimensions must be positive")


class LatestFrame:
    """A thread-safe latest-frame buffer with the newest processed frame only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: object | None = None
        self._timestamp: float | None = None

    def put(self, frame: object, timestamp: float | None = None) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = time.monotonic() if timestamp is None else timestamp

    def get(self) -> tuple[object | None, float | None]:
        with self._lock:
            return self._frame, self._timestamp


class CameraCapture:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.latest = LatestFrame()
        self._capture: CameraFrameProtocol | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cv2: object | None = None

    def _load_cv2(self) -> object:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV is required. Install opencv-python, then retry camera capture."
                ) from exc
            self._cv2 = cv2
        return self._cv2

    def open(self) -> None:
        cv2 = self._load_cv2()
        capture = cv2.VideoCapture(self.config.index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.capture_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.capture_height))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open webcam index {self.config.index}")
        self._capture = capture

    def start(self) -> None:
        if self._capture is None:
            self.open()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="drivemate-camera", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _capture_loop(self) -> None:
        cv2 = self._load_cv2()
        capture = self._capture
        if capture is None:
            return

        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            processed = cv2.resize(
                frame,
                (self.config.process_width, self.config.process_height),
                interpolation=cv2.INTER_AREA,
            )
            self.latest.put(processed)

    def get_latest(self) -> tuple[object | None, float | None]:
        return self.latest.get()


def list_cameras(max_index: int = 10) -> list[int]:
    """Return webcam indices that OpenCV can open on this machine."""
    if max_index < 0:
        raise ValueError("max_index must be non-negative")
    cv2 = CameraCapture()._load_cv2()
    available: list[int] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                available.append(index)
        finally:
            capture.release()
    return available
