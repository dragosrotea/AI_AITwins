import unittest

from backend.cv.camera import CameraConfig, LatestFrame, CameraCapture
from backend.cv.face_landmarks import FaceLandmarker, LandmarkResult


class LatestFrameTest(unittest.TestCase):
    def test_replaces_previous_frame(self) -> None:
        buffer = LatestFrame()
        buffer.put("first", timestamp=1.0)
        buffer.put("second", timestamp=2.0)

        frame, timestamp = buffer.get()

        self.assertEqual(frame, "second")
        self.assertEqual(timestamp, 2.0)


class CameraConfigTest(unittest.TestCase):
    def test_cv_t02_defaults(self) -> None:
        config = CameraConfig()

        self.assertEqual(config.index, 0)
        self.assertEqual((config.capture_width, config.capture_height), (1280, 720))
        self.assertEqual((config.process_width, config.process_height), (640, 360))


class FaceLandmarkerTest(unittest.TestCase):
    def test_missing_model_is_actionable(self) -> None:
        landmarker = FaceLandmarker("models/does-not-exist.task")

        with self.assertRaisesRegex(RuntimeError, "download_datasets.sh mediapipe"):
            landmarker.open()


class CameraCaptureTest(unittest.TestCase):
    def test_missing_opencv_error_is_actionable(self) -> None:
        capture = CameraCapture()

        with self.assertRaisesRegex(RuntimeError, "Install opencv-python"):
            capture.open()


if __name__ == "__main__":
    unittest.main()
