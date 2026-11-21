import unittest
from types import SimpleNamespace
from unittest import mock

from backend.cv.camera import CameraCapture, CameraConfig, LatestFrame
from backend.cv.face_landmarks import FaceLandmarker
from backend.cv.run import fake_sequences, state_dict


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
        self.assertEqual(config.capture_fourcc, "MJPG")
        self.assertEqual((config.process_width, config.process_height), (640, 360))


class FaceLandmarkerTest(unittest.TestCase):
    def test_missing_model_is_actionable(self) -> None:
        landmarker = FaceLandmarker("models/does-not-exist.task")

        with self.assertRaisesRegex(RuntimeError, "download_datasets.sh mediapipe"):
            landmarker.open()


class FakePublisherTest(unittest.TestCase):
    def test_fake_state_has_shared_contract_fields(self) -> None:
        state = next(fake_sequences("normal"))
        data = state_dict(state)

        expected = {
            "face_present",
            "eye_closed_p",
            "eyes_closed_ms",
            "perclos_60s",
            "blink_rate_pm",
            "gaze_dispersion_30s",
            "head",
            "yawn_p",
            "yawns_10m",
            "anger_p",
            "zoning_visual_p",
            "phone_in_hand",
            "fps",
        }
        self.assertEqual(set(data), expected)

    def test_fake_sequences_include_requested_modes(self) -> None:
        for sequence in ("normal", "eyes_closing", "looking_down", "angry", "staring"):
            state = next(fake_sequences(sequence))
            self.assertIn(state.face_present, (True, False))


class CameraCaptureTest(unittest.TestCase):
    def test_requests_configured_capture_format_before_dimensions(self) -> None:
        calls: list[tuple[int, float]] = []

        class Capture:
            def set(self, property_id: int, value: float) -> bool:
                calls.append((property_id, value))
                return True

            def isOpened(self) -> bool:
                return True

            def release(self) -> None:
                pass

        cv2 = SimpleNamespace(
            VideoCapture=lambda _index: Capture(),
            VideoWriter_fourcc=lambda *chars: ord(chars[0]),
            CAP_PROP_FOURCC=1,
            CAP_PROP_FRAME_WIDTH=2,
            CAP_PROP_FRAME_HEIGHT=3,
        )
        capture = CameraCapture()

        with mock.patch.dict("sys.modules", {"cv2": cv2}):
            capture.open()

        self.assertEqual(calls, [(1, float(ord("M"))), (2, 1280.0), (3, 720.0)])
        capture.stop()

    def test_missing_opencv_error_is_actionable(self) -> None:
        capture = CameraCapture()

        with mock.patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "Install opencv-python"):
                capture.open()


if __name__ == "__main__":
    unittest.main()
