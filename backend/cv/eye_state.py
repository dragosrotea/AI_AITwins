from __future__ import annotations

from pathlib import Path

import numpy as np


class EyeStateModel:
    """CPU ONNX inference for the closed-eye probability from one or both eye crops."""

    def __init__(
        self,
        model_path: str | Path,
        closed_threshold: float,
        max_model_size_bytes: int,
    ) -> None:
        if not 0.0 <= closed_threshold <= 1.0:
            raise ValueError("closed_threshold must be between 0 and 1")
        if max_model_size_bytes <= 0:
            raise ValueError("max_model_size_bytes must be positive")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Eye-state ONNX model not found: {path}")
        if path.stat().st_size >= max_model_size_bytes:
            raise ValueError(
                f"Eye-state ONNX model must be under {max_model_size_bytes} bytes: {path}"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Eye-state inference requires onnxruntime; install the runtime dependency."
            ) from exc

        self.closed_threshold = closed_threshold
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"Unsupported eye-state model input/output count: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def predict(self, eye_crops: list[np.ndarray]) -> tuple[float, list[float]]:
        if not eye_crops:
            raise ValueError("At least one eye crop is required")
        batch = np.stack(eye_crops).astype(np.float32, copy=False)
        if batch.ndim != 3 or batch.shape[1:] != (64, 64):
            raise ValueError(f"Eye model expects Nx64x64 crops, got {batch.shape}")
        if not np.isfinite(batch).all() or batch.min() < 0.0 or batch.max() > 1.0:
            raise ValueError("Eye crops must contain finite values in [0, 1]")
        logits = np.asarray(
            self._session.run(
                [self._output_name],
                {self._input_name: batch[:, np.newaxis, :, :]},
            )[0],
            dtype=np.float32,
        )
        if logits.shape != (len(eye_crops), 2):
            raise ValueError(f"Expected Nx2 eye logits, got {logits.shape}")
        if not np.isfinite(logits).all():
            raise ValueError("Eye model returned non-finite logits")
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        closed_probabilities = exponentials[:, 0] / exponentials.sum(axis=1)
        probabilities = [float(value) for value in closed_probabilities]
        return float(np.mean(closed_probabilities)), probabilities

    def is_closed(self, probability: float) -> bool:
        return probability >= self.closed_threshold
