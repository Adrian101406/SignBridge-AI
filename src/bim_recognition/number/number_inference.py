"""SignBridge Number Recognizer V1.

MCIE-facing inference wrapper for the static BIM number model (0-10).
The public ``recognize_bim`` result intentionally matches the shared result
shape used by SignBridgeMedicalRecognizer and SignBridgeGeneralRecognizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

try:
    from .src.landmarks import get_hand_landmarker, process_image
except ImportError:
    from src.landmarks import get_hand_landmarker, process_image


class SignBridgeNumberRecognizer:
    """Recognize one static BIM number sign from a frame or short video."""

    RECOGNIZER_NAME = "SignBridge_Number_Recognizer_V1"
    DOMAIN = "number"

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        hand_task_path: str,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.hand_task_path = Path(hand_task_path)

        for path in (
            self.model_path,
            self.labels_path,
            self.hand_task_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.labels = [
            line.strip()
            for line in self.labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        expected_labels = [str(value) for value in range(11)]
        if self.labels != expected_labels:
            raise ValueError(
                "labels.txt must contain exactly 0,1,2,...,10 in numeric order. "
                f"Received: {self.labels}"
            )

        self.interpreter = tf.lite.Interpreter(
            model_path=str(self.model_path)
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        input_shape = tuple(int(value) for value in self.input_detail["shape"])
        output_shape = tuple(int(value) for value in self.output_detail["shape"])

        if input_shape[-1] != 63:
            raise ValueError(
                f"Expected TFLite input ending in 63, received {input_shape}."
            )
        if output_shape[-1] != len(self.labels):
            raise ValueError(
                "TFLite output count does not match labels.txt: "
                f"{output_shape[-1]} versus {len(self.labels)}."
            )

        self.hand_landmarker = get_hand_landmarker(
            self.hand_task_path,
            num_hands=1,
        )
        self._closed = False

    @staticmethod
    def _quantize_input(
        values: np.ndarray,
        tensor_detail: Dict,
    ) -> np.ndarray:
        dtype = tensor_detail["dtype"]
        values = values.astype(np.float32)

        if np.issubdtype(dtype, np.floating):
            return values.astype(dtype)

        scale, zero_point = tensor_detail["quantization"]
        if scale == 0:
            raise ValueError("Quantized input tensor has scale=0.")

        quantized = np.round(values / scale + zero_point)
        limits = np.iinfo(dtype)
        return np.clip(
            quantized,
            limits.min,
            limits.max,
        ).astype(dtype)

    @staticmethod
    def _dequantize_output(
        values: np.ndarray,
        tensor_detail: Dict,
    ) -> np.ndarray:
        if np.issubdtype(values.dtype, np.floating):
            return values.astype(np.float32)

        scale, zero_point = tensor_detail["quantization"]
        if scale == 0:
            raise ValueError("Quantized output tensor has scale=0.")

        return scale * (
            values.astype(np.float32) - zero_point
        )

    def _infer_frame_probabilities(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._closed:
            raise RuntimeError("Recognizer has already been closed.")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy BGR image.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected a BGR image with shape (H,W,3), got {frame.shape}."
            )

        result = process_image(frame, self.hand_landmarker)
        if result is None:
            return None

        landmarks_63, _handedness = result
        features = np.asarray(
            landmarks_63,
            dtype=np.float32,
        ).reshape(1, 63)

        model_input = self._quantize_input(
            features,
            self.input_detail,
        )
        self.interpreter.set_tensor(
            self.input_detail["index"],
            model_input,
        )
        self.interpreter.invoke()

        raw_output = self.interpreter.get_tensor(
            self.output_detail["index"]
        )[0]
        probabilities = self._dequantize_output(
            raw_output,
            self.output_detail,
        )

        if probabilities.shape != (len(self.labels),):
            raise ValueError(
                f"Unexpected probability shape: {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("Model output contains NaN or Inf.")

        return probabilities.astype(np.float32)

    def _rank_predictions(
        self,
        probabilities: np.ndarray,
        top_k: int,
    ) -> List[Dict]:
        if not 1 <= top_k <= len(self.labels):
            raise ValueError(
                f"top_k must be between 1 and {len(self.labels)}."
            )

        ranked_ids = np.argsort(probabilities)[::-1][:top_k]
        predictions = []

        for rank, class_id in enumerate(ranked_ids, start=1):
            class_id = int(class_id)
            label = self.labels[class_id]
            predictions.append(
                {
                    "rank": rank,
                    "class_id": class_id,
                    "gloss": label,
                    "english": label,
                    "confidence": float(probabilities[class_id]),
                }
            )

        return predictions

    def _build_result(
        self,
        probabilities: np.ndarray,
        top_k: int,
        raw_frames: int,
        detected_frames: int,
    ) -> Dict:
        predictions = self._rank_predictions(probabilities, top_k)
        top1_confidence = float(predictions[0]["confidence"])
        top2_confidence = (
            float(predictions[1]["confidence"])
            if len(predictions) >= 2
            else 0.0
        )

        return {
            "recognizer": self.RECOGNIZER_NAME,
            "domain": self.DOMAIN,
            "top1": {
                "class_id": int(predictions[0]["class_id"]),
                "gloss": predictions[0]["gloss"],
                "english": predictions[0]["english"],
                "confidence": top1_confidence,
            },
            "top_k": predictions,
            "confidence_margin": float(
                top1_confidence - top2_confidence
            ),
            "quality": {
                "hand_presence": float(
                    detected_frames / raw_frames
                    if raw_frames > 0
                    else 0.0
                ),
                "raw_frames": int(raw_frames),
                # Kept for schema compatibility with the existing recognizers.
                # For this static recognizer, these are hand-detected frames.
                "trimmed_frames": int(detected_frames),
            },
        }

    def predict_frame(
        self,
        frame: np.ndarray,
        top_k: int = 3,
    ) -> Optional[Dict]:
        """Predict one BGR frame.

        Returns ``None`` when no hand is detected. Otherwise returns the same
        outer result schema used by ``recognize_bim``.
        """
        probabilities = self._infer_frame_probabilities(frame)
        if probabilities is None:
            return None

        return self._build_result(
            probabilities=probabilities,
            top_k=top_k,
            raw_frames=1,
            detected_frames=1,
        )

    def recognize_bim(
        self,
        video_path: str,
        top_k: int = 3,
        sample_every: int = 1,
    ) -> Dict:
        """Recognize a static number held across a short video.

        Probabilities are averaged over frames in which MediaPipe detects a
        hand. This mirrors the public API of the medical/general recognizers.
        """
        if sample_every < 1:
            raise ValueError("sample_every must be at least 1.")

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        raw_frames = 0
        sampled_frames = 0
        frame_probabilities = []

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                raw_frames += 1
                if (raw_frames - 1) % sample_every != 0:
                    continue

                sampled_frames += 1
                probabilities = self._infer_frame_probabilities(frame)
                if probabilities is not None:
                    frame_probabilities.append(probabilities)
        finally:
            cap.release()

        if raw_frames == 0:
            raise ValueError("The video contains no readable frames.")
        if not frame_probabilities:
            raise ValueError("No hand was detected in the video.")

        averaged_probabilities = np.mean(
            np.stack(frame_probabilities, axis=0),
            axis=0,
        ).astype(np.float32)

        return self._build_result(
            probabilities=averaged_probabilities,
            top_k=top_k,
            raw_frames=sampled_frames,
            detected_frames=len(frame_probabilities),
        )

    def close(self) -> None:
        if not self._closed:
            self.hand_landmarker.close()
            self._closed = True

    def __enter__(self) -> "SignBridgeNumberRecognizer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

