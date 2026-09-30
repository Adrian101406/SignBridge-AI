"""
SignBridge Medical Recognizer V2 — MCIE integration wrapper.

IMPORTANT:
Copy the proven V1 medical_recognizer_v1/recognition_inference.py into this
V2 folder as recognition_core_v1.py. This wrapper deliberately reuses the
exact V1 preprocessing so MediaPipe extraction, trimming, interpolation,
scaling, compact-face features and geometry do not drift between V1 and V2.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RECOGNIZER_NAME = "SignBridge_Medical_Recognizer_V2"
DOMAIN = "medical"

DEFAULT_ACCEPT_CONFIDENCE = 0.80
DEFAULT_ACCEPT_MARGIN = 0.25
DEFAULT_CONFIRM_CONFIDENCE = 0.55
DEFAULT_CONFIRM_MARGIN = 0.10


def _load_core_module(core_path: Path):
    if not core_path.is_file():
        raise FileNotFoundError(
            f"Missing V1 preprocessing core: {core_path}\n"
            "Copy medical_recognizer_v1/recognition_inference.py into the V2 "
            "folder and rename it recognition_core_v1.py."
        )

    spec = importlib.util.spec_from_file_location(
        "signbridge_medical_v1_core", str(core_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inference core from: {core_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "SignBridgeMedicalRecognizer"):
        raise AttributeError(
            "The V1 core does not expose SignBridgeMedicalRecognizer."
        )
    return module


def _normalize_class_map(raw_map: Dict[Any, Any]) -> Dict[int, Dict[str, str]]:
    normalized: Dict[int, Dict[str, str]] = {}

    for raw_id, value in raw_map.items():
        class_id = int(raw_id)

        if isinstance(value, dict):
            gloss = (
                value.get("gloss")
                or value.get("malay")
                or value.get("label")
                or value.get("name")
            )
            english = value.get("english") or value.get("translation") or gloss
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            gloss, english = value[0], value[1]
        else:
            gloss = str(value)
            english = str(value)

        if gloss is None:
            raise ValueError(
                f"Invalid label mapping for class {class_id}: {value}"
            )

        normalized[class_id] = {
            "gloss": str(gloss),
            "english": str(english),
        }

    return normalized


def _load_label_mapping(
    label_mapping_path: Optional[Path],
    core_module: Any,
) -> Dict[int, Dict[str, str]]:
    if label_mapping_path is not None:
        if not label_mapping_path.is_file():
            raise FileNotFoundError(label_mapping_path)

        data = json.loads(label_mapping_path.read_text(encoding="utf-8"))
        if "class_map" in data and isinstance(data["class_map"], dict):
            data = data["class_map"]
        return _normalize_class_map(data)

    if hasattr(core_module, "CLASS_MAP"):
        return _normalize_class_map(core_module.CLASS_MAP)

    raise ValueError(
        "No label_mapping.json supplied and the V1 core has no CLASS_MAP."
    )


class SignBridgeMedicalRecognizer:
    RECOGNIZER_NAME = RECOGNIZER_NAME
    DOMAIN = DOMAIN

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        pose_task_path: str,
        hand_task_path: str,
        face_task_path: str,
        label_mapping_path: Optional[str] = None,
        core_inference_path: Optional[str] = None,
        accept_confidence: float = DEFAULT_ACCEPT_CONFIDENCE,
        accept_margin: float = DEFAULT_ACCEPT_MARGIN,
        confirm_confidence: float = DEFAULT_CONFIRM_CONFIDENCE,
        confirm_margin: float = DEFAULT_CONFIRM_MARGIN,
    ) -> None:
        package_dir = Path(__file__).resolve().parent

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.pose_task_path = Path(pose_task_path)
        self.hand_task_path = Path(hand_task_path)
        self.face_task_path = Path(face_task_path)
        self.core_inference_path = Path(
            core_inference_path
            if core_inference_path is not None
            else package_dir / "recognition_core_v1.py"
        )
        self.label_mapping_path = (
            Path(label_mapping_path) if label_mapping_path else None
        )

        required = {
            "V2 model": self.model_path,
            "feature scaler": self.scaler_path,
            "pose landmarker": self.pose_task_path,
            "hand landmarker": self.hand_task_path,
            "face landmarker": self.face_task_path,
            "V1 preprocessing core": self.core_inference_path,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required Medical Recognizer V2 files:\n"
                + "\n".join(missing)
            )

        self.accept_confidence = float(accept_confidence)
        self.accept_margin = float(accept_margin)
        self.confirm_confidence = float(confirm_confidence)
        self.confirm_margin = float(confirm_margin)

        for name, value in {
            "accept_confidence": self.accept_confidence,
            "accept_margin": self.accept_margin,
            "confirm_confidence": self.confirm_confidence,
            "confirm_margin": self.confirm_margin,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1, got {value}."
                )

        self._core_module = _load_core_module(self.core_inference_path)
        self.class_map = _load_label_mapping(
            self.label_mapping_path, self._core_module
        )

        self.core = self._core_module.SignBridgeMedicalRecognizer(
            model_path=str(self.model_path),
            scaler_path=str(self.scaler_path),
            pose_task_path=str(self.pose_task_path),
            hand_task_path=str(self.hand_task_path),
            face_task_path=str(self.face_task_path),
        )

        self.model = self.core.model

        output_shape = self.model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        n_classes = int(output_shape[-1])
        if n_classes != len(self.class_map):
            raise ValueError(
                "Model output count does not match label mapping: "
                f"{n_classes} outputs vs {len(self.class_map)} labels."
            )

    def preprocess_video(self, video_path: str):
        return self.core.preprocess_video(video_path)

    def _prepare_model_inputs(self, sequence):
        return self.core._prepare_model_inputs(sequence)

    def _decision(self, top1_conf: float, top2_conf: float):
        margin = float(top1_conf - top2_conf)

        if (
            top1_conf >= self.accept_confidence
            and margin >= self.accept_margin
        ):
            return "ACCEPT", margin

        if (
            top1_conf >= self.confirm_confidence
            and margin >= self.confirm_margin
        ):
            return "CONFIRM", margin

        return "VERIFY", margin

    def _predict_from_inputs(
        self,
        model_inputs: Any,
        quality: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        probabilities = np.asarray(
            self.model.predict(model_inputs, verbose=0)[0],
            dtype=np.float32,
        )

        top_k = min(int(top_k), len(probabilities))
        order = np.argsort(probabilities)[::-1][:top_k]

        candidates: List[Dict[str, Any]] = []
        for rank, raw_id in enumerate(order, start=1):
            class_id = int(raw_id)
            label = self.class_map[class_id]

            candidates.append(
                {
                    "rank": rank,
                    "class_id": class_id,
                    "gloss": label["gloss"],
                    "english": label["english"],
                    "confidence": float(probabilities[class_id]),
                }
            )

        top1 = candidates[0]
        top2_conf = (
            candidates[1]["confidence"] if len(candidates) >= 2 else 0.0
        )
        decision, margin = self._decision(
            top1["confidence"], top2_conf
        )

        quality_out: Dict[str, Any] = {}
        if quality:
            for key, value in quality.items():
                if isinstance(value, np.generic):
                    value = value.item()
                quality_out[str(key)] = value

        return {
            "recognizer": self.RECOGNIZER_NAME,
            "domain": self.DOMAIN,
            "top1": {
                "class_id": top1["class_id"],
                "gloss": top1["gloss"],
                "english": top1["english"],
                "confidence": top1["confidence"],
            },
            "top_k": candidates,
            "confidence_margin": margin,
            "decision": decision,
            "quality": quality_out,
        }

    def recognize_bim(
        self,
        video_path: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        sequence, quality = self.preprocess_video(video_path)
        model_inputs = self._prepare_model_inputs(sequence)

        return self._predict_from_inputs(
            model_inputs=model_inputs,
            quality=quality,
            top_k=top_k,
        )

    def recognize_sequence(
        self,
        sequence: np.ndarray,
        quality: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        model_inputs = self._prepare_model_inputs(sequence)
        return self._predict_from_inputs(
            model_inputs=model_inputs,
            quality=quality,
            top_k=top_k,
        )

    def close(self) -> None:
        if hasattr(self.core, "close"):
            self.core.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
