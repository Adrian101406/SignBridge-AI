from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol

from .config import DEFAULT_SETTINGS, Settings
from .contracts import RecognitionResult


class Recognizer(Protocol):
    def recognize_bim(self, video_path: str, top_k: int = 3) -> dict[str, Any]: ...


class RecognitionUnavailable(RuntimeError):
    """A requested Person 1 recognizer could not produce a result."""


@dataclass(frozen=True)
class Person1Paths:
    root: Path
    general_dir: Path
    medical_dir: Path
    number_dir: Path
    mediapipe_dir: Path

    @classmethod
    def from_person1_root(cls, root: Path) -> "Person1Paths":
        root = Path(root).resolve()
        return cls(
            root=root,
            general_dir=root / "general_recognizer_v1",
            medical_dir=root / "medical_recognizer_v2",
            number_dir=root / "number_recognizer_v1",
            mediapipe_dir=root / "mediapipe",
        )

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.general_dir / "best_zenodo_lstm.keras",
            self.general_dir / "zenodo_xyz_scaling.npz",
            self.general_dir / "general_label_mapping.json",
            self.medical_dir / "best_domain_alignment_lambda0005.keras",
            self.medical_dir / "three_branch_feature_scaling.npz",
            self.medical_dir / "label_mapping.json",
            self.number_dir / "models" / "number_model.tflite",
            self.number_dir / "models" / "labels.txt",
            self.mediapipe_dir / "pose_landmarker.task",
            self.mediapipe_dir / "hand_landmarker.task",
            self.mediapipe_dir / "face_landmarker.task",
        )


class Person1RecognizerLoader:
    def __init__(self, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.paths = Person1Paths.from_person1_root(settings.person1_dir)
        self._loaded: dict[str, Recognizer] = {}

    def get(self, domain: str) -> Recognizer:
        normalized = domain.strip().lower()
        if normalized not in {"general", "medical", "number"}:
            raise RecognitionUnavailable(f"Unknown recognizer domain: {domain}")
        if normalized not in self._loaded:
            try:
                self._loaded[normalized] = getattr(self, f"_load_{normalized}")()
            except Exception as error:
                raise RecognitionUnavailable(
                    f"{normalized} recognizer could not load: {error}"
                ) from error
        return self._loaded[normalized]

    def close(self) -> None:
        for recognizer in self._loaded.values():
            close = getattr(recognizer, "close", None)
            if callable(close):
                close()
        self._loaded.clear()

    def _load_general(self) -> Recognizer:
        module = _load_module(
            "signbridge_person1_general",
            self.paths.general_dir / "general_inference.py",
            self.paths.general_dir,
        )
        return module.SignBridgeGeneralRecognizer(
            model_path=str(self.paths.general_dir / "best_zenodo_lstm.keras"),
            scaler_path=str(self.paths.general_dir / "zenodo_xyz_scaling.npz"),
            label_mapping_path=str(
                self.paths.general_dir / "general_label_mapping.json"
            ),
            pose_task_path=str(self.paths.mediapipe_dir / "pose_landmarker.task"),
            hand_task_path=str(self.paths.mediapipe_dir / "hand_landmarker.task"),
        )

    def _load_medical(self) -> Recognizer:
        module = _load_module(
            "signbridge_person1_medical",
            self.paths.medical_dir / "recognition_inference.py",
            self.paths.medical_dir,
        )
        return module.SignBridgeMedicalRecognizer(
            model_path=str(
                self.paths.medical_dir / "best_domain_alignment_lambda0005.keras"
            ),
            scaler_path=str(
                self.paths.medical_dir / "three_branch_feature_scaling.npz"
            ),
            label_mapping_path=str(self.paths.medical_dir / "label_mapping.json"),
            pose_task_path=str(self.paths.mediapipe_dir / "pose_landmarker.task"),
            hand_task_path=str(self.paths.mediapipe_dir / "hand_landmarker.task"),
            face_task_path=str(self.paths.mediapipe_dir / "face_landmarker.task"),
        )

    def _load_number(self) -> Recognizer:
        module = _load_module(
            "signbridge_person1_number",
            self.paths.number_dir / "number_inference.py",
            self.paths.number_dir,
        )
        return module.SignBridgeNumberRecognizer(
            model_path=str(self.paths.number_dir / "models" / "number_model.tflite"),
            labels_path=str(self.paths.number_dir / "models" / "labels.txt"),
            hand_task_path=str(self.paths.mediapipe_dir / "hand_landmarker.task"),
        )


class RecognitionService:
    def __init__(
        self,
        *,
        recognizers: Mapping[str, Recognizer] | None = None,
        loader: Person1RecognizerLoader | None = None,
        temporary_dir: Path | None = None,
        stable_confidence: float = 0.8,
    ) -> None:
        self._recognizers = dict(recognizers or {})
        self._loader = loader
        self._temporary_dir = Path(temporary_dir) if temporary_dir else None
        self._stable_confidence = stable_confidence

    def recognize_clip(
        self,
        data: bytes,
        *,
        suffix: str,
        request_id: str,
        expected_domains: tuple[str, ...],
    ) -> RecognitionResult:
        if not data:
            raise RecognitionUnavailable("Recognition clip cannot be empty")
        if suffix.lower() not in {".webm", ".mp4", ".avi"}:
            raise RecognitionUnavailable(f"Unsupported recognition clip type: {suffix}")
        if not expected_domains:
            raise RecognitionUnavailable("At least one recognizer domain is required")
        if self._temporary_dir:
            self._temporary_dir.mkdir(parents=True, exist_ok=True)

        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix.lower(),
                dir=self._temporary_dir,
                delete=False,
            ) as clip:
                clip.write(data)
                path = Path(clip.name)

            observations: list[RecognitionResult] = []
            errors: list[str] = []
            for domain in expected_domains:
                normalized = domain.strip().lower()
                try:
                    recognizer = self._recognizers.get(normalized)
                    if recognizer is None and self._loader is not None:
                        recognizer = self._loader.get(normalized)
                    if recognizer is None:
                        raise RecognitionUnavailable("recognizer is not configured")
                    raw = recognizer.recognize_bim(str(path), top_k=3)
                    observations.append(
                        self._normalize(raw, normalized, request_id=request_id)
                    )
                except Exception as error:
                    errors.append(f"{normalized}: {error}")
            if not observations:
                raise RecognitionUnavailable("; ".join(errors))
            return max(observations, key=lambda result: result.confidence)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _normalize(
        self,
        raw: Mapping[str, Any],
        fallback_domain: str,
        *,
        request_id: str,
    ) -> RecognitionResult:
        top1 = raw.get("top1")
        if not isinstance(top1, Mapping):
            raise RecognitionUnavailable("recognizer result is missing top1")
        gloss = str(top1.get("gloss", "")).strip()
        confidence = float(top1.get("confidence", 0.0))
        domain = str(raw.get("domain", fallback_domain)).lower()
        decision = raw.get("decision")
        if isinstance(decision, Mapping):
            decision = decision.get("status")
        decision_allows_append = str(decision or "ACCEPT").upper() == "ACCEPT"
        return RecognitionResult(
            gloss=gloss,
            confidence=confidence,
            recognizer_type=domain,
            stable=(confidence >= self._stable_confidence and decision_allows_append),
            timestamp=time.time(),
            request_id=request_id,
        )


class SignStabilizer:
    def __init__(self, required_matches: int = 2) -> None:
        if required_matches < 1:
            raise ValueError("required_matches must be at least 1")
        self._required_matches = required_matches
        self._candidate: str | None = None
        self._matches = 0
        self._latched: str | None = None

    def observe(self, result: RecognitionResult | None) -> str | None:
        if result is None:
            self._candidate = None
            self._matches = 0
            self._latched = None
            return None
        if not result.stable:
            self._candidate = None
            self._matches = 0
            return None
        label = result.gloss
        if label == self._latched:
            return None
        if label != self._candidate:
            self._candidate = label
            self._matches = 1
        else:
            self._matches += 1
        if self._matches < self._required_matches:
            return None
        self._latched = label
        self._candidate = None
        self._matches = 0
        return label


def _load_module(name: str, file_path: Path, import_root: Path) -> ModuleType:
    import_root_text = str(import_root)
    added = import_root_text not in sys.path
    if added:
        sys.path.insert(0, import_root_text)
    try:
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            sys.path.remove(import_root_text)


def main() -> int:
    loader = Person1RecognizerLoader()
    failed = False
    for domain in ("number", "medical", "general"):
        try:
            loader.get(domain)
            print(f"[OK] {domain} recognizer loaded")
        except RecognitionUnavailable as error:
            failed = True
            print(f"[UNAVAILABLE] {error}")
    loader.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
