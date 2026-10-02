from __future__ import annotations

from pathlib import Path

import pytest

from signbridge_app.contracts import RecognitionResult
from signbridge_app.recognition_service import (
    Person1Paths,
    RecognitionService,
    RecognitionUnavailable,
    SignStabilizer,
)


class FakeRecognizer:
    def __init__(self, *, gloss: str, confidence: float, domain: str) -> None:
        self.gloss = gloss
        self.confidence = confidence
        self.domain = domain
        self.seen_path: Path | None = None

    def recognize_bim(self, video_path: str, top_k: int = 3) -> dict:
        self.seen_path = Path(video_path)
        assert self.seen_path.read_bytes() == b"clip-data"
        return {
            "domain": self.domain,
            "top1": {"gloss": self.gloss, "confidence": self.confidence},
            "decision": {"status": "ACCEPT"},
        }


def result(label: str, *, stable: bool = True) -> RecognitionResult:
    return RecognitionResult(
        gloss=label,
        confidence=0.92,
        recognizer_type="medical",
        stable=stable,
        timestamp=1.0,
        request_id="request-1",
    )


def test_recognition_service_normalizes_best_domain_and_removes_clip(tmp_path: Path) -> None:
    medical = FakeRecognizer(gloss="PAIN", confidence=0.91, domain="medical")
    general = FakeRecognizer(gloss="HELLO", confidence=0.72, domain="general")
    service = RecognitionService(
        recognizers={"medical": medical, "general": general},
        temporary_dir=tmp_path,
    )

    observed = service.recognize_clip(
        b"clip-data",
        suffix=".webm",
        request_id="request-1",
        expected_domains=("general", "medical"),
    )

    assert observed.gloss == "PAIN"
    assert observed.recognizer_type == "medical"
    assert observed.stable is True
    assert medical.seen_path is not None
    assert not medical.seen_path.exists()
    assert not general.seen_path.exists()


def test_recognition_service_wraps_person1_failure(tmp_path: Path) -> None:
    class BrokenRecognizer:
        def recognize_bim(self, video_path: str, top_k: int = 3) -> dict:
            raise ValueError("bad scaler")

    service = RecognitionService(
        recognizers={"general": BrokenRecognizer()},
        temporary_dir=tmp_path,
    )

    with pytest.raises(RecognitionUnavailable, match="general.*bad scaler"):
        service.recognize_clip(
            b"clip-data",
            suffix=".webm",
            request_id="request-1",
            expected_domains=("general",),
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_person1_paths_never_use_person2_team_package(tmp_path: Path) -> None:
    paths = Person1Paths.from_person1_root(tmp_path / "uploads" / "person1")

    assert all(
        str(path).startswith(str(tmp_path / "uploads" / "person1"))
        for path in paths.required_files()
    )


def test_held_sign_appends_once_until_release() -> None:
    gate = SignStabilizer(required_matches=2)

    assert gate.observe(result("PAIN")) is None
    assert gate.observe(result("PAIN")) == "PAIN"
    assert gate.observe(result("PAIN")) is None
    assert gate.observe(None) is None
    assert gate.observe(result("PAIN")) is None
    assert gate.observe(result("PAIN")) == "PAIN"


def test_different_sign_must_stabilize_before_append() -> None:
    gate = SignStabilizer(required_matches=2)
    gate.observe(result("PAIN"))
    assert gate.observe(result("PAIN")) == "PAIN"

    assert gate.observe(result("HELLO")) is None
    assert gate.observe(result("HELLO")) == "HELLO"


def test_unstable_result_never_appends() -> None:
    gate = SignStabilizer(required_matches=2)

    assert gate.observe(result("PAIN", stable=False)) is None
    assert gate.observe(result("PAIN", stable=False)) is None

