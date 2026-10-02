from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from signbridge_app.asr_service import (
    AsrService,
    AsrUnavailable,
    UnsupportedAudio,
    validate_wav,
)


def wav_bytes(*, sample_rate: int = 16000, channels: int = 1, frames: int = 320) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames * channels)
    return output.getvalue()


class FakeAsrEngine:
    def __init__(self, result: dict | None = None, error: Exception | None = None) -> None:
        self.result = result or {
            "speech_recognition": {"transcript": "", "speech_segments": 0},
            "quality_check": {"status": "RETRY", "reason": "No speech detected"},
            "healthcare_validation": {
                "validation_status": "RETRY",
                "risk_level": "HIGH",
                "detected_concepts": [],
                "requires_confirmation": True,
                "message": "No usable transcript detected. Please repeat.",
            },
        }
        self.error = error
        self.path: Path | None = None

    def transcribe(self, audio_file: str | Path) -> dict:
        self.path = Path(audio_file)
        assert self.path.is_file()
        if self.error:
            raise self.error
        return self.result


def test_validate_wav_accepts_16khz_mono_pcm() -> None:
    metadata = validate_wav(wav_bytes())

    assert metadata.sample_rate == 16000
    assert metadata.channels == 1
    assert metadata.sample_width == 2
    assert metadata.frame_count == 320


@pytest.mark.parametrize(
    ("data", "message"),
    (
        pytest.param(b"not-wave", "valid WAV", id="malformed"),
        pytest.param(wav_bytes(sample_rate=44100), "16 kHz", id="wrong-rate"),
        pytest.param(wav_bytes(channels=2), "mono", id="stereo"),
    ),
)
def test_validate_wav_rejects_unsupported_audio(data: bytes, message: str) -> None:
    with pytest.raises(UnsupportedAudio, match=message):
        validate_wav(data)


def test_silence_returns_editable_proposal_with_warning(tmp_path: Path) -> None:
    engine = FakeAsrEngine()
    service = AsrService(engine=engine, temporary_dir=tmp_path)

    proposal = service.transcribe_wav(wav_bytes(), request_id="asr-1")

    assert proposal.original_text == ""
    assert proposal.requires_confirmation is True
    assert "No speech detected" in proposal.warnings
    assert engine.path is not None
    assert not engine.path.exists()


def test_medical_transcript_keeps_original_and_warning(tmp_path: Path) -> None:
    engine = FakeAsrEngine(
        result={
            "speech_recognition": {
                "transcript": "take five milligram",
                "speech_segments": 1,
            },
            "quality_check": {"status": "OK", "reason": "usable"},
            "healthcare_validation": {
                "validation_status": "CONFIRM",
                "risk_level": "MEDICAL",
                "detected_concepts": ["MEDICATION"],
                "requires_confirmation": True,
                "message": "Medical information detected.",
            },
        }
    )
    service = AsrService(engine=engine, temporary_dir=tmp_path)

    proposal = service.transcribe_wav(wav_bytes(), request_id="asr-1")

    assert proposal.original_text == "take five milligram"
    assert proposal.warnings == ("Medical information detected.",)


def test_asr_failure_is_wrapped_and_temp_file_is_removed(tmp_path: Path) -> None:
    engine = FakeAsrEngine(error=RuntimeError("GPU failure"))
    service = AsrService(engine=engine, temporary_dir=tmp_path)

    with pytest.raises(AsrUnavailable, match="GPU failure"):
        service.transcribe_wav(wav_bytes(), request_id="asr-1")

    assert tuple(tmp_path.iterdir()) == ()


def test_repetitive_whisper_hallucination_is_rejected_before_review(tmp_path: Path) -> None:
    repeated = ", ".join(["Oh"] * 80)
    engine = FakeAsrEngine(
        result={
            "speech_recognition": {"transcript": repeated, "speech_segments": 1},
            "quality_check": {"status": "OK", "reason": "Transcript contains usable text"},
            "healthcare_validation": {
                "validation_status": "PASS",
                "risk_level": "LOW",
                "detected_concepts": [],
                "requires_confirmation": False,
                "message": "No critical medical concept detected.",
            },
        }
    )
    service = AsrService(engine=engine, temporary_dir=tmp_path)

    with pytest.raises(AsrUnavailable, match="repetitive result"):
        service.transcribe_wav(wav_bytes(frames=16000), request_id="asr-repeat")

    assert tuple(tmp_path.iterdir()) == ()
