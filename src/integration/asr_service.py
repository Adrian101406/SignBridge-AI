from __future__ import annotations

import argparse
import importlib.util
import io
import re
import tempfile
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from .config import DEFAULT_SETTINGS, Settings
from .contracts import TranscriptProposal


class UnsupportedAudio(ValueError):
    """The browser recording does not match the ASR audio contract."""


class AsrUnavailable(RuntimeError):
    """The local ASR engine could not transcribe a valid recording."""


class AsrEngine(Protocol):
    def transcribe(self, audio_file: str | Path) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int


def validate_wav(data: bytes) -> AudioMetadata:
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            metadata = AudioMetadata(
                sample_rate=audio.getframerate(),
                channels=audio.getnchannels(),
                sample_width=audio.getsampwidth(),
                frame_count=audio.getnframes(),
            )
            compression = audio.getcomptype()
    except (EOFError, wave.Error) as error:
        raise UnsupportedAudio("Recording must be a valid WAV file") from error
    if compression != "NONE":
        raise UnsupportedAudio("Recording must use uncompressed PCM WAV")
    if metadata.sample_rate != 16000:
        raise UnsupportedAudio("Recording must use a 16 kHz sample rate")
    if metadata.channels != 1:
        raise UnsupportedAudio("Recording must be mono")
    if metadata.sample_width != 2:
        raise UnsupportedAudio("Recording must use 16-bit PCM samples")
    return metadata


class AsrService:
    def __init__(
        self,
        settings: Settings = DEFAULT_SETTINGS,
        *,
        engine: AsrEngine | None = None,
        temporary_dir: Path | None = None,
        gpu_lock: Lock | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._temporary_dir = Path(temporary_dir) if temporary_dir else None
        self._gpu_lock = gpu_lock or Lock()

    def transcribe_wav(self, data: bytes, *, request_id: str) -> TranscriptProposal:
        validate_wav(data)
        if self._temporary_dir:
            self._temporary_dir.mkdir(parents=True, exist_ok=True)
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                dir=self._temporary_dir,
                delete=False,
            ) as audio:
                audio.write(data)
                path = Path(audio.name)
            try:
                with self._gpu_lock:
                    result = self._get_engine().transcribe(path)
            except Exception as error:
                raise AsrUnavailable(f"ASR failed: {error}") from error
            recognition = result.get("speech_recognition", {})
            text = str(recognition.get("transcript", ""))
            if _is_excessively_repetitive(text):
                raise AsrUnavailable(
                    "ASR produced a repetitive result. Please record again closer to the microphone."
                )
            return TranscriptProposal(
                original_text=text,
                warnings=_warnings_from_result(result),
                requires_confirmation=True,
                request_id=request_id,
            )
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _get_engine(self) -> AsrEngine:
        if self._engine is None:
            module_path = self._settings.person2_dir / "signbridge" / "asr_engine.py"
            spec = importlib.util.spec_from_file_location(
                "signbridge_person3_asr_engine",
                module_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load offline ASR from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._engine = module.OfflineASR(self._settings.whisper_model_dir)
        return self._engine


def _warnings_from_result(result: dict[str, Any]) -> tuple[str, ...]:
    warnings: list[str] = []
    quality = result.get("quality_check", {})
    if quality.get("status") != "OK" and quality.get("reason"):
        warnings.append(str(quality["reason"]))
    healthcare = result.get("healthcare_validation", {})
    if healthcare.get("validation_status") != "PASS" and healthcare.get("message"):
        message = str(healthcare["message"])
        if message not in warnings:
            warnings.append(message)
    return tuple(warnings)


def _is_excessively_repetitive(text: str) -> bool:
    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold(), flags=re.UNICODE)
    if len(words) < 8:
        return False
    counts = Counter(words)
    return len(counts) <= max(2, len(words) // 5) or max(counts.values()) / len(words) >= 0.6


def _silence_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local SignBridge ASR")
    parser.add_argument("--smoke-check", action="store_true")
    args = parser.parse_args()
    if not args.smoke_check:
        parser.print_help()
        return 0
    proposal = AsrService().transcribe_wav(_silence_wav(), request_id="smoke-asr")
    print(f"ASR loaded; silence warnings: {', '.join(proposal.warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
