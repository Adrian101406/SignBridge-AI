from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = PROJECT_ROOT / "uploads" / "person2" / "signbridge" / "asr_engine.py"


def load_engine_module():
    spec = importlib.util.spec_from_file_location("test_person3_asr_engine", ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_whisper_decode_is_bounded_against_repetition(monkeypatch) -> None:
    module = load_engine_module()
    monkeypatch.setattr(module.librosa, "load", lambda *_args, **_kwargs: (np.ones(1600, dtype=np.float32), 16000))
    monkeypatch.setattr(module.librosa, "resample", lambda audio, **_kwargs: audio)
    monkeypatch.setattr(
        module,
        "get_speech_timestamps",
        lambda *_args, **_kwargs: [{"start": 0, "end": 1600}],
    )

    class RunawayUnlessBounded:
        def __call__(self, _audio, *, generate_kwargs=None):
            options = generate_kwargs or {}
            if options.get("max_new_tokens", 448) <= 96 and options.get("no_repeat_ngram_size", 0) >= 3:
                return {"text": "Hello."}
            return {"text": ", ".join(["Oh"] * 80)}

    engine = object.__new__(module.OfflineASR)
    engine.model_dir = Path("malaysian-whisper-small-v3")
    engine.vad_model = object()
    engine.pipe = RunawayUnlessBounded()

    result = engine.transcribe("unused.wav")

    assert result["speech_recognition"]["transcript"] == "Hello."
