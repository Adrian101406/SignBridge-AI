import json
import re
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .healthcare_validation import validate_healthcare_transcript


class SignBridgeASR:
    """Latest runnable ASR pipeline extracted from the SignBridge Colab prototype."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config.json"
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))

        self.model_id = config["model_id"]
        self.target_sample_rate = int(config.get("target_sample_rate", 16000))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=0 if torch.cuda.is_available() else -1
        )
        self.vad_model = load_silero_vad()

    @staticmethod
    def clean_transcript(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    @staticmethod
    def check_transcript_quality(text: str) -> dict:
        text = text.strip()
        if not text:
            return {"status": "RETRY", "reason": "No speech detected"}
        if len(text.split()) < 2:
            return {"status": "CHECK", "reason": "Transcript is very short"}
        return {"status": "OK", "reason": "Transcript contains usable text"}

    def transcribe_audio(self, audio_file: str) -> dict:
        audio, original_sr = librosa.load(audio_file, sr=None, mono=True)
        audio_16k = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=self.target_sample_rate
        )

        if len(audio_16k) == 0:
            return self._empty_result()

        max_amplitude = max(abs(audio_16k))
        if max_amplitude > 0:
            audio_16k = audio_16k / max_amplitude

        with tempfile.TemporaryDirectory(prefix="signbridge_asr_") as tmp:
            processed_file = Path(tmp) / "processed_audio_16k.wav"
            speech_only_file = Path(tmp) / "speech_only.wav"
            sf.write(processed_file, audio_16k, self.target_sample_rate)

            vad_audio = read_audio(str(processed_file), sampling_rate=self.target_sample_rate)
            timestamps = get_speech_timestamps(
                vad_audio,
                self.vad_model,
                sampling_rate=self.target_sample_rate
            )

            if not timestamps:
                return self._empty_result()

            segments = [vad_audio[s["start"]:s["end"]] for s in timestamps]
            speech_only_audio = torch.cat(segments)
            sf.write(speech_only_file, speech_only_audio.numpy(), self.target_sample_rate)

            result = self.pipeline(str(speech_only_file))
            transcript = self.clean_transcript(result["text"])
            quality = self.check_transcript_quality(transcript)
            healthcare_validation = validate_healthcare_transcript(transcript)

            return {
                "speech_recognition": {
                    "model": self.model_id,
                    "transcript": transcript,
                    "speech_segments": len(timestamps)
                },
                "quality_check": quality,
                "healthcare_validation": healthcare_validation
            }

    def _empty_result(self) -> dict:
        return {
            "speech_recognition": {
                "model": self.model_id,
                "transcript": "",
                "speech_segments": 0
            },
            "quality_check": {
                "status": "RETRY",
                "reason": "No speech detected"
            },
            "healthcare_validation": validate_healthcare_transcript("")
        }
