from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Literal

from .config import DEFAULT_SETTINGS, Settings
from .contracts import MedicalProposal
from .medical_guard import compare_critical_terms


# Transformers otherwise initializes TensorFlow too, wasting scarce laptop RAM.
os.environ.setdefault("USE_TF", "0")


MCIE_MAX_NEW_TOKENS = 450

_MEDICAL_GLOSS_CONCEPTS = {
    "badan": "BODY",
    "bahu": "SHOULDER",
    "jururawat": "NURSE",
    "jari": "FINGERS",
    "kaki": "FEET",
    "kulit": "SKIN",
    "leher": "NECK",
    "doktor": "DOCTOR",
    "bibir": "LIPS",
    "gigi": "TEETH",
    "hidung": "NOSE",
    "lidah": "TONGUE",
    "mata": "EYES",
    "mulut": "MOUTH",
    "lemah": "WEAKNESS",
    "tulang": "BONES",
    "sinar-x": "X_RAY",
    "tulang rusuk": "RIB_CAGE",
    "darah": "BLOOD",
    "jantung": "HEART",
    "kesakitan": "PAIN",
    "buah pinggang": "KIDNEY",
    "asma": "ASTHMA",
    "batuk": "COUGH",
    "bengkak": "SWELLING",
    "gatal-gatal": "ITCHING",
    "muntah": "VOMITING",
    "selesema": "FLU",
    "demam": "FEVER",
    "sakit kepala": "HEADACHE",
    "sakit perut": "STOMACH_PAIN",
    "sakit pinggang": "BACK_PAIN",
    "sakit tekak": "SORE_THROAT",
    "bedah": "SURGERY",
    "kecemasan": "EMERGENCY",
    "krim": "CREAM",
    "pil": "PILL",
    "mengandung": "PREGNANCY",
    "penat": "FATIGUE",
    "perut": "STOMACH",
}
_MAX_MEDICAL_GLOSS_WORDS = max(
    len(gloss.split()) for gloss in _MEDICAL_GLOSS_CONCEPTS
)

_PATIENT_SYMPTOM_SENTENCES = {
    "ASTHMA": "I have asthma.",
    "BACK_PAIN": "I have back pain.",
    "BREATHING_DIFFICULTY": "I have difficulty breathing.",
    "CHEST_PAIN": "I have chest pain.",
    "COUGH": "I have a cough.",
    "FATIGUE": "I feel tired.",
    "FEVER": "I have a fever.",
    "FLU": "I have the flu.",
    "HEADACHE": "I have a headache.",
    "ITCHING": "I feel itchy.",
    "PAIN": "I am in pain.",
    "SORE_THROAT": "I have a sore throat.",
    "STOMACH_PAIN": "I have stomach pain.",
    "SWELLING": "I have swelling.",
    "VOMITING": "I am vomiting.",
    "WEAKNESS": "I feel weak.",
}


def _patient_sequences(source_text: str) -> tuple[list[str], list[str]]:
    words = source_text.split()
    glosses: list[str] = []
    concepts: list[str] = []
    index = 0
    while index < len(words):
        matched = False
        max_width = min(_MAX_MEDICAL_GLOSS_WORDS, len(words) - index)
        for width in range(max_width, 0, -1):
            gloss = " ".join(words[index:index + width])
            concept = _MEDICAL_GLOSS_CONCEPTS.get(gloss.casefold())
            if concept is None:
                continue
            glosses.append(gloss)
            concepts.append(concept)
            index += width
            matched = True
            break
        if not matched:
            glosses.append(words[index])
            concepts.append(words[index])
            index += 1
    return glosses, concepts


class McieUnavailable(RuntimeError):
    pass


class LocalGgufLlm:
    """Person 2's generate_json interface backed by a low-memory local GGUF."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama

        self._model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=max(2, (os.cpu_count() or 4) // 2),
            n_gpu_layers=0,
            verbose=False,
        )

    def generate_json(self, *, system_prompt: str, user_payload: dict[str, Any], max_new_tokens: int = 180) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Interpret this SignBridge input:\n" + json.dumps(user_payload, ensure_ascii=False)},
            ],
            max_tokens=max_new_tokens,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = str(response["choices"][0]["message"]["content"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        return {"parsed": parsed, "raw_output": raw, "latency_seconds": time.perf_counter() - started}


class McieService:
    def __init__(
        self,
        settings: Settings,
        *,
        engine: Any | None = None,
        gpu_lock: threading.Lock | None = None,
    ) -> None:
        self.settings = settings
        self._engine = engine
        self._gpu_lock = gpu_lock or threading.Lock()
        self._context_state: Any | None = None

    def propose(
        self,
        source_text: str,
        sender: Literal["patient", "doctor"],
        request_id: str,
    ) -> MedicalProposal:
        untouched_source = str(source_text)
        try:
            with self._gpu_lock:
                engine = self._get_engine()
                context = deepcopy(self._get_context_state())
                if sender == "doctor":
                    result = engine.doctor_text(text=untouched_source, state=context)
                else:
                    glosses, concepts = _patient_sequences(untouched_source)
                    result = self._contextual_duration_result(
                        source_text=untouched_source,
                        context=context,
                        engine=engine,
                    )
                    if result is None:
                        result = engine.patient_bim(
                            gloss_sequence=glosses,
                            concept_sequence=concepts,
                            state=context,
                        )
        except Exception as exc:
            raise McieUnavailable(f"Medical intelligence failed: {exc}") from exc

        output = result.get("output", {}) if isinstance(result, dict) else {}
        suggestion = output.get("simplified_text_en") or output.get("simplified_text_ms")
        if not isinstance(suggestion, str) or not suggestion.strip():
            suggestion = untouched_source

        interpretation = result.get("interpretation", {}) if isinstance(result, dict) else {}
        safety = result.get("safety", {}) if isinstance(result, dict) else {}
        safety_reasons = [str(value) for value in safety.get("reasons", [])]
        if sender == "patient" and safety_reasons:
            suggestion = untouched_source
        elif sender == "patient" and interpretation.get("intent") == "SYMPTOM_REPORT":
            concepts = interpretation.get("medical_concepts", [])
            if isinstance(concepts, list) and len(concepts) == 1:
                suggestion = _PATIENT_SYMPTOM_SENTENCES.get(concepts[0], suggestion)
        flags = list(safety_reasons)
        flags.extend(str(value) for value in interpretation.get("medical_concepts", []))
        return MedicalProposal(
            source_text=untouched_source,
            suggested_text=suggestion,
            medical_flags=tuple(dict.fromkeys(flags)),
            changed_critical_terms=compare_critical_terms(untouched_source, suggestion),
            request_id=request_id,
        )

    def record_confirmed(
        self,
        sender: Literal["patient", "doctor"],
        text: str,
    ) -> None:
        if sender != "doctor":
            return
        with self._gpu_lock:
            self._ensure_person2_import_path()
            rule_module = importlib.import_module("signbridge.rule_engine")
            anchor = rule_module.build_rule_anchor(text)
            self._get_context_state().update_from_doctor_mcie(
                text,
                {
                    "interpretation": {
                        "intent": anchor["intent"],
                        "entities": anchor["entities"],
                        "medical_concepts": anchor["medical_concepts"],
                    }
                },
            )

    def clear_context(self) -> None:
        with self._gpu_lock:
            self._context_state = self._new_context_state()

    def _contextual_duration_result(
        self,
        *,
        source_text: str,
        context: Any,
        engine: Any,
    ) -> dict[str, Any] | None:
        if context.last_doctor_intent != "SYMPTOM_DURATION":
            return None
        number = source_text.strip()
        if not re.fullmatch(r"10|[0-9]", number):
            return None
        question = (context.last_doctor_text or "").casefold()
        units = []
        if re.search(r"\b(?:day|days|hari)\b", question):
            units.append(("DAY", "day"))
        if re.search(r"\b(?:hour|hours|jam)\b", question):
            units.append(("HOUR", "hour"))
        if len(units) != 1:
            return None
        unit_concept, unit_gloss = units[0]
        patient_bim_module = importlib.import_module("signbridge.patient_bim")
        return patient_bim_module.process_patient_token_sequence(
            sign_tokens=[
                {
                    "success": True,
                    "gloss": number,
                    "concept": number,
                    "recognizer_type": "number",
                    "requires_confirmation": False,
                    "requires_retry": False,
                },
                {
                    "success": True,
                    "gloss": unit_gloss,
                    "concept": unit_concept,
                    "recognizer_type": "confirmed_doctor_context",
                    "requires_confirmation": False,
                    "requires_retry": False,
                },
            ],
            state=context,
            mcie=engine,
        )

    def _ensure_person2_import_path(self) -> None:
        person2_root = str(self.settings.person2_dir)
        if person2_root not in sys.path:
            sys.path.insert(0, person2_root)

    def _new_context_state(self) -> Any:
        self._ensure_person2_import_path()
        conversation_module = importlib.import_module("signbridge.conversation")
        return conversation_module.ConversationState()

    def _get_context_state(self) -> Any:
        if self._context_state is None:
            self._context_state = self._new_context_state()
        return self._context_state

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        self._ensure_person2_import_path()
        mcie_module = importlib.import_module("signbridge.mcie_ai")
        llm = LocalGgufLlm(str(self.settings.mcie_gguf_path))
        self._engine = mcie_module.AIEnhancedMCIE(
            llm,
            max_new_tokens=MCIE_MAX_NEW_TOKENS,
        )
        return self._engine


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", metavar="TEXT")
    args = parser.parse_args()
    if args.smoke_check is None:
        parser.error("--smoke-check TEXT is required")
    started = time.perf_counter()
    proposal = McieService(DEFAULT_SETTINGS).propose(args.smoke_check, "doctor", "smoke")
    print(json.dumps(proposal.model_dump(), indent=2))
    try:
        import torch
        print(f"peak_vram_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
    except Exception:
        pass
    print(f"elapsed_seconds={time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
