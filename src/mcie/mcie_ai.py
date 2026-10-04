"""
SignBridge AI - Medical Conversation Intelligence Engine (MCIE)

Developed by Team Silence Love for Project Nexus 2026.

External AI model:
Qwen/Qwen3-4B-Instruct-2507
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

Model inference implemented using Hugging Face Transformers
and PyTorch.

The MCIE logic, rule-based semantic processing, conversation
state management, evidence validation, and safety mechanisms
were developed by the SignBridge AI team.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .conversation import ConversationState
if TYPE_CHECKING:
    from .llm_engine import LocalTransformersLLM
from .rule_engine import ALLOWED_INTENTS, CONCEPT_ALIASES, build_rule_anchor
from .types import build_unified_mcie_output

ALLOWED_MEDICAL_CONCEPTS: Set[str] = {
    "FEVER", "COUGH", "FLU", "VOMITING", "ITCHING", "SWELLING", "WEAKNESS",
    "FATIGUE", "PAIN", "HEADACHE", "STOMACH_PAIN", "BACK_PAIN", "SORE_THROAT",
    "CHEST_PAIN", "BREATHING_DIFFICULTY", "ASTHMA", "PREGNANCY", "ALLERGY",
    "MEDICATION", "PILL", "CREAM", "BLOOD", "HEART", "KIDNEY", "BLOOD_PRESSURE",
    "SURGERY", "X_RAY", "FOLLOW_UP", "EMERGENCY",
}

CRITICAL_CONCEPTS = {
    "EMERGENCY", "ALLERGY", "MEDICATION", "PILL", "CREAM",
    "CHEST_PAIN", "BREATHING_DIFFICULTY",
}

DURATION_UNITS = {"HOUR", "DAY", "WEEK", "MONTH"}
MEDICATION_TIMING = {"AFTER_MEALS", "BEFORE_MEALS", "WITH_MEALS", "MORNING", "NIGHT"}

SYSTEM_PROMPT = r"""
You are the semantic reasoning component of SignBridge MCIE.
SignBridge supports Malaysian healthcare communication between hearing staff and Deaf BIM users.
Inputs may contain Bahasa Melayu, English, or Malaysian code-switching such as U, you, nak, tak, kat, dekat.

You do NOT diagnose and you do NOT prescribe. Preserve the communicative meaning.
Never invent symptoms, diagnoses, medicines, quantities, frequency, timing, duration, allergies, or body locations.
For BIM input, use ONLY concepts supplied in concept_sequence or explicit conversation context.
Never create a recognizer candidate that was not supplied upstream.
Questions stay questions, instructions stay instructions, and patient statements stay patient statements.
"lepas makan" / "selepas makan" means AFTER_MEALS; never change it to AFTER_BREAKFAST.
If meaning is unclear, use intent UNKNOWN and set requires_confirmation true.
Return ONLY valid JSON with no Markdown.

Intent definitions:
SYMPTOM_REPORT = patient states a symptom.
SYMPTOM_CHECK = asks whether a symptom is present.
SYMPTOM_LOCATION = asks where pain/symptom is located.
SYMPTOM_DURATION = asks when it started or how long it has lasted.
MEDICATION_QUERY = asks what medicine the patient is taking/using.
MEDICATION_INSTRUCTION = gives instructions about taking/using medicine.
ALLERGY_CHECK = asks about allergy.
MEDICAL_PROCEDURE = states a measurement, scan, examination or procedure will be done.
FOLLOW_UP_INSTRUCTION = tells the patient to return later.
REST_INSTRUCTION = tells the patient to rest.
EMERGENCY = explicit emergency communication.
GENERAL_QUESTION = question not safely classifiable above.
GENERAL_COMMUNICATION = other general communication.
UNKNOWN = cannot safely determine meaning.

Examples:
"U rasa demam tak?" -> SYMPTOM_CHECK, [FEVER]
"Kat mana rasa sakit?" -> SYMPTOM_LOCATION, []
"U tengah ambil ubat apa?" -> MEDICATION_QUERY, [MEDICATION]
"Ambil satu pil sebelum makan." -> MEDICATION_INSTRUCTION, [PILL], quantity=1, timing=BEFORE_MEALS
"Bila mula batuk?" -> SYMPTOM_DURATION, [COUGH]
"Saya sakit kepala." -> SYMPTOM_REPORT, [HEADACHE]

Required JSON schema:
{
  "intent": "UNKNOWN",
  "medical_concepts": [],
  "entities": {
    "symptoms": [],
    "body_parts": [],
    "medications": [],
    "quantity": null,
    "frequency": null,
    "timing": null,
    "duration": {"quantity": null, "unit": null}
  },
  "simplified_text_en": null,
  "simplified_text_ms": null,
  "context_used": false,
  "evidence": [],
  "uncertainty": [],
  "requires_confirmation": false
}
""".strip()


def _unique_strings(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        if isinstance(value, str) and value not in out:
            out.append(value)
    return out


def _context_concepts(state: Optional[ConversationState]) -> Set[str]:
    if state is None:
        return set()
    return {
        x for x in [state.current_symptom, state.current_medication, state.current_body_location]
        if isinstance(x, str)
    }


def _text_concept_supported(concept: str, text: str, anchor: Dict[str, Any], evidence: Any) -> bool:
    if concept in (anchor.get("medical_concepts") or []):
        return True
    t = text.lower()
    if any(alias.lower() in t for alias in CONCEPT_ALIASES.get(concept, [])):
        return True
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict) or item.get("concept") != concept:
                continue
            source = str(item.get("source", "")).strip().lower()
            if source and source in t:
                return True
    return False


class AIEnhancedMCIE:
    def __init__(self, llm: "LocalTransformersLLM", max_new_tokens: int = 450):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def doctor_text(
        self,
        *,
        text: str,
        state: Optional[ConversationState] = None,
        asr_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(text).strip()
        if not text:
            return build_unified_mcie_output(
                source="doctor_speech",
                success=False,
                decision="RETRY",
                requires_retry=True,
                decision_reasons=["No valid transcript."],
            )

        anchor = build_rule_anchor(text)
        payload = {
            "mode": "doctor_speech",
            "raw_text": text,
            "rule_based_anchor": anchor,
            "conversation_context": state.ai_snapshot() if state else {},
        }
        generation = self.llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_payload=payload,
            max_new_tokens=self.max_new_tokens,
        )
        ai = generation.get("parsed")
        if not isinstance(ai, dict):
            return build_unified_mcie_output(
                source="doctor_speech",
                input_data={"raw_text": text, "rule_anchor": anchor},
                success=False,
                decision="CONFIRM",
                requires_confirmation=True,
                decision_reasons=["AI did not return valid structured JSON."],
                metadata={"raw_ai_output": generation.get("raw_output")},
            )

        issues: List[str] = []
        intent = ai.get("intent", "UNKNOWN")
        if intent not in ALLOWED_INTENTS:
            issues.append(f"Unsupported AI intent removed: {intent}")
            intent = anchor.get("intent", "UNKNOWN")
        if intent == "UNKNOWN" and anchor.get("intent") != "UNKNOWN":
            intent = anchor["intent"]

        final_concepts: List[str] = []
        for c in anchor.get("medical_concepts", []):
            if c in ALLOWED_MEDICAL_CONCEPTS and c not in final_concepts:
                final_concepts.append(c)
        for c in _unique_strings(ai.get("medical_concepts", [])):
            if c not in ALLOWED_MEDICAL_CONCEPTS:
                issues.append(f"Unsupported AI concept removed: {c}")
                continue
            if _text_concept_supported(c, text, anchor, ai.get("evidence", [])):
                if c not in final_concepts:
                    final_concepts.append(c)
            else:
                issues.append(f"AI concept lacked source support and was removed: {c}")

        ai_entities = ai.get("entities") if isinstance(ai.get("entities"), dict) else {}
        entities = dict(ai_entities)
        anchor_entities = anchor.get("entities", {}) or {}
        for field in ["quantity", "frequency", "timing"]:
            if anchor_entities.get(field) is not None:
                entities[field] = anchor_entities[field]
        entities.setdefault("symptoms", anchor_entities.get("symptoms", []))
        entities.setdefault("medications", anchor_entities.get("medications", []))
        entities.setdefault("duration", {"quantity": None, "unit": None})

        if entities.get("timing") not in MEDICATION_TIMING and entities.get("timing") is not None:
            issues.append("Unsupported medication timing removed.")
            entities["timing"] = None
        duration = entities.get("duration")
        if not isinstance(duration, dict):
            duration = {"quantity": None, "unit": None}
        if duration.get("unit") not in DURATION_UNITS and duration.get("unit") is not None:
            issues.append("Unsupported duration unit removed.")
            duration["unit"] = None
        entities["duration"] = duration

        en = ai.get("simplified_text_en") if isinstance(ai.get("simplified_text_en"), str) else None
        ms = ai.get("simplified_text_ms") if isinstance(ai.get("simplified_text_ms"), str) else None

        upstream_confirmed = False
        if isinstance(asr_payload, dict):
            upstream_confirmed = asr_payload.get("confirmation_status") in {"CONFIRMED", "NOT_REQUIRED"}

        requires_confirmation = bool(ai.get("requires_confirmation", False)) or intent == "UNKNOWN"
        if issues:
            # Unsupported concepts are not allowed to silently flow downstream.
            requires_confirmation = True
        if any(c in CRITICAL_CONCEPTS for c in final_concepts) and not upstream_confirmed:
            requires_confirmation = True
            issues.append("Critical medical information requires confirmation.")

        decision = "CONFIRM" if requires_confirmation else "ACCEPT"
        result = build_unified_mcie_output(
            source="doctor_speech",
            intent=intent,
            entities=entities,
            medical_concepts=final_concepts,
            input_data={"raw_text": text, "rule_anchor": anchor},
            context_used=bool(ai.get("context_used", False)),
            simplified_text_en=en,
            simplified_text_ms=ms,
            decision=decision,
            requires_confirmation=requires_confirmation,
            decision_reasons=issues,
            metadata={
                "engine": "AI_ENHANCED_MCIE_V2",
                "latency_seconds": generation.get("latency_seconds"),
                "raw_ai_output": generation.get("raw_output"),
            },
        )
        if state and result.get("success"):
            state.update_from_doctor_mcie(text, result)
        return result

    def patient_bim(
        self,
        *,
        gloss_sequence: List[str],
        concept_sequence: List[str],
        state: Optional[ConversationState] = None,
        upstream_requires_confirmation: bool = False,
        upstream_requires_retry: bool = False,
        recognition_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if upstream_requires_retry:
            return build_unified_mcie_output(
                source="patient_bim",
                input_data={"gloss_sequence": gloss_sequence, "concept_sequence": concept_sequence},
                success=False,
                decision="RETRY",
                requires_retry=True,
                decision_reasons=["Recognition layer requested retry."],
                metadata={"recognition_details": recognition_details or []},
            )

        payload = {
            "mode": "patient_bim",
            "gloss_sequence": gloss_sequence,
            "concept_sequence": concept_sequence,
            "conversation_context": state.ai_snapshot() if state else {},
        }
        generation = self.llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_payload=payload,
            max_new_tokens=self.max_new_tokens,
        )
        ai = generation.get("parsed")
        if not isinstance(ai, dict):
            return build_unified_mcie_output(
                source="patient_bim",
                input_data=payload,
                success=False,
                decision="CONFIRM",
                requires_confirmation=True,
                decision_reasons=["AI did not return valid structured JSON."],
                metadata={"recognition_details": recognition_details or []},
            )

        allowed_evidence = set(concept_sequence) | _context_concepts(state)
        issues: List[str] = []
        concepts: List[str] = []
        for c in _unique_strings(ai.get("medical_concepts", [])):
            if c not in ALLOWED_MEDICAL_CONCEPTS:
                issues.append(f"Unsupported AI concept removed: {c}")
                continue
            if c in allowed_evidence:
                concepts.append(c)
            else:
                issues.append(f"AI attempted to introduce concept absent from BIM/context: {c}")

        intent = ai.get("intent", "UNKNOWN")
        if intent not in ALLOWED_INTENTS:
            intent = "UNKNOWN"
            issues.append("Unsupported AI intent removed.")

        entities = ai.get("entities") if isinstance(ai.get("entities"), dict) else {}
        en = ai.get("simplified_text_en") if isinstance(ai.get("simplified_text_en"), str) else None
        ms = ai.get("simplified_text_ms") if isinstance(ai.get("simplified_text_ms"), str) else None
        requires_confirmation = (
            upstream_requires_confirmation
            or bool(ai.get("requires_confirmation", False))
            or bool(issues)
            or intent == "UNKNOWN"
        )
        decision = "CONFIRM" if requires_confirmation else "ACCEPT"
        result = build_unified_mcie_output(
            source="patient_bim",
            intent=intent,
            entities=entities,
            medical_concepts=concepts,
            input_data={"gloss_sequence": gloss_sequence, "concept_sequence": concept_sequence},
            context_used=bool(ai.get("context_used", False)),
            simplified_text_en=en,
            simplified_text_ms=ms,
            decision=decision,
            requires_confirmation=requires_confirmation,
            decision_reasons=issues,
            metadata={
                "engine": "AI_ENHANCED_MCIE_V2",
                "latency_seconds": generation.get("latency_seconds"),
                "raw_ai_output": generation.get("raw_output"),
                "recognition_details": recognition_details or [],
            },
        )
        if state and result.get("success"):
            state.update_from_patient_mcie(result)
        return result
