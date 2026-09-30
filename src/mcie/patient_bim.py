from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .conversation import ConversationState

if TYPE_CHECKING:
    from .mcie_ai import AIEnhancedMCIE
from .recognizer_adapter import process_recognizer_result
from .types import build_unified_mcie_output


def build_processed_sequence(
    results: List[Dict[str, Any]],
    state: ConversationState,
    *,
    number_is_medical: bool = False,
    number_confirmed: bool = False,
    min_hand_presence: float = 0.50,
    low_confidence: float = 0.50,
    close_margin: float = 0.15,
) -> Dict[str, Any]:
    """Backward-compatible helper for already-collected recognizer results."""
    signs = []
    for result in results:
        signs.append(
            process_recognizer_result(
                result,
                state,
                number_is_medical=number_is_medical,
                number_confirmed=number_confirmed,
                min_hand_presence=min_hand_presence,
                low_confidence=low_confidence,
                close_margin=close_margin,
            )
        )

    return build_sequence_from_tokens(signs)


def build_sequence_from_tokens(signs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one patient turn from isolated-sign tokens already in MCIE form."""
    signs = [deepcopy(s) for s in signs]
    if not signs:
        return {
            "success": False,
            "error": "No patient signs were supplied.",
            "signs": [],
            "gloss_sequence": [],
            "concept_sequence": [],
            "requires_retry": False,
            "requires_confirmation": False,
        }

    if any(not s.get("success", False) for s in signs):
        return {
            "success": False,
            "error": "At least one recognizer result is invalid.",
            "signs": signs,
            "gloss_sequence": [s.get("gloss") for s in signs],
            "concept_sequence": [s.get("concept") for s in signs],
            "requires_retry": True,
            "requires_confirmation": False,
        }

    requires_retry = any(s.get("requires_retry", False) for s in signs)
    requires_confirmation = any(s.get("requires_confirmation", False) for s in signs)
    return {
        "success": not requires_retry,
        "signs": signs,
        "gloss_sequence": [s.get("gloss") for s in signs],
        "concept_sequence": [s.get("concept") for s in signs],
        "requires_retry": requires_retry,
        "requires_confirmation": requires_confirmation,
    }


def _extract_duration(sequence: Dict[str, Any]) -> Dict[str, Any]:
    quantities: List[int] = []
    units: List[str] = []
    for sign in sequence.get("signs", []):
        if sign.get("recognizer_type") == "number":
            try:
                value = int(str(sign.get("gloss", "")).strip())
                if 0 <= value <= 10:
                    quantities.append(value)
            except ValueError:
                pass
        concept = sign.get("concept")
        if concept in {"DAY", "HOUR"}:
            units.append(concept)

    quantities = list(dict.fromkeys(quantities))
    units = list(dict.fromkeys(units))
    return {
        "complete": len(quantities) == 1 and len(units) == 1,
        "quantity": quantities[0] if len(quantities) == 1 else None,
        "unit": units[0] if len(units) == 1 else None,
    }


def _duration_sentence(symptom: str, quantity: int, unit: str) -> Dict[str, Optional[str]]:
    en_symptom = {
        "HEADACHE": "a headache", "FEVER": "a fever", "COUGH": "a cough",
        "STOMACH_PAIN": "stomach pain", "BACK_PAIN": "back pain",
        "SORE_THROAT": "a sore throat", "PAIN": "pain",
    }.get(symptom)
    ms_symptom = {
        "HEADACHE": "sakit kepala", "FEVER": "demam", "COUGH": "batuk",
        "STOMACH_PAIN": "sakit perut", "BACK_PAIN": "sakit pinggang",
        "SORE_THROAT": "sakit tekak", "PAIN": "kesakitan",
    }.get(symptom)
    if not en_symptom or not ms_symptom:
        return {"en": None, "ms": None}
    en_unit = "day" if unit == "DAY" else "hour"
    if quantity != 1:
        en_unit += "s"
    ms_unit = "hari" if unit == "DAY" else "jam"
    return {
        "en": f"I have had {en_symptom} for {quantity} {en_unit}.",
        "ms": f"Saya mengalami {ms_symptom} selama {quantity} {ms_unit}.",
    }


def process_patient_token_sequence(
    *,
    sign_tokens: List[Dict[str, Any]],
    state: ConversationState,
    mcie: "AIEnhancedMCIE",
) -> Dict[str, Any]:
    """Process the patient's complete explicitly-finished sentence.

    The sign tokens should already have been reviewed/accepted by the patient
    UI. This function never guesses sentence completion. It is called only
    after the UI sends an explicit Finish Sentence action.
    """
    sequence = build_sequence_from_tokens(sign_tokens)
    if not sequence.get("success", False):
        return build_unified_mcie_output(
            source="patient_bim",
            input_data={
                "gloss_sequence": sequence.get("gloss_sequence", []),
                "concept_sequence": sequence.get("concept_sequence", []),
            },
            success=False,
            decision="RETRY",
            requires_retry=True,
            decision_reasons=[sequence.get("error", "Recognition sequence requires retry.")],
            metadata={"recognition_details": sequence.get("signs", [])},
        )

    # For duration, keep quantity semantics deterministic. The LLM is not
    # allowed to modify a recognized medical quantity.
    if state.last_doctor_intent == "SYMPTOM_DURATION":
        duration = _extract_duration(sequence)
        symptom = state.current_symptom
        complete = bool(duration["complete"] and symptom)
        sentence = (
            _duration_sentence(symptom, duration["quantity"], duration["unit"])
            if complete else {"en": None, "ms": None}
        )
        requires_confirmation = sequence.get("requires_confirmation", False) or not complete
        decision = "CONFIRM" if requires_confirmation else "ACCEPT"
        result = build_unified_mcie_output(
            source="patient_bim",
            intent="SYMPTOM_DURATION",
            entities={
                "symptom": symptom,
                "duration": {"quantity": duration["quantity"], "unit": duration["unit"]},
            },
            medical_concepts=[symptom] if symptom else [],
            input_data={
                "gloss_sequence": sequence.get("gloss_sequence", []),
                "concept_sequence": sequence.get("concept_sequence", []),
            },
            context_used=True,
            simplified_text_en=sentence["en"],
            simplified_text_ms=sentence["ms"],
            decision=decision,
            requires_confirmation=requires_confirmation,
            decision_reasons=[] if complete else ["Duration or symptom context is incomplete."],
            metadata={"recognition_details": sequence.get("signs", [])},
        )
        state.update_from_patient_mcie(result)
        return result

    return mcie.patient_bim(
        gloss_sequence=sequence.get("gloss_sequence", []),
        concept_sequence=sequence.get("concept_sequence", []),
        state=state,
        upstream_requires_confirmation=sequence.get("requires_confirmation", False),
        upstream_requires_retry=sequence.get("requires_retry", False),
        recognition_details=sequence.get("signs", []),
    )


def process_patient_bim_turn(
    *,
    recognition_results: List[Dict[str, Any]],
    state: ConversationState,
    mcie: "AIEnhancedMCIE",
    number_is_medical: bool = False,
    number_confirmed: bool = False,
    min_hand_presence: float = 0.50,
    low_confidence: float = 0.50,
    close_margin: float = 0.15,
) -> Dict[str, Any]:
    """Backward-compatible one-shot path.

    New UI code should prefer the sentence-buffer API in SignBridgeOrchestrator:
    start_patient_sentence -> patient_preview_sign -> patient_accept_sign ->
    finish_patient_sentence.
    """
    sequence = build_processed_sequence(
        recognition_results,
        state,
        number_is_medical=number_is_medical,
        number_confirmed=number_confirmed,
        min_hand_presence=min_hand_presence,
        low_confidence=low_confidence,
        close_margin=close_margin,
    )
    return process_patient_token_sequence(
        sign_tokens=sequence.get("signs", []),
        state=state,
        mcie=mcie,
    )
