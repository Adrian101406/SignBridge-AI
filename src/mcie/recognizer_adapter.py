from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

from .conversation import ConversationState

RECOGNIZER_TYPES = {
    "SignBridge_Medical_Recognizer_V1": "medical",
    "SignBridge_Medical_Recognizer_V2": "medical",
    "SignBridge_General_Recognizer_V1": "general",
    "SignBridge_Number_Recognizer_V1": "number",
}

MEDICAL_BIM_CONCEPT_MAP = {
    "Demam": "FEVER", "Batuk": "COUGH", "Selesema": "FLU", "Muntah": "VOMITING",
    "Gatal-Gatal": "ITCHING", "Bengkak": "SWELLING", "Lemah": "WEAKNESS", "Penat": "FATIGUE",
    "Asma": "ASTHMA", "Kesakitan": "PAIN", "Sakit Kepala": "HEADACHE",
    "Sakit Perut": "STOMACH_PAIN", "Sakit Pinggang": "BACK_PAIN", "Sakit Tekak": "SORE_THROAT",
    "Badan": "BODY", "Bahu": "SHOULDER", "Jari": "FINGER", "Kaki": "LEG", "Kulit": "SKIN",
    "Leher": "NECK", "Bibir": "LIP", "Gigi": "TOOTH", "Hidung": "NOSE", "Lidah": "TONGUE",
    "Mata": "EYE", "Mulut": "MOUTH", "Tulang": "BONE", "Tulang Rusuk": "RIB",
    "Perut": "ABDOMEN", "Darah": "BLOOD", "Jantung": "HEART", "Buah Pinggang": "KIDNEY",
    "Mengandung": "PREGNANCY", "Doktor": "DOCTOR", "Jururawat": "NURSE",
    "Kecemasan": "EMERGENCY", "Bedah": "SURGERY", "Sinar-X": "X_RAY", "Pil": "PILL", "Krim": "CREAM",
}

GENERAL_BIM_CONCEPT_MAP = {
    "saya": "I", "awak": "YOU", "apa": "WHAT", "bagaimana": "HOW", "bila": "WHEN",
    "berapa": "HOW_MANY", "mana": "WHERE_WHICH", "siapa": "WHO", "ada": "HAVE_EXIST",
    "sudah": "ALREADY", "ambil": "TAKE", "bawa": "BRING", "makan": "EAT", "minum": "DRINK",
    "tidur": "SLEEP", "pergi": "GO", "pergi_2": "GO", "mari": "COME", "mari_2": "COME",
    "jumpa": "MEET", "tanya": "ASK", "tolong": "HELP", "tolong_2": "HELP", "mohon": "REQUEST",
    "mohon_2": "REQUEST", "jangan": "DO_NOT", "boleh": "CAN", "hari": "DAY", "jam": "HOUR",
    "masa": "TIME", "esok": "TOMORROW", "hospital": "HOSPITAL", "tandas": "TOILET",
    "kesakitan": "PAIN", "panas": "HOT", "panas_2": "HOT", "sejuk": "COLD", "masalah": "PROBLEM",
    "perlahan": "SLOW", "perlahan_2": "SLOW", "bahasa_isyarat": "SIGN_LANGUAGE", "hi": "HELLO",
    "selamat_pagi": "GOOD_MORNING", "terima_kasih": "THANK_YOU", "apa_khabar": "HOW_ARE_YOU",
    "khabar_baik": "I_AM_FINE",
}

CONTEXT_COMPATIBILITY = {
    "SYMPTOM_LOCATION": {
        "BODY", "SHOULDER", "FINGER", "LEG", "SKIN", "NECK", "LIP", "TOOTH", "NOSE", "TONGUE",
        "EYE", "MOUTH", "BONE", "RIB", "ABDOMEN",
    },
    "SYMPTOM_CHECK": {
        "FEVER", "COUGH", "FLU", "VOMITING", "ITCHING", "SWELLING", "WEAKNESS", "FATIGUE",
        "HEADACHE", "STOMACH_PAIN", "BACK_PAIN", "SORE_THROAT", "CHEST_PAIN", "PAIN", "ASTHMA",
    },
    "MEDICATION_QUERY": {"PILL", "CREAM", "MEDICATION"},
    "ALLERGY_CHECK": {"ALLERGY", "MEDICATION"},
}

CRITICAL_RECOGNITION_CONCEPTS = {"EMERGENCY", "PILL", "CREAM", "MEDICATION"}


def recognizer_type(result: Dict[str, Any]) -> Optional[str]:
    domain = result.get("domain")
    if domain in {"medical", "general", "number"}:
        return str(domain)
    return RECOGNIZER_TYPES.get(result.get("recognizer"))


def normalize_gloss(gloss: str, rtype: str) -> str:
    gloss = str(gloss).strip()
    if rtype == "number":
        return gloss
    if rtype == "medical":
        return MEDICAL_BIM_CONCEPT_MAP.get(gloss, gloss.upper().replace("-", "_").replace(" ", "_"))
    if rtype == "general":
        clean = gloss.lower().strip()
        return GENERAL_BIM_CONCEPT_MAP.get(clean, clean.upper())
    return gloss.upper().replace(" ", "_")


def validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {"valid": False, "reason": "Recognizer result is not a dictionary."}

    rtype = recognizer_type(result)
    if rtype is None:
        return {"valid": False, "reason": "Unknown recognizer/domain."}

    top1 = result.get("top1")
    top_k = result.get("top_k")
    if not isinstance(top1, dict):
        return {"valid": False, "reason": "Missing top1 object."}
    if not isinstance(top_k, list) or not top_k:
        return {"valid": False, "reason": "Missing top_k candidates."}

    for field in ("class_id", "gloss", "confidence"):
        if field not in top1:
            return {"valid": False, "reason": f"top1 missing {field}."}

    try:
        confidence = float(top1["confidence"])
    except (TypeError, ValueError):
        return {"valid": False, "reason": "Invalid top1 confidence."}
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return {"valid": False, "reason": "top1 confidence must be finite and between 0 and 1."}

    for candidate in top_k:
        if not isinstance(candidate, dict):
            return {"valid": False, "reason": "Invalid Top-K candidate."}
        if "gloss" not in candidate or "confidence" not in candidate:
            return {"valid": False, "reason": "Top-K candidate missing gloss/confidence."}
        try:
            candidate_conf = float(candidate["confidence"])
        except (TypeError, ValueError):
            return {"valid": False, "reason": "Invalid candidate confidence."}
        if not math.isfinite(candidate_conf) or not 0.0 <= candidate_conf <= 1.0:
            return {"valid": False, "reason": "Candidate confidence must be finite and between 0 and 1."}

    if rtype == "number":
        gloss = str(top1.get("gloss", "")).strip()
        if gloss not in {str(i) for i in range(11)}:
            return {"valid": False, "reason": "Number recognizer gloss must be 0 through 10."}

    return {"valid": True, "recognizer_type": rtype}


def extract_candidates(result: Dict[str, Any], state: Optional[ConversationState] = None) -> List[Dict[str, Any]]:
    check = validate_result(result)
    if not check["valid"]:
        return []
    rtype = check["recognizer_type"]
    doctor_intent = state.last_doctor_intent if state else None
    compatible_set = CONTEXT_COMPATIBILITY.get(doctor_intent, set())
    out = []
    for item in result.get("top_k", []):
        concept = normalize_gloss(item.get("gloss", ""), rtype)
        out.append({
            "rank": int(item.get("rank", len(out) + 1)),
            "class_id": item.get("class_id"),
            "gloss": item.get("gloss"),
            "english": item.get("english"),
            "confidence": float(item.get("confidence", 0.0)),
            "concept": concept,
            "context_compatibility": 1 if compatible_set and concept in compatible_set else 0,
        })
    return out


def select_contextual_candidate(result: Dict[str, Any], state: Optional[ConversationState] = None) -> Dict[str, Any]:
    candidates = extract_candidates(result, state)
    if not candidates:
        return {"success": False, "error": "No candidates."}
    original = candidates[0]
    selected = original
    basis = "RECOGNIZER_TOP1"
    compatible = [c for c in candidates if c["context_compatibility"] == 1]
    if compatible:
        if original["context_compatibility"] == 1:
            basis = "TOP1_CONTEXT_COMPATIBLE"
        else:
            selected = sorted(compatible, key=lambda x: x["rank"])[0]
            basis = "CONTEXTUAL_RERANK"
    return {
        "success": True,
        "original_top1": original,
        "selected_candidate": selected,
        "selection_basis": basis,
        "selection_changed": selected["rank"] != original["rank"],
        "all_candidates": candidates,
    }


def assess_recognition_decision(
    result: Dict[str, Any],
    selection: Dict[str, Any],
    state: Optional[ConversationState] = None,
    *,
    min_hand_presence: float = 0.50,
    low_confidence: float = 0.50,
    close_margin: float = 0.15,
) -> Dict[str, Any]:
    if not selection.get("success"):
        return {
            "status": "RETRY",
            "requires_confirmation": False,
            "requires_retry": True,
            "reasons": ["Selection failed."],
        }

    quality = result.get("quality") or {}
    hand_presence = float(quality.get("hand_presence", 0.0))
    if hand_presence < min_hand_presence:
        return {
            "status": "RETRY",
            "requires_confirmation": False,
            "requires_retry": True,
            "reasons": ["Hand detection quality is too low."],
        }

    reasons: List[str] = []
    selected = selection["selected_candidate"]
    margin = float(result.get("confidence_margin", 0.0))
    recognizer_name = result.get("recognizer")
    upstream_decision = str(result.get("decision", "")).upper().strip()

    # Conversation context safety remains an MCIE responsibility.
    if state and state.last_doctor_intent in CONTEXT_COMPATIBILITY:
        if not any(c["context_compatibility"] == 1 for c in selection.get("all_candidates", [])):
            reasons.append("No returned Top-K candidate matches the active conversation context.")

    if selection.get("selection_changed"):
        reasons.append("Contextual reranking changed the recognizer Top-1 candidate.")

    # Medical Recognizer V2 already exposes its own temporary model-specific
    # ACCEPT / CONFIRM / VERIFY policy. Preserve it instead of applying the
    # generic V1 thresholds a second time. Context reranking, quality, and
    # critical-concept rules still apply on top of the upstream decision.
    if recognizer_name == "SignBridge_Medical_Recognizer_V2":
        if upstream_decision == "CONFIRM":
            reasons.append("Medical Recognizer V2 requests confirmation.")
        elif upstream_decision == "VERIFY":
            reasons.append("Medical Recognizer V2 marked the recognition for verification.")
        elif upstream_decision not in {"ACCEPT", ""}:
            reasons.append(f"Unknown Medical Recognizer V2 decision: {upstream_decision}.")
    else:
        if margin < close_margin:
            reasons.append("Top recognition candidates are close.")
        if float(selected.get("confidence", 0.0)) < low_confidence:
            reasons.append("Selected recognition confidence is low.")

    if selected.get("concept") in CRITICAL_RECOGNITION_CONCEPTS:
        reasons.append("Critical medical concept requires verification.")

    if reasons:
        return {
            "status": "CONFIRM",
            "requires_confirmation": True,
            "requires_retry": False,
            "reasons": reasons,
            "upstream_recognizer_decision": upstream_decision or None,
        }

    return {
        "status": "ACCEPT",
        "requires_confirmation": False,
        "requires_retry": False,
        "reasons": [],
        "upstream_recognizer_decision": upstream_decision or None,
    }

def process_recognizer_result(
    result: Dict[str, Any],
    state: Optional[ConversationState] = None,
    *,
    number_is_medical: bool = False,
    number_confirmed: bool = False,
    min_hand_presence: float = 0.50,
    low_confidence: float = 0.50,
    close_margin: float = 0.15,
) -> Dict[str, Any]:
    check = validate_result(result)
    if not check["valid"]:
        return {"success": False, "error": check["reason"]}
    rtype = check["recognizer_type"]

    if rtype == "number":
        quality = result.get("quality") or {}
        top1 = result.get("top1") or {}
        reasons: List[str] = []
        hand = float(quality.get("hand_presence", 0.0))
        if hand < min_hand_presence:
            return {
                "success": True, "recognizer_type": "number", "gloss": str(top1.get("gloss", "")),
                "concept": str(top1.get("gloss", "")), "confidence": float(top1.get("confidence", 0.0)),
                "candidates": result.get("top_k", []), "decision": "RETRY", "requires_confirmation": False,
                "requires_retry": True, "decision_reasons": ["Hand detection quality is too low."],
            }
        if float(top1.get("confidence", 0.0)) < low_confidence:
            reasons.append("Number recognition confidence is low.")
        if float(result.get("confidence_margin", 0.0)) < close_margin:
            reasons.append("Top number candidates are close.")
        if number_is_medical and not number_confirmed:
            reasons.append("Medical quantity requires human confirmation.")
        return {
            "success": True, "recognizer_type": "number", "gloss": str(top1.get("gloss", "")),
            "english": str(top1.get("english", top1.get("gloss", ""))), "concept": str(top1.get("gloss", "")),
            "confidence": float(top1.get("confidence", 0.0)), "candidates": result.get("top_k", []),
            "decision": "CONFIRM" if reasons else "ACCEPT", "requires_confirmation": bool(reasons),
            "requires_retry": False, "decision_reasons": reasons,
        }

    selection = select_contextual_candidate(result, state)
    safety = assess_recognition_decision(
        result, selection, state,
        min_hand_presence=min_hand_presence,
        low_confidence=low_confidence,
        close_margin=close_margin,
    )
    if not selection.get("success"):
        return selection
    selected = selection["selected_candidate"]
    return {
        "success": True,
        "recognizer_type": rtype,
        "recognizer_name": result.get("recognizer"),
        "upstream_recognizer_decision": safety.get("upstream_recognizer_decision"),
        "gloss": selected.get("gloss"),
        "english": selected.get("english"),
        "concept": selected.get("concept"),
        "confidence": selected.get("confidence"),
        "candidates": selection.get("all_candidates", []),
        "selection_basis": selection.get("selection_basis"),
        "selection_changed": selection.get("selection_changed"),
        "decision": safety["status"],
        "requires_confirmation": safety["requires_confirmation"],
        "requires_retry": safety["requires_retry"],
        "decision_reasons": safety["reasons"],
    }
