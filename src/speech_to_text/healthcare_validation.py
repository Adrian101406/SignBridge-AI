import json
from pathlib import Path

_CONCEPTS_PATH = Path(__file__).with_name("critical_medical_concepts.json")
CRITICAL_MEDICAL_CONCEPTS = json.loads(_CONCEPTS_PATH.read_text(encoding="utf-8"))


def validate_healthcare_transcript(text: str) -> dict:
    text_lower = text.lower()
    detected_concepts = []

    for concept, phrases in CRITICAL_MEDICAL_CONCEPTS.items():
        for phrase in phrases:
            if phrase in text_lower:
                detected_concepts.append(concept)
                break

    detected_concepts = list(dict.fromkeys(detected_concepts))

    if len(text.strip()) == 0:
        return {
            "validation_status": "RETRY",
            "risk_level": "HIGH",
            "detected_concepts": [],
            "requires_confirmation": True,
            "message": "No usable transcript detected. Please repeat."
        }

    if detected_concepts:
        return {
            "validation_status": "CONFIRM",
            "risk_level": "MEDICAL",
            "detected_concepts": detected_concepts,
            "requires_confirmation": True,
            "message": "Medical information detected. Please confirm the transcript before continuing."
        }

    return {
        "validation_status": "PASS",
        "risk_level": "LOW",
        "detected_concepts": [],
        "requires_confirmation": False,
        "message": "No critical medical concept detected."
    }


def prepare_mcie_input(asr_result: dict) -> dict:
    transcript = asr_result["speech_recognition"]["transcript"]
    quality = asr_result["quality_check"]
    validation = asr_result["healthcare_validation"]

    if quality["status"] == "RETRY" or validation["validation_status"] == "RETRY":
        action, ready = "RETRY_SPEECH", False
    elif validation["requires_confirmation"]:
        action, ready = "CONFIRM_TRANSCRIPT", False
    else:
        action, ready = "SEND_TO_MCIE", True

    return {
        "transcript": transcript,
        "asr_status": quality["status"],
        "medical_validation": {
            "status": validation["validation_status"],
            "risk_level": validation["risk_level"],
            "detected_concepts": validation["detected_concepts"]
        },
        "ready_for_mcie": ready,
        "next_action": action
    }


def handle_transcript_confirmation(mcie_input: dict, doctor_confirmed: bool) -> dict:
    output = mcie_input.copy()
    if doctor_confirmed:
        output["ready_for_mcie"] = True
        output["next_action"] = "SEND_TO_MCIE"
        output["confirmation_status"] = "CONFIRMED"
    else:
        output["ready_for_mcie"] = False
        output["next_action"] = "RETRY_SPEECH"
        output["confirmation_status"] = "REJECTED"
    return output


def build_mcie_payload(confirmed_result: dict, model_id: str) -> dict:
    if not confirmed_result["ready_for_mcie"]:
        return {"success": False, "message": "Transcript is not ready for MCIE."}

    # PASS cases may not need a manual confirmation click.
    confirmation_status = confirmed_result.get("confirmation_status", "NOT_REQUIRED")

    return {
        "success": True,
        "source": "doctor_speech",
        "transcript": confirmed_result["transcript"],
        "asr_information": {
            "model": model_id,
            "status": confirmed_result["asr_status"]
        },
        "medical_validation": confirmed_result["medical_validation"],
        "confirmation_status": confirmation_status
    }
