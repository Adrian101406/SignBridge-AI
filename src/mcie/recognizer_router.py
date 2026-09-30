from __future__ import annotations

from typing import List

from .conversation import ConversationState


def expected_domains(state: ConversationState, default_domain: str = "medical") -> List[str]:
    """Context routing only. Never compare confidence across recognizers."""
    if state.expected_response_type == "DURATION" or state.last_doctor_intent == "SYMPTOM_DURATION":
        return ["number", "general"]
    if state.expected_response_type == "BODY_LOCATION" or state.last_doctor_intent == "SYMPTOM_LOCATION":
        return ["medical", "general"]
    if state.last_doctor_intent in {"MEDICATION_QUERY", "ALLERGY_CHECK"}:
        return ["medical", "general"]
    if state.expected_response_type == "GENERAL_RESPONSE" or state.last_doctor_intent == "SYMPTOM_CHECK":
        return ["general"]
    return [default_domain]
