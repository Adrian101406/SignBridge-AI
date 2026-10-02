from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSON2_ROOT = PROJECT_ROOT / "uploads" / "person2"
if str(PERSON2_ROOT) not in sys.path:
    sys.path.insert(0, str(PERSON2_ROOT))

from signbridge.rule_engine import build_rule_anchor  # noqa: E402
from signbridge.conversation import ConversationState  # noqa: E402


def test_fever_how_many_days_is_a_duration_question():
    anchor = build_rule_anchor("How many days have you had a fever?")

    assert anchor["intent"] == "SYMPTOM_DURATION"
    assert anchor["medical_concepts"] == ["FEVER"]


def test_ai_context_includes_the_confirmed_doctor_question():
    state = ConversationState(
        current_symptom="FEVER",
        last_doctor_text="How many days have you had a fever?",
        last_doctor_intent="SYMPTOM_DURATION",
        expected_response_type="DURATION",
    )

    assert state.ai_snapshot() == {
        "current_symptom": "FEVER",
        "last_doctor_text": "How many days have you had a fever?",
        "last_doctor_intent": "SYMPTOM_DURATION",
        "expected_response_type": "DURATION",
    }
