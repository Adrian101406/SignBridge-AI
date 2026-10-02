from __future__ import annotations

import pytest
from pydantic import ValidationError

from signbridge_app.contracts import ConfirmedMessage, RecognitionResult


def test_recognition_confidence_must_be_normalized() -> None:
    with pytest.raises(ValidationError):
        RecognitionResult(
            gloss="PAIN",
            confidence=1.4,
            recognizer_type="medical",
            stable=True,
            timestamp=1.0,
            request_id="r1",
        )


def test_confirmed_message_keeps_original_and_final_text() -> None:
    message = ConfirmedMessage(
        original_text="NO PAIN",
        final_text="No pain.",
        sender="patient",
        message_id="m1",
        session_id="s1",
        confirmed_at="2026-09-25T00:00:00Z",
    )

    assert message.original_text == "NO PAIN"
    assert message.final_text == "No pain."


def test_contracts_reject_blank_identifiers() -> None:
    with pytest.raises(ValidationError):
        ConfirmedMessage(
            original_text="Pain",
            final_text="Pain",
            sender="patient",
            message_id=" ",
            session_id="s1",
            confirmed_at="2026-09-25T00:00:00Z",
        )

