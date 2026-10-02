from __future__ import annotations

import pytest

from signbridge_app.contracts import TranscriptProposal
from signbridge_app.conversation import ConversationManager, TurnStatus


def test_newer_request_rejects_late_result() -> None:
    manager = ConversationManager(session_id="session-1")
    old_request = manager.begin_request("asr")
    new_request = manager.begin_request("asr")

    accepted_old = manager.accept_result(
        old_request,
        TranscriptProposal(
            original_text="old",
            warnings=(),
            requires_confirmation=True,
            request_id=old_request,
        ),
    )
    accepted_new = manager.accept_result(
        new_request,
        TranscriptProposal(
            original_text="new",
            warnings=(),
            requires_confirmation=True,
            request_id=new_request,
        ),
    )

    assert accepted_old is False
    assert accepted_new is True
    assert manager.snapshot().status is TurnStatus.REVIEW_REQUIRED


def test_proposal_does_not_enter_confirmed_conversation() -> None:
    manager = ConversationManager(session_id="session-1")
    request_id = manager.begin_request("asr")
    manager.accept_result(
        request_id,
        TranscriptProposal(
            original_text="Please rest",
            warnings=(),
            requires_confirmation=True,
            request_id=request_id,
        ),
    )

    assert manager.snapshot().messages == ()


def test_blank_confirmation_is_rejected() -> None:
    manager = ConversationManager(session_id="session-1")

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.confirm_doctor("   ", "message-1")


def test_duplicate_confirmation_is_idempotent() -> None:
    manager = ConversationManager(session_id="session-1")

    first = manager.confirm_patient("I HAVE PAIN", "message-1")
    second = manager.confirm_patient("I HAVE PAIN", "message-1")

    assert first is second
    assert manager.snapshot().messages == (first,)


def test_doctor_confirmation_preserves_asr_original() -> None:
    manager = ConversationManager(session_id="session-1")
    request_id = manager.begin_request("asr")
    manager.accept_result(
        request_id,
        TranscriptProposal(
            original_text="take five milligram",
            warnings=("medical terms require review",),
            requires_confirmation=True,
            request_id=request_id,
        ),
    )

    message = manager.confirm_doctor("Take 5 mg.", "message-1")

    assert message.original_text == "take five milligram"
    assert message.final_text == "Take 5 mg."
    assert manager.snapshot().status is TurnStatus.CONFIRMED


def test_clear_confirmed_messages_removes_history_and_allows_message_id_reuse() -> None:
    manager = ConversationManager(session_id="session-1")
    manager.confirm_patient("Three days", "message-1")
    manager.confirm_doctor("Please rest", "message-2")

    cleared = manager.clear_confirmed_messages()

    assert cleared == 2
    assert manager.snapshot().messages == ()
    replacement = manager.confirm_patient("Four days", "message-1")
    assert replacement.final_text == "Four days"
