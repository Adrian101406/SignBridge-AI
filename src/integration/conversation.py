from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .contracts import (
    ConfirmedMessage,
    ConversationSnapshot,
    MedicalProposal,
    TranscriptProposal,
    TurnStatus,
)


class ConversationManager:
    def __init__(self, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        self._session_id = session_id.strip()
        self._status = TurnStatus.IDLE
        self._active_request_ids: dict[str, str] = {}
        self._pending: dict[str, Any] = {}
        self._messages: list[ConfirmedMessage] = []
        self._messages_by_id: dict[str, ConfirmedMessage] = {}
        self._doctor_original_text = ""

    def begin_request(self, kind: str) -> str:
        normalized_kind = kind.strip()
        if not normalized_kind:
            raise ValueError("Request kind cannot be empty")
        request_id = f"{normalized_kind}-{uuid4().hex}"
        self._active_request_ids[normalized_kind] = request_id
        self._pending.pop(normalized_kind, None)
        self._status = TurnStatus.PROCESSING
        return request_id

    def accept_result(self, request_id: str, payload: object) -> bool:
        kind = next(
            (
                active_kind
                for active_kind, active_id in self._active_request_ids.items()
                if active_id == request_id
            ),
            None,
        )
        if kind is None:
            return False
        payload_request_id = getattr(payload, "request_id", request_id)
        if payload_request_id != request_id:
            return False
        self._pending[kind] = payload
        if isinstance(payload, TranscriptProposal):
            self._doctor_original_text = payload.original_text
        elif isinstance(payload, MedicalProposal) and kind == "doctor_mcie":
            self._doctor_original_text = payload.source_text
        self._status = TurnStatus.REVIEW_REQUIRED
        return True

    def confirm_patient(self, text: str, message_id: str) -> ConfirmedMessage:
        proposal = self._pending.get("patient_mcie")
        original_text = proposal.source_text if isinstance(proposal, MedicalProposal) else text
        message = self._confirm(
            original_text=original_text,
            final_text=text,
            sender="patient",
            message_id=message_id,
        )
        self._pending.pop("patient_mcie", None)
        return message

    def confirm_doctor(self, text: str, message_id: str) -> ConfirmedMessage:
        return self._confirm(
            original_text=self._doctor_original_text or text,
            final_text=text,
            sender="doctor",
            message_id=message_id,
        )

    def snapshot(self) -> ConversationSnapshot:
        return ConversationSnapshot(
            session_id=self._session_id,
            status=self._status,
            active_request_ids=dict(self._active_request_ids),
            messages=tuple(self._messages),
        )

    def clear_confirmed_messages(self) -> int:
        cleared = len(self._messages)
        self._messages.clear()
        self._messages_by_id.clear()
        return cleared

    def _confirm(
        self,
        *,
        original_text: str,
        final_text: str,
        sender: str,
        message_id: str,
    ) -> ConfirmedMessage:
        if not final_text.strip():
            raise ValueError("Confirmed message cannot be empty")
        existing = self._messages_by_id.get(message_id.strip())
        if existing is not None:
            if existing.sender != sender or existing.final_text != final_text.strip():
                raise ValueError("Message ID was already used for different content")
            return existing
        message = ConfirmedMessage(
            original_text=original_text,
            final_text=final_text,
            sender=sender,
            message_id=message_id,
            session_id=self._session_id,
            confirmed_at=datetime.now(timezone.utc),
        )
        self._messages.append(message)
        self._messages_by_id[message.message_id] = message
        self._status = TurnStatus.CONFIRMED
        return message
