from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TurnStatus(str, Enum):
    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    PROCESSING = "PROCESSING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True)


class RecognitionResult(FrozenContract):
    gloss: NonBlank
    confidence: float = Field(ge=0.0, le=1.0)
    recognizer_type: Literal["medical", "general", "number"]
    stable: bool
    timestamp: float
    request_id: NonBlank


class TranscriptProposal(FrozenContract):
    original_text: str
    warnings: tuple[str, ...] = ()
    requires_confirmation: Literal[True] = True
    request_id: NonBlank


class MedicalProposal(FrozenContract):
    source_text: str
    suggested_text: str
    medical_flags: tuple[str, ...] = ()
    changed_critical_terms: tuple[str, ...] = ()
    requires_confirmation: Literal[True] = True
    request_id: NonBlank


class ConfirmedMessage(FrozenContract):
    original_text: str
    final_text: NonBlank
    sender: Literal["patient", "doctor"]
    message_id: NonBlank
    session_id: NonBlank
    confirmed_at: datetime


class ConversationSnapshot(FrozenContract):
    session_id: NonBlank
    status: TurnStatus
    active_request_ids: dict[str, str]
    messages: tuple[ConfirmedMessage, ...]
