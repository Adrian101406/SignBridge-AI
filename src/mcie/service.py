from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from .conversation import ConversationState
from .mcie_ai import AIEnhancedMCIE
from .patient_bim import process_patient_token_sequence
from .recognizer_adapter import process_recognizer_result
from .recognizer_router import expected_domains


class MCIERuntime:
    """Public integration API for the SignBridge MCIE component.

    This class deliberately does not load ASR, camera, BIM recognizer, or Unity
    components. It accepts their documented payloads and returns the stable MCIE
    output contract. This keeps the MCIE source code independently testable.
    """

    def __init__(self, mcie: AIEnhancedMCIE, state: Optional[ConversationState] = None):
        self.mcie = mcie
        self.state = state or ConversationState()

    # ------------------------------------------------------------------
    # Doctor speech -> MCIE
    # ------------------------------------------------------------------
    def process_doctor_asr_payload(self, asr_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(asr_payload, dict):
            raise TypeError("asr_payload must be a dictionary")
        if not asr_payload.get("success", False):
            return {
                "success": False,
                "source": "doctor_speech",
                "safety": {
                    "decision": "RETRY",
                    "requires_confirmation": False,
                    "requires_retry": True,
                    "reasons": ["ASR payload is not successful."],
                },
                "ready_for_output": False,
                "ready_for_translation": False,
            }

        transcript = str(asr_payload.get("transcript", "")).strip()
        confirmation_status = str(asr_payload.get("confirmation_status", "")).upper()
        validation = asr_payload.get("medical_validation") or {}

        # The upstream ASR design may require human confirmation before MCIE.
        if validation.get("status") == "CONFIRM" and confirmation_status != "CONFIRMED":
            return {
                "success": False,
                "source": "doctor_speech",
                "input": {"raw_text": transcript},
                "safety": {
                    "decision": "CONFIRM",
                    "requires_confirmation": True,
                    "requires_retry": False,
                    "reasons": ["ASR transcript requires doctor confirmation before MCIE."],
                },
                "ready_for_output": False,
                "ready_for_translation": False,
            }

        return self.mcie.doctor_text(
            text=transcript,
            state=self.state,
            asr_payload=asr_payload,
        )

    # ------------------------------------------------------------------
    # Recognizer routing / isolated sign processing
    # ------------------------------------------------------------------
    def expected_recognizer_domains(self, default_domain: str = "medical"):
        return expected_domains(self.state, default_domain=default_domain)

    def preview_patient_sign(
        self,
        recognizer_result: Dict[str, Any],
        *,
        number_is_medical: bool = False,
        number_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Convert one recognizer result into an MCIE sign token.

        The token is only a preview. It is not stored until ``accept_patient_sign``.
        """
        return process_recognizer_result(
            recognizer_result,
            self.state,
            number_is_medical=number_is_medical,
            number_confirmed=number_confirmed,
        )

    # ------------------------------------------------------------------
    # Patient sentence buffer
    # ------------------------------------------------------------------
    def start_patient_sentence(self, *, reset: bool = False) -> Dict[str, Any]:
        return self.state.start_patient_sentence(reset=reset)

    def accept_patient_sign(
        self,
        sign_token: Dict[str, Any],
        *,
        human_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Add one reviewed isolated sign to the current sentence buffer.

        If upstream recognition requires confirmation, the UI must pass
        ``human_confirmed=True`` after the patient explicitly confirms the sign.
        """
        token = deepcopy(sign_token)
        if token.get("requires_retry", False):
            return {"success": False, "reason": "Retry the sign before adding it."}

        if token.get("requires_confirmation", False):
            if not human_confirmed:
                return {
                    "success": False,
                    "reason": "This recognized sign requires human confirmation.",
                }
            token.setdefault("metadata", {})
            token["metadata"]["human_confirmed"] = True
            token["metadata"]["preconfirmation_decision"] = token.get("decision")
            token["requires_confirmation"] = False
            token["decision"] = "ACCEPT"

        return self.state.add_patient_sign(token)

    def remove_last_patient_sign(self) -> Dict[str, Any]:
        return self.state.remove_last_patient_sign()

    def cancel_patient_sentence(self) -> Dict[str, Any]:
        return self.state.cancel_patient_sentence()

    def patient_sentence_preview(self) -> Dict[str, Any]:
        return self.state.patient_buffer_snapshot()

    def finish_patient_sentence(self) -> Dict[str, Any]:
        """Explicitly finish and interpret the buffered patient sentence.

        MCIE never guesses sentence completion from a pause or timeout. The UI
        calls this method only when the patient presses Finish Sentence.
        """
        committed = self.state.commit_patient_sentence()
        if not committed.get("success"):
            return committed

        return process_patient_token_sequence(
            sign_tokens=committed["signs"],
            state=self.state,
            mcie=self.mcie,
        )
