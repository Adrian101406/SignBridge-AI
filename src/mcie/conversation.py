from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationState:
    current_topic: Optional[str] = None
    current_symptom: Optional[str] = None
    current_body_location: Optional[str] = None
    current_medication: Optional[str] = None
    last_doctor_text: Optional[str] = None
    last_doctor_intent: Optional[str] = None
    last_doctor_entities: Dict[str, Any] = field(default_factory=dict)
    last_patient_glosses: List[str] = field(default_factory=list)
    last_patient_concepts: List[str] = field(default_factory=list)
    last_patient_intent: Optional[str] = None
    expected_response_type: Optional[str] = None

    # ------------------------------------------------------------
    # Patient isolated-sign sentence buffer
    # ------------------------------------------------------------
    # The current recognizers classify one isolated sign per capture.
    # Accepted signs are accumulated here until the patient explicitly
    # presses Finish Sentence in the UI.
    patient_sign_buffer: List[Dict[str, Any]] = field(default_factory=list)
    patient_sentence_in_progress: bool = False
    last_patient_signs: List[Dict[str, Any]] = field(default_factory=list)

    turn_count: int = 0

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)

    def ai_snapshot(self) -> Dict[str, Any]:
        # Do not expose the entire Top-K/recognizer metadata to the LLM.
        # The semantic context is enough; the deterministic recognition layer
        # keeps ownership of the evidence and candidate restrictions.
        keys = [
            "current_topic",
            "current_symptom",
            "current_body_location",
            "current_medication",
            "last_doctor_intent",
            "last_patient_intent",
            "expected_response_type",
        ]
        raw = self.snapshot()
        return {k: raw[k] for k in keys if raw.get(k) is not None}

    # ============================================================
    # PATIENT SENTENCE BUFFER
    # ============================================================

    def start_patient_sentence(self, *, reset: bool = False) -> Dict[str, Any]:
        """Begin a new patient BIM sentence.

        If a non-empty sentence is already in progress, callers must either
        continue it or explicitly pass reset=True. This prevents accidental
        loss of recognized signs.
        """
        if self.patient_sentence_in_progress and self.patient_sign_buffer and not reset:
            return {
                "success": False,
                "reason": "A patient sentence is already in progress.",
                **self.patient_buffer_snapshot(),
            }

        if reset:
            self.patient_sign_buffer = []

        if not self.patient_sentence_in_progress:
            # A previous completed sentence never remains in this buffer, but
            # clearing here makes the lifecycle explicit and robust.
            self.patient_sign_buffer = []

        self.patient_sentence_in_progress = True
        return {"success": True, **self.patient_buffer_snapshot()}

    def add_patient_sign(self, sign_token: Dict[str, Any]) -> Dict[str, Any]:
        """Append one already-reviewed sign token to the current sentence."""
        if not isinstance(sign_token, dict):
            return {"success": False, "reason": "Sign token must be a dictionary."}
        if not sign_token.get("success", False):
            return {"success": False, "reason": "Cannot add an unsuccessful sign token."}
        if sign_token.get("requires_retry", False):
            return {"success": False, "reason": "A sign requiring retry cannot be added."}
        if sign_token.get("requires_confirmation", False):
            return {
                "success": False,
                "reason": "Confirm the recognized sign before adding it to the sentence.",
            }

        if not self.patient_sentence_in_progress:
            self.start_patient_sentence(reset=True)

        self.patient_sign_buffer.append(deepcopy(sign_token))
        return {"success": True, **self.patient_buffer_snapshot()}

    def remove_last_patient_sign(self) -> Dict[str, Any]:
        if not self.patient_sign_buffer:
            return {
                "success": False,
                "reason": "The patient sentence buffer is empty.",
                **self.patient_buffer_snapshot(),
            }
        removed = self.patient_sign_buffer.pop()
        return {
            "success": True,
            "removed_sign": removed,
            **self.patient_buffer_snapshot(),
        }

    def cancel_patient_sentence(self) -> Dict[str, Any]:
        self.patient_sign_buffer = []
        self.patient_sentence_in_progress = False
        return {"success": True, **self.patient_buffer_snapshot()}

    def patient_buffer_snapshot(self) -> Dict[str, Any]:
        signs = deepcopy(self.patient_sign_buffer)
        return {
            "sentence_in_progress": self.patient_sentence_in_progress,
            "sign_count": len(signs),
            "signs": signs,
            "gloss_sequence": [s.get("gloss") for s in signs],
            "concept_sequence": [s.get("concept") for s in signs],
        }

    def commit_patient_sentence(self) -> Dict[str, Any]:
        """Store the completed patient turn and clear the temporary buffer."""
        if not self.patient_sign_buffer:
            return {
                "success": False,
                "reason": "No patient signs are available to finish.",
                **self.patient_buffer_snapshot(),
            }

        completed_signs = deepcopy(self.patient_sign_buffer)
        self.last_patient_signs = completed_signs
        self.last_patient_glosses = [s.get("gloss") for s in completed_signs]
        self.last_patient_concepts = [s.get("concept") for s in completed_signs]
        self.patient_sign_buffer = []
        self.patient_sentence_in_progress = False

        return {
            "success": True,
            "signs": deepcopy(self.last_patient_signs),
            "gloss_sequence": list(self.last_patient_glosses),
            "concept_sequence": list(self.last_patient_concepts),
            "sentence_in_progress": False,
            "sign_count": len(self.last_patient_signs),
        }

    # ============================================================
    # CONVERSATION CONTEXT UPDATES
    # ============================================================

    def update_from_doctor_mcie(self, text: str, mcie_output: Dict[str, Any]) -> None:
        interp = mcie_output.get("interpretation", {})
        intent = interp.get("intent")
        entities = interp.get("entities", {}) or {}
        concepts = interp.get("medical_concepts", []) or []

        self.last_doctor_text = text
        self.last_doctor_intent = intent
        self.last_doctor_entities = entities
        self.turn_count += 1

        symptom_candidates = entities.get("symptoms") or []
        if not symptom_candidates and entities.get("symptom"):
            symptom_candidates = [entities.get("symptom")]

        if symptom_candidates:
            self.current_symptom = symptom_candidates[0]
            self.current_topic = self.current_symptom
        else:
            symptom_like = [
                c for c in concepts
                if c in {
                    "FEVER", "COUGH", "FLU", "VOMITING", "ITCHING",
                    "SWELLING", "WEAKNESS", "FATIGUE", "PAIN", "HEADACHE",
                    "STOMACH_PAIN", "BACK_PAIN", "SORE_THROAT", "CHEST_PAIN",
                    "BREATHING_DIFFICULTY", "ASTHMA",
                }
            ]
            if symptom_like:
                self.current_symptom = symptom_like[0]
                self.current_topic = self.current_symptom

        if intent == "SYMPTOM_DURATION":
            self.expected_response_type = "DURATION"
        elif intent == "SYMPTOM_LOCATION":
            self.expected_response_type = "BODY_LOCATION"
        elif intent in {"MEDICATION_QUERY", "ALLERGY_CHECK"}:
            self.expected_response_type = "MEDICAL_OR_GENERAL"
        elif intent == "SYMPTOM_CHECK":
            self.expected_response_type = "GENERAL_RESPONSE"
        else:
            self.expected_response_type = None

    def update_from_patient_mcie(self, mcie_output: Dict[str, Any]) -> None:
        interp = mcie_output.get("interpretation", {})
        input_data = mcie_output.get("input", {}) or {}
        self.last_patient_intent = interp.get("intent")
        self.last_patient_glosses = input_data.get("gloss_sequence", []) or []
        self.last_patient_concepts = input_data.get("concept_sequence", []) or []
        self.turn_count += 1
        self.expected_response_type = None
