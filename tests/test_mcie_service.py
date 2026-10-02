from signbridge_app.config import DEFAULT_SETTINGS, Settings
import signbridge_app.mcie_service as mcie_service
from signbridge_app.mcie_service import McieService, McieUnavailable


class FakeMcie:
    def __init__(self, result=None, error=None):
        self.result = result or {
            "success": True,
            "interpretation": {"medical_concepts": ["CHEST_PAIN"]},
            "output": {"simplified_text_en": "I have chest pain."},
            "safety": {"reasons": []},
        }
        self.error = error
        self.calls = []

    def doctor_text(self, **kwargs):
        self.calls.append(("doctor", kwargs))
        if self.error:
            raise self.error
        return self.result

    def patient_bim(self, **kwargs):
        self.calls.append(("patient", kwargs))
        if self.error:
            raise self.error
        return self.result


def test_doctor_proposal_preserves_source_and_routes(tmp_path):
    engine = FakeMcie()
    service = McieService(Settings.from_project_root(tmp_path), engine=engine)

    proposal = service.propose("I got chest pain", "doctor", "req-1")

    assert proposal.source_text == "I got chest pain"
    assert proposal.suggested_text == "I have chest pain."
    assert proposal.request_id == "req-1"
    assert engine.calls[0][0] == "doctor"


def test_patient_proposal_routes_glosses(tmp_path):
    engine = FakeMcie()
    service = McieService(Settings.from_project_root(tmp_path), engine=engine)

    service.propose("CHEST PAIN", "patient", "req-2")

    _, call = engine.calls[0]
    assert call["gloss_sequence"] == ["CHEST", "PAIN"]
    assert call["concept_sequence"] == ["CHEST", "PAIN"]


def test_confirmed_doctor_context_is_passed_to_patient_mcie():
    engine = FakeMcie()
    service = McieService(DEFAULT_SETTINGS, engine=engine)

    service.record_confirmed("doctor", "How long have you had a fever?")
    service.propose("3", "patient", "req-context")

    _, call = engine.calls[-1]
    assert call["state"].ai_snapshot() == {
        "current_topic": "FEVER",
        "current_symptom": "FEVER",
        "last_doctor_text": "How long have you had a fever?",
        "last_doctor_intent": "SYMPTOM_DURATION",
        "expected_response_type": "DURATION",
    }


def test_clearing_mcie_context_removes_confirmed_doctor_context():
    engine = FakeMcie()
    service = McieService(DEFAULT_SETTINGS, engine=engine)
    service.record_confirmed("doctor", "How long have you had a fever?")

    service.clear_context()
    service.propose("3", "patient", "req-cleared")

    _, call = engine.calls[-1]
    assert call["state"].ai_snapshot() == {}


def test_confirmed_days_question_turns_numeric_patient_answer_into_duration_sentence():
    engine = FakeMcie()
    service = McieService(DEFAULT_SETTINGS, engine=engine)
    service.record_confirmed("doctor", "How many days have you had a fever?")

    proposal = service.propose("3", "patient", "req-duration")

    assert proposal.suggested_text == "I have had a fever for 3 days."
    assert proposal.medical_flags == ("FEVER",)
    assert engine.calls == []


def test_single_patient_symptom_is_formatted_as_a_first_person_sentence():
    engine = FakeMcie(result={
        "success": True,
        "interpretation": {
            "intent": "SYMPTOM_REPORT",
            "medical_concepts": ["HEADACHE"],
        },
        "output": {"simplified_text_en": "Headache"},
        "safety": {"reasons": []},
    })
    service = McieService(DEFAULT_SETTINGS, engine=engine)

    proposal = service.propose("Sakit Kepala", "patient", "req-first-person")

    assert proposal.suggested_text == "I have a headache."
    assert proposal.medical_flags == ("HEADACHE",)


def test_patient_proposal_preserves_multiword_medical_glosses(tmp_path):
    engine = FakeMcie()
    service = McieService(Settings.from_project_root(tmp_path), engine=engine)

    service.propose(
        "Lidah Sakit Kepala Tulang Rusuk Buah Pinggang Leher",
        "patient",
        "req-multiword",
    )

    _, call = engine.calls[0]
    assert call["gloss_sequence"] == [
        "Lidah",
        "Sakit Kepala",
        "Tulang Rusuk",
        "Buah Pinggang",
        "Leher",
    ]
    assert call["concept_sequence"] == [
        "TONGUE",
        "HEADACHE",
        "RIB_CAGE",
        "KIDNEY",
        "NECK",
    ]


def test_patient_proposal_with_safety_reasons_does_not_use_ai_suggestion(tmp_path):
    engine = FakeMcie(result={
        "success": True,
        "interpretation": {"medical_concepts": ["HEADACHE"]},
        "output": {"simplified_text_en": "Headache and invented jaw pain."},
        "safety": {"reasons": ["Unsupported AI concept removed: JAW_PAIN"]},
    })
    service = McieService(Settings.from_project_root(tmp_path), engine=engine)

    proposal = service.propose("Sakit Kepala", "patient", "req-unsafe")

    assert proposal.suggested_text == "Sakit Kepala"
    assert "Unsupported AI concept removed: JAW_PAIN" in proposal.medical_flags


def test_default_mcie_engine_allows_complete_structured_output(tmp_path, monkeypatch):
    class LengthSensitiveEngine:
        def __init__(self, max_new_tokens):
            self.max_new_tokens = max_new_tokens

        def patient_bim(self, **kwargs):
            if self.max_new_tokens < 450:
                return {
                    "success": False,
                    "output": {},
                    "safety": {"reasons": ["AI did not return valid structured JSON."]},
                }
            return {
                "success": True,
                "interpretation": {"medical_concepts": ["HEADACHE"]},
                "output": {"simplified_text_en": "Headache"},
                "safety": {"reasons": []},
            }

    class FakeMcieModule:
        @staticmethod
        def AIEnhancedMCIE(llm, max_new_tokens):
            return LengthSensitiveEngine(max_new_tokens)

    real_import_module = mcie_service.importlib.import_module
    monkeypatch.setattr(
        mcie_service.importlib,
        "import_module",
        lambda name: FakeMcieModule if name == "signbridge.mcie_ai" else real_import_module(name),
    )
    monkeypatch.setattr(mcie_service, "LocalGgufLlm", lambda path: object())

    proposal = McieService(Settings.from_project_root(tmp_path)).propose(
        "Sakit Kepala", "patient", "req-complete"
    )

    assert proposal.suggested_text == "Headache"


def test_malformed_ai_output_falls_back_to_source(tmp_path):
    engine = FakeMcie(result={"success": False, "safety": {"reasons": ["invalid JSON"]}})
    proposal = McieService(Settings.from_project_root(tmp_path), engine=engine).propose(
        "Take one pill", "doctor", "req-3"
    )
    assert proposal.suggested_text == "Take one pill"
    assert "invalid JSON" in proposal.medical_flags


def test_model_error_is_mapped(tmp_path):
    engine = FakeMcie(error=RuntimeError("GPU failed"))
    service = McieService(Settings.from_project_root(tmp_path), engine=engine)
    try:
        service.propose("hello", "doctor", "req-4")
    except McieUnavailable as exc:
        assert "GPU failed" in str(exc)
    else:
        raise AssertionError("McieUnavailable was not raised")


def test_critical_changes_are_exposed(tmp_path):
    engine = FakeMcie(result={
        "success": True,
        "interpretation": {"medical_concepts": ["MEDICATION"]},
        "output": {"simplified_text_en": "Take 10 mg ibuprofen"},
        "safety": {"reasons": []},
    })
    proposal = McieService(Settings.from_project_root(tmp_path), engine=engine).propose(
        "Take 5 mg aspirin", "doctor", "req-5"
    )
    assert set(proposal.changed_critical_terms) >= {"number", "medication"}
