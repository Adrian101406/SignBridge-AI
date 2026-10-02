from contextlib import contextmanager

from fastapi.testclient import TestClient

from signbridge_app.api import create_app
from signbridge_app.contracts import MedicalProposal, RecognitionResult, TranscriptProposal
from signbridge_app.conversation import ConversationManager
from signbridge_app.runtime import EventHub, Runtime


class FakeRecognition:
    def recognize_clip(self, data, *, suffix, request_id, expected_domains):
        return RecognitionResult(gloss="PAIN", confidence=.94, recognizer_type="medical", stable=True, timestamp=1, request_id=request_id)


class RecognitionWithoutNumber:
    def recognize_clip(self, data, *, suffix, request_id, expected_domains):
        if "number" in expected_domains:
            raise RuntimeError("number recognizer must not be connected")
        return RecognitionResult(gloss="PAIN", confidence=.94, recognizer_type="medical", stable=True, timestamp=1, request_id=request_id)


class CapturingRecognition:
    def __init__(self):
        self.expected_domains = None

    def recognize_clip(self, data, *, suffix, request_id, expected_domains):
        self.expected_domains = expected_domains
        domain = expected_domains[0]
        return RecognitionResult(gloss="3" if domain == "number" else "PAIN", confidence=.94, recognizer_type=domain, stable=True, timestamp=1, request_id=request_id)


class FakeAsr:
    def transcribe_wav(self, data, request_id):
        return TranscriptProposal(original_text="Take 5 mg aspirin", request_id=request_id)


class FakeMcie:
    def __init__(self):
        self.confirmed = []
        self.clear_count = 0

    def propose(self, source_text, sender, request_id):
        return MedicalProposal(source_text=source_text, suggested_text="Take 5 mg aspirin after meals", medical_flags=("MEDICATION",), request_id=request_id)

    def record_confirmed(self, sender, text):
        self.confirmed.append((sender, text))

    def clear_context(self):
        self.clear_count += 1


@contextmanager
def client():
    runtime = Runtime(
        conversation=ConversationManager("test-session"),
        recognition=FakeRecognition(),
        asr=FakeAsr(),
        mcie=FakeMcie(),
        events=EventHub(),
    )
    with TestClient(create_app(runtime)) as api:
        yield api


def test_health_does_not_load_models():
    with client() as api:
        response = api.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_clip_rejects_wrong_content_type_and_large_body():
    with client() as api:
        assert api.post("/api/patient/clips", content=b"x", headers={"content-type": "text/plain"}).status_code == 415
        accepted = api.post("/api/patient/clips", content=b"x" * (8 * 1024 * 1024 + 1), headers={"content-type": "video/webm"})
        assert accepted.status_code == 200
        response = api.post(
            "/api/patient/clips",
            content=b"x",
            headers={"content-type": "video/webm", "content-length": str(100 * 1024 * 1024 + 1)},
        )
        assert response.status_code == 413


def test_patient_clip_returns_normalized_result():
    with client() as api:
        response = api.post("/api/patient/clips", content=b"clip", headers={"content-type": "video/webm"})
        assert response.status_code == 200
        assert response.json()["gloss"] == "PAIN"


def test_patient_clip_does_not_connect_number_recognizer():
    runtime = Runtime(
        conversation=ConversationManager("test-session"),
        recognition=RecognitionWithoutNumber(),
        asr=FakeAsr(),
        mcie=FakeMcie(),
        events=EventHub(),
    )
    with TestClient(create_app(runtime)) as api:
        response = api.post(
            "/api/patient/clips",
            content=b"clip",
            headers={"content-type": "video/webm"},
        )

    assert response.status_code == 200
    assert response.json()["recognizer_type"] == "medical"


def test_patient_clip_uses_only_number_recognizer_when_number_mode_is_requested():
    recognition = CapturingRecognition()
    runtime = Runtime(
        conversation=ConversationManager("test-session"),
        recognition=recognition,
        asr=FakeAsr(),
        mcie=FakeMcie(),
        events=EventHub(),
    )
    with TestClient(create_app(runtime)) as api:
        response = api.post(
            "/api/patient/clips",
            content=b"clip",
            headers={"content-type": "video/webm", "x-recognition-mode": "number"},
        )

    assert response.status_code == 200
    assert recognition.expected_domains == ("number",)
    assert response.json()["gloss"] == "3"


def test_patient_clip_rejects_unknown_recognition_mode():
    with client() as api:
        response = api.post(
            "/api/patient/clips",
            content=b"clip",
            headers={"content-type": "video/webm", "x-recognition-mode": "anything"},
        )

    assert response.status_code == 422


def test_doctor_asr_returns_original_and_separate_medical_proposal():
    with client() as api:
        response = api.post("/api/doctor/asr", content=b"wav", headers={"content-type": "audio/wav"})
        payload = response.json()
        assert payload["transcript"]["original_text"] == "Take 5 mg aspirin"
        assert payload["medical_proposal"]["suggested_text"].endswith("after meals")


def test_only_confirmed_doctor_text_enters_mcie_context():
    runtime = Runtime(
        conversation=ConversationManager("test-session"),
        recognition=FakeRecognition(),
        asr=FakeAsr(),
        mcie=FakeMcie(),
        events=EventHub(),
    )
    with TestClient(create_app(runtime)) as api:
        api.post("/api/doctor/asr", content=b"wav", headers={"content-type": "audio/wav"})
        assert runtime.mcie.confirmed == []

        response = api.post(
            "/api/doctor/confirm",
            json={"final_text": "How long have you had a fever?", "message_id": "d-context"},
        )

    assert response.status_code == 200
    assert runtime.mcie.confirmed == [("doctor", "How long have you had a fever?")]


def test_patient_mcie_proposes_without_sending_and_confirmation_preserves_bim_source():
    with client() as api:
        proposal = api.post("/api/patient/mcie", json={"source_text": "CHEST PAIN"})
        assert proposal.status_code == 200
        assert proposal.json()["source_text"] == "CHEST PAIN"
        assert api.get("/api/session").json()["messages"] == []

        confirmed = api.post(
            "/api/patient/confirm",
            json={"final_text": "I have chest pain", "message_id": "p1"},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["original_text"] == "CHEST PAIN"
        assert confirmed.json()["final_text"] == "I have chest pain"


def test_blank_confirm_is_rejected_and_duplicate_is_idempotent():
    with client() as api:
        assert api.post("/api/patient/confirm", json={"final_text": " ", "message_id": "m1"}).status_code == 422
        first = api.post("/api/patient/confirm", json={"final_text": "PAIN", "message_id": "m1"})
        second = api.post("/api/patient/confirm", json={"final_text": "PAIN", "message_id": "m1"})
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()


def test_websocket_replays_state_and_delivers_confirmed_message():
    with client() as api, api.websocket_connect("/ws/session") as socket:
        replay = socket.receive_json()
        assert replay["event"] == "session.snapshot"
        api.post("/api/doctor/confirm", json={"final_text": "Please rest", "message_id": "d1"})
        event = socket.receive_json()
        assert event["event"] == "message.confirmed"
        assert event["payload"]["final_text"] == "Please rest"


def test_duplicate_confirmation_is_not_broadcast_twice():
    class RecordingEvents(EventHub):
        def __init__(self):
            super().__init__()
            self.published = []

        async def publish(self, event):
            self.published.append(event)
            await super().publish(event)

    events = RecordingEvents()
    runtime = Runtime(ConversationManager("test-session"), FakeRecognition(), FakeAsr(), FakeMcie(), events)
    with TestClient(create_app(runtime)) as api:
        command = {"final_text": "PAIN", "message_id": "same-id"}
        assert api.post("/api/patient/confirm", json=command).status_code == 200
        assert api.post("/api/patient/confirm", json=command).status_code == 200
    assert len(events.published) == 1


def test_clear_conversation_removes_confirmed_history_and_broadcasts_to_both_displays():
    runtime = Runtime(
        conversation=ConversationManager("test-session"),
        recognition=FakeRecognition(),
        asr=FakeAsr(),
        mcie=FakeMcie(),
        events=EventHub(),
    )
    with TestClient(create_app(runtime)) as api, api.websocket_connect("/ws/session") as socket:
        socket.receive_json()
        api.post("/api/patient/confirm", json={"final_text": "Three days", "message_id": "p1"})
        confirmed = socket.receive_json()
        assert confirmed["event"] == "message.confirmed"

        response = api.post("/api/session/clear")
        assert response.status_code == 200
        cleared = socket.receive_json()

        assert response.json() == {"cleared_messages": 1}
        assert cleared["event"] == "conversation.cleared"
        assert cleared["payload"] == {"cleared_messages": 1}
        assert api.get("/api/session").json()["messages"] == []
        assert runtime.mcie.clear_count == 1
