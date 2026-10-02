from fastapi.testclient import TestClient

from signbridge_app.api import create_app
from signbridge_app.contracts import MedicalProposal, TranscriptProposal
from signbridge_app.conversation import ConversationManager
from signbridge_app.runtime import EventHub, Runtime


class NoRecognition:
    pass


class Asr:
    def transcribe_wav(self, data, request_id):
        return TranscriptProposal(original_text="Do not take 5 mg aspirin", request_id=request_id)


class Mcie:
    def propose(self, source_text, sender, request_id):
        return MedicalProposal(
            source_text=source_text,
            suggested_text="Do not take 5 mg aspirin after meals",
            medical_flags=("MEDICATION",),
            request_id=request_id,
        )

    def record_confirmed(self, sender, text):
        pass

    def clear_context(self):
        pass


def test_confirmed_two_way_conversation_and_doctor_review_flow():
    runtime = Runtime(ConversationManager("e2e"), NoRecognition(), Asr(), Mcie(), EventHub())
    with TestClient(create_app(runtime)) as api:
        patient = api.post("/api/patient/confirm", json={"final_text": "CHEST PAIN", "message_id": "p1"})
        assert patient.json()["final_text"] == "CHEST PAIN"

        review = api.post("/api/doctor/asr", content=b"wav", headers={"content-type": "audio/wav"}).json()
        assert review["transcript"]["original_text"] == "Do not take 5 mg aspirin"
        assert review["medical_proposal"]["suggested_text"].endswith("after meals")

        doctor = api.post("/api/doctor/confirm", json={"final_text": "Do not take 5 mg aspirin", "message_id": "d1"})
        assert doctor.json()["original_text"] == "Do not take 5 mg aspirin"
        assert doctor.json()["final_text"] == "Do not take 5 mg aspirin"
        messages = api.get("/api/session").json()["messages"]
        assert [message["message_id"] for message in messages] == ["p1", "d1"]
