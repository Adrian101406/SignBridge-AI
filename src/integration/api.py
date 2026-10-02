from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from .runtime import Runtime


MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class ConfirmCommand(BaseModel):
    final_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    message_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PatientMcieCommand(BaseModel):
    source_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _event(name: str, runtime: Runtime, payload, request_id: str = "") -> dict:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return {
        "event": name,
        "request_id": request_id,
        "session_id": runtime.conversation.snapshot().session_id,
        "payload": payload,
    }


async def _body(request: Request, allowed: tuple[str, ...]) -> tuple[bytes, str]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in allowed:
        raise HTTPException(415, detail={"code": "unsupported_media_type"})
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail={"code": "upload_too_large"})
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail={"code": "upload_too_large"})
    if not data:
        raise HTTPException(422, detail={"code": "empty_upload"})
    return data, content_type


def create_app(runtime: Runtime, *, project_root: Path | None = None) -> FastAPI:
    app = FastAPI(title="SignBridge AI", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "session_id": runtime.conversation.snapshot().session_id}

    @app.get("/api/session")
    async def session():
        return runtime.conversation.snapshot()

    @app.post("/api/session/clear")
    async def clear_session():
        payload = {"cleared_messages": runtime.conversation.clear_confirmed_messages()}
        runtime.mcie.clear_context()
        await runtime.events.publish(_event("conversation.cleared", runtime, payload))
        return payload

    @app.post("/api/patient/clips")
    async def patient_clip(request: Request):
        data, content_type = await _body(request, ("video/webm", "video/mp4"))
        recognition_mode = request.headers.get("x-recognition-mode", "default").strip().lower()
        if recognition_mode not in {"default", "number"}:
            raise HTTPException(422, detail={"code": "invalid_recognition_mode"})
        expected_domains = ("number",) if recognition_mode == "number" else ("medical", "general")
        request_id = runtime.conversation.begin_request("patient_recognition")
        suffix = ".webm" if content_type == "video/webm" else ".mp4"
        try:
            result = await asyncio.to_thread(
                runtime.recognition.recognize_clip,
                data,
                suffix=suffix,
                request_id=request_id,
                expected_domains=expected_domains,
            )
        except Exception as exc:
            raise HTTPException(503, detail={"code": "recognition_failed", "message": str(exc)}) from exc
        if not runtime.conversation.accept_result(request_id, result):
            raise HTTPException(409, detail={"code": "stale_request"})
        await runtime.events.publish(_event("recognition.result", runtime, result, request_id))
        return result

    @app.post("/api/doctor/asr")
    async def doctor_asr(request: Request):
        data, _ = await _body(request, ("audio/wav", "audio/x-wav"))
        request_id = runtime.conversation.begin_request("doctor_asr")
        try:
            transcript = await asyncio.to_thread(
                runtime.asr.transcribe_wav,
                data,
                request_id=request_id,
            )
        except Exception as exc:
            raise HTTPException(503, detail={"code": "asr_failed", "message": str(exc)}) from exc
        if not runtime.conversation.accept_result(request_id, transcript):
            raise HTTPException(409, detail={"code": "stale_request"})
        mcie_id = runtime.conversation.begin_request("doctor_mcie")
        try:
            proposal = await asyncio.to_thread(
                runtime.mcie.propose,
                transcript.original_text,
                "doctor",
                mcie_id,
            )
        except Exception as exc:
            raise HTTPException(503, detail={"code": "mcie_failed", "message": str(exc)}) from exc
        if not runtime.conversation.accept_result(mcie_id, proposal):
            raise HTTPException(409, detail={"code": "stale_request"})
        payload = {"transcript": transcript.model_dump(mode="json"), "medical_proposal": proposal.model_dump(mode="json")}
        await runtime.events.publish(_event("doctor.proposal", runtime, payload, request_id))
        return payload

    @app.post("/api/patient/mcie")
    async def patient_mcie(command: PatientMcieCommand):
        request_id = runtime.conversation.begin_request("patient_mcie")
        try:
            proposal = await asyncio.to_thread(
                runtime.mcie.propose,
                command.source_text,
                "patient",
                request_id,
            )
        except Exception as exc:
            raise HTTPException(503, detail={"code": "mcie_failed", "message": str(exc)}) from exc
        if not runtime.conversation.accept_result(request_id, proposal):
            raise HTTPException(409, detail={"code": "stale_request"})
        await runtime.events.publish(_event("patient.proposal", runtime, proposal, request_id))
        return proposal

    async def confirm(command: ConfirmCommand, sender: Literal["patient", "doctor"]):
        already_confirmed = any(
            message.message_id == command.message_id
            for message in runtime.conversation.snapshot().messages
        )
        try:
            message = (
                runtime.conversation.confirm_patient(command.final_text, command.message_id)
                if sender == "patient"
                else runtime.conversation.confirm_doctor(command.final_text, command.message_id)
            )
        except ValueError as exc:
            raise HTTPException(409, detail={"code": "confirmation_conflict", "message": str(exc)}) from exc
        if not already_confirmed:
            runtime.mcie.record_confirmed(sender, message.final_text)
            await runtime.events.publish(_event("message.confirmed", runtime, message))
        return message

    @app.post("/api/patient/confirm")
    async def confirm_patient(command: ConfirmCommand):
        return await confirm(command, "patient")

    @app.post("/api/doctor/confirm")
    async def confirm_doctor(command: ConfirmCommand):
        return await confirm(command, "doctor")

    @app.websocket("/ws/session")
    async def websocket_session(socket: WebSocket):
        await socket.accept()
        queue = runtime.events.subscribe()
        try:
            await socket.send_json(_event("session.snapshot", runtime, runtime.conversation.snapshot()))
            while True:
                event_task = asyncio.create_task(queue.get())
                receive_task = asyncio.create_task(socket.receive())
                done, pending = await asyncio.wait(
                    (event_task, receive_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if receive_task in done:
                    incoming = receive_task.result()
                    if incoming["type"] == "websocket.disconnect":
                        break
                if event_task in done:
                    await socket.send_json(event_task.result())
        except WebSocketDisconnect:
            pass
        finally:
            runtime.events.unsubscribe(queue)

    root = (project_root or Path(__file__).resolve().parents[1]) / "ui"
    app.mount("/ui", StaticFiles(directory=root, html=True), name="ui")
    return app
