# SignBridge AI - Person 4 Source Code

## Role

**Person 4: System Integration and User Interface**

This package contains the source code developed for the integration and user-interface layer of the SignBridge AI prototype. It connects the independently developed BIM recognition, Automatic Speech Recognition (ASR), and Medical Conversation Intelligence Engine (MCIE) modules through explicit local interfaces. It also provides the two-display Patient and Doctor workflow.

## Included source code

| Path | Responsibility |
| --- | --- |
| `main.py` | Local application entry point, preflight validation, and Uvicorn server startup. |
| `signbridge_app/api.py` | FastAPI endpoints, upload validation, stale-request protection, confirmation endpoints, and WebSocket events. |
| `signbridge_app/runtime.py` | Composition root that connects recognition, ASR, MCIE, conversation state, and event delivery. |
| `signbridge_app/conversation.py` | Confirmed-message state, request lifecycle, duplicate confirmation handling, and conversation clearing. |
| `signbridge_app/contracts.py` | Stable data contracts between the independently developed AI modules and the UI. |
| `signbridge_app/recognition_service.py` | Adapter and routing layer for Person 1 medical, general, and number recognizers. |
| `signbridge_app/asr_service.py` | Adapter for the local Person 3 ASR pipeline. |
| `signbridge_app/mcie_service.py` | Adapter for Person 2 MCIE, confirmed conversation context, and safe patient sentence generation. |
| `signbridge_app/medical_guard.py` | Checks for changes to medical terms, medication details, quantities, units, and negation. |
| `signbridge_app/config.py` | Local model and teammate-module path configuration. |
| `signbridge_app/preflight.py` | Required UI, model, and dependency checks before startup. |
| `ui/patient.html` | Single-screen patient interface for camera/video input, recognition review, sentence construction, confirmation, chat, TTS, and avatar playback. |
| `ui/doctor.html` | Single-screen doctor interface for microphone input, ASR/MCIE review, editable confirmation, chat, history saving, and clearing. |
| `ui/js/` | UI state machines, browser-device adapters, backend communication, stale-result handling, touch keyboard, and confirmed-message synchronization. |
| `ui/css/styles.css` | Responsive dual-display layout and accessibility-oriented control styling. |
| `scripts/` | Windows setup, local model download, and offline launcher scripts. |
| `tests/` and `ui/tests/` | Python integration/regression tests and browser-based UI tests. |

## Implemented workflow

### Patient to doctor

1. The patient opens the camera or uploads a local video.
2. A recognition result is shown with its confidence.
3. The result is appended only after the patient selects **Confirm**; **Try again** discards it.
4. Confirmed BIM words build an editable sentence.
5. MCIE can propose a medical sentence while the original BIM sequence remains visible.
6. The patient reviews or edits the proposal and selects **Confirm & send**.
7. Only the confirmed final text is delivered to the Doctor display.

### Doctor to patient

1. The doctor records speech locally.
2. The original ASR transcript and MCIE suggestion are displayed separately.
3. The doctor edits the final message.
4. Only **Confirm & send** commits and delivers the message.
5. The patient can play the confirmed reply through local TTS or a supported local avatar clip.

## Important integration boundaries

The Person 1, Person 2, and Person 3 implementation files and trained model weights are intentionally not duplicated in this Person 4 package. In the complete project, they are supplied through `uploads/` and `models/`. This package contains the adapters and contracts used to call them.

The excluded items are:

- teammate-owned notebook/module source under `uploads/`;
- trained model weights under `models/`;
- datasets and training artifacts;
- the local `.venv` environment;
- caches and generated temporary files.

The package therefore documents and submits Person 4's work without claiming ownership of the other team members' model implementations. To run the complete AI pipeline, place this source in the full `SignBridge_AI_Rebuild` project that contains the required local teammate modules and model assets.

## Local execution

From the complete project folder on Windows PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" main.py --port 5501
```

Then open:

- Doctor display: `http://127.0.0.1:5501/ui/doctor.html`
- Patient display: `http://127.0.0.1:5501/ui/patient.html`

All runtime communication uses the local loopback server. The application does not require a cloud API during normal operation.

## Verification evidence

The source package includes automated coverage for API contracts, confirmation gating, stale-result rejection, conversation synchronization, offline configuration, medical-term protection, UI state transitions, device-adapter behavior, and responsive layout.

At packaging time, the complete project passed:

- **78 Python tests**
- **87 browser UI tests**

Physical camera, microphone, touchscreen, GPU, TTS voice, and real-model accuracy still require hardware acceptance testing on the demonstration computer.

## Submission note

This package is the **Source Code** contribution for Person 4 under the Phase 3 proposal requirement. The proposal should describe the complete team system separately and credit the Person 1, Person 2, and Person 3 AI modules to their respective authors.
