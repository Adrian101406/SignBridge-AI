# SignBridge AI System Architecture

## 1. Overview

SignBridge AI is a modular, offline-first bidirectional communication prototype for interactions between Deaf patients using Bahasa Isyarat Malaysia (BIM) and medical professionals.

The architecture intentionally separates:

1. input acquisition;
2. AI perception;
3. conversation intelligence;
4. safety and human verification;
5. translation;
6. user output.

This separation prevents uncertain perception results from being treated automatically as final medical communication.

---

## 2. Patient → Medical Professional

```text
Patient BIM
   ↓
Camera
   ↓
Context / Expected-Slot Router
   ↓
┌──────────────────────────────────────┐
│ Medical V2 │ General V1 │ Number V1 │
└──────────────────────────────────────┘
   ↓
Top-K Recognition Result
   ↓
Recognizer Decision + Quality Checks
   ↓
Patient Sign Confirmation
   ↓
Patient Sentence Buffer
(Add Sign / Delete Last / Finish Sentence)
   ↓
MCIE
   ↓
Structured Meaning
(intent / concepts / entities / duration)
   ↓
Safe Malay / English Text
   ↓
Medical Professional Display
```

### Important scope

The current recognizers process **one isolated sign per capture**. The prototype does not claim continuous BIM sentence segmentation.

A multi-sign message is built by confirming and adding isolated signs to the patient sentence buffer. The sentence is processed only after the user selects **Finish Sentence**.

---

## 3. Medical Professional → Patient

```text
Doctor Speech
   ↓
Microphone
   ↓
Audio Pre-processing / VAD
   ↓
Malaysian Whisper ASR
   ↓
Healthcare Transcript Validation
   ↓
Doctor Confirmation when required
   ↓
MCIE
   ↓
Structured Medical Meaning
   ↓
Text-to-BIM
   ↓
Supported BIM Gloss Sequence
   ↓
Avatar-Ready Safety Check
   ↓
Unity BIM Avatar
```

The ASR healthcare-validation step is upstream of MCIE. MCIE should consume the confirmed/validated transcript payload rather than duplicate the ASR validation logic.

---

## 4. Medical Conversation Intelligence Engine (MCIE)

```text
Input Adapters
      ↓
Conversation State & Dialogue Memory
      ↓
Rule-Based Semantic Anchor
      ↓
Context-Aware Recognition Intelligence
      ↓
Local Qwen Semantic Reasoning
      ↓
Evidence Validator + Controlled Medical Mappings
      ↓
Safety Decision
ACCEPT / CONFIRM / RETRY
      ↓
Unified MCIE Output
```

### MCIE responsibilities

- preserve conversation context across turns;
- identify intent, medical concepts and entities;
- interpret confirmed patient sign sequences;
- handle number + time-unit combinations such as `2 + DAY`;
- use an LLM for flexible Malay / English / code-switched understanding;
- validate AI output against available evidence;
- trigger confirmation for uncertain or critical information.

### MCIE safety rules

- The LLM must not invent recognizer candidates.
- Contextual reranking is restricted to candidates returned in Top-K.
- Critical medical information can require human confirmation.
- Unsupported evidence should not be converted into a confident medical claim.
- AI output is assistive communication, not diagnosis or prescribing.

---

## 5. Recognizer Routing

The Medical, General and Number models are separately trained models. Their raw softmax confidence scores are therefore **not directly comparable**.

Routing is based on conversational context / expected response type.

Examples:

```text
SYMPTOM_CHECK       → Medical
SYMPTOM_LOCATION    → Medical
SYMPTOM_DURATION    → Number, then General time unit
NUMBER expected     → Number
TIME_UNIT expected  → General
GENERAL response    → General
```

See [`../configs/routing.yaml`](../configs/routing.yaml).

---

## 6. Text-to-BIM Safety

Text-to-BIM only produces avatar output when the concept is supported by the prototype vocabulary.

Example:

```text
Supported concept
FEVER → DEMAM → READY_FOR_AVATAR
```

An unsupported concept must be blocked rather than approximated using another sign.

```text
CHEST_PAIN
→ UNSUPPORTED_CONCEPT
→ avatar_ready = false
```

This avoids silently changing the medical meaning.

---

## 7. Offline Deployment

The target runtime keeps the following components local:

- BIM recognizer models;
- MediaPipe task files;
- Malaysian Whisper;
- Qwen MCIE model;
- controlled medical mappings;
- Text-to-BIM vocabulary;
- Unity avatar assets.

No cloud API is required for the intended offline prototype runtime.

---

## 8. Current Limitations

- isolated-sign recognition only;
- limited prototype BIM vocabulary;
- Number recognizer limited to static signs 0–10;
- no expert-validated full BIM grammar generation;
- hardware latency depends on the machine running Whisper and Qwen;
- full real-world end-to-end testing with diverse users remains required.
