# SignBridge-AI

SignBridge AI is an offline AI-powered bidirectional communication
platform designed to support communication between Deaf patients
using Bahasa Isyarat Malaysia (BIM) and medical professionals.

## System Architecture

### Patient → Medical Professional
Camera
→ BIM Recognition
→ MCIE
→ Malay/English Text

### Medical Professional → Patient
Microphone
→ Malaysian Whisper ASR
→ MCIE
→ Text-to-BIM
→ Unity BIM Avatar

## Main Modules

- Medical BIM Recognizer V2
- General BIM Recognizer V1
- Number Recognizer V1
- Malaysian Whisper ASR
- Medical Conversation Intelligence Engine (MCIE)
- Text-to-BIM Translation
- BIM-facing patient output
- Offline Integration

## Offline Operation

The prototype is designed to operate locally without continuous
Internet connectivity.


## Datasets

### Medical BIM Dataset

The Medical Recognizer was developed using the MSL Medical dataset hosted on Kaggle.

- Purpose: Medical BIM recognition
- Classes used: 40 individual medical glosses
- Processing: Video → MediaPipe pose, hand and face landmarks
- Source: Kaggle
- Dataset: `arkuuu21/msl-medical`

The dataset itself is not redistributed in this repository.

### General BIM Dataset

The General Recognizer was developed using the Malaysian Sign Language dataset hosted on Zenodo.

- Purpose: General BIM recognition
- Classes used: 117
- Processed samples: 15,266
- Representation used: `keypoints_258`
- DOI: 10.5281/zenodo.21631884

The dataset itself is not redistributed in this repository.

## External Models and Dependencies

The Medical Conversation Intelligence Engine uses the following
open-source technologies:

- Qwen3-4B-Instruct-2507 — Qwen Team
- Hugging Face Transformers
- PyTorch
- bitsandbytes (for optional 4-bit quantization)

All SignBridge-specific MCIE logic, including conversation state,
rule-based semantic anchoring, controlled medical mappings,
evidence validation, recognizer-context processing, and
ACCEPT / CONFIRM / RETRY safety logic, was developed by the team.

### Malaysian Whisper Small v3
Used for Malay/English automatic speech recognition.

Source:
- Model/repository: Malaysian Whisper Small v3
- Provider: Mesolitica
- Model ID: `mesolitica/malaysian-whisper-small-v3`

Used in:
`src/speech_to_text/`

The pretrained model is not redistributed in this repository.

## Team

Team Silence Love
Universiti Teknologi PETRONAS
Project Nexus 2026
