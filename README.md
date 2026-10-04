# SignBridge AI

**AI-Powered Bidirectional Communication System for Malaysian Sign Language (BIM) in Healthcare**

SignBridge AI is an AI-assisted communication system designed to reduce communication barriers between **Malaysian Sign Language (Bahasa Isyarat Malaysia, BIM) users and healthcare professionals**.

The system supports two-way communication:

- **Patient → Healthcare Professional:** BIM gestures are recognized and converted into readable text.
- **Healthcare Professional → Patient:** Spoken communication is transcribed, processed into simplified BIM-compatible content, and mapped to BIM output.

The project combines **computer vision, deep learning, automatic speech recognition (ASR), medical context processing, and BIM language mapping** into a unified healthcare communication prototype.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Main Modules](#main-modules)
- [BIM Recognition Models](#bim-recognition-models)
- [Automatic Speech Recognition](#automatic-speech-recognition)
- [Medical Context Interpretation Engine](#medical-context-interpretation-engine)
- [Text-to-BIM](#text-to-bim)
- [User Interface](#user-interface)
- [Installation](#installation)
- [Running the Prototype](#running-the-prototype)
- [Datasets](#datasets)
- [External Models and Dependencies](#external-model-and-dependencies)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Ethical and Clinical Considerations](#ethical-and-clinical-considerations)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

# Project Overview

Communication between Deaf or hard-of-hearing patients and healthcare professionals can be difficult when a qualified sign-language interpreter is unavailable.

In Malaysia, this challenge is particularly relevant because existing sign-language recognition systems frequently focus on American Sign Language (ASL) or other international sign languages rather than **Bahasa Isyarat Malaysia (BIM)**.

SignBridge AI aims to provide an assistive communication interface that supports basic healthcare-related conversations using Malaysian Sign Language.

The system integrates multiple AI components instead of relying on a single recognition model.

### Patient-to-Healthcare Professional

```text
Patient performs BIM gesture
        ↓
Camera captures gesture
        ↓
MediaPipe landmark extraction
        ↓
Context-aware recognizer routing
        ↓
Medical / General / Number recognizer
        ↓
Predicted BIM gloss
        ↓
Readable text
        ↓
Healthcare professional
```

### Healthcare Professional-to-Patient

```text
Healthcare professional speaks
        ↓
Microphone input
        ↓
Automatic Speech Recognition
        ↓
Medical Context Interpretation Engine
        ↓
Text simplification and BIM-compatible processing
        ↓
Text-to-BIM mapping
        ↓
BIM-facing patient output
```

---

# Problem Statement

Healthcare interactions require accurate and timely communication.

However, Deaf and hard-of-hearing patients may experience communication barriers when healthcare professionals are unable to communicate using Malaysian Sign Language.

Depending solely on interpreters may also be difficult because interpreters may not always be immediately available.

Existing automated sign-language systems have several limitations:

- Many focus on ASL rather than BIM.
- General sign-language models may not contain sufficient medical vocabulary.
- Medical terminology can be difficult for general speech-recognition and language-processing systems.
- Similar gestures may be difficult to differentiate.
- Different recognition models may produce confidence scores that are not directly comparable.
- Continuous and sentence-level sign-language recognition remains significantly more difficult than isolated-sign recognition.

SignBridge AI explores a modular AI architecture specifically designed for BIM-based healthcare communication.

---

# Objectives

The main objectives of SignBridge AI are:

1. Develop an AI-based system capable of recognizing selected Malaysian Sign Language gestures.
2. Develop a dedicated **Medical BIM Recognizer** for healthcare-related signs.
3. Develop a **General BIM Recognizer** for commonly used BIM vocabulary.
4. Develop a dedicated **Number Recognizer** for numerical communication.
5. Implement context-aware routing between different BIM recognizers.
6. Convert healthcare professionals' speech into text using Automatic Speech Recognition.
7. Interpret healthcare-related speech using a Medical Context Interpretation Engine.
8. Convert processed text into BIM-compatible output.
9. Integrate all modules into a functional bidirectional healthcare communication prototype.

---

# System Architecture

SignBridge AI uses a modular architecture.

```text
                         SIGNBRIDGE AI

        ┌─────────────────────────────────────────────┐
        │      Patient → Healthcare Professional      │
        └─────────────────────────────────────────────┘

                         Camera
                           │
                           ▼
                MediaPipe Landmark Extraction
                           │
                           ▼
                 Context Routing / MCIE
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Medical       General      Number
          Recognizer    Recognizer    Recognizer
              │            │            │
              └────────────┼────────────┘
                           ▼
                    BIM Gloss / Text
                           │
                           ▼
                Healthcare Professional


        ┌─────────────────────────────────────────────┐
        │      Healthcare Professional → Patient      │
        └─────────────────────────────────────────────┘

                       Microphone
                           │
                           ▼
                  Speech Recognition
                           │
                           ▼
          Medical Context Interpretation Engine
                           │
                           ▼
                 Text Simplification
                           │
                           ▼
                    Text-to-BIM
                           │
                           ▼
                  BIM Patient Output
```

---

# Main Modules

The prototype consists of several major components.

| Module | Purpose |
|---|---|
| Medical BIM Recognizer | Recognizes healthcare-related BIM signs |
| General BIM Recognizer | Recognizes general BIM vocabulary |
| Number Recognizer | Recognizes numerical signs |
| MediaPipe Processing | Extracts body, hand and facial landmarks |
| ASR | Converts healthcare professional speech into text |
| MCIE | Interprets medical context and assists routing |
| Text-to-BIM | Converts processed text into BIM-compatible glosses |
| SignBridge Application | Integrates the different AI components |
| User Interface | Provides interaction between patient and healthcare professional |

---

# BIM Recognition Models

SignBridge AI separates BIM recognition into multiple specialized models instead of using one large classifier.

This approach allows models to be trained specifically for different types of signs.

## 1. Medical BIM Recognizer

The Medical BIM Recognizer focuses on healthcare-related vocabulary.

Examples may include signs associated with:

- symptoms,
- pain,
- medication,
- body conditions,
- healthcare communication,
- clinical questions.

The current model contains **40 medical BIM classes**.

Video frames are processed using MediaPipe to extract landmarks before classification.

---

## 2. General BIM Recognizer

The General BIM Recognizer handles common Malaysian Sign Language vocabulary outside the specialized medical vocabulary.

The current General Recognizer contains **117 BIM classes**.

The training pipeline uses landmark-based temporal sequences generated from the original videos.

The processed representation includes MediaPipe keypoints rather than directly training on raw video frames.

---

## 3. Number Recognizer

A separate Number Recognizer is used for numerical signs.

Numbers are particularly important in healthcare communication for information such as:

- age,
- dosage,
- duration,
- pain scales,
- frequency,
- appointment dates,
- measurements.

Using a dedicated model improves number recognition without requiring numerical classes to compete directly with the much larger general or medical vocabularies.

---

# Context-Aware Recognizer Routing

An earlier system design considered selecting the prediction with the highest confidence across all recognizers.

However, confidence scores produced by independently trained models are **not necessarily calibrated or directly comparable**.

For example:

```text
Medical model confidence = 0.78
General model confidence = 0.92
```

does not necessarily mean that the General model prediction is more reliable.

For this reason, SignBridge AI uses **context-aware routing** rather than directly comparing confidence scores between recognizers.

The routing logic considers the expected communication context and selects the appropriate recognizer before interpreting the model confidence.

The configuration therefore follows the principle:

```yaml
compare_cross_model_confidence: false
context_gating: true
```

This reduces bias toward recognizers that naturally produce higher softmax confidence values.

---

# Landmark Extraction

MediaPipe is used to transform video frames into numerical landmark representations.

Depending on the recognizer, extracted features may contain information from:

- left hand,
- right hand,
- pose,
- selected facial landmarks.

Using landmarks instead of raw RGB images offers several advantages:

- Reduced input dimensionality
- Lower computational requirements
- Greater emphasis on body and hand movement
- Reduced dependence on background appearance
- Improved suitability for temporal sequence modelling

---

# Automatic Speech Recognition

For the **Healthcare Professional → Patient** communication direction, spoken language is first converted into text.

The ASR module is designed to process healthcare-related speech and Malaysian conversational speech.

```text
Speech
  ↓
Microphone
  ↓
ASR
  ↓
Transcribed text
```

The resulting transcript is passed to the Medical Context Interpretation Engine for further processing.

ASR performance may vary depending on:

- microphone quality,
- background noise,
- accent,
- speaking speed,
- medical terminology,
- pronunciation.

---

# Medical Context Interpretation Engine

The **Medical Context Interpretation Engine (MCIE)** acts as an intermediate intelligence layer between the different modules.

Its responsibilities include:

- interpreting healthcare-related context,
- simplifying spoken sentences,
- identifying important medical concepts,
- assisting recognizer routing,
- preparing text for BIM mapping,
- reducing unnecessary linguistic complexity.

For example:

```text
Original speech:
"Have you been experiencing a fever for more than three days?"

Processed representation:
FEVER DURATION MORE THAN THREE DAY?
```

The goal is not word-for-word translation.

Instead, MCIE attempts to preserve the **intended meaning** while producing content that is easier to map into the available BIM vocabulary.

---

# Text-to-BIM

The Text-to-BIM module converts processed text into BIM-compatible output.

The system uses available vocabulary mappings to identify BIM glosses corresponding to the processed sentence.

Example:

```text
Input:
How long have you had fever?

Processed:
FEVER HOW-LONG?

BIM mapping:
FEVER → HOW-LONG
```

The system currently focuses on vocabulary-level and phrase-level mapping.

Full grammatical BIM sentence generation remains an area for future development.

---

# User Interface

The user interface provides interaction for both communication directions.

The interface is designed around two primary users:

### Patient

The patient can:

- perform BIM signs in front of the camera,
- view recognized signs,
- receive BIM-compatible communication from the healthcare professional.

### Healthcare Professional

The healthcare professional can:

- view recognized BIM output,
- speak using a microphone,
- review speech-to-text output,
- communicate healthcare-related information to the patient.

The UI is intended as a **prototype assistive interface**, not a replacement for professional medical interpreters.

---


# Installation

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

---

## 2. Create a virtual environment

Using Python `venv`:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Model Setup

Some trained models or third-party model files may not be included directly in the GitHub repository because of:

- file-size limitations,
- third-party licences,
- model-distribution restrictions.

Depending on the module, required files may include:

```text
.keras
.tflite
.task
.npz
.json
```

Place model files in the directories specified by the corresponding module documentation.

Do not rename model files unless the relevant configuration is also updated.

---

# Running the Prototype

After installing all required dependencies and model files, run:

```bash
python main.py
```

The main application integrates components from:

```text
src/signbridge_app/
```

including components such as:

```python
from signbridge_app.api import create_app
from signbridge_app.preflight import run_preflight
from signbridge_app.runtime import Runtime
```

If Python cannot locate the `src` directory, ensure that the project environment is configured so that `src` is available on the Python module path.

For development environments, this may require configuring `PYTHONPATH` or installing the project as a package.

---

# Datasets

SignBridge AI uses external datasets for model development.

The datasets are **not redistributed directly through this repository unless their licences explicitly permit redistribution**.

Users should obtain the original datasets from their respective sources.

---

## Medical BIM Dataset

The Medical BIM Recognizer was developed using a Malaysian medical sign-language dataset available through Kaggle.

**Source:** Kaggle  
**Dataset:** MSL Medical  
**Kaggle identifier:** `arkuuu21/msl-medical`

### Usage in SignBridge AI

The dataset was used for:

- Medical BIM recognition
- 40 medical sign classes
- MediaPipe landmark extraction
- Temporal gesture classification
- Model evaluation

The dataset itself is not redistributed through this repository.

Please refer to the original Kaggle dataset page for its licence and usage conditions.

---

## General BIM Dataset

The General BIM Recognizer was developed using a Malaysian Sign Language dataset hosted on Zenodo.

**Source:** Zenodo  
**DOI:** `10.5281/zenodo.21631884`

### Usage in SignBridge AI

The dataset was used for:

- General BIM recognition
- 117 BIM classes
- Landmark preprocessing
- Temporal gesture classification

The processed training representation uses MediaPipe-derived keypoints.

The original dataset is not redistributed through this repository.

Please refer to the Zenodo record for the dataset licence and citation requirements.

---

## Number Dataset

The Number Recognizer uses numerical BIM training data prepared for numerical gesture recognition.

Details regarding:

- dataset source,
- number of samples,
- preprocessing procedure,
- train/validation/test split,

should be documented in:

```text
src/bim_recognition/number/README.md
```

---

# Dataset Attribution

If you use this repository or reproduce the models, please also cite the **original datasets** used to train the models.

Dataset citation is separate from citation of the SignBridge AI source code.

For academic reports, publications, competitions, or demonstrations, the original Kaggle and Zenodo dataset sources should be acknowledged.

---

## External Models and Dependencies

### Malaysian Whisper Small v3
Used for Malay/English automatic speech recognition.

Source:
- Model/repository: Malaysian Whisper Small v3
- Provider: Mesolitica
- Model ID: `mesolitica/malaysian-whisper-small-v3`

Used in:
`src/speech_to_text/`

The pretrained model is not redistributed in this repository.

### MCIE
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

# Testing

The project includes tests for individual modules and integrated workflows.

Where applicable, tests cover:

- recognizer loading,
- configuration loading,
- routing,
- ASR processing,
- MCIE processing,
- Text-to-BIM mapping,
- application integration,
- end-to-end workflows.

Run the test suite using:

```bash
pytest
```

or:

```bash
python -m pytest
```

depending on the project environment.

---

# Known Limitations

SignBridge AI is currently a research and prototype system.

Several limitations remain.

## 1. Limited Vocabulary

The recognizers only support signs included in their respective training vocabularies.

Unknown signs cannot currently be interpreted reliably.

---

## 2. Isolated Sign Recognition

Most recognition currently focuses on isolated signs or short controlled sequences.

Natural BIM communication contains:

- transitions between signs,
- facial expressions,
- body posture,
- spatial grammar,
- sentence-level context.

Continuous sign-language recognition remains future work.

---

## 3. Real-World Generalization

Model performance may decrease when used with:

- unseen users,
- different camera positions,
- different lighting conditions,
- partially occluded hands,
- different signing speeds,
- different backgrounds.

---

## 4. Recognition Latency

Depending on hardware and the active AI modules, the complete system may experience noticeable latency.

Further optimization is required for real-time clinical deployment.

---

## 5. Cross-Model Confidence

Confidence scores from independently trained recognizers are not directly comparable.

SignBridge AI therefore avoids choosing recognizers solely based on maximum cross-model confidence.

Context-aware routing is used instead.

---

## 6. Medical Vocabulary

The current medical BIM vocabulary is limited to the available training data.

It does not cover all:

- diseases,
- symptoms,
- medications,
- procedures,
- specialist terminology.

---

## 7. Speech Recognition Errors

ASR may incorrectly transcribe:

- uncommon medical terminology,
- heavily accented speech,
- noisy environments,
- rapidly spoken sentences.

---

## 8. BIM Grammar

The current Text-to-BIM system does not provide complete natural-language-to-BIM grammatical translation.

It primarily performs:

- simplification,
- concept extraction,
- vocabulary mapping,
- BIM gloss generation.

---

## 9. Clinical Validation

The current prototype has not been validated as a certified medical communication device.

It should not be relied upon for critical clinical decisions.

---

# Future Improvements

Future development may include:

### Continuous BIM Recognition

Move from isolated sign recognition toward continuous sentence-level BIM recognition.

### Larger Medical Vocabulary

Collect and label additional BIM healthcare signs.

### Signer-Independent Evaluation

Evaluate the system using participants who were not included in the training dataset.

### Confidence Calibration

Develop calibrated confidence estimates for individual recognition models.

### Improved Context Routing

Use conversational context to determine whether Medical, General, or Number recognition should be activated.

### Faster Inference

Optimize:

- model architecture,
- landmark extraction,
- model loading,
- inference pipeline,
- hardware acceleration.

### Improved ASR

Fine-tune speech recognition for Malaysian healthcare conversations and medical vocabulary.

### Improved BIM Translation

Develop more sophisticated language modelling for Malaysian Sign Language grammar.

### Expanded BIM Output

Future versions may integrate a more advanced animated or 3D BIM avatar system.

---

# Ethical and Clinical Considerations

SignBridge AI is intended as an **assistive communication technology**.

It is not intended to replace:

- qualified BIM interpreters,
- healthcare professionals,
- professional medical advice,
- emergency communication procedures.

Incorrect interpretation of medical information can have serious consequences.

For high-risk or critical communication, healthcare institutions should continue to use qualified interpreters and established accessibility procedures.

Users should also be informed when communication is being interpreted by an AI system.

---

# Privacy

Video, audio and healthcare-related communication may contain sensitive information.

Deployments of SignBridge AI should consider:

- patient consent,
- data minimization,
- secure storage,
- access control,
- encryption,
- retention policies,
- applicable privacy regulations.

Where possible, visual and audio processing should be performed locally without unnecessarily storing raw patient recordings.

---

# Reproducibility

To improve reproducibility, this repository aims to document:

- dataset sources,
- preprocessing procedures,
- class mappings,
- model architectures,
- trained model formats,
- evaluation metrics,
- routing configuration,
- runtime dependencies.

External datasets and third-party models should be obtained from their original sources whenever redistribution is not permitted.

---

# Contributors

SignBridge AI was developed as a collaborative project involving work across:

- Computer Vision
- Machine Learning
- Malaysian Sign Language Recognition
- Automatic Speech Recognition
- Natural Language Processing
- Medical Context Interpretation
- System Integration
- User Interface Development

### Project Team

Team Silence Love
Universiti Teknologi PETRONAS
Project Nexus 2026

Team members :

```text
Tong Zi Yi
Alden Ting Tiew Hui
Adrian Wong Chee Yuen
Tee Quan Sheng
```


# Acknowledgements

The SignBridge AI team would like to acknowledge:

- the creators and contributors of the Malaysian Sign Language datasets used in this project,
- Kaggle dataset contributors,
- Zenodo dataset contributors,
- MediaPipe,
- open-source machine-learning communities,
- researchers working on sign-language recognition,
- BIM users and communities whose language and communication needs motivate this work.

We also acknowledge the open-source libraries and pretrained technologies that support the development of this prototype.

---

# Citation

If SignBridge AI is used in academic work, please cite the project repository together with the original datasets used for model training.

A project citation can be added once publication information is available.

Example:

```text
SignBridge AI Team. (2026). SignBridge AI: AI-Powered Bidirectional
Communication System for Malaysian Sign Language in Healthcare.
GitHub repository.
```

Always cite the original dataset publications or repository records separately.

---

# License

The source code developed specifically for SignBridge AI is released under
the MIT License. See the [LICENSE](LICENSE) file for details.

This license applies only to original SignBridge AI source code.

External datasets, pretrained models, libraries, BIM media assets, and
third-party resources remain subject to their respective licenses and terms
of use.

In particular, the datasets obtained from Kaggle and Zenodo are not
relicensed by this repository. Users must follow the terms specified by the
original dataset providers.

# Disclaimer

**SignBridge AI is a research prototype and is not a certified medical device.**

The system may produce incorrect sign recognition, speech transcription, medical interpretation, or BIM mapping.

It should not be used as the sole method of communication for emergency, diagnostic, treatment, medication, consent, or other safety-critical medical decisions.

---

## SignBridge AI

**Bridging communication between Malaysian Sign Language users and healthcare professionals through AI.**
