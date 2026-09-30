# SignBridge Medical Recognizer V2

## Purpose

`medical_recognizer_v2` is the MCIE-facing package for the current medical BIM recognizer.

The current V2 model is the **three-branch pose/hand + compact-face + geometry model trained with class-conditional domain alignment (`lambda_domain = 0.005`)**.

Current external development benchmark:

- Top-1: **13/30 = 43.33%**
- Top-3: **20/30 = 66.67%**

The package recognizes **one isolated medical BIM gloss per capture**. Continuous multi-sign segmentation is not part of this recognizer.

## Important V2 Design Decision

V2 uses a newer model but must preserve the exact proven V1 preprocessing pipeline.

Do **not** rewrite MediaPipe extraction, active-sign trimming, 64-frame interpolation, scaling, compact-face extraction, or geometry generation simply because the model file changed.

Copy:

```text
medical_recognizer_v1/recognition_inference.py
```

into this folder and rename it:

```text
recognition_core_v1.py
```

The new V2 `recognition_inference.py` is an MCIE-facing adapter around that proven core.

## Recommended Folder

The V2 recognizer should be stored inside the shared `signbridge_recognizers` directory, together with the other recognizer packages.

Recommended project structure:

```text
signbridge_recognizers/
│
├── medical_recognizer_v1/
│   └── ...
│
├── medical_recognizer_v2/
│   ├── recognition_inference.py
│   ├── recognition_core_v1.py
│   ├── best_domain_alignment_lambda0005.keras
│   ├── three_branch_feature_scaling.npz
│   ├── label_mapping.json
│   └── README.md
│
├── general_recognizer/
│   └── ...
│
└── number_recognizer/
    └── ...
```

The MediaPipe task files are also stored inside the shared `signbridge_recognizers` directory.

Recommended full structure:

```text
signbridge_recognizers/
│
├── medical_recognizer_v1/
│   └── ...
│
├── medical_recognizer_v2/
│   ├── recognition_inference.py
│   ├── recognition_core_v1.py
│   ├── best_domain_alignment_lambda0005.keras
│   ├── three_branch_feature_scaling.npz
│   ├── label_mapping.json
│   └── README.md
│
├── general_recognizer/
│   └── ...
│
├── number_recognizer/
│   └── ...
│
└── mediapipe/
    ├── pose_landmarker.task
    ├── hand_landmarker.task
    └── face_landmarker.task
```

All recognizers can reuse the same shared MediaPipe task files from `signbridge_recognizers/mediapipe/`.

## Is `label_mapping.json` the Same as V1?

**Yes.**

The V2 domain-alignment model uses the same 40 medical output classes in the same class-index order as V1. Domain alignment changed learned weights/representations, not the output-label order.

Therefore, reuse the V1 medical label mapping unchanged if its IDs are:

| ID | BIM gloss | English |
|---:|---|---|
| 0 | Badan | Body |
| 1 | Bahu | Shoulder |
| 2 | Jururawat | Nurse |
| 3 | Jari | Fingers |
| 4 | Kaki | Feet |
| 5 | Kulit | Skin |
| 6 | Leher | Neck |
| 7 | Doktor | Doctor |
| 8 | Bibir | Lips |
| 9 | Gigi | Teeth |
| 10 | Hidung | Nose |
| 11 | Lidah | Tongue |
| 12 | Mata | Eyes |
| 13 | Mulut | Mouth |
| 14 | Lemah | Weak |
| 15 | Tulang | Bones |
| 16 | Sinar-X | X-ray |
| 17 | Tulang Rusuk | Rib Cage |
| 18 | Darah | Blood |
| 19 | Jantung | Heart |
| 20 | Kesakitan | Pain |
| 21 | Buah Pinggang | Kidney |
| 22 | Asma | Asthma |
| 23 | Batuk | Cough |
| 24 | Bengkak | Swollen |
| 25 | Gatal-Gatal | Itchy |
| 26 | Muntah | Vomit |
| 27 | Selesema | Flu |
| 28 | Demam | Fever |
| 29 | Sakit Kepala | Headache |
| 30 | Sakit Perut | Stomach pain |
| 31 | Sakit Pinggang | Waist pain |
| 32 | Sakit Tekak | Sore throat |
| 33 | Bedah | Surgery |
| 34 | Kecemasan | Emergency |
| 35 | Krim | Cream |
| 36 | Pil | Pill |
| 37 | Mengandung | Pregnant |
| 38 | Penat | Tired |
| 39 | Perut | Stomach |

## Required Runtime Files

```text
recognition_inference.py
recognition_core_v1.py
best_domain_alignment_lambda0005.keras
three_branch_feature_scaling.npz
label_mapping.json
pose_landmarker.task
hand_landmarker.task
face_landmarker.task
```

Dependencies:

```text
tensorflow
mediapipe
opencv-python
numpy
```

## Model Inputs

```text
pose_hand_input: (None, 64, 300)
face_input:      (None, 64, 60)
geometry_input:  (None, 64, 240)
```

The processed sequence remains:

```text
64 frames × 90 landmarks × 4 values
```

where the 90 landmarks are:

```text
33 pose
21 left hand
21 right hand
15 compact face
```

## Public API

```python
from recognition_inference import SignBridgeMedicalRecognizer

recognizer = SignBridgeMedicalRecognizer(
    model_path="signbridge_recognizers/medical_recognizer_v2/best_domain_alignment_lambda0005.keras",
    scaler_path="signbridge_recognizers/medical_recognizer_v2/three_branch_feature_scaling.npz",
    label_mapping_path="signbridge_recognizers/medical_recognizer_v2/label_mapping.json",
    core_inference_path="signbridge_recognizers/medical_recognizer_v2/recognition_core_v1.py",
    pose_task_path="signbridge_recognizers/mediapipe/pose_landmarker.task",
    hand_task_path="signbridge_recognizers/mediapipe/hand_landmarker.task",
    face_task_path="signbridge_recognizers/mediapipe/face_landmarker.task",
)
```

Recognize one isolated sign:

```python
result = recognizer.recognize_bim(
    "sample_sign.mp4",
    top_k=3,
)
```

## MCIE-Compatible Output

```json
{
  "recognizer": "SignBridge_Medical_Recognizer_V2",
  "domain": "medical",
  "top1": {
    "class_id": 28,
    "gloss": "Demam",
    "english": "Fever",
    "confidence": 0.82
  },
  "top_k": [
    {
      "rank": 1,
      "class_id": 28,
      "gloss": "Demam",
      "english": "Fever",
      "confidence": 0.82
    }
  ],
  "confidence_margin": 0.72,
  "decision": "ACCEPT",
  "quality": {
    "hand_presence": 0.95,
    "face_presence": 0.99
  }
}
```

MCIE should consume **Top-K**, not only Top-1.

## Temporary Confidence Policy

```text
ACCEPT:
Top-1 confidence >= 0.80
AND Top1 - Top2 margin >= 0.25

CONFIRM:
Top-1 confidence >= 0.55
AND Top1 - Top2 margin >= 0.10

VERIFY:
otherwise
```

These values are temporary and should be recalibrated after enough real webcam validation data is collected.

## MCIE / Router Guidance

Do not directly compare raw softmax values from:

```text
medical recognizer
general recognizer
number recognizer
```

They were trained separately and are not calibrated against each other.

Recommended flow:

```text
Patient START
    ↓
one isolated sign
    ↓
Patient STOP
    ↓
shared landmark extraction
    ↓
Medical Top-3 / General Top-3 / Number Top-3
    ↓
Recognizer Router
    ↓
MCIE conversation context
    ↓
final candidate or confirmation
```

## Future Low-Latency Integration

For final prototype latency reduction, extract MediaPipe landmarks during live recording and keep them in RAM.

Then call:

```python
recognizer.recognize_sequence(
    sequence,
    quality=quality,
    top_k=3,
)
```

This avoids:

```text
record → encode MP4 → save → reopen → decode → MediaPipe
```

The sequence must remain fully compatible with the proven V1 preprocessing representation.

## What to Pass to the MCIE Team

Minimum package:

```text
recognition_inference.py
recognition_core_v1.py
best_domain_alignment_lambda0005.keras
three_branch_feature_scaling.npz
label_mapping.json
README.md
```

Plus the MediaPipe `.task` files if the integration machine handles raw video.

The MCIE developer should integrate against:

```python
result["domain"]
result["top1"]
result["top_k"]
result["confidence_margin"]
result["decision"]
result["quality"]
```

They should not need to access model branches, scalers, geometry calculations, or MediaPipe internals directly.
