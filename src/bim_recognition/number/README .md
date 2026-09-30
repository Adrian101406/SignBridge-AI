# SignBridge Number Recognizer V1

## Purpose

This package recognizes isolated, static Bahasa Isyarat Malaysia number signs from `0` through `10`. It provides the same MCIE-facing Top-K result structure as the SignBridge medical and general recognizers.

It is intended to be activated when SignBridge expects a numeric answer, such as age, quantity, pain scale, or dosage. It should not continuously compete with the medical and general recognizers using raw confidence alone.

## Recommended Folder Structure

```text
SignBridge/
├── medical_recognizer_v1/
├── general_recognizer_v1/
├── number_recognizer_v1/
│   ├── number_inference.py
│   ├── example_number_usage.py
│   ├── README_SignBridge_Number_Recognizer_V1.md
│   ├── models/
│   │   ├── number_model.tflite
│   │   └── labels.txt
│   └── src/
│       ├── __init__.py
│       └── landmarks.py
└── signbridge_models/
    └── mediapipe/
        ├── pose_landmarker.task
        ├── hand_landmarker.task
        └── face_landmarker.task
```

Copy the exact `src/landmarks.py` used to extract the number model's training landmarks. Its preprocessing must not be replaced or modified independently of the model.

The number recognizer uses only `hand_landmarker.task`. The shared pose and face task files remain available for the medical/general recognizers.

## Required Runtime Files

```text
number_inference.py
models/number_model.tflite
models/labels.txt
src/landmarks.py
src/__init__.py
../signbridge_models/mediapipe/hand_landmarker.task
```

The expected label order is:

```text
0
1
2
3
4
5
6
7
8
9
10
```

## Dependencies

Use the environment already used by SignBridge, containing at minimum:

```text
tensorflow
mediapipe
opencv-python
numpy
```

## Public API

### Constructor

```python
from number_inference import SignBridgeNumberRecognizer

recognizer = SignBridgeNumberRecognizer(
    model_path="models/number_model.tflite",
    labels_path="models/labels.txt",
    hand_task_path="../signbridge_models/mediapipe/hand_landmarker.task",
)
```

The constructor validates:

- all runtime files exist;
- labels are exactly `0` to `10` in numeric order;
- the TFLite input ends in 63 features;
- the TFLite output contains 11 classes.

### Video-level recognition

This method matches the medical/general recognizer API:

```python
result = recognizer.recognize_bim(
    "number_5.mp4",
    top_k=3,
)
```

The signer should hold one static number sign across a short video. Probabilities are averaged across hand-detected frames.

### Frame-level recognition

```python
result = recognizer.predict_frame(
    bgr_frame,
    top_k=3,
)
```

The input is one OpenCV BGR NumPy frame. The method returns `None` when no hand is detected. Camera capture and UI remain outside this reusable module.

### Cleanup

```python
recognizer.close()
```

Or use a context manager:

```python
with SignBridgeNumberRecognizer(...) as recognizer:
    result = recognizer.recognize_bim("number_5.mp4")
```

## MCIE-Compatible Output

Example:

```json
{
  "recognizer": "SignBridge_Number_Recognizer_V1",
  "domain": "number",
  "top1": {
    "class_id": 5,
    "gloss": "5",
    "english": "5",
    "confidence": 0.93
  },
  "top_k": [
    {
      "rank": 1,
      "class_id": 5,
      "gloss": "5",
      "english": "5",
      "confidence": 0.93
    }
  ],
  "confidence_margin": 0.81,
  "quality": {
    "hand_presence": 0.96,
    "raw_frames": 60,
    "trimmed_frames": 58
  }
}
```

`quality.trimmed_frames` represents hand-detected frames for this static recognizer. The name is retained for compatibility with existing consumers.

## MCIE Compatibility and Crash Prevention

The number output uses the same required structure as the established recognizers:

| Field | Medical | General | Number |
|---|---:|---:|---:|
| `recognizer` | Yes | Yes | Yes |
| `top1.class_id` | Yes | Yes | Yes |
| `top1.gloss` | Yes | Yes | Yes |
| `top1.english` | Yes | Yes | Yes |
| `top1.confidence` | Yes | Yes | Yes |
| `top_k[]` | Yes | Yes | Yes |
| `confidence_margin` | Yes | Yes | Yes |
| `quality.hand_presence` | Yes | Yes | Yes |
| `quality.raw_frames` | Yes | Yes | Yes |
| `quality.trimmed_frames` | Yes | Yes | Yes |

Therefore, MCIE code that reads the shared fields will not require a number-specific parser. The `domain` field is additive metadata and should be read with `result.get("domain")` rather than treated as mandatory for older medical outputs.

Recommended safe MCIE access:

```python
def recognizer_result_to_mcie(result: dict) -> dict:
    top1 = result.get("top1") or {}
    quality = result.get("quality") or {}

    return {
        "speaker": "patient",
        "input_type": "bim",
        "recognizer": result.get("recognizer", "unknown"),
        "domain": result.get("domain", "unknown"),
        "raw_text": top1.get("gloss"),
        "recognized_text": top1.get("gloss"),
        "confidence": float(top1.get("confidence", 0.0)),
        "confidence_margin": float(result.get("confidence_margin", 0.0)),
        "hand_presence": float(quality.get("hand_presence", 0.0)),
        "top_k": result.get("top_k", []),
        "status": "awaiting_confirmation",
    }
```

Do not access optional fields using direct indexing. For example, older medical output may not contain `domain`, while medical quality contains `face_presence` and the other recognizers do not. Those differences are safe when optional fields use `.get(...)`.

## Routing Requirement

The number recognizer is a closed 11-class classifier. Any detected hand will be forced toward one of `0` to `10`, even when the sign is not numeric. For that reason:

- enable it when MCIE/session context expects a number;
- do not select it merely because its confidence is higher than another model;
- retain human confirmation for medically meaningful quantities;
- treat low hand presence or low confidence as insufficient evidence.

## Run the Example

From `number_recognizer_v1`:

```powershell
python example_number_usage.py --video "path\to\number_5.mp4"
```

The example prints the complete result and saves:

```text
example_mcie_number_output.json
```

## Limitations

- Static signs only; not continuous digit sequences.
- One hand per frame.
- Classes are restricted to `0` through `10`.
- Training data appears to be dominated by one signer.
- Held-out image accuracy is not unseen-signer accuracy.
- Confidence alone cannot reliably reject arbitrary non-number signs.

