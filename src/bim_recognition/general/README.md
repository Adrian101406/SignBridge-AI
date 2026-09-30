# SignBridge General Recognizer V1

This package wraps the existing Zenodo general BIM recognizer using the same MCIE-facing output style as the medical recognizer.

## Expected runtime files

Place these in the `general_recognizer_v1/` folder:

- `general_inference.py`
- `example_general_usage.py`
- `best_zenodo_lstm.keras`
- `zenodo_xyz_scaling.npz`
- `general_label_mapping.json`
- `pose_landmarker.task`
- `hand_landmarker.task`

## Important

The general recognizer is separate from `medical_recognizer_v1`.

Do not compare raw softmax confidence from the general recognizer directly against raw softmax confidence from the medical recognizer to choose which model wins. They were trained on different datasets and probability scales are not necessarily calibrated against each other.

For now, route by session/domain:

```python
if session_mode == "medical":
    result = medical_recognizer.recognize_bim(video_path)

elif session_mode == "general":
    result = general_recognizer.recognize_bim(video_path)
```

Both recognizers should return the same core MCIE fields:

- `top1`
- `top_k`
- `confidence_margin`
- `quality`

## Colab

Install dependencies:

```python
!pip install -q mediapipe opencv-python-headless
```

Then run:

```python
%run /content/drive/MyDrive/general_recognizer_v1/example_general_usage.py
```

## Expected general model representation

The wrapper assumes the existing general model was trained on:

- 64 frames
- 33 pose landmarks
- 21 left-hand landmarks
- 21 right-hand landmarks
- `[x, y, z, validity]`
- shape `(64, 75, 4)`
- flattened model input `(64, 300)`

If your saved general model or scaler uses a different preprocessing convention, verify the wrapper against the known general-model test accuracy before handing it to MCIE.
