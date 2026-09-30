# Local Model Assets

The `models/` directory is reserved for runtime model assets.

Large language and speech models should normally **not** be committed directly to GitHub. Store them locally and document how they are obtained.

Recommended structure:

```text
models/
├── medical/
│   ├── best_domain_alignment_lambda0005.keras
│   ├── three_branch_feature_scaling.npz
│   └── label_mapping.json
├── general/
│   ├── best_zenodo_lstm.keras
│   ├── zenodo_xyz_scaling.npz
│   └── general_label_mapping.json
├── number/
│   ├── number_model.tflite
│   └── labels.txt
├── mediapipe/
│   ├── pose_landmarker.task
│   ├── hand_landmarker.task
│   └── face_landmarker.task
├── whisper/
│   └── <local Malaysian Whisper model files>
└── qwen/
    └── <local Qwen model files>
```

## Current Model Identifiers

### Speech Recognition

```text
mesolitica/malaysian-whisper-small-v3
```

For offline use, download the model during setup and point the runtime to the local model directory.

### MCIE Local LLM

```text
Qwen/Qwen3-4B-Instruct-2507
```

The development design supports local loading. A 4-bit configuration may be used where compatible hardware and `bitsandbytes` are available.

## GitHub Guidance

Before committing any model file:

1. confirm that your team has permission to redistribute it;
2. check the file size;
3. do not upload confidential training data or participant recordings;
4. use Git LFS / a model host / release asset for files that are too large;
5. keep large Qwen and Whisper model directories out of normal Git history.

Small recognizer assets may be committed if licensing allows and they are within GitHub limits.

## Offline Setup

After model assets are stored locally, the runtime can be configured for offline Transformers loading:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

The source code should use local paths and `local_files_only=True` for offline model loading where applicable.
