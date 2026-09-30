"""
SignBridge General Recognizer V1
Updated example usage for a fresh Google Colab session.

Folder layout used by this example:

/content/drive/MyDrive/
├── general_recognizer_v1/
│   ├── general_inference.py
│   ├── best_zenodo_lstm.keras
│   ├── zenodo_xyz_scaling.npz
│   └── general_label_mapping.json
│
└── signbridge_models/
    └── mediapipe/
        ├── pose_landmarker.task
        ├── hand_landmarker.task
        └── face_landmarker.task   # not used by the general recognizer

Change GENERAL_DIR or MEDIAPIPE_DIR below if your folders are elsewhere.
"""

import os
import sys
import json

from google.colab import drive, files

# ============================================================
# 1. MOUNT DRIVE
# ============================================================

drive.mount("/content/drive")

# ============================================================
# 2. FOLDERS
# ============================================================

GENERAL_DIR = (
    "/content/drive/MyDrive/"
    "general_recognizer_v1"
)

MEDIAPIPE_DIR = (
    "/content/drive/MyDrive/"
    "signbridge_models/mediapipe"
)

if not os.path.isdir(GENERAL_DIR):
    raise FileNotFoundError(
        f"General recognizer folder not found:\n{GENERAL_DIR}"
    )

if not os.path.isdir(MEDIAPIPE_DIR):
    raise FileNotFoundError(
        f"MediaPipe folder not found:\n{MEDIAPIPE_DIR}"
    )

if GENERAL_DIR not in sys.path:
    sys.path.insert(0, GENERAL_DIR)

# ============================================================
# 3. IMPORT WRAPPER
# ============================================================

from general_inference import SignBridgeGeneralRecognizer

# ============================================================
# 4. FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    GENERAL_DIR,
    "best_zenodo_lstm.keras"
)

SCALER_PATH = os.path.join(
    GENERAL_DIR,
    "zenodo_xyz_scaling.npz"
)

LABEL_MAPPING_PATH = os.path.join(
    GENERAL_DIR,
    "general_label_mapping.json"
)

POSE_TASK_PATH = os.path.join(
    MEDIAPIPE_DIR,
    "pose_landmarker.task"
)

HAND_TASK_PATH = os.path.join(
    MEDIAPIPE_DIR,
    "hand_landmarker.task"
)

# face_landmarker.task is intentionally not used here.
# The original general recognizer uses pose + hands only.

# ============================================================
# 5. VERIFY FILES
# ============================================================

required = {
    "general_inference.py":
        os.path.join(GENERAL_DIR, "general_inference.py"),

    "model":
        MODEL_PATH,

    "scaler":
        SCALER_PATH,

    "label mapping":
        LABEL_MAPPING_PATH,

    "pose landmarker":
        POSE_TASK_PATH,

    "hand landmarker":
        HAND_TASK_PATH,
}

print("\nChecking general recognizer files...")

missing = []

for name, path in required.items():
    exists = os.path.exists(path)
    print(
        f"{name:24s}: "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:
        missing.append(path)

if missing:
    raise FileNotFoundError(
        "\nMissing required files:\n"
        + "\n".join(missing)
        + "\n\n"
        "If general_label_mapping.json is the only missing file, "
        "recover/create the exact original 117-class label mapping "
        "before running inference."
    )

# ============================================================
# 6. LOAD RECOGNIZER
# ============================================================

recognizer = SignBridgeGeneralRecognizer(
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    label_mapping_path=LABEL_MAPPING_PATH,
    pose_task_path=POSE_TASK_PATH,
    hand_task_path=HAND_TASK_PATH,
)

print("\nGeneral recognizer loaded successfully.")
print("Number of classes:", len(recognizer.class_map))

# ============================================================
# 7. UPLOAD TEST VIDEO
# ============================================================

print("\nUpload one general BIM video.")

uploaded = files.upload()

video_extensions = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
)

video_names = [
    name
    for name in uploaded.keys()
    if name.lower().endswith(video_extensions)
]

if not video_names:
    raise ValueError("No supported video uploaded.")

VIDEO_PATH = os.path.join(
    "/content",
    video_names[0]
)

# ============================================================
# 8. RUN RECOGNITION
# ============================================================

result = recognizer.recognize_bim(
    VIDEO_PATH,
    top_k=3
)

# ============================================================
# 9. PRINT OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("GENERAL RECOGNIZER OUTPUT")
print("=" * 60)

print(
    json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    )
)

print("\nTop-3 candidates:")

for item in result["top_k"]:
    print(
        f"#{item['rank']} "
        f"{item['gloss']} "
        f"- {item['confidence'] * 100:.2f}%"
    )
