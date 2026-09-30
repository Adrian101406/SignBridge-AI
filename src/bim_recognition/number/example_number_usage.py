"""Example usage for SignBridge Number Recognizer V1.

Run from the number_recognizer_v1 folder:
    python example_number_usage.py --video path/to/number_sign.mp4
"""

import argparse
import json
from pathlib import Path

from number_inference import SignBridgeNumberRecognizer


NUMBER_DIR = Path(__file__).resolve().parent
SHARED_MEDIAPIPE_DIR = (
    NUMBER_DIR.parent / "signbridge_models" / "mediapipe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the SignBridge BIM number recognizer."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Short video containing one static BIM number sign.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidates to return (default: 3).",
    )
    parser.add_argument(
        "--output",
        default=str(NUMBER_DIR / "example_mcie_number_output.json"),
        help="Destination JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = NUMBER_DIR / "models" / "number_model.tflite"
    labels_path = NUMBER_DIR / "models" / "labels.txt"
    hand_task_path = SHARED_MEDIAPIPE_DIR / "hand_landmarker.task"

    required = {
        "number_inference.py": NUMBER_DIR / "number_inference.py",
        "number model": model_path,
        "labels": labels_path,
        "shared hand landmarker": hand_task_path,
        "test video": Path(args.video),
    }

    print("\nChecking Number Recognizer V1 files...")
    missing = []
    for name, path in required.items():
        exists = path.is_file()
        print(f"{name:26s}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    print("\nLoading SignBridge Number Recognizer V1...")

    with SignBridgeNumberRecognizer(
        model_path=str(model_path),
        labels_path=str(labels_path),
        hand_task_path=str(hand_task_path),
    ) as recognizer:
        result = recognizer.recognize_bim(
            video_path=args.video,
            top_k=args.top_k,
        )

    print("\n" + "=" * 60)
    print("NUMBER RECOGNIZER OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print("\nShort summary")
    print("Top-1:", result["top1"]["gloss"])
    print(
        "Top-1 confidence:",
        f"{result['top1']['confidence'] * 100:.2f}%",
    )
    print(
        "Confidence margin:",
        f"{result['confidence_margin']:.4f}",
    )
    print("Top candidates:")
    for item in result["top_k"]:
        print(
            f"#{item['rank']} {item['gloss']} "
            f"- {item['confidence'] * 100:.2f}%"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSaved MCIE-compatible output:", output_path)


if __name__ == "__main__":
    main()

