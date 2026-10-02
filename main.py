from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from signbridge_app.api import create_app
from signbridge_app.preflight import run_preflight
from signbridge_app.runtime import Runtime


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_UI_FILES = (
    Path("ui/doctor.html"),
    Path("ui/patient.html"),
    Path("ui/css/styles.css"),
    Path("ui/js/doctor-ui.js"),
    Path("ui/js/patient-ui.js"),
)


def parse_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def check_ui_files() -> list[Path]:
    return [relative for relative in REQUIRED_UI_FILES if not (PROJECT_ROOT / relative).is_file()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local SignBridge AI UI")
    parser.add_argument("--host", default="127.0.0.1", help="local interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=parse_port, default=5500, help="local port (default: 5500)")
    parser.add_argument("--check", action="store_true", help="verify required UI files without starting a server")
    parser.add_argument("--skip-models", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    missing = check_ui_files()
    if missing:
        print("UI check failed. Missing required files:")
        for path in missing:
            print(f"- {path.as_posix()}")
        return 1

    report = run_preflight(check_models=not args.skip_models)
    if not report.ok:
        for item in report.failures:
            print(f"Preflight failed: {item.code}: {item.detail}")
        return 1
    print("UI check passed: ui/doctor.html and ui/patient.html are present.")
    if args.check:
        return 0

    print(f"Doctor UI: http://{args.host}:{args.port}/ui/doctor.html")
    print(f"Patient UI: http://{args.host}:{args.port}/ui/patient.html")
    print("Press Ctrl+C to stop the local server.")
    uvicorn.run(create_app(Runtime.create(), project_root=PROJECT_ROOT), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
