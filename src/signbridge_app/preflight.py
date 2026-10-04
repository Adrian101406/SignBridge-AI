from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from .config import DEFAULT_SETTINGS, Settings


@dataclass(frozen=True)
class PreflightItem:
    code: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    items: tuple[PreflightItem, ...]

    @property
    def failures(self) -> tuple[PreflightItem, ...]:
        return tuple(item for item in self.items if not item.ok)

    @property
    def ok(self) -> bool:
        return not self.failures


def _path_item(code: str, path: Path, label: str) -> PreflightItem:
    return PreflightItem(code=code, ok=path.exists(), detail=f"{label}: {path}")


def _model_item(code: str, directory: Path, label: str) -> PreflightItem:
    config = directory / "config.json"
    weights = tuple(directory.glob("*.safetensors")) if directory.is_dir() else ()
    ok = config.is_file() and bool(weights)
    return PreflightItem(code=code, ok=ok, detail=f"{label}: {directory}")


def run_preflight(
    settings: Settings = DEFAULT_SETTINGS,
    *,
    check_models: bool = True,
) -> PreflightReport:
    items = [
        _path_item(
            f"ui_file_{index}",
            settings.project_root / relative,
            "Required UI file",
        )
        for index, relative in enumerate(settings.required_ui_files, start=1)
    ]
    if check_models:
        items.extend(
            (
                _model_item(
                    "whisper_model_missing",
                    settings.whisper_model_dir,
                    "Malaysian Whisper model",
                ),
                _model_item(
                    "mcie_model_missing",
                    settings.mcie_model_dir,
                    "Qwen MCIE model",
                ),
                _path_item(
                    "mcie_gguf_missing",
                    settings.mcie_gguf_path,
                    "Memory-efficient Qwen MCIE GGUF",
                ),
                PreflightItem(
                    code="silero_vad_missing",
                    ok=find_spec("silero_vad") is not None,
                    detail="Package-bundled Silero VAD",
                ),
            )
        )
    return PreflightReport(items=tuple(items))


def main() -> int:
    report = run_preflight()
    for item in report.items:
        status = "OK" if item.ok else "MISSING"
        print(f"[{status}] {item.code}: {item.detail}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
