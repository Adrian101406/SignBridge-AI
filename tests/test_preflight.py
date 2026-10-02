from __future__ import annotations

from pathlib import Path

from signbridge_app.config import Settings
from signbridge_app.preflight import run_preflight


def test_settings_keep_models_inside_project(tmp_path: Path) -> None:
    settings = Settings.from_project_root(tmp_path)

    assert settings.whisper_model_dir == (
        tmp_path / "models" / "whisper" / "malaysian-whisper-small-v3"
    )
    assert settings.mcie_model_dir == (
        tmp_path / "models" / "mcie" / "Qwen3-4B-Instruct-2507"
    )


def test_preflight_reports_missing_model_files(tmp_path: Path) -> None:
    settings = Settings.from_project_root(tmp_path)

    report = run_preflight(settings)

    assert not report.ok
    assert {item.code for item in report.failures} >= {
        "whisper_model_missing",
        "mcie_model_missing",
    }


def test_preflight_can_skip_model_checks(tmp_path: Path) -> None:
    settings = Settings.from_project_root(tmp_path)
    for relative in settings.required_ui_files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    report = run_preflight(settings, check_models=False)

    assert report.ok
