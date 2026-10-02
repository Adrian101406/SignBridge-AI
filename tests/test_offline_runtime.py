from pathlib import Path

from signbridge_app.config import Settings
from signbridge_app.preflight import run_preflight
from signbridge_app.runtime import Runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_composition_is_lazy_and_does_not_load_models():
    runtime = Runtime.create(Settings.from_project_root(PROJECT_ROOT))
    assert runtime.recognition._recognizers == {}
    assert runtime.asr._engine is None
    assert runtime.mcie._engine is None


def test_local_assets_pass_preflight():
    report = run_preflight(Settings.from_project_root(PROJECT_ROOT))
    assert report.ok, [item.detail for item in report.failures]


def test_run_script_forces_offline_model_mode():
    script = (PROJECT_ROOT / "scripts" / "run.ps1").read_text(encoding="utf-8")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HOME"):
        assert name in script
    assert "main.py --check" in script
