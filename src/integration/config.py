from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    whisper_model_dir: Path
    mcie_model_dir: Path
    mcie_gguf_path: Path
    person1_dir: Path
    person2_dir: Path
    required_ui_files: tuple[Path, ...]

    @classmethod
    def from_project_root(cls, root: Path) -> "Settings":
        project_root = Path(root).resolve()
        return cls(
            project_root=project_root,
            whisper_model_dir=(
                project_root
                / "models"
                / "whisper"
                / "malaysian-whisper-small-v3"
            ),
            mcie_model_dir=(
                project_root
                / "models"
                / "mcie"
                / "Qwen3-4B-Instruct-2507"
            ),
            mcie_gguf_path=(
                project_root
                / "models"
                / "mcie"
                / "Qwen3-4B-Instruct-2507-GGUF"
                / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
            ),
            person1_dir=project_root / "uploads" / "person1",
            person2_dir=project_root / "uploads" / "person2",
            required_ui_files=(
                Path("ui/doctor.html"),
                Path("ui/patient.html"),
                Path("ui/css/styles.css"),
                Path("ui/js/doctor-ui.js"),
                Path("ui/js/patient-ui.js"),
            ),
        )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = Settings.from_project_root(PROJECT_ROOT)
