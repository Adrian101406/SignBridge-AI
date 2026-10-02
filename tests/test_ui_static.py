from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = PROJECT_ROOT / "ui"

PATIENT_IDS = {
    "camera-preview",
    "camera-toggle",
    "video-upload-button",
    "video-upload",
    "recognition-start",
    "recognition-stop",
    "recognition-number-mode",
    "recognition-mode",
    "recognition-decision-actions",
    "confirm-recognition",
    "retry-recognition",
    "current-detection",
    "recognition-confidence",
    "recognition-status",
    "patient-draft",
    "patient-original-bim",
    "undo-word",
    "clear-draft",
    "send-patient-message",
    "conversation-log",
    "avatar-placeholder",
    "avatar-video",
    "avatar-status",
    "play-avatar-video",
    "play-doctor-reply",
    "touch-keyboard",
}

DOCTOR_IDS = {
    "conversation-log",
    "save-conversation-history",
    "clear-conversation",
    "record-start",
    "record-stop",
    "recording-status",
    "original-transcript",
    "doctor-draft",
    "medical-review",
    "cancel-doctor-draft",
    "confirm-doctor-message",
    "audio-visualizer",
}


class UiStaticContractTests(unittest.TestCase):
    def test_fever_duration_avatar_asset_is_packaged_locally(self):
        self.assertTrue((UI_ROOT / "assets" / "avatar" / "fever-duration.mp4").is_file())

    def _read_page(self, filename: str) -> str:
        path = UI_ROOT / filename
        self.assertTrue(path.is_file(), f"missing UI page: {path}")
        return path.read_text(encoding="utf-8")

    def test_patient_page_exposes_required_controls(self) -> None:
        html = self._read_page("patient.html")
        ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', html))
        self.assertEqual(set(), PATIENT_IDS - ids)

    def test_doctor_page_exposes_required_controls(self) -> None:
        html = self._read_page("doctor.html")
        ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', html))
        self.assertEqual(set(), DOCTOR_IDS - ids)

    def test_doctor_layout_follows_review_order_without_duplicate_patient_panel(self) -> None:
        html = self._read_page("doctor.html")
        self.assertNotIn("Latest confirmed message", html)
        self.assertNotIn('id="patient-message"', html)
        self.assertLess(html.index('id="conversation-log"'), html.index('id="audio-visualizer"'))
        self.assertLess(html.index('id="original-transcript"'), html.index('id="medical-review"'))
        self.assertLess(html.index('id="medical-review"'), html.index('id="doctor-draft"'))

    def test_pages_use_only_local_assets_and_define_viewport(self) -> None:
        for filename in ("doctor.html", "patient.html"):
            with self.subTest(filename=filename):
                html = self._read_page(filename)
                self.assertEqual(1, len(re.findall(r'name=["\']viewport["\']', html)))
                self.assertNotRegex(html, r'(?:src|href)=["\']https?://')
                self.assertNotIn("localStorage", html)
                self.assertNotIn("sessionStorage", html)

                asset_paths = re.findall(r'(?:src|href)=["\']([^"\']+)["\']', html)
                self.assertTrue(asset_paths, "each page must load local CSS or JavaScript")
                for asset_path in asset_paths:
                    self.assertFalse(asset_path.startswith(("//", "/")), asset_path)

    def test_shared_stylesheet_exists(self) -> None:
        self.assertTrue((UI_ROOT / "css" / "styles.css").is_file())

    def test_runtime_javascript_stays_offline_and_ephemeral(self) -> None:
        forbidden_patterns = {
            "persistent local storage": r"\blocalStorage\b",
            "persistent session storage": r"\bsessionStorage\b",
            "XML HTTP request": r"\bXMLHttpRequest\b",
            "remote URL": r"https?://",
        }
        runtime_files = sorted((UI_ROOT / "js").rglob("*.js"))
        self.assertTrue(runtime_files, "runtime JavaScript modules are missing")
        for path in runtime_files:
            source = path.read_text(encoding="utf-8")
            for description, pattern in forbidden_patterns.items():
                with self.subTest(path=path.name, rule=description):
                    self.assertNotRegex(source, pattern)

        backend_source = (UI_ROOT / "js" / "adapters" / "backend-client.js").read_text(encoding="utf-8")
        self.assertIn("/api/", backend_source)
        self.assertIn("/ws/session", backend_source)

    def test_readme_lists_vscode_environment_and_both_pages(self) -> None:
        readme_path = PROJECT_ROOT / "README.md"
        self.assertTrue(readme_path.is_file(), f"missing setup guide: {readme_path}")
        readme = readme_path.read_text(encoding="utf-8")
        required_text = (
            "Python 3.11 or 3.12",
            "Microsoft Edge",
            "Google Chrome",
            "Live Server",
            "http://127.0.0.1:5500/ui/doctor.html",
            "http://127.0.0.1:5500/ui/patient.html",
            "camera permission",
            "Node.js is not required",
            "React is not required",
            "Unity is not required",
        )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, readme)

    def test_main_check_validates_ui_without_starting_server(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), "--check"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("doctor.html", result.stdout)
        self.assertIn("patient.html", result.stdout)
        self.assertIn("UI check passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
