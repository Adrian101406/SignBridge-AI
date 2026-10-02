from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSON2_ROOT = PROJECT_ROOT / "uploads" / "person2"
if str(PERSON2_ROOT) not in sys.path:
    sys.path.insert(0, str(PERSON2_ROOT))

from signbridge.mcie_ai import AIEnhancedMCIE  # noqa: E402


class FakeLlm:
    def generate_json(self, **kwargs):
        return {
            "parsed": {
                "intent": "SYMPTOM_REPORT",
                "medical_concepts": ["TONGUE", "HEADACHE"],
                "entities": {
                    "symptoms": ["HEADACHE"],
                    "body_parts": ["TONGUE"],
                    "medications": [],
                    "quantity": None,
                    "frequency": None,
                    "timing": None,
                    "duration": {"quantity": None, "unit": None},
                },
                "simplified_text_en": "I have a headache and tongue pain.",
                "simplified_text_ms": None,
                "context_used": False,
                "evidence": [],
                "uncertainty": [],
                "requires_confirmation": True,
            },
            "raw_output": "{}",
            "latency_seconds": 0.1,
        }


def test_patient_bim_keeps_anatomy_concepts_that_came_from_recognised_signs():
    result = AIEnhancedMCIE(FakeLlm()).patient_bim(
        gloss_sequence=["Lidah", "Sakit Kepala"],
        concept_sequence=["TONGUE", "HEADACHE"],
    )

    assert result["interpretation"]["medical_concepts"] == ["TONGUE", "HEADACHE"]
    assert result["output"]["simplified_text_en"] == "I have a headache and tongue pain."
    assert result["safety"]["reasons"] == []
