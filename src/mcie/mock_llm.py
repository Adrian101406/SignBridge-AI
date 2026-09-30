"""Deterministic mock used only for simulations and unit tests.

This is not the production AI model. Production uses LocalTransformersLLM with
Qwen3-4B-Instruct-2507 loaded from a local directory.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


class DeterministicMockLLM:
    def __init__(self, response: Optional[Dict[str, Any]] = None):
        self.response = response or {
            "intent": "UNKNOWN",
            "medical_concepts": [],
            "entities": {
                "symptoms": [], "body_parts": [], "medications": [],
                "quantity": None, "frequency": None, "timing": None,
                "duration": {"quantity": None, "unit": None},
            },
            "simplified_text_en": None,
            "simplified_text_ms": None,
            "context_used": False,
            "evidence": [],
            "uncertainty": [],
            "requires_confirmation": False,
        }

    def set_response(self, response: Dict[str, Any]) -> None:
        self.response = response

    def generate_json(self, *, system_prompt: str, user_payload: Dict[str, Any], max_new_tokens: int = 450):
        raw = json.dumps(self.response, ensure_ascii=False)
        return {"parsed": self.response, "raw_output": raw, "latency_seconds": 0.0}
