from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_unified_mcie_output(
    *,
    source: str,
    intent: Optional[str] = None,
    entities: Optional[Dict[str, Any]] = None,
    medical_concepts: Optional[List[str]] = None,
    input_data: Any = None,
    context_used: bool = False,
    simplified_text_en: Optional[str] = None,
    simplified_text_ms: Optional[str] = None,
    decision: str = "ACCEPT",
    requires_confirmation: bool = False,
    requires_retry: bool = False,
    decision_reasons: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
) -> Dict[str, Any]:
    """One stable contract for doctor-speech and patient-BIM MCIE outputs."""
    entities = entities or {}
    medical_concepts = medical_concepts or []
    decision_reasons = decision_reasons or []
    metadata = metadata or {}

    ready = (
        success
        and not requires_confirmation
        and not requires_retry
        and (simplified_text_en is not None or simplified_text_ms is not None)
    )

    # ready_for_translation is retained because the existing Text-to-BIM
    # teammate handoff expects that field.
    return {
        "success": success,
        "source": source,
        "input": input_data,
        "interpretation": {
            "intent": intent,
            "medical_concepts": medical_concepts,
            "entities": entities,
            "context_used": context_used,
        },
        "output": {
            "simplified_text_en": simplified_text_en,
            "simplified_text_ms": simplified_text_ms,
        },
        "safety": {
            "decision": decision,
            "requires_confirmation": requires_confirmation,
            "requires_retry": requires_retry,
            "reasons": decision_reasons,
        },
        "metadata": metadata,
        "ready_for_output": ready,
        "ready_for_translation": ready,
    }
