"""Offline runtime safeguards for SignBridge MCIE.

Import this module before loading Hugging Face/Transformers models in the final
prototype. The actual LLM loader also uses ``local_files_only=True``.
"""
from __future__ import annotations

import os


def enable_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
