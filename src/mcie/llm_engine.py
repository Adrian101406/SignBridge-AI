from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


class LocalTransformersLLM:
    """Local-only Qwen-style chat model. No network calls are permitted."""

    def __init__(self, model_dir: Path, use_4bit_if_available: bool = True):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Local MCIE LLM folder not found: {self.model_dir}. "
                "Download/copy the complete model before offline use."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )

        quant = None
        if use_4bit_if_available and torch.cuda.is_available() and BitsAndBytesConfig is not None:
            try:
                import bitsandbytes  # noqa: F401
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                quant = None

        kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
        }
        if quant is not None:
            kwargs["quantization_config"] = quant
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            **kwargs,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_payload: Dict[str, Any],
        max_new_tokens: int = 450,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Interpret the following SignBridge input.\n\n"
                + json.dumps(user_payload, ensure_ascii=False, indent=2),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output[0][prompt_len:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "parsed": extract_first_json_object(raw),
            "raw_output": raw,
            "latency_seconds": time.time() - started,
        }
