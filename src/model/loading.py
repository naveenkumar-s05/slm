"""Load the base SLM and optional PEFT adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_quantization_config(model_cfg: dict[str, Any]) -> BitsAndBytesConfig:
    qcfg = model_cfg.get("quantization", {})
    compute_dtype = getattr(torch, qcfg.get("compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=qcfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(qcfg.get("use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_cfg: dict[str, Any]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(
    model_cfg: dict[str, Any],
    *,
    quantized: bool = True,
    dtype: str | None = None,
):
    kwargs: dict[str, Any] = {
        "revision": model_cfg.get("revision", "main"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "device_map": model_cfg.get("device_map", "auto"),
    }
    if quantized:
        kwargs["quantization_config"] = build_quantization_config(model_cfg)
    else:
        kwargs["torch_dtype"] = getattr(torch, dtype or model_cfg.get("dtype", "bfloat16"))

    model = AutoModelForCausalLM.from_pretrained(model_cfg["base_model"], **kwargs)
    model.eval()
    return model


def load_adapter_model(
    model_cfg: dict[str, Any],
    adapter_path: str | Path,
    *,
    quantized: bool = True,
):
    model = load_base_model(model_cfg, quantized=quantized)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model
