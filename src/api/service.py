"""Inference service with input validation, guardrails, logging and metrics."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.model.inference import SLMGenerator
from src.model.loading import load_adapter_model, load_tokenizer
from src.model.prompting import CATEGORIES, apply_guardrails, format_target, is_unsafe_request, safe_refusal
from src.utils import ROOT, cuda_memory_mb, load_yaml, now_iso, write_json


class AnswerService:
    def __init__(self, api_config: str = "configs/api.yaml", model_config: str = "configs/model.yaml") -> None:
        self.api_cfg = load_yaml(ROOT / api_config)
        self.model_cfg = load_yaml(ROOT / model_config)
        self.adapter_path = ROOT / self.api_cfg.get("adapter_path", "models/cyberaware-qlora-v2")
        self.model_name = self.model_cfg["base_model"]
        self._generator: SLMGenerator | None = None
        self._load_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._log_path = ROOT / self.api_cfg.get("log_path", "experiments/logs/api_requests.jsonl")
        self._metrics_path = ROOT / self.api_cfg.get("metrics_path", "experiments/results/api_metrics.json")
        self._started = time.perf_counter()
        self._latencies: list[float] = []
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "invalid_responses": 0,
            "guardrail_requests": 0,
            "error_requests": 0,
        }

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None

    def _load(self) -> SLMGenerator:
        if self._generator is not None:
            return self._generator
        with self._load_lock:
            if self._generator is not None:
                return self._generator
            tokenizer = load_tokenizer(self.model_cfg)
            model = load_adapter_model(self.model_cfg, self.adapter_path, quantized=True)
            self._generator = SLMGenerator(
                tokenizer,
                model,
                max_new_tokens=int(self.api_cfg.get("max_new_tokens", 140)),
                repetition_penalty=1.05,
            )
            self._load_seconds = round(time.perf_counter() - self._started, 2)
            self._load_memory = cuda_memory_mb()
            return self._generator

    def answer(self, question: str) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        warnings: list[str] = []
        guardrail = False
        raw = ""
        try:
            if is_unsafe_request(question):
                parsed = safe_refusal(question)
                parsed["raw"] = format_target(parsed)
                guardrail = True
            else:
                generator = self._load()
                result = generator.generate_one(question)
                raw = result["raw_response"]
                parsed = apply_guardrails(question, result["parsed"], raw)
                guardrail = bool(parsed.get("guardrail_triggered"))
                if guardrail:
                    raw = parsed["raw"]
            valid = bool(parsed.get("valid"))
            if not valid:
                warnings.extend(parsed.get("errors", []))
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            answer_payload = (
                {
                    "category": parsed.get("category"),
                    "risk": parsed.get("risk"),
                    "advice": parsed.get("advice"),
                    "action": parsed.get("action"),
                }
                if valid
                else None
            )
            response = {
                "request_id": request_id,
                "answer": answer_payload,
                "raw_response": raw or parsed.get("raw") or None,
                "valid": valid,
                "guardrail_triggered": guardrail,
                "latency_ms": latency_ms,
                "model": self.model_name,
                "warnings": warnings,
            }
            self._record(request_id, question, response, None)
            return response
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            self._record(request_id, question, None, f"{type(exc).__name__}: {exc}", latency_ms)
            raise

    def _record(
        self,
        request_id: str,
        question: str,
        response: dict[str, Any] | None,
        error: str | None,
        latency_ms: float | None = None,
    ) -> None:
        with self._log_lock:
            self._metrics["total_requests"] += 1
            if error:
                self._metrics["error_requests"] += 1
            elif response and response.get("valid"):
                self._metrics["successful_requests"] += 1
            else:
                self._metrics["invalid_responses"] += 1
            if response and response.get("guardrail_triggered"):
                self._metrics["guardrail_requests"] += 1
            if latency_ms is None and response:
                latency_ms = response["latency_ms"]
            if latency_ms is not None:
                self._latencies.append(float(latency_ms))
                self._latencies = self._latencies[-1000:]

            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp_utc": now_iso(),
                "request_id": request_id,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "question_chars": len(question),
                "latency_ms": latency_ms,
                "valid": bool(response and response.get("valid")),
                "guardrail_triggered": bool(response and response.get("guardrail_triggered")),
                "category": (response or {}).get("answer", {}).get("category") if response and response.get("answer") else None,
                "error": error,
            }
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._write_metrics()

    def _write_metrics(self) -> None:
        snapshot = self.snapshot()
        write_json(self._metrics_path, snapshot)

    def snapshot(self) -> dict[str, Any]:
        latencies = sorted(self._latencies)
        if latencies:
            p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
        else:
            p95 = 0.0
        total = self._metrics["total_requests"]
        return {
            **self._metrics,
            "valid_rate": round(self._metrics["successful_requests"] / total, 4) if total else 0.0,
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p95": round(p95, 2),
                "max": round(max(latencies), 2) if latencies else 0.0,
            },
            "uptime_s": round(time.perf_counter() - self._started, 2),
        }

    def model_info(self) -> dict[str, Any]:
        qcfg = self.model_cfg.get("quantization", {})
        return {
            "base_model": self.model_name,
            "adapter": str(self.adapter_path.relative_to(ROOT)).replace("\\", "/"),
            "license": self.model_cfg.get("license", "apache-2.0"),
            "quantization": f"{qcfg.get('quant_type', 'nf4')} 4-bit"
            if qcfg.get("load_in_4bit", True)
            else "none",
            "categories": list(CATEGORIES),
            "max_input_chars": int(self.api_cfg.get("max_input_chars", 500)),
        }
