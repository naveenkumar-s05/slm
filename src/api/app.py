"""FastAPI application for the CyberAware assistant."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api.schemas import (
    AnswerResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    QuestionRequest,
)
from src.api.service import AnswerService


class FakeService:
    """Test double used by API tests; never loaded in production."""

    def __init__(self) -> None:
        self.is_loaded = True

    def answer(self, question: str) -> dict[str, Any]:
        if "unsafe" in question.lower():
            return {
                "request_id": "test",
                "answer": {
                    "category": "unsafe-request",
                    "risk": "high",
                    "advice": "I cannot help with harmful activity. I can explain defensive practices instead.",
                    "action": "Ask for defensive guidance.",
                },
                "raw_response": "",
                "valid": True,
                "guardrail_triggered": True,
                "latency_ms": 1.0,
                "model": "fake",
                "warnings": [],
            }
        return {
            "request_id": "test",
            "answer": {
                "category": "phishing",
                "risk": "medium",
                "advice": "Check the sender and avoid clicking unexpected links.",
                "action": "Report the message to your security team.",
            },
            "raw_response": "",
            "valid": True,
            "guardrail_triggered": False,
            "latency_ms": 1.0,
            "model": "fake",
            "warnings": [],
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_requests": 1,
            "successful_requests": 1,
            "invalid_responses": 0,
            "guardrail_requests": 0,
            "error_requests": 0,
            "valid_rate": 1.0,
            "latency_ms": {"mean": 1.0, "p95": 1.0, "max": 1.0},
            "uptime_s": 1.0,
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "base_model": "fake",
            "adapter": "fake",
            "license": "apache-2.0",
            "quantization": "nf4 4-bit",
            "categories": ["phishing"],
            "max_input_chars": 500,
        }


def get_service(request: Request) -> Any:
    return request.app.state.service


def create_app(service: Any | None = None) -> FastAPI:
    application = FastAPI(
        title="CyberAware SLM API",
        version="1.0.0",
        description="Defensive cybersecurity awareness assistant powered by a fine-tuned Qwen2.5-1.5B SLM.",
    )
    application.state.service = service or AnswerService()

    @application.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!doctype html><html><head><meta charset="utf-8"><title>CyberAware</title>
<style>body{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px}textarea{width:100%;height:90px}
button{padding:10px 18px;margin-top:8px}pre{background:#f4f4f4;padding:16px;white-space:pre-wrap}</style></head>
<body><h1>CyberAware</h1><p>Cybersecurity awareness assistant (local SLM).</p>
<textarea id="q">How can I tell if an email is phishing?</textarea><br>
<button onclick="ask()">Ask</button><pre id="out"></pre>
<script>async function ask(){const q=document.getElementById('q').value;document.getElementById('out').textContent='Thinking...';
const r=await fetch('/v1/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);}</script></body></html>"""

    @application.get("/health", response_model=HealthResponse)
    def health(svc: Any = Depends(get_service)) -> dict[str, Any]:
        return {"status": "ok", "model_loaded": bool(getattr(svc, "is_loaded", False)), "model": "cyberaware-qwen2.5-1.5b"}

    @application.get("/v1/model", response_model=ModelInfoResponse)
    def model_info(svc: Any = Depends(get_service)) -> dict[str, Any]:
        return svc.model_info()

    @application.get("/metrics", response_model=MetricsResponse)
    def metrics(svc: Any = Depends(get_service)) -> dict[str, Any]:
        return svc.snapshot()

    @application.post("/v1/answer", response_model=AnswerResponse)
    def answer(payload: QuestionRequest, svc: Any = Depends(get_service)) -> dict[str, Any]:
        try:
            return svc.answer(payload.question)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"inference timed out: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - API boundary
            raise HTTPException(status_code=500, detail=f"inference failed: {type(exc).__name__}") from exc

    return application


app = create_app()
