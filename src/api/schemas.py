"""Pydantic request/response schemas and input validation."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=500, examples=["How can I spot a phishing email?"])

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        if len(value) > 500:
            raise ValueError("question must be 500 characters or fewer")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            raise ValueError("question contains unsupported control characters")
        if not re.search(r"[A-Za-z0-9]", value):
            raise ValueError("question must contain letters or digits")
        return value


class AnswerPayload(BaseModel):
    category: str
    risk: str
    advice: str
    action: str


class AnswerResponse(BaseModel):
    request_id: str
    answer: AnswerPayload | None
    raw_response: str | None
    valid: bool
    guardrail_triggered: bool
    latency_ms: float
    model: str
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str


class ModelInfoResponse(BaseModel):
    base_model: str
    adapter: str
    license: str
    quantization: str
    categories: list[str]
    max_input_chars: int


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    invalid_responses: int
    guardrail_requests: int
    error_requests: int
    valid_rate: float
    latency_ms: dict[str, float]
    uptime_s: float
