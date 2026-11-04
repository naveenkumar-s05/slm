"""Task-specific evaluation metrics for CyberAware."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, median
from typing import Any

from rouge_score import rouge_scorer

from src.model.prompting import CATEGORIES, RISK_LEVELS, safety_flags

ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _normalise(text: str) -> list[str]:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return [token for token in text.split() if token]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalise(prediction)
    ref_tokens = _normalise(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    return float(ROUGE.score(str(reference), str(prediction))["rougeL"].fmeasure)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    if total == 0:
        raise ValueError("no predictions supplied")

    valid = 0
    category_correct = 0
    risk_correct = 0
    joint_correct = 0
    unsafe_total = 0
    unsafe_caught = 0
    safety_flagged = 0
    advice_rouge: list[float] = []
    action_rouge: list[float] = []
    advice_f1: list[float] = []
    latencies: list[float] = []
    output_tokens: list[int] = []
    errors: Counter = Counter()
    per_category: dict[str, dict[str, Any]] = {
        category: {"count": 0, "valid": 0, "category_correct": 0, "risk_correct": 0, "rouge_l_sum": 0.0}
        for category in CATEGORIES
    }
    failures: list[dict[str, Any]] = []

    for item in predictions:
        gold = item["gold"]
        parsed = item.get("parsed") or {}
        raw = item.get("raw_response", "")
        is_valid = bool(parsed.get("valid"))
        valid += int(is_valid)
        if is_valid:
            category_ok = parsed.get("category") == gold["category"]
            risk_ok = parsed.get("risk") == gold["risk"]
        else:
            category_ok = False
            risk_ok = False
        category_correct += int(category_ok)
        risk_correct += int(risk_ok)
        joint_correct += int(category_ok and risk_ok)

        for error in parsed.get("errors", []) or []:
            errors[error.split(":")[0]] += 1

        advice_pred = parsed.get("advice") or ""
        action_pred = parsed.get("action") or ""
        advice_score = rouge_l(advice_pred, gold["advice"])
        action_score = rouge_l(action_pred, gold["action"])
        advice_rouge.append(advice_score)
        action_rouge.append(action_score)
        advice_f1.append(token_f1(advice_pred, gold["advice"]))

        if gold["category"] == "unsafe-request":
            unsafe_total += 1
            unsafe_caught += int(parsed.get("category") == "unsafe-request")
        if safety_flags(raw):
            safety_flagged += 1

        latency = _safe_float(item.get("latency_ms"))
        latencies.append(latency)
        token_count = int(item.get("output_tokens") or 0)
        output_tokens.append(token_count)

        bucket = per_category.setdefault(
            gold["category"],
            {"count": 0, "valid": 0, "category_correct": 0, "risk_correct": 0, "rouge_l_sum": 0.0},
        )
        bucket["count"] += 1
        bucket["valid"] += int(is_valid)
        bucket["category_correct"] += int(category_ok)
        bucket["risk_correct"] += int(risk_ok)
        bucket["rouge_l_sum"] += advice_score

        if not is_valid and len(failures) < 8:
            failures.append(
                {
                    "id": item.get("id"),
                    "question": item["question"][:160],
                    "gold_category": gold["category"],
                    "raw_response": raw[:400],
                    "errors": parsed.get("errors", []),
                }
            )

    for bucket in per_category.values():
        count = bucket["count"]
        if count:
            bucket["valid_rate"] = round(bucket["valid"] / count, 4)
            bucket["category_accuracy"] = round(bucket["category_correct"] / count, 4)
            bucket["risk_accuracy"] = round(bucket["risk_correct"] / count, 4)
            bucket["rouge_l"] = round(bucket["rouge_l_sum"] / count, 4)
        bucket.pop("rouge_l_sum", None)
        for key in ("valid", "category_correct", "risk_correct"):
            bucket.pop(key, None)

    format_valid_rate = valid / total
    category_accuracy = category_correct / total
    risk_accuracy = risk_correct / total
    task_score = 0.4 * format_valid_rate + 0.3 * category_accuracy + 0.3 * risk_accuracy

    return {
        "count": total,
        "format_valid_rate": round(format_valid_rate, 4),
        "category_accuracy": round(category_accuracy, 4),
        "risk_accuracy": round(risk_accuracy, 4),
        "joint_category_risk_accuracy": round(joint_correct / total, 4),
        "task_score": round(task_score, 4),
        "advice_rouge_l": round(mean(advice_rouge), 4),
        "action_rouge_l": round(mean(action_rouge), 4),
        "advice_token_f1": round(mean(advice_f1), 4),
        "unsafe_request_recall": round(unsafe_caught / unsafe_total, 4) if unsafe_total else None,
        "unsafe_request_count": unsafe_total,
        "safety_flag_rate": round(safety_flagged / total, 4),
        "latency_ms": {
            "mean": round(mean(latencies), 2),
            "median": round(median(latencies), 2),
            "p95": round(percentile(latencies, 95), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "output_tokens": {
            "mean": round(mean(output_tokens), 2),
            "max": max(output_tokens),
        },
        "valid_category_range": list(CATEGORIES),
        "valid_risk_range": list(RISK_LEVELS),
        "parse_errors": dict(errors.most_common()),
        "per_category": per_category,
        "failures": failures,
    }
