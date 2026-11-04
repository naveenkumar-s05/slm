from src.evaluation.metrics import compute_metrics


def prediction(gold_category, gold_risk, parsed, raw="", latency=100.0, output_tokens=50):
    return {
        "id": "x",
        "question": "sample question",
        "gold": {
            "category": gold_category,
            "risk": gold_risk,
            "advice": "Use multi-factor authentication and a unique password.",
            "action": "Enable MFA on the account.",
        },
        "raw_response": raw,
        "parsed": parsed,
        "latency_ms": latency,
        "output_tokens": output_tokens,
    }


def test_metrics_combine_valid_and_invalid():
    valid = prediction(
        "passwords",
        "low",
        {
            "valid": True,
            "category": "passwords",
            "risk": "low",
            "advice": "Use multi-factor authentication and a unique password.",
            "action": "Enable MFA on the account.",
            "errors": [],
        },
    )
    invalid = prediction("phishing", "high", {"valid": False, "errors": ["missing Category"]})
    metrics = compute_metrics([valid, invalid])
    assert metrics["count"] == 2
    assert metrics["format_valid_rate"] == 0.5
    assert metrics["category_accuracy"] == 0.5
    assert metrics["risk_accuracy"] == 0.5
    assert metrics["latency_ms"]["mean"] == 100.0


def test_unsafe_recall():
    item = prediction(
        "unsafe-request",
        "high",
        {
            "valid": True,
            "category": "unsafe-request",
            "risk": "high",
            "advice": "I cannot help with harmful activity.",
            "action": "Ask for defensive guidance.",
            "errors": [],
        },
    )
    metrics = compute_metrics([item])
    assert metrics["unsafe_request_recall"] == 1.0
