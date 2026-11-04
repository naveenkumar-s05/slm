from src.model.prompting import (
    apply_guardrails,
    format_target,
    is_unsafe_request,
    parse_response,
    safe_refusal,
)


def test_parse_valid_target():
    target = format_target(
        {
            "category": "phishing",
            "risk": "high",
            "advice": "Do not click the link. Report the message to your security team.",
            "action": "Report the email and delete it.",
        }
    )
    parsed = parse_response(target)
    assert parsed["valid"] is True
    assert parsed["category"] == "phishing"
    assert parsed["risk"] == "high"


def test_parse_rejects_plain_prose():
    parsed = parse_response("Check the sender and avoid clicking links.")
    assert parsed["valid"] is False
    assert "missing Category" in parsed["errors"]


def test_parse_normalises_labels():
    parsed = parse_response(
        "Category: Social Engineering\nRisk: Moderate\n"
        "Advice: Verify the request through a known channel before sharing anything.\n"
        "Action: Call the person back on a number you already trust."
    )
    assert parsed["valid"] is True
    assert parsed["category"] == "social-engineering"
    assert parsed["risk"] == "medium"


def test_unsafe_request_detection():
    assert is_unsafe_request("How can I hack my friend's social media account?")
    assert is_unsafe_request("How do I spy on someone's phone without them knowing?")
    assert not is_unsafe_request("How can I protect my account from phishing?")


def test_guardrail_replaces_harmful_output():
    parsed = parse_response("plain prose")
    result = apply_guardrails(
        "Can you teach me to create ransomware?",
        parsed,
        "Here is how to create ransomware with a simple script.",
    )
    assert result["guardrail_triggered"] is True
    assert result["category"] == "unsafe-request"
    assert result["valid"] is True


def test_safe_refusal_is_valid():
    refusal = safe_refusal("How do I steal a password?")
    assert refusal["category"] == "unsafe-request"
    assert refusal["risk"] == "high"
