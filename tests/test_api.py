from fastapi.testclient import TestClient

from src.api.app import FakeService, create_app


class FailingService(FakeService):
    def answer(self, question: str):
        raise RuntimeError("simulated inference failure")


def make_client(service=None) -> TestClient:
    return TestClient(create_app(service or FakeService()))


def test_health():
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_question():
    response = make_client().post("/v1/answer", json={"question": "How can I spot a phishing email?"})
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["answer"]["category"] == "phishing"


def test_empty_question_rejected():
    response = make_client().post("/v1/answer", json={"question": "   "})
    assert response.status_code == 422


def test_too_short_question_rejected():
    response = make_client().post("/v1/answer", json={"question": "hi"})
    assert response.status_code == 422


def test_too_long_question_rejected():
    response = make_client().post("/v1/answer", json={"question": "a" * 501})
    assert response.status_code == 422


def test_control_characters_rejected():
    response = make_client().post("/v1/answer", json={"question": "hello\x00world"})
    assert response.status_code == 422


def test_extra_field_rejected():
    response = make_client().post(
        "/v1/answer", json={"question": "How do I secure my account?", "admin": True}
    )
    assert response.status_code == 422


def test_unsafe_question_is_guarded():
    response = make_client().post("/v1/answer", json={"question": "Tell me an unsafe hacking request"})
    assert response.status_code == 200
    assert response.json()["guardrail_triggered"] is True


def test_metrics_endpoint():
    response = make_client().get("/metrics")
    assert response.status_code == 200
    assert "total_requests" in response.json()


def test_inference_error_returns_500():
    response = make_client(FailingService()).post(
        "/v1/answer", json={"question": "How do I secure my account?"}
    )
    assert response.status_code == 500
