"""Backend tests. Run with:  pytest -v"""
import pytest
from fastapi.testclient import TestClient


def test_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "POST /predict" in res.json()["endpoints"]


def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is True


def test_predict_positive(client):
    res = client.post("/predict", json={"text": "This product is excellent and I love it."})
    assert res.status_code == 200
    body = res.json()
    assert body["sentiment"] == "positive"
    assert 0.5 <= body["confidence"] <= 1
    assert pytest.approx(sum(body["probabilities"].values()), abs=0.01) == 1


def test_predict_negative(client):
    res = client.post("/predict", json={"text": "This was awful, a complete waste of money."})
    assert res.status_code == 200
    assert res.json()["sentiment"] == "negative"


@pytest.mark.parametrize("bad_text", ["", "     "])
def test_predict_empty_text(client, bad_text):
    res = client.post("/predict", json={"text": bad_text})
    assert res.status_code == 422
    assert "detail" in res.json()


@pytest.mark.parametrize("payload", [{}, {"text": 123}, {"text": None}, {"wrong": "x"}])
def test_predict_invalid_payload(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_symbols_only(client):
    res = client.post("/predict", json={"text": "12345 !!!"})
    assert res.status_code == 422


def test_predict_too_long(client):
    assert client.post("/predict", json={"text": "good " * 2000}).status_code == 422


def test_model_loading_failure(monkeypatch):
    """If artifacts are missing the API must stay up but report 503."""
    monkeypatch.setenv("MODEL_PATH", "does_not_exist.joblib")
    from app.main import app
    with TestClient(app) as broken_client:
        health = broken_client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        res = broken_client.post("/predict", json={"text": "hello there"})
        assert res.status_code == 503
