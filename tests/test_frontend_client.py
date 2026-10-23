"""Tests for the frontend's API helper (connection failures, server errors)."""
import pytest
import requests

from api_client import ApiError, predict_sentiment


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_post(monkeypatch, result):
    def fake_post(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(requests, "post", fake_post)


def test_success(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(200, {"sentiment": "positive", "confidence": 0.9}))
    assert predict_sentiment("good", "http://x")["sentiment"] == "positive"


def test_connection_error(monkeypatch):
    _patch_post(monkeypatch, requests.exceptions.ConnectionError())
    with pytest.raises(ApiError, match="Cannot reach"):
        predict_sentiment("good", "http://x")


def test_timeout(monkeypatch):
    _patch_post(monkeypatch, requests.exceptions.Timeout())
    with pytest.raises(ApiError, match="too long"):
        predict_sentiment("good", "http://x")


def test_validation_error_message(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(422, {"detail": "text: Text must not be empty."}))
    with pytest.raises(ApiError, match="must not be empty"):
        predict_sentiment("", "http://x")


def test_server_error(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(500))
    with pytest.raises(ApiError, match="internal error"):
        predict_sentiment("good", "http://x")


def test_model_not_ready(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(503, {"detail": "x"}))
    with pytest.raises(ApiError, match="not ready"):
        predict_sentiment("good", "http://x")
