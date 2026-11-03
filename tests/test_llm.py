from types import SimpleNamespace

from google.genai import errors

import llm

DAILY = errors.ClientError(429, {"error": {"code": 429, "message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})


class FakeModels:
    def __init__(self):
        self.tried = []

    def generate_content(self, model, **kw):
        self.tried.append(model)
        if model == llm.CHAT_MODELS[0]:
            raise DAILY
        return SimpleNamespace(text="Answer [1]")


def test_daily_limit_moves_to_next_model_without_waiting(monkeypatch):
    models = FakeModels()
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(llm.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    assert llm.answer("q", [])["text"] == "Answer [1]"
    assert models.tried == llm.CHAT_MODELS[:2]


def test_network_error_becomes_unavailable(monkeypatch):
    import httpx
    import pytest

    def fail(**kw):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=fail)))
    with pytest.raises(llm.Unavailable):
        llm.answer("q", [])


def test_overload_gives_up_quickly_for_visitors(monkeypatch):
    import pytest
    slept = []

    def busy(**kw):
        raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=busy)))
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    with pytest.raises(llm.Unavailable):
        llm.answer("q", [])
    assert sum(slept) <= 60  # all three models together, not minutes per model
