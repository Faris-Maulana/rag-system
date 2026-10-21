from types import SimpleNamespace

import httpx
import pytest
from groq import RateLimitError

import generate
from generate import Generator


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"  {text}  "))])


def _rate_limit():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(generate.time, "sleep", lambda _: None)


def test_prompt_contains_context_and_question():
    gen = Generator(client=FakeClient([]))
    prompt = gen.format_prompt("What is RAG?", ["chunk one", "chunk two"])
    assert "chunk one\n\nchunk two" in prompt
    assert "Question: What is RAG?" in prompt


def test_generate_returns_stripped_answer_at_temperature_zero():
    client = FakeClient([_response("RAG grounds answers.")])
    assert Generator(client=client).generate("q", ["c"]) == "RAG grounds answers."
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_generate_retries_on_rate_limit():
    client = FakeClient([_rate_limit(), _rate_limit(), _response("ok")])
    assert Generator(client=client).generate("q", ["c"]) == "ok"
    assert len(client.calls) == 3


def test_generate_gives_up_after_max_retries():
    client = FakeClient([_rate_limit()] * 3)
    assert Generator(client=client).generate("q", ["c"]).startswith("Error: Max retries")


def test_generate_does_not_retry_non_rate_limit_errors():
    client = FakeClient([ValueError("bad request"), _response("never reached")])
    assert Generator(client=client).generate("q", ["c"]) == "Error: LLM call failed (ValueError)."
    assert len(client.calls) == 1


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Generator()
