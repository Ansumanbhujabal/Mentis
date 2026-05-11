from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from mentis.llm import LLMClient, LLMConfig, SafetyBlockedException


class TinyOut(BaseModel):
    items: list[str]


def _fake_response(content: str):
    return type(
        "R",
        (),
        {
            "choices": [
                type("C", (), {"message": type("M", (), {"content": content})()})()
            ]
        },
    )()


def _safety_blocked():
    raise SafetyBlockedException("blocked by provider safety filter")


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(providers=["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"])


@pytest.mark.asyncio
async def test_primary_succeeds_no_retries(config: LLMConfig) -> None:
    client = LLMClient(config)
    mock = AsyncMock(return_value=_fake_response('{"items": ["a"]}'))
    with patch("mentis.llm.acompletion", new=mock):
        out, used, trace = await client.complete_with_safety(
            system="sys", user="usr", schema=TinyOut, prompt_version="v1"
        )
    assert out == TinyOut(items=["a"])
    assert used == "gemini/gemini-2.0-flash"
    assert trace.retry_count == 0
    assert mock.await_count == 1


@pytest.mark.asyncio
async def test_safety_relax_retry(config: LLMConfig) -> None:
    """Primary fails with safety block → retry with relaxed settings succeeds."""
    client = LLMClient(config)
    side_effects = [_safety_blocked, _fake_response('{"items": ["x"]}')]

    async def mock_call(*args, **kwargs):
        effect = side_effects.pop(0)
        if callable(effect) and effect.__name__ == "_safety_blocked":
            effect()
        return effect

    with patch("mentis.llm.acompletion", new=AsyncMock(side_effect=mock_call)):
        out, used, trace = await client.complete_with_safety(
            system="sys", user="usr", schema=TinyOut, prompt_version="v1"
        )
    assert out == TinyOut(items=["x"])
    assert used == "gemini/gemini-2.0-flash"
    assert trace.retry_count == 1
    assert "relaxing_filters" in trace.actions


@pytest.mark.asyncio
async def test_all_paths_blocked_raises(config: LLMConfig) -> None:
    """All 4 escalation paths fail with safety block → raises SafetyBlockedException."""
    client = LLMClient(config)

    async def always_block(*args, **kwargs):
        _safety_blocked()

    with patch("mentis.llm.acompletion", new=AsyncMock(side_effect=always_block)), pytest.raises(
        SafetyBlockedException
    ):
        await client.complete_with_safety(
            system="sys", user="usr", schema=TinyOut, prompt_version="v1"
        )


@pytest.mark.asyncio
async def test_trace_does_not_claim_reframe_when_no_template(config: LLMConfig) -> None:
    """Without reframe_template, trace.actions must not contain 'reframing_prompt'."""
    client = LLMClient(config)

    block_count = [0]

    async def mock_call(*args, **kwargs):
        block_count[0] += 1
        # block step 1 and 2; step 4 (fallback) succeeds
        if block_count[0] <= 2:
            _safety_blocked()
        return _fake_response('{"items": ["x"]}')

    with patch("mentis.llm.acompletion", new=AsyncMock(side_effect=mock_call)):
        out, used, trace = await client.complete_with_safety(
            system="sys",
            user="usr",
            schema=TinyOut,
            prompt_version="v1",
            # reframe_template intentionally omitted
        )
    assert used == "groq/llama-3.3-70b-versatile"
    assert "reframing_prompt" not in trace.actions, (
        f"trace must not falsely claim reframing when no template provided: {trace.actions}"
    )
    assert "provider_fallback" in trace.actions
