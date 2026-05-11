"""Async LiteLLM client with safety-aware fallback chain.

Escalation order on safety block:
  1. Primary call (default safety settings)
  2. Retry with relaxed safety settings on Gemini
  3. Retry with prompt reframed via safety_reframe.v1.j2 + relaxed settings
  4. Provider fallback (Groq)
  5. Raise SafetyBlockedException — caller handles (e.g., raw-snippets fallback)
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TypeVar

import litellm
from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Per-pipeline-run cost accumulator. ContextVar so asyncio.gather() tasks
# inherit the same context and all concurrent LLM calls contribute.
_PIPELINE_COST_USD: contextvars.ContextVar[float] = contextvars.ContextVar(
    "_PIPELINE_COST_USD", default=0.0
)


def add_pipeline_cost(usd: float) -> None:
    """Add to the per-context cost accumulator."""
    try:
        _PIPELINE_COST_USD.set(_PIPELINE_COST_USD.get() + float(usd))
    except Exception:
        pass


def consume_pipeline_cost() -> float:
    """Return the current accumulator value and reset it to 0."""
    total = _PIPELINE_COST_USD.get()
    _PIPELINE_COST_USD.set(0.0)
    return total

T = TypeVar("T", bound=BaseModel)


class SafetyBlockedException(Exception):
    """Raised when an LLM call is blocked by the provider's safety filter."""


RELAXED_PROCUREMENT_SETTINGS = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


@dataclass
class SafetyTrace:
    retry_count: int = 0
    actions: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    providers: list[str] = field(
        default_factory=lambda: ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"]
    )
    timeout_s: float = 180.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        primary = os.environ.get("MENTIS_PRIMARY_MODEL", "gemini/gemini-2.0-flash")
        csv = os.environ.get("MENTIS_PROVIDERS", "gemini,groq").split(",")
        azure_model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o")
        m = {
            "azure": f"azure/{azure_model}",
            "gemini": "gemini/gemini-2.0-flash",
            "groq": "groq/llama-3.3-70b-versatile",
        }
        ordered = [primary] + [m[p] for p in csv if m.get(p) and m[p] != primary]
        return cls(providers=ordered)


def _is_safety_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("safety", "blocked", "content_filter", "harm", "policy violation")
    )


def _strip_md_fences(content: str) -> str:
    """Strip ```json ... ``` markdown code fences some providers wrap JSON output in."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("rate limit", "resource_exhausted", "quota", "429", "rate_limit_exceeded")
    )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def _call(
        self,
        *,
        provider: str,
        system: str,
        user: str,
        schema: type[T],
        prompt_version: str,
        safety_settings: list | None = None,
    ) -> T:
        kwargs = {
            "model": provider,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "timeout": self.config.timeout_s,
            "metadata": {"prompt_version": prompt_version, "mentis_provider": provider},
        }
        if safety_settings is not None and provider.startswith("gemini"):
            kwargs["safety_settings"] = safety_settings
        if provider.startswith("azure/"):
            # LiteLLM expects api_key / api_base / api_version explicitly for Azure
            if api_key := os.environ.get("AZURE_OPENAI_API_KEY"):
                kwargs["api_key"] = api_key
            if api_base := os.environ.get("AZURE_OPENAI_ENDPOINT"):
                kwargs["api_base"] = api_base
            if api_version := os.environ.get("AZURE_OPENAI_API_VERSION"):
                kwargs["api_version"] = api_version

        max_rate_retries = 3
        for attempt in range(max_rate_retries):
            try:
                resp = await acompletion(**kwargs)
                # Track per-call cost in the pipeline accumulator
                try:
                    call_cost = float(litellm.completion_cost(completion_response=resp))
                except Exception:
                    call_cost = 0.0
                add_pipeline_cost(call_cost)
                content = resp.choices[0].message.content
                return schema.model_validate_json(_strip_md_fences(content))
            except Exception as e:
                if _is_safety_error(e):
                    raise SafetyBlockedException(str(e)) from e
                if _is_rate_limit_error(e) and attempt < max_rate_retries - 1:
                    # Try to parse retry-delay from message; default to exponential backoff
                    msg = str(e)
                    delay_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', msg)
                    delay = (
                        int(delay_match.group(1)) + 2
                        if delay_match
                        else 5 * (2 ** attempt)
                    )
                    logger.warning(
                        f"rate-limited on {provider}, waiting {delay}s "
                        f"(attempt {attempt + 1}/{max_rate_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"rate-limited too many times on {provider}")

    async def complete_with_safety(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        prompt_version: str,
        reframe_template: str | None = None,
    ) -> tuple[T, str, SafetyTrace]:
        """Run the 4-step safety escalation chain.

        Returns (parsed, provider_used, trace).
        Raises SafetyBlockedException if all paths fail.
        """
        trace = SafetyTrace()
        primary = self.config.providers[0]

        # 1. Primary call
        try:
            out = await self._call(
                provider=primary,
                system=system,
                user=user,
                schema=schema,
                prompt_version=prompt_version,
            )
            return out, primary, trace
        except SafetyBlockedException:
            trace.retry_count += 1
            trace.actions.append("relaxing_filters")

        # 2. Retry with relaxed safety on Gemini
        try:
            out = await self._call(
                provider=primary,
                system=system,
                user=user,
                schema=schema,
                prompt_version=prompt_version,
                safety_settings=RELAXED_PROCUREMENT_SETTINGS,
            )
            return out, primary, trace
        except SafetyBlockedException:
            trace.retry_count += 1
            # Action describes the NEXT step we'll attempt
            if reframe_template:
                trace.actions.append("reframing_prompt")
            else:
                trace.actions.append("provider_fallback")

        # 3. Reframe user prompt + relaxed safety (only if template provided)
        if reframe_template:
            reframed = reframe_template.replace("{{ original }}", user)
            try:
                out = await self._call(
                    provider=primary,
                    system=system,
                    user=reframed,
                    schema=schema,
                    prompt_version=prompt_version,
                    safety_settings=RELAXED_PROCUREMENT_SETTINGS,
                )
                return out, primary, trace
            except SafetyBlockedException:
                trace.retry_count += 1
                trace.actions.append("provider_fallback")

        # 4. Provider fallback
        for fb in self.config.providers[1:]:
            try:
                out = await self._call(
                    provider=fb,
                    system=system,
                    user=user,
                    schema=schema,
                    prompt_version=prompt_version,
                )
                return out, fb, trace
            except SafetyBlockedException:
                continue

        # 5. All paths failed
        trace.retry_count += 1
        trace.actions.append("all_blocked")
        raise SafetyBlockedException("all providers blocked; surface raw snippets")
