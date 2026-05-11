"""Async LiteLLM client with safety-aware fallback chain.

Escalation order on safety block:
  1. Primary call (default safety settings)
  2. Retry with relaxed safety settings on Gemini
  3. Retry with prompt reframed via safety_reframe.v1.j2 + relaxed settings
  4. Provider fallback (Groq)
  5. Raise SafetyBlockedException — caller handles (e.g., raw-snippets fallback)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TypeVar

from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        primary = os.environ.get("MENTIS_PRIMARY_MODEL", "gemini/gemini-2.0-flash")
        csv = os.environ.get("MENTIS_PROVIDERS", "gemini,groq").split(",")
        m = {
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
        try:
            resp = await acompletion(**kwargs)
        except Exception as e:
            if _is_safety_error(e):
                raise SafetyBlockedException(str(e)) from e
            raise
        content = resp.choices[0].message.content
        return schema.model_validate_json(content)

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
