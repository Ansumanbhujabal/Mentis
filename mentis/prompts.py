"""Versioned prompt loader.

Source of truth: in-repo Jinja2 files at prompts/{name}.{version}.j2
Runtime: try Langfuse first → fall back to in-repo file.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template_text: str
    source: str  # "langfuse" or "local"

    def render(self, **kwargs: object) -> str:
        return Template(self.template_text).render(**kwargs)


class PromptRegistry:
    _FILE_RE = re.compile(r"^(?P<name>.+)\.(?P<version>v\d+)\.j2$")

    def __init__(self, in_repo_dir: Path) -> None:
        self.in_repo_dir = in_repo_dir
        self._langfuse_enabled = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        )

    def _local_candidates(self, name: str) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        for p in self.in_repo_dir.glob(f"{name}.*.j2"):
            m = self._FILE_RE.match(p.name)
            if m and m.group("name") == name:
                out.append((m.group("version"), p))
        return sorted(out, key=lambda t: int(t[0][1:]))

    def _try_langfuse(self, name: str, version: str | None) -> Prompt | None:
        if not self._langfuse_enabled:
            return None
        try:
            from langfuse import Langfuse

            lf = Langfuse()
            # Langfuse rejects passing both version + label. Prefer label when version unspecified.
            if version:
                p = lf.get_prompt(name=name, version=int(version[1:]))
            else:
                p = lf.get_prompt(name=name, label="production")
            return Prompt(
                name=name,
                version=f"v{p.version}",
                template_text=p.prompt,
                source="langfuse",
            )
        except Exception as e:
            logger.warning(f"Langfuse fetch failed for {name}: {e!r}; falling back to local")
            return None

    def get(self, name: str, version: str | None = None) -> Prompt:
        # 1. Try Langfuse
        if lf_prompt := self._try_langfuse(name, version):
            return lf_prompt

        # 2. Local fallback
        candidates = self._local_candidates(name)
        if not candidates:
            raise FileNotFoundError(f"no prompt named {name!r} in {self.in_repo_dir}")
        if version is None:
            chosen_version, chosen_path = candidates[-1]
        else:
            matches = [c for c in candidates if c[0] == version]
            if not matches:
                raise FileNotFoundError(f"no prompt {name!r} version {version!r}")
            chosen_version, chosen_path = matches[0]
        return Prompt(
            name=name, version=chosen_version, template_text=chosen_path.read_text(), source="local"
        )
