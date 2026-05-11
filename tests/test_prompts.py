from __future__ import annotations

from pathlib import Path

import pytest

from mentis.prompts import PromptRegistry


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "x.v1.j2").write_text("hello {{ name }}")
    (p / "x.v2.j2").write_text("hi {{ name }}")
    return p


def test_loads_local_prompt(prompt_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompt_dir)
    rendered = reg.get("x", version="v1").render(name="Mentis")
    assert rendered == "hello Mentis"


def test_default_version_is_highest(prompt_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompt_dir)
    p = reg.get("x")
    assert p.version == "v2"


def test_missing_prompt_raises(prompt_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompt_dir)
    with pytest.raises(FileNotFoundError):
        reg.get("nope")
