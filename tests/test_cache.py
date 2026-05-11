from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from mentis.cache import Cache


class Toy(BaseModel):
    n: int


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(root=tmp_path)


def test_set_and_get(cache: Cache) -> None:
    cache.set("retrieve", "key1", Toy(n=42))
    assert cache.get("retrieve", "key1", Toy) == Toy(n=42)


def test_miss_returns_none(cache: Cache) -> None:
    assert cache.get("retrieve", "absent", Toy) is None


def test_namespace_isolation(cache: Cache) -> None:
    cache.set("a", "k", Toy(n=1))
    cache.set("b", "k", Toy(n=2))
    assert cache.get("a", "k", Toy) == Toy(n=1)
    assert cache.get("b", "k", Toy) == Toy(n=2)


def test_compute_hash_stable_and_version_aware() -> None:
    assert Cache.compute_hash("q", "v1") == Cache.compute_hash("q", "v1")
    assert Cache.compute_hash("q", "v1") != Cache.compute_hash("q", "v2")


def test_hf_spaces_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    c = Cache.from_env()
    assert str(tmp_path / "hf") in str(c.root)
