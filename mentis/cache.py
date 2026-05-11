"""File-based snippet cache with version-keyed invalidation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Cache:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return cls(root=Path(hf_home) / ".cache" / "mentis")
        return cls(root=Path.home() / ".cache" / "mentis")

    @staticmethod
    def compute_hash(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()[:16]

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str, model_cls: type[T]) -> T | None:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        return model_cls.model_validate_json(p.read_text())

    def set(self, namespace: str, key: str, value: BaseModel) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value.model_dump_json(indent=2))

    def clear(self, namespace: str | None = None) -> None:
        import shutil

        target = self.root / namespace if namespace else self.root
        if target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
