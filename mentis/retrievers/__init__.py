"""Retriever protocol — every retriever implements .search(query, n) -> list[Snippet]."""
from __future__ import annotations

from typing import Protocol

from mentis.schemas import Snippet


class BaseRetriever(Protocol):
    name: str

    async def search(self, query: str, n: int = 5) -> list[Snippet]: ...
