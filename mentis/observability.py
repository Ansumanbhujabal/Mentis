"""Langfuse initialization + LiteLLM callback wiring.

init_observability() is called once at app/CLI startup.
If Langfuse keys are absent, this is a no-op.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def init_observability() -> None:
    if not is_langfuse_configured():
        logger.debug("Langfuse keys not set; tracing disabled.")
        return

    import litellm

    litellm.success_callback = list(set((litellm.success_callback or []) + ["langfuse"]))
    litellm.failure_callback = list(set((litellm.failure_callback or []) + ["langfuse"]))
    logger.info("Langfuse observability enabled for LiteLLM.")
