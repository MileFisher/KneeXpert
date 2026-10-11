"""Readiness reporting for the clinical-note layer, safe to call from /health.

``/health`` is hit by the frontend on page load, so every probe here is bounded.
The generation host is the dangerous one: it may be a rented GPU box that is
switched off, and a naive synchronous probe against a dead host would hang the
whole dashboard. So the probe is async, has a short timeout, and its result is
cached for a while — a health check must never cost more than it reports.

Model-weight presence is reported from the HuggingFace cache *without loading
anything*, so an operator can see before a demo whether the first request will
stall on a multi-hundred-megabyte download.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from rag.config import (
    EMBED_MODEL,
    INDEX_DIR,
    LLM_PROBE_CACHE_TTL_S,
    LLM_PROBE_CONNECT_TIMEOUT_S,
    LLM_PROBE_TIMEOUT_S,
    NLI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    rag_corpus_chunk_count,
    rag_index_ready,
)

_probe_cache: dict[str, Any] = {"at": 0.0, "value": None}


def _hf_cache_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def model_weights_cached() -> dict[str, bool]:
    """Whether each local model is already in the HuggingFace cache.

    Nothing is loaded and nothing is downloaded — this only looks for the cache
    directory, so it is cheap enough for a health check.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = HF_HUB_CACHE
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        return {"embedder": False, "nli": False}

    from pathlib import Path

    root = Path(cache_root)
    return {
        "embedder": (root / _hf_cache_dir_name(EMBED_MODEL)).is_dir(),
        "nli": (root / _hf_cache_dir_name(NLI_MODEL)).is_dir(),
    }


async def probe_llm(force: bool = False) -> dict[str, Any]:
    """Check whether the generation host is reachable and has the model.

    Cached for :data:`LLM_PROBE_CACHE_TTL_S`. Never raises: an unreachable host
    is a reported state, not an error, because the note layer degrades rather
    than failing when generation is unavailable.
    """
    now = time.monotonic()
    cached = _probe_cache["value"]
    if not force and cached is not None and now - _probe_cache["at"] < LLM_PROBE_CACHE_TTL_S:
        return dict(cached)

    result: dict[str, Any] = {
        "available": False,
        "model_present": False,
        "models": [],
        "error": None,
    }
    try:
        timeout = httpx.Timeout(
            LLM_PROBE_TIMEOUT_S, connect=LLM_PROBE_CONNECT_TIMEOUT_S
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            payload = response.json()
        names = [m.get("name", "") for m in payload.get("models", [])]
        result["available"] = True
        result["models"] = names
        result["model_present"] = any(
            name == OLLAMA_MODEL or name.startswith(f"{OLLAMA_MODEL}:")
            for name in names
        )
    except (httpx.HTTPError, ValueError, KeyError) as e:
        result["error"] = f"{type(e).__name__}: {e}"

    _probe_cache["at"] = now
    _probe_cache["value"] = result
    return dict(result)


async def rag_health() -> dict[str, Any]:
    """The clinical-note layer's slice of ``/health``.

    Field naming follows the convention already used for model weights:
    ``<thing>_available`` / ``<thing>_ready`` booleans alongside the detail.
    """
    llm = await probe_llm()
    cached = model_weights_cached()
    chunk_count = rag_corpus_chunk_count()

    return {
        "rag_index_ready": rag_index_ready(),
        "rag_index_path": str(INDEX_DIR),
        "rag_chunk_count": chunk_count,
        "rag_models_cached": cached,
        "rag_models_ready": all(cached.values()),
        "llm_available": llm["available"],
        "llm_model": OLLAMA_MODEL,
        "llm_model_present": llm["model_present"],
        "llm_base_url": OLLAMA_BASE_URL,
        "llm_error": llm["error"],
    }
