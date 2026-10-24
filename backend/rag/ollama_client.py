"""HTTP client for the generation host.

Generation is the one part of this layer that runs elsewhere — on a rented GPU
box, an institutional server, or a local Ollama. Everything else (retrieval,
embedding, the safety gate) runs in-process, so this module is the only thing
that fails when that host is unreachable.

Unreachability is a *reported state*, not an exception the caller must handle as
an error: the note layer withholds a note and says why rather than failing the
request.
"""

from __future__ import annotations

from typing import Any

import httpx

from rag.config import (
    GEN_SEED,
    HTTP_CONNECT_TIMEOUT_S,
    LLM_PROBE_TIMEOUT_S,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_S,
)


class OllamaUnavailable(RuntimeError):
    """The generation host could not be reached."""


class OllamaTimeout(RuntimeError):
    """The host accepted the request but did not finish in time."""


def _timeout(total_s: float) -> httpx.Timeout:
    """Timeout with a separate, tighter connect budget.

    On Windows an unreachable host can burn an entire single timeout on an IPv6
    attempt before falling back to IPv4, so one overall value understates the
    real wall-clock cost of a dead host.
    """
    return httpx.Timeout(total_s, connect=HTTP_CONNECT_TIMEOUT_S)


def is_available(timeout_s: float = LLM_PROBE_TIMEOUT_S) -> bool:
    """Whether the host answers and is serving models."""
    try:
        with httpx.Client(timeout=_timeout(timeout_s)) as client:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    seed: int = GEN_SEED,
    num_ctx: int = OLLAMA_NUM_CTX,
    num_predict: int = OLLAMA_NUM_PREDICT,
    json_schema: dict[str, Any] | None = None,
    timeout_s: float = OLLAMA_TIMEOUT_S,
    model: str = OLLAMA_MODEL,
) -> str:
    """Generate one completion and return its raw text.

    ``num_ctx`` is always sent explicitly and must stay that way. Ollama defaults
    it to 2048 regardless of the model's advertised context length, then silently
    truncates the prompt **from the left** — which discards the system message's
    safety rules first and leaves no error behind. Six retrieved passages plus the
    system prompt and payload overflow 2048 comfortably.

    Raises:
        OllamaUnavailable: the host could not be reached.
        OllamaTimeout: generation did not finish within ``timeout_s``.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "repeat_penalty": 1.15,
        },
    }
    if json_schema is not None:
        payload["format"] = json_schema

    try:
        with httpx.Client(timeout=_timeout(timeout_s)) as client:
            response = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as e:
        raise OllamaTimeout(f"generation exceeded {timeout_s}s: {e}") from e
    except httpx.HTTPError as e:
        raise OllamaUnavailable(f"{OLLAMA_BASE_URL} unreachable: {e}") from e

    message = body.get("message") or {}
    return str(message.get("content", ""))
