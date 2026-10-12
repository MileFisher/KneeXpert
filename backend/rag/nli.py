"""Natural-language inference scorer for the safety gate.

Runs in-process rather than on the generation host, so the gate keeps working
when that host is unreachable — the same isolation that puts retrieval here.
A note that cannot be generated and a note that cannot be *checked* are
different failures, and only the first should depend on remote hardware.

Deliberately a dedicated entailment model, not the generating model judging its
own output: self-assessment is optimistic, doubles the slowest stage, and the
project's §2.9 specifies NLI models explicitly. Embedding cosine was also
rejected — similarity is not entailment, and a sentence that *contradicts* a
passage is highly similar to it.

Follows the lazy-singleton shape of ``mri/deit_classifier.py``: cheap
construction, weights on first use, module-level cache.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rag.config import NLI_BATCH_SIZE, NLI_MAX_LENGTH, NLI_MODEL

# MNLI label order for the DeBERTa-v3 checkpoints used here. Read from the
# model's own config at load time rather than assumed, because the ordering
# differs between MNLI checkpoints and getting it wrong silently inverts the
# gate: contradictions would score as support.
FALLBACK_LABEL_ORDER = ("entailment", "neutral", "contradiction")


class NliWeightsMissing(RuntimeError):
    """The entailment model could not be loaded. A deployment fault -> 503."""


class NliScorer:
    """Cross-encoder scoring (premise, hypothesis) pairs for entailment."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = NLI_MODEL
        self.tokenizer: Any = None
        self.model: Any = None
        self.label_order: tuple[str, ...] = FALLBACK_LABEL_ORDER
        self._loaded = False

    def load(self) -> None:
        """Load tokenizer and weights. Idempotent.

        Raises:
            NliWeightsMissing: the model could not be fetched or loaded.
        """
        if self._loaded:
            return
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            print(f"[rag] loading NLI {self.model_name} on {self.device}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            model.to(self.device)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
        except Exception as e:  # noqa: BLE001 - surfaced as a deployment fault
            raise NliWeightsMissing(
                f"Could not load entailment model {self.model_name!r}: {e}. "
                "It downloads on first use; check network access, or set "
                "RAG_NLI_MODEL to a locally cached model."
            ) from e

        # Trust the checkpoint's own id2label where it is present.
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) == 3:
            self.label_order = tuple(
                str(id2label[i]).lower() for i in sorted(id2label)
            )
        self._loaded = True

    def score_pairs(
        self, premises: list[str], hypotheses: list[str]
    ) -> np.ndarray:
        """Probabilities over (entailment, neutral, contradiction) per pair.

        Columns follow :attr:`label_order`. Returns shape ``(len(pairs), 3)``.
        """
        if len(premises) != len(hypotheses):
            raise ValueError(
                f"premise/hypothesis count mismatch: "
                f"{len(premises)} vs {len(hypotheses)}"
            )
        if not premises:
            return np.zeros((0, 3), dtype=np.float32)

        self.load()
        assert self.tokenizer is not None and self.model is not None

        out: list[np.ndarray] = []
        for start in range(0, len(premises), NLI_BATCH_SIZE):
            batch_p = premises[start : start + NLI_BATCH_SIZE]
            batch_h = hypotheses[start : start + NLI_BATCH_SIZE]
            encoded = self.tokenizer(
                batch_p,
                batch_h,
                truncation=True,
                max_length=NLI_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())

        return np.concatenate(out, axis=0).astype(np.float32)

    def entailment_index(self) -> int:
        """Column holding the entailment probability."""
        self.load()
        try:
            return self.label_order.index("entailment")
        except ValueError:
            return 0


_scorer: NliScorer | None = None


def get_nli() -> NliScorer:
    """Module-level cached scorer."""
    global _scorer
    if _scorer is None:
        _scorer = NliScorer()
    return _scorer


def nli_weights_cached() -> bool:
    """Whether the entailment model is already in the local cache.

    Checked without loading, so ``/health`` can warn before a demo that the
    first request will stall on a download.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:  # pragma: no cover
        return False
    from pathlib import Path

    folder = "models--" + NLI_MODEL.replace("/", "--")
    return (Path(HF_HUB_CACHE) / folder).is_dir()
