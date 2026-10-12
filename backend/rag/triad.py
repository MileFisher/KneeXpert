"""RAG Triad metrics: context relevance, answer relevance, groundedness.

All three are computed and recorded; only groundedness gates. The other two are
diagnostic — they distinguish "the model wrote badly" from "retrieval fetched the
wrong passages", which a single number cannot.

Groundedness is the expensive one. Each sentence is scored against only the few
most similar passages rather than all of them: a 184M-parameter cross-encoder at
100-400 ms per pair makes the difference between 48 pairs and 24 the difference
between a usable endpoint and an unusable one.
"""

from __future__ import annotations

import numpy as np

from rag.config import NLI_PREMISE_TOP_N
from rag.gate import SentenceClass, SentenceScore
from rag.prompt import CITATION_MARKER
from rag.schemas import RetrievedPassage


def context_relevance(passages: list[RetrievedPassage]) -> float | None:
    """Mean retrieval score of the supplied passages.

    Returns None — not zero — when nothing was supplied as context, because the
    metric is *undefined* for the ungrounded configuration rather than scoring
    badly. Reporting zero there would read as "retrieval failed" in §4.3 instead
    of "not applicable".
    """
    if not passages:
        return None
    return float(sum(p.score for p in passages) / len(passages))


def answer_relevance(query: str, note_text: str) -> float:
    """Similarity between the diagnostic payload and the note that answers it.

    Both sides are embedded as passages, since both are prose; the query is the
    natural-language rendering used for retrieval, not a serialised dict.
    """
    if not query.strip() or not note_text.strip():
        return 0.0
    from rag.embedder import get_embedder

    return float(get_embedder().cosine(query, note_text))


def _cited_marker(sentence: str) -> int | None:
    """The first ``[n]`` the model wrote in this sentence, if any."""
    match = CITATION_MARKER.search(sentence)
    return int(match.group(1)) if match else None


def _premise_candidates(
    sentence: str, passages: list[RetrievedPassage], top_n: int
) -> list[RetrievedPassage]:
    """The most similar passages to one sentence, by embedding cosine.

    A cheap pre-filter in front of an expensive cross-encoder. Similarity is a
    poor judge of entailment but a decent judge of *topicality*, which is all
    that is being asked of it here.

    The patient-findings premise is always kept, never ranked. It is the record
    the note is about, so it is relevant to every sentence by construction;
    making it compete for a top-N slot on cosine similarity against prose-heavy
    literature excluded it from most sentences, which defeated the point of
    adding it.
    """
    always = [p for p in passages if p.source_type == "findings"]
    rankable = [p for p in passages if p.source_type != "findings"]

    if len(rankable) <= top_n:
        return always + rankable

    from rag.embedder import get_embedder

    embedder = get_embedder()
    vectors = embedder.embed_passages([sentence] + [p.text for p in rankable])
    sims = vectors[1:] @ vectors[0]
    order = np.argsort(-sims)[:top_n]
    return always + [rankable[i] for i in order]


FINDINGS_PASSAGE_ID = "patient-findings"


def findings_premise(req: object) -> RetrievedPassage:
    """The patient's own findings, as a premise the gate can score against.

    §2.9 requires every statement to be supported by "the underlying patient
    record **and** retrieved guidelines". Scoring only against retrieved
    literature implements half of that, and the half it omits is the half most of
    a note is about: a synthesis sentence restates *this patient's* grade and
    labels, which no general article entails. Measured on real output, grounded
    synthesis sentences scored 0.06 against literature alone while their
    recommendations scored 0.98 — the note was rejected for being about the
    patient.

    Deliberately not added to the response's ``retrieved_passages``, and excluded
    from context relevance: it was not retrieved, and reporting it as though it
    were would overstate what retrieval found.
    """
    from rag.prompt import render_findings_premise

    return RetrievedPassage(
        passage_id=FINDINGS_PASSAGE_ID,
        source="Patient findings (this study)",
        text=render_findings_premise(req),  # type: ignore[arg-type]
        score=1.0,
        licence="n/a",
        source_type="findings",
    )


def groundedness(
    sentences: list[tuple[str, SentenceClass]],
    passages: list[RetrievedPassage],
) -> list[SentenceScore]:
    """Score each sentence against its best-supporting passage.

    Returns one :class:`SentenceScore` per sentence, carrying the
    entailment-derived supporting passage. That attribution — not the marker the
    model wrote — is what the UI surfaces, because it cannot be fabricated.

    Every per-sentence score is returned rather than only the aggregate, because
    the offline threshold sweep replays these and is impossible without them.
    """
    if not sentences:
        return []
    if not passages:
        return [
            SentenceScore(
                text=text,
                sentence_class=cls,
                groundedness=0.0,
                label="neutral",
                supporting_passage_id=None,
                cited_marker=_cited_marker(text),
            )
            for text, cls in sentences
        ]

    from rag.nli import get_nli

    scorer = get_nli()
    entail_col = scorer.entailment_index()

    premise_sets = [
        _premise_candidates(text, passages, NLI_PREMISE_TOP_N)
        for text, _ in sentences
    ]

    # One flat batch across all (sentence, premise) pairs, so the cross-encoder
    # is invoked once rather than per sentence.
    premises: list[str] = []
    hypotheses: list[str] = []
    spans: list[tuple[int, int]] = []
    for (text, _), candidates in zip(sentences, premise_sets, strict=True):
        start = len(premises)
        for passage in candidates:
            premises.append(passage.text)
            hypotheses.append(text)
        spans.append((start, len(premises)))

    probs = scorer.score_pairs(premises, hypotheses)

    scores: list[SentenceScore] = []
    for index, ((text, cls), (start, end)) in enumerate(
        zip(sentences, spans, strict=True)
    ):
        window = probs[start:end]
        candidates = premise_sets[index]
        if window.size == 0 or not candidates:
            scores.append(
                SentenceScore(
                    text=text,
                    sentence_class=cls,
                    groundedness=0.0,
                    label="neutral",
                    cited_marker=_cited_marker(text),
                )
            )
            continue

        best = int(np.argmax(window[:, entail_col]))
        row = window[best]
        label = scorer.label_order[int(np.argmax(row))]
        scores.append(
            SentenceScore(
                text=text,
                sentence_class=cls,
                groundedness=float(row[entail_col]),
                label=label,  # type: ignore[arg-type]
                supporting_passage_id=candidates[best].passage_id,
                cited_marker=_cited_marker(text),
            )
        )
    return scores


def note_sentences(
    synthesis: str, next_steps: list[str]
) -> list[tuple[str, SentenceClass]]:
    """Flatten a note into classified sentences for scoring.

    Each next step is one unit even when it contains several sentences: it is one
    recommendation, and splitting it would let half a recommendation pass.
    """
    from rag.sentences import split_sentences

    items: list[tuple[str, SentenceClass]] = [
        (sentence, "synthesis") for sentence in split_sentences(synthesis)
    ]
    items.extend((step.strip(), "next_step") for step in next_steps if step.strip())
    return items
