"""Run the three-configuration ablation.

    python -m rag.eval.run_ablation                       # all configs, resuming
    python -m rag.eval.run_ablation --limit 3             # smoke test first
    python -m rag.eval.run_ablation --config rag_gate
    python -m rag.eval.run_ablation --fresh               # ignore prior results

Three configurations, mirroring the artifact-removal ablation the project already
reports:

======================  ====================  ===========
configuration           passages in prompt    gate scores
======================  ====================  ===========
``ungrounded``          no                    yes
``rag_no_gate``         yes                   yes
``rag_gate``            yes                   yes
======================  ====================  ===========

**Retrieval runs in every configuration.** Only the two RAG configurations put
the passages in the prompt; in ``ungrounded`` they serve purely as scoring
premises. Without that, groundedness would have nothing to score the ungrounded
arm against and the comparison would be ill-posed — and it is exactly the number
that distinguishes "our retrieval worked" from "the base model is cautious
anyway".

**Every attempt is generated unconditionally**, not only after a failure. The
offline threshold sweep replays stored per-sentence scores, so a run that stopped
early would only be honest at or below the threshold it ran at: there would be no
record of what attempt 2 *would* have produced for a case that passed at attempt
1. That is the difference between a real sweep curve and a misleading one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from rag.config import (
    EVAL_RESULTS_DIR,
    MAX_RETRIES,
    NEXT_STEPS_TAU,
    NLI_MODEL,
    OLLAMA_MODEL,
    SYNTHESIS_TAU,
    TOP_K,
)
from rag.eval.payloads import PayloadCase, load_payloads
from rag.gate import plan_attempt, rejected_texts
from rag.prompt import revision_hint_for
from rag.schemas import FeedbackGenerateRequest, NoteConfig
from rag.service import AttemptResult, run_attempt, to_schema

CONFIGS: tuple[NoteConfig, ...] = ("ungrounded", "rag_no_gate", "rag_gate")
RUNS_PATH = EVAL_RESULTS_DIR / "runs.jsonl"


def record_key(case_id: str, config: str, attempt: int) -> str:
    """Resume key. Stable, because a rented box can vanish mid-run."""
    return f"{case_id}|{config}|{attempt}"


def load_resumed(path: Path) -> dict[str, dict[str, Any]]:
    """Rows already recorded, keyed for resume.

    Returns the rows themselves rather than just their keys: a resumed attempt
    needs its predecessor's rejected sentences to rebuild the revision hint, and
    dropping them silently changed the prompt on resumed runs.
    """
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial final line. Skip it
                # rather than refusing to resume.
                continue
            rows[record_key(row["case_id"], row["config"], row["attempt"])] = row
    return rows


def revision_hint_from_row(row: dict[str, Any]) -> str | None:
    """Rebuild a recorded attempt's revision hint from its stored scores.

    Uses the thresholds the row was *recorded at*, not the current ones, so a
    resumed run reproduces the prompt the original run would have produced. Rows
    from before per-class gating carry one threshold, which applied to both.
    """
    tau = row.get("run_time_tau")
    scores = row.get("sentence_scores") or []
    if tau is None or not scores:
        return None
    tau_next_steps = row.get("run_time_tau_next_steps", tau)
    rejected = rejected_texts(
        (
            (
                s["text"],
                s.get("sentence_class", "synthesis"),
                s.get("groundedness") or 0.0,
            )
            for s in scores
        ),
        tau,
        tau_next_steps,
    )
    return revision_hint_for(rejected) if rejected else None


def _record(
    case: PayloadCase, config: NoteConfig, result: AttemptResult
) -> dict[str, Any]:
    """One JSONL row: everything the sweep and the results table need.

    Per-sentence scores are stored in full. The sweep is a pure replay over them,
    so anything omitted here cannot be recovered without paying for the run again.
    """
    outcome = result.outcome
    triad = result.triad
    note = result.note
    return {
        "case_id": case["case_id"],
        "config": config,
        "attempt": result.attempt,
        "modality": case["modality"],
        "grade": case["grade"],
        "confidence": case["confidence"],
        "label_count": len(case["positive_labels"]),
        "retrieval_k": result.retrieval_k,
        "run_time_tau": SYNTHESIS_TAU,
        "run_time_tau_next_steps": NEXT_STEPS_TAU,
        "failure": result.failure,
        "parse_mode": result.parse_mode,
        "passed_at_run_time_tau": bool(outcome and outcome.passed),
        # All three RAG Triad dimensions. context_relevance is deliberately None
        # — not zero — for the ungrounded arm, where no context was supplied and
        # the metric is undefined; zero would read as "retrieval failed".
        "context_relevance": triad.context_relevance if triad else None,
        "answer_relevance": triad.answer_relevance if triad else None,
        "groundedness": outcome.groundedness if outcome else None,
        "synthesis_groundedness": outcome.synthesis_groundedness if outcome else None,
        "next_steps_groundedness": (
            outcome.next_steps_groundedness if outcome else None
        ),
        "citation_agreement_rate": (
            outcome.citation_agreement_rate if outcome else None
        ),
        "sentence_scores": [
            {
                "text": s.text,
                "sentence_class": s.sentence_class,
                "groundedness": s.groundedness,
                "label": s.label,
                "supporting_passage_id": s.supporting_passage_id,
                "cited_marker": s.cited_marker,
            }
            for s in (outcome.sentence_scores if outcome else [])
        ],
        "synthesis": note.synthesis if note else None,
        "next_steps": note.next_steps if note else [],
        "passage_ids": [p.passage_id for p in result.passages],
        "passage_scores": [round(p.score, 6) for p in result.passages],
        "passage_source_types": [p.source_type for p in result.passages],
        "generation_ms": result.generation_ms,
        "gate_ms": result.gate_ms,
        "models": {"llm": OLLAMA_MODEL, "nli": NLI_MODEL},
    }


def run_case(
    case: PayloadCase, config: NoteConfig, resumed: dict[str, dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Generate and score every attempt for one case under one configuration."""
    from rag.retriever import retrieve

    req = FeedbackGenerateRequest(
        modality=case["modality"],  # type: ignore[arg-type]
        grade=case["grade"],
        confidence=case["confidence"],
        positive_labels=case["positive_labels"],  # type: ignore[arg-type]
        patient_context=case["patient_context"],  # type: ignore[arg-type]
        config=config,
        case_id=case["case_id"],
    )
    include_passages = config != "ungrounded"

    passages = to_schema(retrieve(req, k=TOP_K))
    if not passages:
        yield {
            "case_id": case["case_id"],
            "config": config,
            "attempt": 0,
            "failure": "empty_corpus",
            "sentence_scores": [],
        }
        return

    revision: str | None = None
    for attempt in range(MAX_RETRIES + 1):
        plan = plan_attempt(attempt, MAX_RETRIES)
        if plan.retrieval_k != TOP_K:
            passages = to_schema(retrieve(req, k=plan.retrieval_k))

        # A recorded attempt is not regenerated, but the loop must still walk
        # through it to carry the revision hint forward. Skipping straight past
        # it meant a resumed attempt 2 ran with no REVISION block, so resumed
        # rows came from a different prompt than fresh ones and the results file
        # mixed two conditions invisibly. The recorded row supplies the hint.
        recorded = resumed.get(record_key(case["case_id"], config, attempt))
        if recorded is not None:
            revision = revision_hint_from_row(recorded) or revision
            continue

        result = run_attempt(
            req,
            passages,
            attempt,
            include_passages,
            revision_hint=revision if plan.include_revision_hint else None,
        )
        yield _record(case, config, result)

        # The hint for the *next* attempt names what this one got wrong. Computed
        # even when this attempt passed, because later attempts are generated
        # unconditionally and must still receive a realistic prompt.
        if result.outcome and result.outcome.rejected_sentences:
            revision = revision_hint_for(result.outcome.rejected_sentences)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", choices=[*CONFIGS, "all"], default="all", help="which arm to run"
    )
    parser.add_argument("--limit", type=int, help="first N cases only (smoke test)")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore prior results and start over"
    )
    parser.add_argument("--out", type=Path, default=RUNS_PATH)
    args = parser.parse_args(argv)

    from rag import ollama_client

    if not ollama_client.is_available():
        print(
            "Generation host unreachable. Start the dev stub\n"
            "  python -m uvicorn rag.dev.fake_ollama:app --port 11434\n"
            "or point OLLAMA_BASE_URL at a real host."
        )
        return 1

    cases = load_payloads()
    if args.limit:
        cases = cases[: args.limit]
    configs = CONFIGS if args.config == "all" else (args.config,)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    resumed = {} if args.fresh else load_resumed(args.out)
    mode = "w" if args.fresh else "a"

    total = len(cases) * len(configs) * (MAX_RETRIES + 1)
    print(
        f"{len(cases)} cases x {len(configs)} config(s) x "
        f"{MAX_RETRIES + 1} attempts = {total} generations"
    )
    if resumed:
        print(f"resuming: {len(resumed)} already recorded")

    written = 0
    started = time.monotonic()
    with args.out.open(mode, encoding="utf-8", newline="\n") as fh:
        for index, case in enumerate(cases, start=1):
            for config in configs:
                for row in run_case(case, config, resumed):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    # Flushed per row: a rented instance can disappear without
                    # notice, and --resume is only as good as what reached disk.
                    fh.flush()
                    written += 1
            elapsed = time.monotonic() - started
            print(
                f"  [{index}/{len(cases)}] {case['case_id']}  "
                f"{written} rows  {elapsed / 60:.1f} min"
            )

    print(f"\nWrote {written} row(s) to {args.out}")
    print("Next: python -m rag.eval.sweep_tau")
    return 0


if __name__ == "__main__":
    sys.exit(main())
