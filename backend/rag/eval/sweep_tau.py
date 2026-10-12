"""Replay the ablation at every threshold, offline.

    python -m rag.eval.sweep_tau
    python -m rag.eval.sweep_tau --no-plot

Pure replay over the stored per-sentence scores: no model is loaded, nothing is
regenerated, no network is touched. That is the point — a threshold chosen by
re-running inference at each candidate value would cost hours of GPU time and
still only answer for the attempts that happened to be generated.

This is what makes the operating point defensible rather than fitted. The
provisional default was set from two hand-written examples; a value read off this
curve is read from the whole payload set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rag.config import EVAL_RESULTS_DIR, MAX_RETRIES

RUNS_PATH = EVAL_RESULTS_DIR / "runs.jsonl"
SWEEP_CSV = EVAL_RESULTS_DIR / "tau_sweep.csv"
SWEEP_PNG = EVAL_RESULTS_DIR / "tau_sweep.png"


@dataclass(frozen=True)
class SweepRow:
    """System behaviour at one candidate threshold."""

    tau: float
    served: int
    withheld: int
    cases: int
    mean_attempts: float
    mean_served_groundedness: float | None

    @property
    def serve_rate(self) -> float:
        return self.served / self.cases if self.cases else 0.0

    @property
    def withhold_rate(self) -> float:
        return self.withheld / self.cases if self.cases else 0.0

    @property
    def regeneration_rate(self) -> float:
        """Share of cases that needed more than one attempt."""
        if not MAX_RETRIES:
            return 0.0
        return max(0.0, (self.mean_attempts - 1.0) / MAX_RETRIES)


def load_runs(path: Path = RUNS_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No results at {path}. Run `python -m rag.eval.run_ablation` first."
        )
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def replay_config(rows: list[dict], config: str, tau: float) -> SweepRow:
    """What the gated pipeline would have done at ``tau``.

    Walks each case's attempts in order and serves the first whose stored
    groundedness clears ``tau`` — exactly the production policy, applied to
    scores that were already computed.
    """
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("config") != config:
            continue
        by_case.setdefault(row["case_id"], []).append(row)

    served = 0
    attempts_used: list[int] = []
    served_scores: list[float] = []

    for attempts in by_case.values():
        ordered = sorted(attempts, key=lambda r: r["attempt"])
        used = 0
        accepted: dict | None = None
        for row in ordered:
            used += 1
            score = row.get("groundedness")
            if score is not None and score >= tau:
                accepted = row
                break
        attempts_used.append(used)
        if accepted is not None:
            served += 1
            served_scores.append(float(accepted["groundedness"]))

    cases = len(by_case)
    return SweepRow(
        tau=tau,
        served=served,
        withheld=cases - served,
        cases=cases,
        mean_attempts=sum(attempts_used) / len(attempts_used) if attempts_used else 0.0,
        mean_served_groundedness=(
            sum(served_scores) / len(served_scores) if served_scores else None
        ),
    )


def sweep(
    rows: list[dict],
    config: str,
    lo: float = 0.50,
    hi: float = 0.95,
    step: float = 0.01,
) -> list[SweepRow]:
    taus = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    return [replay_config(rows, config, tau) for tau in taus]


def config_summary(rows: list[dict], config: str) -> dict[str, float | int | None]:
    """Per-configuration figures for the results table, at first attempt only.

    First attempt, because the retry schedule is part of the gated pipeline
    rather than a property of the configuration being compared.
    """
    first = [
        r for r in rows if r.get("config") == config and r.get("attempt") == 0
    ]
    if not first:
        return {"cases": 0}

    def mean(key: str) -> float | None:
        values = [r[key] for r in first if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    failures = [r for r in first if r.get("failure")]
    prose = [r for r in first if r.get("parse_mode") == "prose_fallback"]
    return {
        "cases": len(first),
        "groundedness": mean("groundedness"),
        "synthesis_groundedness": mean("synthesis_groundedness"),
        "next_steps_groundedness": mean("next_steps_groundedness"),
        "citation_agreement_rate": mean("citation_agreement_rate"),
        "generation_ms": mean("generation_ms"),
        "gate_ms": mean("gate_ms"),
        "failure_rate": len(failures) / len(first),
        "prose_fallback_rate": len(prose) / len(first),
    }


def write_csv(sweeps: dict[str, list[SweepRow]], path: Path = SWEEP_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "config",
                "tau",
                "cases",
                "served",
                "withheld",
                "serve_rate",
                "withhold_rate",
                "mean_attempts",
                "regeneration_rate",
                "mean_served_groundedness",
            ]
        )
        for config, series in sweeps.items():
            for row in series:
                writer.writerow(
                    [
                        config,
                        f"{row.tau:.2f}",
                        row.cases,
                        row.served,
                        row.withheld,
                        f"{row.serve_rate:.4f}",
                        f"{row.withhold_rate:.4f}",
                        f"{row.mean_attempts:.3f}",
                        f"{row.regeneration_rate:.4f}",
                        ""
                        if row.mean_served_groundedness is None
                        else f"{row.mean_served_groundedness:.4f}",
                    ]
                )


def write_plot(sweeps: dict[str, list[SweepRow]], path: Path = SWEEP_PNG) -> bool:
    """Plot serve rate against threshold. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for config, series in sweeps.items():
        axis.plot(
            [r.tau for r in series],
            [r.serve_rate for r in series],
            label=config,
            linewidth=1.8,
        )
    axis.set_xlabel("groundedness threshold (tau)")
    axis.set_ylabel("share of cases served")
    axis.set_title("Notes served vs groundedness threshold")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_PATH)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = load_runs(args.runs)
    except FileNotFoundError as e:
        print(e)
        return 1

    configs = sorted({r["config"] for r in rows if r.get("config")})
    print(f"{len(rows)} rows across {len(configs)} config(s): {', '.join(configs)}\n")

    print("Per-configuration means (first attempt):")
    print(
        f"  {'config':14s} {'cases':>5s} {'ground':>7s} "
        f"{'synth':>7s} {'steps':>7s} {'cite':>6s} {'prose':>6s}"
    )
    for config in configs:
        s = config_summary(rows, config)
        print(
            f"  {config:14s} {s['cases']:>5} "
            f"{_fmt(s.get('groundedness')):>7s} "
            f"{_fmt(s.get('synthesis_groundedness')):>7s} "
            f"{_fmt(s.get('next_steps_groundedness')):>7s} "
            f"{_fmt(s.get('citation_agreement_rate'), 2):>6s} "
            f"{_fmt(s.get('prose_fallback_rate'), 2):>6s}"
        )

    sweeps = {config: sweep(rows, config) for config in configs}
    write_csv(sweeps)
    print(f"\nSweep written to {SWEEP_CSV}")

    gated = sweeps.get("rag_gate")
    if gated:
        print("\nThreshold sweep (rag_gate):")
        print(f"  {'tau':>5s} {'served':>7s} {'withheld':>9s} {'attempts':>9s}")
        for row in gated:
            if abs(row.tau * 100 - round(row.tau * 100)) < 1e-6 and (
                round(row.tau * 100) % 5 == 0
            ):
                print(
                    f"  {row.tau:>5.2f} {row.serve_rate:>6.1%} "
                    f"{row.withhold_rate:>8.1%} {row.mean_attempts:>9.2f}"
                )

    if not args.no_plot:
        if write_plot(sweeps):
            print(f"\nPlot written to {SWEEP_PNG}")
        else:
            print("\nmatplotlib not installed; skipped the plot (CSV still written).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
