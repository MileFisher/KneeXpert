"""Replay the ablation at every threshold pair, offline.

    python -m rag.eval.sweep_tau
    python -m rag.eval.sweep_tau --no-plot

Pure replay over the stored per-sentence scores: no model is loaded, nothing is
regenerated, no network is touched. That is the point — a threshold chosen by
re-running inference at each candidate value would cost hours of GPU time and
still only answer for the attempts that happened to be generated.

The gate has two thresholds (decision 17): one for the synthesis, one for the
next steps. Two outputs follow from that:

* ``tau_sweep.csv`` — the diagonal, both thresholds equal. The one-number curve.
* ``tau_grid.csv`` — the whole (synthesis, next-steps) plane. Its ``tau_next_steps
  = 0`` column is option (a), gating the synthesis alone, so one run compares
  every policy that was on the table.

This is what makes the operating point defensible rather than fitted. The
provisional default was set from two hand-written examples; a value read off
this grid is read from the whole payload set.

One limit to state alongside the numbers: pass/fail replays exactly, but the
retry prompts do not. Attempts 2 and 3 were prompted with the sentences rejected
at the run-time thresholds, so a cell far from those thresholds describes
attempts generated under slightly different feedback than it would have given.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rag.config import EVAL_RESULTS_DIR
from rag.gate import passes

RUNS_PATH = EVAL_RESULTS_DIR / "runs.jsonl"
SWEEP_CSV = EVAL_RESULTS_DIR / "tau_sweep.csv"
SWEEP_PNG = EVAL_RESULTS_DIR / "tau_sweep.png"
GRID_CSV = EVAL_RESULTS_DIR / "tau_grid.csv"
GRID_PNG = EVAL_RESULTS_DIR / "tau_grid.png"


def _steps(lo: float, hi: float, step: float) -> list[float]:
    return [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]


GRID_SYNTHESIS_TAUS = _steps(0.50, 0.95, 0.05)
# Down to zero, because zero is option (a): any recommendation passes.
GRID_NEXT_STEPS_TAUS = _steps(0.00, 0.95, 0.05)


@dataclass(frozen=True)
class SweepRow:
    """System behaviour at one candidate threshold pair."""

    tau: float
    """The synthesis threshold."""
    next_steps_tau: float
    served: int
    withheld: int
    cases: int
    mean_attempts: float
    mean_served_groundedness: float | None
    regenerated: int
    """Cases that needed more than one attempt. Counted, not inferred."""
    first_failed_synthesis: int
    """Cases whose first attempt had a synthesis below its threshold."""
    first_failed_next_steps: int
    """Cases whose first attempt had next steps below theirs.

    Counted independently of the synthesis, so the two answer "what does the gate
    fire on" without one hiding the other.
    """

    @property
    def serve_rate(self) -> float:
        return self.served / self.cases if self.cases else 0.0

    @property
    def withhold_rate(self) -> float:
        return self.withheld / self.cases if self.cases else 0.0

    @property
    def regeneration_rate(self) -> float:
        """Share of cases that needed more than one attempt.

        Counted directly. Deriving it from the mean attempt count measured a
        different quantity and understated it by up to the retry budget.
        """
        return self.regenerated / self.cases if self.cases else 0.0


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


def row_passes(row: dict, tau: float, next_steps_tau: float) -> bool:
    """Whether a stored attempt clears both thresholds.

    Delegates to the live gate's own rule, so a replayed verdict cannot drift
    from a production one. A failed attempt carries no per-class scores and so
    never passes.
    """
    return passes(
        row.get("synthesis_groundedness"),
        row.get("next_steps_groundedness"),
        tau,
        next_steps_tau,
    )


def replay_config(
    rows: list[dict],
    config: str,
    tau: float,
    next_steps_tau: float | None = None,
) -> SweepRow:
    """What the gated pipeline would have done at this threshold pair.

    Walks each case's attempts in order and serves the first that clears both
    thresholds — exactly the production policy, applied to scores that were
    already computed. ``next_steps_tau`` defaults to ``tau``.
    """
    steps_tau = tau if next_steps_tau is None else next_steps_tau
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("config") != config:
            continue
        by_case.setdefault(row["case_id"], []).append(row)

    served = 0
    regenerated = 0
    failed_synthesis = 0
    failed_next_steps = 0
    attempts_used: list[int] = []
    served_scores: list[float] = []

    for attempts in by_case.values():
        ordered = sorted(attempts, key=lambda r: r["attempt"])

        first = ordered[0]
        synthesis = first.get("synthesis_groundedness")
        next_steps = first.get("next_steps_groundedness")
        if synthesis is None or synthesis < tau:
            failed_synthesis += 1
        if next_steps is not None and next_steps < steps_tau:
            failed_next_steps += 1

        used = 0
        accepted: dict | None = None
        for row in ordered:
            used += 1
            if row_passes(row, tau, steps_tau):
                accepted = row
                break
        attempts_used.append(used)
        if used > 1:
            regenerated += 1
        if accepted is not None:
            served += 1
            if accepted.get("groundedness") is not None:
                served_scores.append(float(accepted["groundedness"]))

    cases = len(by_case)
    return SweepRow(
        tau=tau,
        next_steps_tau=steps_tau,
        served=served,
        withheld=cases - served,
        cases=cases,
        mean_attempts=sum(attempts_used) / len(attempts_used) if attempts_used else 0.0,
        mean_served_groundedness=(
            sum(served_scores) / len(served_scores) if served_scores else None
        ),
        regenerated=regenerated,
        first_failed_synthesis=failed_synthesis,
        first_failed_next_steps=failed_next_steps,
    )


def sweep(
    rows: list[dict],
    config: str,
    lo: float = 0.50,
    hi: float = 0.95,
    step: float = 0.01,
) -> list[SweepRow]:
    """The diagonal: both thresholds moved together."""
    return [replay_config(rows, config, tau) for tau in _steps(lo, hi, step)]


def sweep_grid(
    rows: list[dict],
    config: str,
    synthesis_taus: list[float] = GRID_SYNTHESIS_TAUS,
    next_steps_taus: list[float] = GRID_NEXT_STEPS_TAUS,
) -> list[SweepRow]:
    """Every (synthesis, next-steps) pair. Cheap: it is arithmetic over rows."""
    return [
        replay_config(rows, config, tau, steps_tau)
        for tau in synthesis_taus
        for steps_tau in next_steps_taus
    ]


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

    def percentile(key: str, fraction: float) -> float | None:
        values = sorted(r[key] for r in first if r.get(key) is not None)
        if not values:
            return None
        index = min(len(values) - 1, int(round(fraction * (len(values) - 1))))
        return float(values[index])

    failures = [r for r in first if r.get("failure")]
    prose = [r for r in first if r.get("parse_mode") == "prose_fallback"]
    return {
        "cases": len(first),
        # All three RAG Triad dimensions. context_relevance is None for the
        # ungrounded arm, where it is undefined rather than zero.
        "context_relevance": mean("context_relevance"),
        "answer_relevance": mean("answer_relevance"),
        "groundedness": mean("groundedness"),
        "synthesis_groundedness": mean("synthesis_groundedness"),
        "next_steps_groundedness": mean("next_steps_groundedness"),
        "citation_agreement_rate": mean("citation_agreement_rate"),
        "generation_ms": mean("generation_ms"),
        "gate_ms": mean("gate_ms"),
        "generation_ms_p50": percentile("generation_ms", 0.50),
        "generation_ms_p95": percentile("generation_ms", 0.95),
        "gate_ms_p95": percentile("gate_ms", 0.95),
        "failure_rate": len(failures) / len(first),
        "prose_fallback_rate": len(prose) / len(first),
    }


_CSV_HEADER = [
    "config",
    "tau",
    "tau_next_steps",
    "cases",
    "served",
    "withheld",
    "serve_rate",
    "withhold_rate",
    "mean_attempts",
    "regeneration_rate",
    "mean_served_groundedness",
    "first_failed_synthesis",
    "first_failed_next_steps",
]


def _csv_row(config: str, row: SweepRow) -> list[object]:
    return [
        config,
        f"{row.tau:.2f}",
        f"{row.next_steps_tau:.2f}",
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
        row.first_failed_synthesis,
        row.first_failed_next_steps,
    ]


def write_csv(sweeps: dict[str, list[SweepRow]], path: Path = SWEEP_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for config, series in sweeps.items():
            for row in series:
                writer.writerow(_csv_row(config, row))


def write_plot(sweeps: dict[str, list[SweepRow]], path: Path = SWEEP_PNG) -> bool:
    """Plot serve rate against the shared threshold. False if matplotlib is absent."""
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
    axis.set_xlabel("groundedness threshold, both parts (tau)")
    axis.set_ylabel("share of cases served")
    axis.set_title("Notes served vs groundedness threshold")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def write_grid_plot(grid: list[SweepRow], path: Path = GRID_PNG) -> bool:
    """Heatmap of serve rate over the threshold plane. False without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    synthesis_taus = sorted({r.tau for r in grid})
    steps_taus = sorted({r.next_steps_tau for r in grid})
    cell = {(r.tau, r.next_steps_tau): r.serve_rate for r in grid}
    matrix = [[cell[(s, n)] for s in synthesis_taus] for n in steps_taus]

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        extent=(
            synthesis_taus[0] - 0.025,
            synthesis_taus[-1] + 0.025,
            steps_taus[0] - 0.025,
            steps_taus[-1] + 0.025,
        ),
    )
    lo = max(synthesis_taus[0], steps_taus[0])
    hi = min(synthesis_taus[-1], steps_taus[-1])
    axis.plot([lo, hi], [lo, hi], color="white", linewidth=1, linestyle="--")
    axis.set_xlabel("synthesis threshold")
    axis.set_ylabel("next-steps threshold (0 = synthesis-only gating)")
    axis.set_title("Share of cases served (rag_gate)")
    figure.colorbar(image, ax=axis, label="serve rate")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _on_grid(value: float, every: float) -> bool:
    scaled = value / every
    return abs(scaled - round(scaled)) < 1e-6


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
    print(f"\nDiagonal sweep written to {SWEEP_CSV}")

    grids = {config: sweep_grid(rows, config) for config in configs}
    with GRID_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for config, grid in grids.items():
            for row in grid:
                writer.writerow(_csv_row(config, row))
    print(f"Threshold grid written to {GRID_CSV}")

    gated = sweeps.get("rag_gate")
    if gated:
        print("\nDiagonal sweep (rag_gate, both thresholds equal):")
        print(
            f"  {'tau':>5s} {'served':>7s} {'withheld':>9s} {'attempts':>9s} "
            f"{'synth-fail':>11s} {'steps-fail':>11s}"
        )
        for row in gated:
            if _on_grid(row.tau, 0.05):
                print(
                    f"  {row.tau:>5.2f} {row.serve_rate:>6.1%} "
                    f"{row.withhold_rate:>8.1%} {row.mean_attempts:>9.2f} "
                    f"{row.first_failed_synthesis:>11d} "
                    f"{row.first_failed_next_steps:>11d}"
                )

    grid = grids.get("rag_gate")
    if grid:
        columns = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        cell = {(r.tau, r.next_steps_tau): r for r in grid}
        print("\nServe rate over the threshold plane (rag_gate):")
        print("  rows: synthesis threshold; columns: next-steps threshold")
        print("  column 0.00 is option (a), synthesis-only gating")
        print("  " + " " * 6 + "".join(f"{c:>7.2f}" for c in columns))
        for tau in GRID_SYNTHESIS_TAUS:
            if not _on_grid(tau, 0.10):
                continue
            cells = [cell.get((tau, c)) for c in columns]
            print(
                f"  {tau:>5.2f} "
                + "".join(
                    f"{c.serve_rate:>7.1%}" if c else f"{'':>7s}" for c in cells
                )
            )

    if not args.no_plot:
        if write_plot(sweeps):
            print(f"\nPlot written to {SWEEP_PNG}")
            if grid and write_grid_plot(grid):
                print(f"Heatmap written to {GRID_PNG}")
        else:
            print("\nmatplotlib not installed; skipped the plots (CSVs still written).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
