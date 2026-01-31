"""Summarize TinyStories runs for the RMSNorm ablation.

Filters runs logged by experiments/train_tinystories_lm.py where `use_rmsnorm == False`.
Prints best (lowest) final_valid_loss per lr_max and highlights the overall best.

Usage:
  uv run python experiments/layer_norm_ablation_results.py

Optional:
  uv run python experiments/layer_norm_ablation_results.py --runs-dir outputs/runs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize RMSNorm-ablation runs")
    p.add_argument("--runs-dir", type=str, default="outputs/runs")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    rows: list[tuple[float, float, str, int, int]] = []
    diverged: list[tuple[float, str, int | None]] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_json = run_dir / "run.json"
        summary_json = run_dir / "summary.json"
        if not run_json.exists() or not summary_json.exists():
            continue

        run = _load_json(run_json)
        summary = _load_json(summary_json)

        if run.get("task") != "tinystories_train":
            continue
        if run.get("use_rmsnorm", True) is True:
            continue

        lr_max = run.get("lr_max")
        if lr_max is None:
            continue

        steps = int(run.get("total_steps", 0) or 0)
        bs = int(run.get("batch_size", 0) or 0)

        if summary.get("diverged"):
            diverged.append((float(lr_max), run_dir.name, summary.get("diverged_step")))
            continue

        final = summary.get("final_valid_loss")
        if final is None:
            continue

        rows.append((float(final), float(lr_max), run_dir.name, bs, steps))

    if not rows and not diverged:
        print("No ablation runs found (use_rmsnorm == False).")
        return

    by_lr: dict[float, list[tuple[float, str, int, int]]] = defaultdict(list)
    for final, lr, run_name, bs, steps in rows:
        by_lr[lr].append((final, run_name, bs, steps))

    print("RMSNorm ablation: best final_valid_loss per lr_max")
    print("lr_max\tbest_final_valid_loss\trun\tbatch_size\tsteps")

    best_overall: tuple[float, float, str, int, int] | None = None
    for lr in sorted(by_lr):
        best = min(by_lr[lr], key=lambda t: t[0])
        final, run_name, bs, steps = best
        print(f"{lr:.6g}\t{final:.6f}\t{run_name}\t{bs}\t{steps}")
        cand = (final, lr, run_name, bs, steps)
        if best_overall is None or cand[0] < best_overall[0]:
            best_overall = cand

    if best_overall is not None:
        final, lr, run_name, bs, steps = best_overall
        print("\nOverall best (ablation)")
        print(
            f"final_valid_loss={final:.6f} lr_max={lr:.6g} run={run_name} bs={bs} steps={steps}"
        )

    if diverged:
        print("\nDiverged ablation runs")
        print("lr_max\trun\tdiverged_step")
        for lr, run_name, step in sorted(diverged, key=lambda x: x[0]):
            print(f"{lr:.6g}\t{run_name}\t{step}")


if __name__ == "__main__":
    main()
