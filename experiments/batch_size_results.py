"""Summarize TinyStories runs by batch size.

Reads `outputs/runs/*/{run.json,summary.json}` and prints:
- best (lowest) final_valid_loss per batch_size
- which lr_max achieved it

Usage:
  uv run python experiments/batch_size_results.py

This is a helper for writing up Problem (batch_size_experiment).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    runs_dir = Path("outputs/runs")
    if not runs_dir.exists():
        raise SystemExit("outputs/runs not found")

    by_bs: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    diverged: list[tuple[int, float, str, int | None]] = []

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

        bs = run.get("batch_size")
        lr_max = run.get("lr_max")
        if bs is None or lr_max is None:
            continue

        if summary.get("diverged"):
            diverged.append(
                (int(bs), float(lr_max), run_dir.name, summary.get("diverged_step"))
            )
            continue

        final = summary.get("final_valid_loss")
        if final is None:
            continue

        by_bs[int(bs)].append((float(final), float(lr_max), run_dir.name))

    print("Best final_valid_loss per batch_size")
    print("batch_size\tbest_final_valid_loss\tlr_max\trun")
    for bs in sorted(by_bs):
        best = min(by_bs[bs], key=lambda t: t[0])
        final, lr_max, run_name = best
        print(f"{bs}\t{final:.6f}\t{lr_max:.6g}\t{run_name}")

    if diverged:
        print("\nDiverged runs")
        print("batch_size\tlr_max\trun\tdiverged_step")
        for bs, lr, run_name, step in sorted(diverged, key=lambda x: (x[0], x[1])):
            print(f"{bs}\t{lr:.6g}\t{run_name}\t{step}")


if __name__ == "__main__":
    main()
