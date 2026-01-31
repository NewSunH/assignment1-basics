"""Extract simple train/valid loss TSVs from metrics.jsonl.

Writes:
  - <run_dir>/train_loss.tsv  (columns: step, loss)
  - <run_dir>/valid_loss.tsv  (columns: step, loss)

This keeps plotting (e.g. with pgfplots) trivial.

Usage:
  uv run python experiments/metrics_split_to_tsv.py outputs/runs/<RUN>/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write(path: Path, rows: list[tuple[int, float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("step\tloss\n")
        for step, loss in rows:
            f.write(f"{step}\t{loss}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Split metrics.jsonl into train/valid TSVs")
    p.add_argument("metrics", type=str, help="Path to metrics.jsonl")
    args = p.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"metrics.jsonl not found: {metrics_path}")

    run_dir = metrics_path.parent
    train_rows: list[tuple[int, float]] = []
    valid_rows: list[tuple[int, float]] = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            split = rec.get("split")
            step = rec.get("step")
            loss = rec.get("loss")
            if step is None or loss is None:
                continue
            if split == "train":
                train_rows.append((int(step), float(loss)))
            elif split == "valid":
                valid_rows.append((int(step), float(loss)))

    out_train = run_dir / "train_loss.tsv"
    out_valid = run_dir / "valid_loss.tsv"
    _write(out_train, train_rows)
    _write(out_valid, valid_rows)

    print(out_train)
    print(out_valid)


if __name__ == "__main__":
    main()
