"""Convert one or more ExperimentLogger metrics.jsonl files to TSV.

This is meant to produce files that are easy to plot with pgfplots in LaTeX.

Example:
  uv run experiments/metrics_to_tsv.py outputs/runs/20250101T000000Z/metrics.jsonl

Outputs:
  - <run_dir>/metrics.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_one(metrics_path: Path) -> Path:
    run_dir = metrics_path.parent
    out_path = run_dir / "metrics.tsv"

    # Stable column order for plotting.
    columns = [
        "step",
        "wallclock_s",
        "split",
        "loss",
        "lr",
        "grad_norm",
        "iter_s",
    ]

    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = {c: rec.get(c, "") for c in columns}
            rows.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Convert metrics.jsonl to metrics.tsv")
    p.add_argument("metrics", nargs="+", help="Path(s) to metrics.jsonl")
    args = p.parse_args()

    for m in args.metrics:
        out = convert_one(Path(m))
        print(out)


if __name__ == "__main__":
    main()
