"""Summarize NoPE (no RoPE) TinyStories runs.

Looks for run.json with use_rope=false.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runs_root = Path("outputs/runs")
    if not runs_root.exists():
        raise SystemExit("outputs/runs not found")

    rows: list[tuple[float, str, bool, int | None, float | None]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_json = run_dir / "run.json"
        summary_json = run_dir / "summary.json"
        if not run_json.exists() or not summary_json.exists():
            continue

        run_cfg = _load_json(run_json)
        if run_cfg.get("task") != "tinystories_train":
            continue
        if bool(run_cfg.get("use_rope", True)) is True:
            continue

        summary = _load_json(summary_json)
        lr = float(run_cfg.get("lr_max"))
        rows.append(
            (
                lr,
                str(run_dir),
                bool(summary.get("diverged", False)),
                summary.get("diverged_step"),
                summary.get("final_valid_loss"),
            )
        )

    if not rows:
        print("No NoPE runs found (use_rope=false).")
        return

    rows.sort(key=lambda r: r[0])
    print("NoPE runs (use_rope=false):")
    for lr, run_dir, diverged, diverged_step, final_valid_loss in rows:
        if diverged:
            print(f"  lr_max={lr:g}  DIVERGED at step={diverged_step}  run={run_dir}")
        else:
            print(
                f"  lr_max={lr:g}  final_valid_loss={final_valid_loss:.6f}  run={run_dir}"
            )

    stable = [r for r in rows if not r[2] and r[4] is not None]
    if stable:
        best = min(stable, key=lambda r: float(r[4]))
        print(
            f"Best NoPE stable: lr_max={best[0]:g} final_valid_loss={best[4]:.6f} run={best[1]}"
        )


if __name__ == "__main__":
    main()
