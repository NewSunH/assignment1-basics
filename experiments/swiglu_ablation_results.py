"""Summarize SwiGLU vs SiLU FFN ablation runs.

Looks for run.json with ffn_variant in {swiglu, silu}.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    runs_root = Path("outputs/runs")
    if not runs_root.exists():
        raise SystemExit("outputs/runs not found")

    # variant -> lr -> list[entry]
    buckets: dict[
        str, dict[float, list[tuple[str, bool, int | None, float | None]]]
    ] = defaultdict(lambda: defaultdict(list))

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

        variant = str(run_cfg.get("ffn_variant", "swiglu"))
        if variant not in {"swiglu", "silu"}:
            continue

        lr = float(run_cfg.get("lr_max"))
        summary = _load_json(summary_json)
        buckets[variant][lr].append(
            (
                str(run_dir),
                bool(summary.get("diverged", False)),
                summary.get("diverged_step"),
                summary.get("final_valid_loss"),
            )
        )

    if not buckets:
        print("No swiglu/silu runs found.")
        return

    for variant in ("swiglu", "silu"):
        if variant not in buckets:
            continue
        print(f"FFN variant: {variant}")
        for lr in sorted(buckets[variant].keys()):
            entries = buckets[variant][lr]
            # choose best stable entry for this lr
            stable = [e for e in entries if (not e[1]) and (e[3] is not None)]
            if stable:
                best = min(stable, key=lambda e: float(e[3]))
                print(
                    f"  lr_max={lr:g}  best_final_valid_loss={float(best[3]):.6f}  run={best[0]}"
                )
            else:
                # show first divergence
                d = next((e for e in entries if e[1]), None)
                if d is None:
                    print(f"  lr_max={lr:g}  no stable results")
                else:
                    print(f"  lr_max={lr:g}  DIVERGED at step={d[2]}  run={d[0]}")

        # overall best stable
        all_stable: list[tuple[float, str, float]] = []
        for lr, entries in buckets[variant].items():
            for run_dir, diverged, _, final_valid_loss in entries:
                if diverged or final_valid_loss is None:
                    continue
                all_stable.append((float(lr), str(run_dir), float(final_valid_loss)))
        if all_stable:
            lr_best, run_best, loss_best = min(all_stable, key=lambda t: t[2])
            print(
                f"Best {variant} stable: lr_max={lr_best:g} final_valid_loss={loss_best:.6f} run={run_best}"
            )
        print()


if __name__ == "__main__":
    main()
