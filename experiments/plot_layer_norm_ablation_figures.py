"""Generate external learning-curve figures for the RMSNorm ablation.

This avoids pgfplots/tikz inside LaTeX and produces PDF/PNG images that can be
included via \\includegraphics.

Expected inputs are TSV files produced by experiments/metrics_split_to_tsv.py:
- train_loss.tsv columns: step<TAB>loss
- valid_loss.tsv columns: step<TAB>loss

Example:
  uv run python experiments/plot_layer_norm_ablation_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. If you only have metrics.jsonl, run: "
            f"uv run python experiments/metrics_split_to_tsv.py --run-dir {path.parent}"
        )

    # TSV has a header row: step\tloss
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    steps = data[:, 0]
    loss = data[:, 1]
    return steps, loss


def _truncate(
    steps: np.ndarray, loss: np.ndarray, xmax: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = steps <= xmax
    return steps[mask], loss[mask]


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=Path("outputs/runs/20260130T032122Z"),
        help="Baseline run directory (with RMSNorm)",
    )
    parser.add_argument(
        "--diverged-run-dir",
        type=Path,
        default=Path("outputs/runs/layer_norm_ablation_norn_lr3e-3_steps2000_seed0"),
        help="No-RMSNorm run directory at previous optimal LR (diverges)",
    )
    parser.add_argument(
        "--stable-run-dir",
        type=Path,
        default=Path("outputs/runs/layer_norm_ablation_norn_lr1e-3_steps2000_seed0"),
        help="No-RMSNorm run directory at best stable LR",
    )
    parser.add_argument(
        "--xmax",
        type=int,
        default=2000,
        help="Max step to display (baseline is truncated to this for comparison)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("answer/figures"),
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # Lazy import so the script errors clearly if matplotlib isn't available.
    import matplotlib.pyplot as plt

    out_dir: Path = args.out_dir
    xmax: int = args.xmax

    # --- Figure A: diverged run vs baseline (train only) ---
    div_train_steps, div_train_loss = _load_tsv(
        args.diverged_run_dir / "train_loss.tsv"
    )
    base_train_steps, base_train_loss = _load_tsv(
        args.baseline_run_dir / "train_loss.tsv"
    )
    base_train_steps, base_train_loss = _truncate(
        base_train_steps, base_train_loss, xmax=xmax
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(div_train_steps, div_train_loss, label="no RMSNorm (train)")
    ax.plot(base_train_steps, base_train_loss, linestyle="--", label="baseline (train)")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("RMSNorm removed @ lr_max=3e-3 (diverges)")
    fig.tight_layout()
    _save(fig, out_dir / "layer_norm_ablation_lr3e-3")
    plt.close(fig)

    # --- Figure B: stable run vs baseline (train+valid) ---
    stab_train_steps, stab_train_loss = _load_tsv(
        args.stable_run_dir / "train_loss.tsv"
    )
    stab_valid_steps, stab_valid_loss = _load_tsv(
        args.stable_run_dir / "valid_loss.tsv"
    )

    base_valid_steps, base_valid_loss = _load_tsv(
        args.baseline_run_dir / "valid_loss.tsv"
    )
    base_train_steps, base_train_loss = _truncate(
        base_train_steps, base_train_loss, xmax=xmax
    )
    base_valid_steps, base_valid_loss = _truncate(
        base_valid_steps, base_valid_loss, xmax=xmax
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(stab_train_steps, stab_train_loss, label="no RMSNorm (train)")
    ax.plot(stab_valid_steps, stab_valid_loss, label="no RMSNorm (valid)")
    ax.plot(base_train_steps, base_train_loss, linestyle="--", label="baseline (train)")
    ax.plot(base_valid_steps, base_valid_loss, linestyle="--", label="baseline (valid)")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("RMSNorm removed @ lr_max=1e-3 (stable)")
    fig.tight_layout()
    _save(fig, out_dir / "layer_norm_ablation_lr1e-3")
    plt.close(fig)

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
