"""Plot SwiGLU vs SiLU FFN learning curves and save to answer/figures.

Inputs are TSV files produced by experiments/metrics_split_to_tsv.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1]


def _truncate(
    steps: np.ndarray, loss: np.ndarray, xmax: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = steps <= xmax
    return steps[mask], loss[mask]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--swiglu-run-dir",
        type=Path,
        required=True,
        help="Run directory with ffn_variant=swiglu",
    )
    p.add_argument(
        "--silu-run-dir",
        type=Path,
        required=True,
        help="Run directory with ffn_variant=silu",
    )
    p.add_argument("--xmax", type=int, default=2000)
    p.add_argument("--out-dir", type=Path, default=Path("answer/figures"))
    args = p.parse_args()

    import matplotlib.pyplot as plt

    xmax = int(args.xmax)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sw_train_s, sw_train_l = _load_tsv(args.swiglu_run_dir / "train_loss.tsv")
    sw_valid_s, sw_valid_l = _load_tsv(args.swiglu_run_dir / "valid_loss.tsv")
    si_train_s, si_train_l = _load_tsv(args.silu_run_dir / "train_loss.tsv")
    si_valid_s, si_valid_l = _load_tsv(args.silu_run_dir / "valid_loss.tsv")

    sw_train_s, sw_train_l = _truncate(sw_train_s, sw_train_l, xmax)
    sw_valid_s, sw_valid_l = _truncate(sw_valid_s, sw_valid_l, xmax)
    si_train_s, si_train_l = _truncate(si_train_s, si_train_l, xmax)
    si_valid_s, si_valid_l = _truncate(si_valid_s, si_valid_l, xmax)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(sw_train_s, sw_train_l, linestyle="--", label="SwiGLU (train)")
    ax.plot(sw_valid_s, sw_valid_l, linestyle="--", label="SwiGLU (valid)")
    ax.plot(si_train_s, si_train_l, label="SiLU (train)")
    ax.plot(si_valid_s, si_valid_l, label="SiLU (valid)")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("SwiGLU vs SiLU (matched params)")
    fig.tight_layout()

    base = out_dir / "swiglu_ablation_swiglu_vs_silu"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {base}.pdf/.png")


if __name__ == "__main__":
    main()
