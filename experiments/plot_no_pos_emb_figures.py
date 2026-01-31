"""Plot RoPE vs NoPE learning curves and save to answer/figures.

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
        "--rope-run-dir", type=Path, default=Path("outputs/runs/20260130T032122Z")
    )
    p.add_argument(
        "--nope-run-dir",
        type=Path,
        required=True,
        help="A run directory with use_rope=false (NoPE)",
    )
    p.add_argument("--xmax", type=int, default=2000)
    p.add_argument("--out-dir", type=Path, default=Path("answer/figures"))
    args = p.parse_args()

    import matplotlib.pyplot as plt

    xmax = int(args.xmax)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rope_train_s, rope_train_l = _load_tsv(args.rope_run_dir / "train_loss.tsv")
    rope_valid_s, rope_valid_l = _load_tsv(args.rope_run_dir / "valid_loss.tsv")
    rope_train_s, rope_train_l = _truncate(rope_train_s, rope_train_l, xmax)
    rope_valid_s, rope_valid_l = _truncate(rope_valid_s, rope_valid_l, xmax)

    nope_train_s, nope_train_l = _load_tsv(args.nope_run_dir / "train_loss.tsv")
    nope_valid_s, nope_valid_l = _load_tsv(args.nope_run_dir / "valid_loss.tsv")
    nope_train_s, nope_train_l = _truncate(nope_train_s, nope_train_l, xmax)
    nope_valid_s, nope_valid_l = _truncate(nope_valid_s, nope_valid_l, xmax)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(rope_train_s, rope_train_l, linestyle="--", label="RoPE (train)")
    ax.plot(rope_valid_s, rope_valid_l, linestyle="--", label="RoPE (valid)")
    ax.plot(nope_train_s, nope_train_l, label="NoPE (train)")
    ax.plot(nope_valid_s, nope_valid_l, label="NoPE (valid)")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("RoPE vs NoPE")
    fig.tight_layout()

    base = out_dir / "no_pos_emb_rope_vs_nope"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {base}.pdf/.png")


if __name__ == "__main__":
    main()
