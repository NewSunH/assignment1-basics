"""Train the 17M-ish TinyStories Transformer LM.

This is a *reproducible experiment script*, not part of the graded library API.
It uses your from-scratch model/optimizer implementations in `cs336_basics/`.

Defaults match the PDF guidance:
- vocab_size: 10_000 (use your TinyStories BPE)
- context_length: 256
- d_model: 512
- d_ff: 1344
- layers: 4
- heads: 16
- rope_theta: 10000
- tokens processed: batch_size * steps * context_length ~= 327,680,000
  (defaults: batch=128, steps=10_000, ctx=256)

It logs train/val curves to `outputs/runs/<name>/metrics.jsonl`.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import typing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.experiment_logging import ExperimentLogger, make_run_dir
from cs336_basics.train_transformer import (
    AdamW,
    cross_entropy,
    data_loading,
    gradient_clipping,
    learning_rate_schedule,
    save_checkpoint,
)
from cs336_basics.transformer import TransformerLm


@dataclass(frozen=True)
class TrainConfig:
    train_tokens_path: str
    valid_tokens_path: str

    vocab_size: int = 10_000
    context_length: int = 256

    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10000.0

    use_rmsnorm: bool = True
    use_rope: bool = True
    ffn_variant: str = "swiglu"  # swiglu|silu

    batch_size: int = 128
    total_steps: int = 10_000

    lr_max: float = 3e-4
    lr_min_ratio: float = 0.1
    warmup_steps: int = 200

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.1

    grad_clip_norm: float = 1.0

    log_interval: int = 10
    eval_interval: int = 200
    eval_batches: int = 200

    seed: int = 0
    dtype: str = "bf16"  # bf16|fp32

    compile: bool = False
    checkpoint_interval: int = 1000


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _torch_dtype(dtype: str, device: torch.device) -> torch.dtype:
    dtype = dtype.lower().strip()
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        # bfloat16 makes sense on H100; on CPU you may prefer fp32.
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    raise ValueError("dtype must be bf16 or fp32")


def _mmap_tokens(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Token file not found: {path}")

    # Infer dtype from filename convention.
    if p.name.endswith(".uint16"):
        dtype = np.uint16
    else:
        dtype = np.int32

    arr = np.memmap(p, mode="r", dtype=dtype)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


@torch.no_grad()
def estimate_loss(
    *,
    model: TransformerLm,
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(int(eval_batches)):
        x, y = data_loading(
            tokens, batch_size=batch_size, context_length=context_length, device=device
        )
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def train_one_run(*, cfg: TrainConfig, run_dir: Path) -> dict[str, Any]:
    device = _device()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    train_tokens = _mmap_tokens(cfg.train_tokens_path)
    valid_tokens = _mmap_tokens(cfg.valid_tokens_path)

    model_dtype = _torch_dtype(cfg.dtype, device)
    model = TransformerLm(
        vocab_size=int(cfg.vocab_size),
        context_length=int(cfg.context_length),
        d_model=int(cfg.d_model),
        num_layers=int(cfg.num_layers),
        num_heads=int(cfg.num_heads),
        d_ff=int(cfg.d_ff),
        rope_theta=float(cfg.rope_theta),
        use_rmsnorm=bool(cfg.use_rmsnorm),
        use_rope=bool(cfg.use_rope),
        ffn_variant=str(cfg.ffn_variant),
        device=device,
        dtype=model_dtype,
    )

    if cfg.compile and device.type == "cuda":
        model = typing.cast(TransformerLm, torch.compile(model))

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.lr_max),
        betas=tuple(map(float, cfg.betas)),
        eps=float(cfg.eps),
        weight_decay=float(cfg.weight_decay),
    )

    logger = ExperimentLogger(run_dir)
    logger.save_run_config(
        {"task": "tinystories_train", **asdict(cfg), "device": str(device)}
    )
    logger.note(
        f"tokens_processed_target={cfg.batch_size * cfg.total_steps * cfg.context_length:,d}"
    )

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lr_min = float(cfg.lr_max) * float(cfg.lr_min_ratio)

    def set_lr(step: int) -> float:
        lr = learning_rate_schedule(
            step,
            lr_max=float(cfg.lr_max),
            lr_min=lr_min,
            warmup_iters=int(cfg.warmup_steps),
            cosine_cycle_iters=int(cfg.total_steps),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
        return float(lr)

    t0 = time.perf_counter()
    last_log = time.perf_counter()

    diverged = False
    diverged_step: int | None = None

    model.train()
    for step in range(1, int(cfg.total_steps) + 1):
        lr = set_lr(step)

        x, y = data_loading(
            train_tokens,
            batch_size=int(cfg.batch_size),
            context_length=int(cfg.context_length),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits, y)

        if not torch.isfinite(loss):
            diverged = True
            diverged_step = int(step)
            logger.note(f"DIVERGED: loss={loss.item()} at step={step}")
            break

        loss.backward()
        grad_norm = float(
            gradient_clipping(model.parameters(), max_norm=float(cfg.grad_clip_norm))
        )
        optimizer.step()

        if step % int(cfg.log_interval) == 0:
            now = time.perf_counter()
            it_s = (now - last_log) / max(1, int(cfg.log_interval))
            last_log = now
            logger.log(
                step=int(step),
                split="train",
                loss=float(loss.item()),
                lr=float(lr),
                grad_norm=float(grad_norm),
                iter_s=float(it_s),
            )

        if step % int(cfg.eval_interval) == 0:
            val = estimate_loss(
                model=model,
                tokens=valid_tokens,
                batch_size=int(cfg.batch_size),
                context_length=int(cfg.context_length),
                eval_batches=int(cfg.eval_batches),
                device=device,
            )
            logger.log(step=int(step), split="valid", loss=float(val), lr=float(lr))
            model.train()

        if step % int(cfg.checkpoint_interval) == 0:
            save_checkpoint(
                model, optimizer, int(step), ckpt_dir / f"step_{step:07d}.pt"
            )

    # final eval (if not diverged)
    final_val = math.nan
    if not diverged:
        final_val = estimate_loss(
            model=model,
            tokens=valid_tokens,
            batch_size=int(cfg.batch_size),
            context_length=int(cfg.context_length),
            eval_batches=int(cfg.eval_batches),
            device=device,
        )
        logger.log(
            step=int(step), split="valid", loss=float(final_val), lr=float(set_lr(step))
        )

        save_checkpoint(model, optimizer, int(step), ckpt_dir / "final.pt")

    elapsed_s = time.perf_counter() - t0
    summary = {
        "run_dir": str(run_dir),
        "diverged": bool(diverged),
        "diverged_step": diverged_step,
        "final_valid_loss": float(final_val) if not math.isnan(final_val) else None,
        "elapsed_s": float(elapsed_s),
        "tokens_processed": int(cfg.batch_size)
        * int(min(step, cfg.total_steps))
        * int(cfg.context_length),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train the TinyStories 17M Transformer LM.")
    p.add_argument(
        "--train-tokens",
        required=True,
        help="Flat token file from pretokenize_dataset.py",
    )
    p.add_argument(
        "--valid-tokens",
        required=True,
        help="Flat token file from pretokenize_dataset.py",
    )

    p.add_argument(
        "--run-name", default=None, help="Run directory name under outputs/runs/"
    )

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)

    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--no-rmsnorm",
        action="store_true",
        help="If set, removes all RMSNorm layers (ablation).",
    )

    p.add_argument(
        "--no-rope",
        action="store_true",
        help="If set, disables RoPE (NoPE ablation: removes positional information).",
    )

    p.add_argument(
        "--ffn-variant",
        choices=["swiglu", "silu"],
        default="swiglu",
        help="FFN variant to use: swiglu (default) or silu (non-gated ablation).",
    )

    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument(
        "--compile", action="store_true", help="Use torch.compile (CUDA only)."
    )
    p.add_argument("--ckpt-interval", type=int, default=1000)

    args = p.parse_args()

    cfg = TrainConfig(
        train_tokens_path=str(args.train_tokens),
        valid_tokens_path=str(args.valid_tokens),
        context_length=int(args.context_length),
        batch_size=int(args.batch_size),
        total_steps=int(args.steps),
        lr_max=float(args.lr),
        lr_min_ratio=float(args.min_lr_ratio),
        warmup_steps=int(args.warmup_steps),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.eps),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip),
        log_interval=int(args.log_interval),
        eval_interval=int(args.eval_interval),
        eval_batches=int(args.eval_batches),
        seed=int(args.seed),
        dtype=str(args.dtype),
        use_rmsnorm=not bool(args.no_rmsnorm),
        use_rope=not bool(args.no_rope),
        ffn_variant=str(args.ffn_variant),
        compile=bool(args.compile),
        checkpoint_interval=int(args.ckpt_interval),
    )

    run_dir = make_run_dir(name=args.run_name)
    summary = train_one_run(cfg=cfg, run_dir=run_dir)
    print(summary)


if __name__ == "__main__":
    main()
