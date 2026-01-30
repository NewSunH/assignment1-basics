"""Learning-rate sweep for TinyStories (Problem: learning_rate).

Runs multiple training jobs (one per LR) and saves their learning curves.

By default it runs sequentially in the current process (backend=local). If you
have Slurm, you can also use submitit (backend=slurm).

Core deliverables:
- learning curves for multiple learning rates (and at least one divergent run)
- final losses / divergence notes
- a run reaching per-token TinyStories val loss <= 1.45

See also: experiments/train_tinystories_lm.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cs336_basics.experiment_logging import make_run_dir

from experiments.train_tinystories_lm import TrainConfig, train_one_run


def _parse_lrs(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--lrs must be a comma-separated list")
    return [float(p) for p in parts]


def _write_sweep_summary(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": rows}, f, indent=2, sort_keys=True)
        f.write("\n")


def run_local(
    *, base_cfg: TrainConfig, lrs: list[float], name_prefix: str
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for lr in lrs:
        run_name = f"{name_prefix}_lr{lr:g}_seed{base_cfg.seed}"
        run_dir = make_run_dir(name=run_name)
        cfg = TrainConfig(**{**asdict(base_cfg), "lr_max": float(lr)})
        summary = train_one_run(cfg=cfg, run_dir=run_dir)
        summary = {"lr": float(lr), **summary}
        summaries.append(summary)
    return summaries


def run_slurm(
    *,
    base_cfg: TrainConfig,
    lrs: list[float],
    name_prefix: str,
    slurm_folder: str,
    slurm_partition: str | None,
    timeout_min: int,
    cpus_per_task: int,
    mem_gb: int,
) -> list[dict[str, Any]]:
    import submitit

    folder = Path(slurm_folder)
    folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(folder))
    params: dict[str, Any] = {
        "timeout_min": int(timeout_min),
        "cpus_per_task": int(cpus_per_task),
        "mem_gb": int(mem_gb),
        "gpus_per_node": 1,
        "tasks_per_node": 1,
    }
    if slurm_partition:
        params["slurm_partition"] = str(slurm_partition)

    executor.update_parameters(**params)

    jobs = []
    for lr in lrs:
        run_name = f"{name_prefix}_lr{lr:g}_seed{base_cfg.seed}"
        run_dir = make_run_dir(name=run_name)
        cfg = TrainConfig(**{**asdict(base_cfg), "lr_max": float(lr)})
        job = executor.submit(train_one_run, cfg=cfg, run_dir=run_dir)
        jobs.append((lr, run_dir, job))

    summaries: list[dict[str, Any]] = []
    for lr, run_dir, job in jobs:
        summary = job.result()
        summaries.append(
            {"lr": float(lr), **summary, "job_id": job.job_id, "run_dir": str(run_dir)}
        )
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(description="TinyStories learning-rate sweep")
    p.add_argument("--train-tokens", required=True)
    p.add_argument("--valid-tokens", required=True)
    p.add_argument(
        "--lrs",
        required=True,
        help="Comma-separated learning rates, e.g. 1e-4,2e-4,3e-4,5e-4,1e-3",
    )
    p.add_argument("--name-prefix", default="learning_rate")

    p.add_argument("--backend", choices=["local", "slurm"], default="local")
    p.add_argument("--slurm-folder", default="outputs/submitit")
    p.add_argument("--slurm-partition", default=None)
    p.add_argument("--timeout-min", type=int, default=120)
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--mem-gb", type=int, default=32)

    # training hyperparams
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)

    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)

    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=200)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--compile", action="store_true")
    p.add_argument("--ckpt-interval", type=int, default=1000)

    p.add_argument("--summary-out", default="outputs/learning_rate_sweep_summary.json")

    args = p.parse_args()

    lrs = _parse_lrs(args.lrs)

    base_cfg = TrainConfig(
        train_tokens_path=str(args.train_tokens),
        valid_tokens_path=str(args.valid_tokens),
        context_length=int(args.context_length),
        batch_size=int(args.batch_size),
        total_steps=int(args.steps),
        lr_max=float(lrs[0]),  # overridden per-run
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
        compile=bool(args.compile),
        checkpoint_interval=int(args.ckpt_interval),
    )

    if args.backend == "local":
        summaries = run_local(
            base_cfg=base_cfg, lrs=lrs, name_prefix=str(args.name_prefix)
        )
    else:
        summaries = run_slurm(
            base_cfg=base_cfg,
            lrs=lrs,
            name_prefix=str(args.name_prefix),
            slurm_folder=str(args.slurm_folder),
            slurm_partition=args.slurm_partition,
            timeout_min=int(args.timeout_min),
            cpus_per_task=int(args.cpus_per_task),
            mem_gb=int(args.mem_gb),
        )

    out_path = Path(args.summary_out)
    _write_sweep_summary(out_path, summaries)
    print(out_path)


if __name__ == "__main__":
    main()
