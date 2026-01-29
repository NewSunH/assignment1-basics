"""learning_rate_tuning (SGD toy example).

Runs the toy SGD loop from the assignment with several learning rates and prints
how the loss evolves.

Usage:
  uv run experiments/learning_rate_tuning.py

Notes:
- Uses an in-place SGD update with per-parameter step counter t:
    p -= lr / sqrt(t + 1) * grad
- Runs for 10 iterations as required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ToySGDState:
    t: int = 0


def run(lr: float, *, steps: int, init: torch.Tensor) -> list[float]:
    weights = torch.nn.Parameter(init.clone())
    state = ToySGDState(t=0)

    losses: list[float] = []
    for _ in range(steps):
        if weights.grad is not None:
            weights.grad.zero_()

        loss = (weights**2).mean()
        losses.append(float(loss.detach().cpu().item()))
        loss.backward()

        with torch.no_grad():
            grad = weights.grad
            assert grad is not None
            weights -= (lr / math.sqrt(state.t + 1)) * grad
            state.t += 1

    return losses


def main() -> None:
    torch.manual_seed(0)
    init = 5 * torch.randn((10, 10), dtype=torch.float32)

    steps = 10
    lrs = [1.0, 1e1, 1e2, 1e3]

    print(f"steps={steps}")
    for lr in lrs:
        losses = run(lr, steps=steps, init=init)
        trend = "decrease" if losses[-1] < losses[0] else "increase"
        print(f"\nlr={lr:g} ({trend})")
        for i, v in enumerate(losses):
            print(f"  t={i:02d}: loss={v:.6e}")


if __name__ == "__main__":
    main()
