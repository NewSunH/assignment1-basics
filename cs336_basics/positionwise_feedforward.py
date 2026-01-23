import torch
from torch import nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
from typing import Callable
import math


def silu(x: Float) -> Float:
    return x * torch.sigmoid(x)


class FFN(nn.Module):
    weight1: Float[Tensor, "d_ff d_model"]
    weight2: Float[Tensor, "d_model d_ff"]
    weight3: Float[Tensor, "d_ff d_model"]
    activation: Callable[[Float], Float]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        activation: Callable[[Float], Float],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff == None:
            raw_d_ff = int(8 * d_model / 3)
            d_ff = ((raw_d_ff + 63) // 64) * 64

        sigma = math.sqrt(2 / (d_ff + d_model))
        self.weight1 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        self.weight2 = nn.Parameter(
            torch.empty(d_model, d_ff, device=device, dtype=dtype)
        )
        self.weight3 = nn.Parameter(
            torch.empty(d_ff, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(
            self.weight1, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.weight2, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        nn.init.trunc_normal_(
            self.weight3, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        self.activation = activation

    def forward(self, x: Float[Tensor, "d_model"]) -> Float[Tensor, "d_model"]:
        u = x @ self.weight1.T
        v = x @ self.weight3.T
        gate = self.activation(u)
        h = gate * v
        y = h @ self.weight2.T
        return y
