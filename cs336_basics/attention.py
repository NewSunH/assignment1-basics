from __future__ import annotations


import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from einops import einsum


def softmax(v: Float[Tensor, "... d"], i: int) -> Float[Tensor, "..."]:
    v_max = torch.max(v, dim=i, keepdim=True).values
    v_stable = v - v_max
    exp_v = torch.exp(v_stable)

    return exp_v / exp_v.sum(dim=i, keepdim=True)


def scaled_dot_product_attention(
    keys: Float[Tensor, "batch_size ... seq_len d_key"],
    queries: Float[Tensor, "batch_size ... seq_len d_key"],
    value: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_key = queries.shape[-1]
    scores = (
        einsum(
            queries,
            keys,
            "batch_size ... q d, batch_size ... k d -> batch_size ... q k",
        )
    ) / math.sqrt(d_key)
    if mask is not None:
        # mask == False -> -inf
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    return softmax(scores, -1) @ value
