from __future__ import annotations

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int

from .attention import MultipleHeadSelfAttention as Mhsa
from .attention import softmax
from .normalization import RmsNorm
from .positionwise_feedforward import FFN, silu
from .embedding import Embedding
from .linear import Linear


from cs336_basics import linear


class TransformerBlock(nn.Module):
    mhsa: Mhsa
    rmsnorm1: RmsNorm
    rmsnorm2: RmsNorm
    ffn: FFN

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mhsa = Mhsa(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.rmsnorm1 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = FFN(
            d_model=d_model, d_ff=d_ff, activation=silu, device=device, dtype=dtype
        )

    def forward(
        self, x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        y = x + self.mhsa.forward(self.rmsnorm1.forward(x), rope=True)
        z = y + self.ffn.forward(self.rmsnorm2.forward(y))

        return z


class TransformerLm(nn.Module):
    embedding: Embedding
    transformer_blocks: nn.ModuleList
    norm: RmsNorm
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, token: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        data = self.embedding.forward(token_ids=token)
        for block in self.transformer_blocks:
            assert isinstance(block, TransformerBlock), "幽默ModuleList不是Generic[T]"
            data = block.forward(data)
        data = self.norm.forward(data)
        logits = self.lm_head.forward(data)
        return logits
