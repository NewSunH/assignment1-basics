from torch import nn, Tensor
from jaxtyping import Float

from cs336_basics import rope
from .attention import MultipleHeadSelfAttention as Mhsa
from .normalization import RmsNorm
from .positionwise_feedforward import FFN, silu


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.mhsa = Mhsa(
            d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len
        )
        self.rmsnorm1 = RmsNorm(d_model=d_model)
        self.rmsnorm2 = RmsNorm(d_model=d_model)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, activation=silu)

    def forward(
        self, x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        y = x + self.mhsa.forward(self.rmsnorm1.forward(x), rope=True)
        z = y + self.ffn.forward(self.rmsnorm2.forward(y))

        return z
