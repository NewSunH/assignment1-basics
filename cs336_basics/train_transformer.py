import torch
from torch import nn, Tensor

from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Tensor:
    vocab_dim = -1
    max_logits = logits.max(dim=vocab_dim, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=vocab_dim))
    target_logits = logits.gather(
        dim=vocab_dim,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    loss = -target_logits + max_logits.squeeze(-1) + log_sum_exp
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                param.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                if weight_decay != 0:
                    param.data.add_(param.data, alpha=-lr * weight_decay)

        return loss
