"""Generate text from a trained TinyStories checkpoint.

Usage (GPU if available):
  uv run python experiments/generate_text.py \
    --run-dir outputs/runs/20260130T032122Z \
    --prompt "Once upon a time, " \
    --max-new-tokens 256 \
    --temperature 0.9 \
    --top-p 0.95

This script prints the generated text and also saves it to
`<run_dir>/generated_text.txt` for easy copy/paste into the writeup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLm


def _find_eos_id(tok: Tokenizer, eos_token: str) -> int | None:
    target = eos_token.encode("utf-8")
    for token_id, b in tok.vocab.items():
        if b == target:
            return int(token_id)
    return None


def _load_checkpoint_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        # fall back: some checkpoints might save raw state_dict
        return ckpt
    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")


@torch.no_grad()
def _generate_with_optional_ban_eos(
    *,
    model: TransformerLm,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_token_id: int | None,
    ban_eos: bool,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return input_ids

    generated = input_ids
    for _ in range(int(max_new_tokens)):
        idx_cond = generated[:, -model.context_length :]
        logits = model(idx_cond)[:, -1, :]

        if ban_eos and eos_token_id is not None:
            logits[:, int(eos_token_id)] = -torch.inf

        if temperature is None or float(temperature) <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / float(temperature)

            if top_k is not None:
                k = min(int(top_k), logits.shape[-1])
                topk_vals, _ = torch.topk(logits, k, dim=-1)
                kth = topk_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < kth, torch.full_like(logits, -torch.inf), logits
                )

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)

                to_remove = cumprobs > float(top_p)
                to_remove[..., 0] = False
                sorted_logits = torch.where(
                    to_remove,
                    torch.full_like(sorted_logits, -torch.inf),
                    sorted_logits,
                )
                logits = torch.full_like(logits, -torch.inf).scatter(
                    -1, sorted_idx, sorted_logits
                )

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token.to(dtype=torch.long)], dim=-1)

        if (not ban_eos) and eos_token_id is not None:
            if bool((next_token.squeeze(-1) == int(eos_token_id)).all()):
                break

    return generated


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate text from a TinyStories Transformer LM checkpoint."
    )
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (default: <run_dir>/checkpoints/final.pt)",
    )
    p.add_argument("--vocab", type=str, default="outputs/tinystories_vocab.json")
    p.add_argument("--merges", type=str, default="outputs/tinystories_merges.txt")
    p.add_argument("--eos-token", type=str, default="<|endoftext|>")
    p.add_argument("--prompt", type=str, default="Once upon a time, ")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--ban-eos",
        action="store_true",
        help="If set, forbids sampling the <|endoftext|> token so generation doesn't stop immediately.",
    )
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise SystemExit(f"run.json not found in {run_dir}")

    run_cfg = json.loads(run_json.read_text())

    device = (
        torch.device("cuda")
        if (args.device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )
    if args.device == "cuda":
        device = torch.device("cuda")
    if args.device == "cpu":
        device = torch.device("cpu")

    torch.manual_seed(int(args.seed))

    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[args.eos_token])
    eos_id = _find_eos_id(tok, args.eos_token)

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else (run_dir / "checkpoints" / "final.pt")
    )
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # Build model from the run config.
    model = TransformerLm(
        vocab_size=int(run_cfg["vocab_size"]),
        context_length=int(run_cfg["context_length"]),
        d_model=int(run_cfg["d_model"]),
        num_layers=int(run_cfg["num_layers"]),
        num_heads=int(run_cfg["num_heads"]),
        d_ff=int(run_cfg["d_ff"]),
        rope_theta=float(run_cfg["rope_theta"]),
        use_rmsnorm=bool(run_cfg.get("use_rmsnorm", True)),
        use_rope=bool(run_cfg.get("use_rope", True)),
        ffn_variant=str(run_cfg.get("ffn_variant", "swiglu")),
        device=device,
        dtype=(
            torch.bfloat16
            if (device.type == "cuda" and run_cfg.get("dtype") == "bf16")
            else torch.float32
        ),
    )

    state_dict = _load_checkpoint_state_dict(ckpt_path, device=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    prompt_ids = tok.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    top_k = int(args.top_k) if int(args.top_k) > 0 else None
    top_p = float(args.top_p) if args.top_p is not None else None

    y = _generate_with_optional_ban_eos(
        model=model,
        input_ids=x,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_id,
        ban_eos=bool(args.ban_eos),
    )

    out_ids = y[0].tolist()
    text = tok.decode(out_ids)

    out_file = run_dir / "generated_text.txt"
    out_file.write_text(text, encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"checkpoint={ckpt_path}")
    print(f"device={device}")
    print(f"prompt_tokens={len(prompt_ids)}")
    print(f"total_tokens={len(out_ids)}")
    if eos_id is not None:
        print(f"eos_token_id={eos_id}")
    print(f"ban_eos={bool(args.ban_eos)}")
    print("--- GENERATED TEXT START ---")
    print(text)
    print("--- GENERATED TEXT END ---")
    print(f"saved_to={out_file}")


if __name__ == "__main__":
    main()
