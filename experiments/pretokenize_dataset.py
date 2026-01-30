"""Pretokenize a text dataset into a flat token-id binary file.

This is intentionally dependency-light (no datasets library). It is designed to
make TinyStories training fast by avoiding tokenization in the training loop.

Typical usage (TinyStories):

  uv run experiments/pretokenize_dataset.py \
    --vocab outputs/tinystories_vocab.json \
    --merges outputs/tinystories_merges.txt \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output outputs/tinystories_train_tokens.uint16 \
    --add-eos

Then the training script can memory-map the output and sample windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _encode_one(tokenizer: Tokenizer, text: str, eos_id: int | None) -> list[int]:
    ids = tokenizer.encode(text)
    if eos_id is not None:
        ids.append(int(eos_id))
    return ids


def pretokenize_file(
    *,
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    dtype: np.dtype,
    add_eos: bool,
    every_n_lines_flush: int = 128,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite to avoid accidental mixing.
    if output_path.exists():
        output_path.unlink()

    eos_id: int | None = None
    if add_eos:
        eos_ids = tokenizer.encode("<|endoftext|>")
        if len(eos_ids) != 1:
            raise RuntimeError("Expected <|endoftext|> to tokenize to a single id.")
        eos_id = int(eos_ids[0])

    n_lines = 0
    n_tokens = 0

    buffered: list[int] = []
    with (
        open(input_path, "r", encoding="utf-8", errors="replace") as f_in,
        open(output_path, "ab") as f_out,
    ):
        for line in f_in:
            # Keep line endings out; we add our own EOS if requested.
            line = line.rstrip("\n")
            ids = _encode_one(tokenizer, line, eos_id)
            buffered.extend(ids)
            n_lines += 1
            n_tokens += len(ids)

            if n_lines % every_n_lines_flush == 0 and buffered:
                arr = np.asarray(buffered, dtype=dtype)
                f_out.write(arr.tobytes(order="C"))
                buffered.clear()

        if buffered:
            arr = np.asarray(buffered, dtype=dtype)
            f_out.write(arr.tobytes(order="C"))
            buffered.clear()

    meta = {
        "input": str(input_path),
        "output": str(output_path),
        "dtype": str(np.dtype(dtype)),
        "add_eos": bool(add_eos),
        "lines": int(n_lines),
        "tokens": int(n_tokens),
        "vocab_size": int(len(tokenizer.vocab)),
    }
    with open(
        output_path.with_suffix(output_path.suffix + ".json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pretokenize a text file to a flat token-id file."
    )
    p.add_argument("--vocab", required=True, help="Path to vocab.json")
    p.add_argument("--merges", required=True, help="Path to merges.txt")
    p.add_argument("--input", required=True, help="Input UTF-8 text file")
    p.add_argument(
        "--output", required=True, help="Output token file (.uint16 recommended)"
    )
    p.add_argument(
        "--dtype",
        default="uint16",
        choices=["uint16", "int32"],
        help="Output dtype for token IDs.",
    )
    p.add_argument(
        "--add-eos",
        action="store_true",
        help="Append <|endoftext|> after each input line.",
    )
    args = p.parse_args()

    dtype = np.uint16 if args.dtype == "uint16" else np.int32

    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=["<|endoftext|>"],
    )

    meta = pretokenize_file(
        tokenizer=tokenizer,
        input_path=Path(args.input),
        output_path=Path(args.output),
        dtype=np.dtype(dtype),
        add_eos=bool(args.add_eos),
    )

    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
