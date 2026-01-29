"""Compare TinyStories vs OpenWebText tokenizers.

Computes bytes/token (lower is better) for a 2x2 grid:
- TinyStories tokenizer on TinyStories sample
- TinyStories tokenizer on OWT sample
- OWT tokenizer on TinyStories sample
- OWT tokenizer on OWT sample

Example:
  uv run experiments/compare_tokenizers.py --lines 2000

By default, reads tokenizers from outputs/ and corpora from data/.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from cs336_basics.tokenizer import Tokenizer


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


@dataclass(frozen=True)
class TokenizerFiles:
    vocab_json: str
    merges_txt: str


@dataclass(frozen=True)
class CorpusFiles:
    tinystories: str
    owt: str


def _default_tokenizer_files(outputs_dir: str) -> dict[str, TokenizerFiles]:
    return {
        "tinystories": TokenizerFiles(
            vocab_json=os.path.join(outputs_dir, "tinystories_vocab.json"),
            merges_txt=os.path.join(outputs_dir, "tinystories_merges.txt"),
        ),
        "owt": TokenizerFiles(
            vocab_json=os.path.join(outputs_dir, "owt_vocab.json"),
            merges_txt=os.path.join(outputs_dir, "owt_merges.txt"),
        ),
    }


def _default_corpora(data_dir: str) -> CorpusFiles:
    return CorpusFiles(
        tinystories=os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt"),
        owt=os.path.join(data_dir, "owt_valid.txt"),
    )


def _bytes_per_token(
    tokenizer: Tokenizer,
    *,
    corpus_path: str,
    lines: int,
    max_bytes: int | None,
) -> float:
    total_bytes = 0
    total_tokens = 0

    line_limit: int | None = None if lines <= 0 else lines

    with open(corpus_path, "rb") as f:
        for idx, line in enumerate(f):
            if line_limit is not None and idx >= line_limit:
                break
            if max_bytes is not None and total_bytes >= max_bytes:
                break

            total_bytes += len(line)
            # Tokenizer expects text; decode with replacement to preserve length-ish.
            text = line.decode("utf-8", errors="replace")
            token_ids = tokenizer.encode(text)
            total_tokens += len(token_ids)

    if total_tokens == 0:
        raise ValueError("No tokens produced; check corpus/lines.")

    return total_bytes / total_tokens


def _require_file(path: str, desc: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default=os.path.join(root, "outputs"),
        help="Directory containing *_vocab.json and *_merges.txt",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(root, "data"),
        help="Directory containing corpus .txt files",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=2000,
        help="Number of lines to sample from each corpus (0 = all lines)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Optional global byte cap per corpus sample",
    )
    parser.add_argument(
        "--save",
        default=os.path.join(root, "outputs", "compare_tokenizers.txt"),
        help="Optional output file to write the results table",
    )
    args = parser.parse_args()

    tok_files = _default_tokenizer_files(args.outputs_dir)
    corpora = _default_corpora(args.data_dir)

    for name, files in tok_files.items():
        _require_file(files.vocab_json, f"{name} vocab")
        _require_file(files.merges_txt, f"{name} merges")

    _require_file(corpora.tinystories, "TinyStories corpus")
    _require_file(corpora.owt, "OWT corpus")

    ts_tok = Tokenizer.from_files(
        vocab_filepath=tok_files["tinystories"].vocab_json,
        merges_filepath=tok_files["tinystories"].merges_txt,
    )
    owt_tok = Tokenizer.from_files(
        vocab_filepath=tok_files["owt"].vocab_json,
        merges_filepath=tok_files["owt"].merges_txt,
    )

    results = {
        "TS tok on TS": _bytes_per_token(
            ts_tok,
            corpus_path=corpora.tinystories,
            lines=args.lines,
            max_bytes=args.max_bytes,
        ),
        "TS tok on OWT": _bytes_per_token(
            ts_tok, corpus_path=corpora.owt, lines=args.lines, max_bytes=args.max_bytes
        ),
        "OWT tok on TS": _bytes_per_token(
            owt_tok,
            corpus_path=corpora.tinystories,
            lines=args.lines,
            max_bytes=args.max_bytes,
        ),
        "OWT tok on OWT": _bytes_per_token(
            owt_tok, corpus_path=corpora.owt, lines=args.lines, max_bytes=args.max_bytes
        ),
    }

    lines_out = [
        "bytes/token (lower is better)",
        f"sample_lines={args.lines}, max_bytes={args.max_bytes}",
        f"tinystories_corpus={corpora.tinystories}",
        f"owt_corpus={corpora.owt}",
        "",
    ]
    for k, v in results.items():
        lines_out.append(f"{k}: {v:.6f}")

    table = "\n".join(lines_out)
    print(table)

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(table)
            f.write("\n")


if __name__ == "__main__":
    main()
