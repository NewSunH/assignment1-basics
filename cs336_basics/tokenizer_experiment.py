import argparse
import os
import time

from cs336_basics.bpe import BpeTokenizer


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def _default_paths() -> dict[str, str]:
    root = _repo_root()
    return {
        "tinystories": os.path.join(root, "data", "TinyStoriesV2-GPT4-train.txt"),
        "owt": os.path.join(root, "data", "owt_train.txt"),
    }


def _print_longest_token(tokenizer: BpeTokenizer) -> None:
    # tokenizer.vocab: dict[int, bytes]
    vocab_items = list(tokenizer.vocab.items())
    if not vocab_items:
        print("Vocab is empty.")
        return

    longest_id, longest_bytes = max(vocab_items, key=lambda kv: len(kv[1]))
    longest_text = longest_bytes.decode("utf-8", errors="replace")
    print(
        "Longest token (overall): "
        f"id={longest_id}, bytes_len={len(longest_bytes)}, text={longest_text!r}"
    )

    special_bytes = {
        s.encode("utf-8") for s in getattr(tokenizer, "special_tokens", [])
    }
    non_special = [(i, b) for i, b in vocab_items if b not in special_bytes]
    if non_special:
        ns_id, ns_bytes = max(non_special, key=lambda kv: len(kv[1]))
        ns_text = ns_bytes.decode("utf-8", errors="replace")
        print(
            "Longest token (non-special): "
            f"id={ns_id}, bytes_len={len(ns_bytes)}, text={ns_text!r}"
        )


def train_bpe(
    *,
    corpus_path: str,
    vocab_size: int,
    jobs: int,
    out_prefix: str,
    special_tokens: list[str] | None = None,
    progress: bool = True,
) -> BpeTokenizer:
    tokenizer = BpeTokenizer(
        special_tokens=special_tokens or ["<|endoftext|>"],
        vocab_size=vocab_size,
    )

    with open(corpus_path, "rb") as f:
        tokenizer.pretokenize_jobs = jobs
        tokenizer.show_progress = True
        t0 = time.perf_counter()
        tokenizer.train_from_file(f, progress=progress)
        elapsed = time.perf_counter() - t0

    mins = int(elapsed // 60)
    secs = elapsed - 60 * mins
    print(f"Training finished in {elapsed:.2f}s ({mins:d}m{secs:05.2f}s)")

    out_dir = os.path.join(_repo_root(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_merges_txt(os.path.join(out_dir, f"{out_prefix}_merges.txt"))
    tokenizer.save_vocab_json(os.path.join(out_dir, f"{out_prefix}_vocab.json"))

    _print_longest_token(tokenizer)

    return tokenizer


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on TinyStories or OpenWebText (OWT)."
    )
    parser.add_argument(
        "--corpus",
        choices=["owt", "tinystories"],
        default="owt",
        help="Which built-in corpus path to use.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Override corpus path (defaults to the repo's data/* files).",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Pretokenization worker processes (higher = faster but heavier).",
    )
    args = parser.parse_args()

    corpus_path = args.path or defaults[args.corpus]
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. "
            "Use --path to provide a valid file path."
        )

    out_prefix = "owt" if args.corpus == "owt" else "tinystories"
    print(f"Training BPE on {args.corpus}: {corpus_path}")

    train_bpe(
        corpus_path=corpus_path,
        vocab_size=args.vocab_size,
        jobs=max(1, int(args.jobs)),
        out_prefix=out_prefix,
    )


if __name__ == "__main__":
    main()
