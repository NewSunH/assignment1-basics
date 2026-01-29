"""Tokenizer experiments for CS336 assignment.

Implements the deliverables for `tokenizer_experiments`:
(a) Sample 10 documents from TinyStories and OpenWebText, compute bytes/token.
(b) Tokenize OWT sample with TinyStories tokenizer and compare.
(c) Estimate throughput (bytes/s) and extrapolate time for 825GB (Pile).
(d) (Optional) Encode train/dev datasets into uint16 token id arrays.

Typical usage:
  uv run experiments/tokenizer_experiments.py --sample-docs 10 --seed 0
  uv run experiments/tokenizer_experiments.py --sample-docs 10 --seed 0 --throughput-bytes 50000000

To encode datasets (can be slow; two-pass writes .npy with exact length):
  uv run experiments/tokenizer_experiments.py --encode tinystories --splits train valid
  uv run experiments/tokenizer_experiments.py --encode owt --splits valid
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(__file__))


@dataclass(frozen=True)
class TokenizerPaths:
    vocab: str
    merges: str


@dataclass(frozen=True)
class CorpusPaths:
    train: str
    valid: str


def _tokenizer_paths(outputs_dir: str) -> dict[str, TokenizerPaths]:
    return {
        "tinystories": TokenizerPaths(
            vocab=os.path.join(outputs_dir, "tinystories_vocab.json"),
            merges=os.path.join(outputs_dir, "tinystories_merges.txt"),
        ),
        "owt": TokenizerPaths(
            vocab=os.path.join(outputs_dir, "owt_vocab.json"),
            merges=os.path.join(outputs_dir, "owt_merges.txt"),
        ),
    }


def _corpus_paths(data_dir: str) -> dict[str, CorpusPaths]:
    return {
        "tinystories": CorpusPaths(
            train=os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt"),
            valid=os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt"),
        ),
        "owt": CorpusPaths(
            train=os.path.join(data_dir, "owt_train.txt"),
            valid=os.path.join(data_dir, "owt_valid.txt"),
        ),
    }


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def _reservoir_sample_lines(path: str, *, k: int, rng: random.Random) -> list[bytes]:
    sample: list[bytes] = []
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i < k:
                sample.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    sample[j] = line
    if len(sample) < k:
        raise ValueError(f"File has only {len(sample)} lines, need {k}: {path}")
    return sample


def _bytes_per_token(tokenizer: Tokenizer, docs: list[bytes]) -> float:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc)
        text = doc.decode("utf-8", errors="replace")
        total_tokens += len(tokenizer.encode(text))
    if total_tokens == 0:
        raise ValueError("No tokens produced")
    return total_bytes / total_tokens


def _measure_throughput(
    tokenizer: Tokenizer,
    *,
    corpus_path: str,
    target_bytes: int,
) -> float:
    processed_bytes = 0
    processed_tokens = 0

    t0 = time.perf_counter()
    with open(corpus_path, "rb") as f:
        for line in f:
            processed_bytes += len(line)
            text = line.decode("utf-8", errors="replace")
            processed_tokens += len(tokenizer.encode(text))
            if processed_bytes >= target_bytes:
                break
    dt = max(1e-9, time.perf_counter() - t0)

    # Return bytes/sec; also sanity-check tokens > 0
    if processed_tokens == 0:
        raise ValueError("No tokens produced during throughput run")
    return processed_bytes / dt


def _encode_to_uint16_npy(
    tokenizer: Tokenizer,
    *,
    input_path: str,
    output_path: str,
    two_pass: bool,
) -> None:
    # uint16 is only valid if vocab ids fit
    max_id = max(tokenizer.vocab.keys())
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"Vocab id {max_id} does not fit uint16")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def iter_ids():
        with open(input_path, "rb") as f:
            for line in f:
                text = line.decode("utf-8", errors="replace")
                for tid in tokenizer.encode(text):
                    yield tid

    if two_pass:
        # Pass 1: count
        count = 0
        for _ in iter_ids():
            count += 1
        arr = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.uint16, shape=(count,)
        )
        # Pass 2: fill
        idx = 0
        for tid in iter_ids():
            arr[idx] = tid
            idx += 1
        arr.flush()
    else:
        # One-pass: write to .bin; but we keep the API consistent and still write .npy by buffering chunks.
        # (This may use more RAM; keep chunk size moderate.)
        chunk: list[int] = []
        chunks: list[np.ndarray] = []
        chunk_size = 1_000_000
        for tid in iter_ids():
            chunk.append(tid)
            if len(chunk) >= chunk_size:
                chunks.append(np.asarray(chunk, dtype=np.uint16))
                chunk.clear()
        if chunk:
            chunks.append(np.asarray(chunk, dtype=np.uint16))
        arr = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.uint16)
        np.save(output_path, arr)


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=os.path.join(root, "outputs"))
    parser.add_argument("--data-dir", default=os.path.join(root, "data"))
    parser.add_argument("--sample-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--throughput-bytes",
        type=int,
        default=50_000_000,
        help="Bytes to process for throughput estimate (default 50MB). Set 0 to skip.",
    )
    parser.add_argument(
        "--encode",
        choices=["tinystories", "owt"],
        default=None,
        help="Optionally encode a dataset into uint16 token ids.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid"],
        choices=["train", "valid"],
        help="Which splits to encode when using --encode.",
    )
    parser.add_argument(
        "--tokenized-dir",
        default=os.path.join(root, "outputs", "tokenized"),
        help="Where to write encoded .npy arrays.",
    )
    parser.add_argument(
        "--two-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Two-pass exact-length .npy writing (slower but memory-safe).",
    )
    args = parser.parse_args()

    tok_paths = _tokenizer_paths(args.outputs_dir)
    corp_paths = _corpus_paths(args.data_dir)

    for name, p in tok_paths.items():
        _require(p.vocab)
        _require(p.merges)

    for name, p in corp_paths.items():
        _require(p.train)
        _require(p.valid)

    ts_tok = Tokenizer.from_files(
        vocab_filepath=tok_paths["tinystories"].vocab,
        merges_filepath=tok_paths["tinystories"].merges,
    )
    owt_tok = Tokenizer.from_files(
        vocab_filepath=tok_paths["owt"].vocab,
        merges_filepath=tok_paths["owt"].merges,
    )

    rng = random.Random(args.seed)
    ts_docs = _reservoir_sample_lines(
        corp_paths["tinystories"].valid, k=args.sample_docs, rng=rng
    )
    owt_docs = _reservoir_sample_lines(
        corp_paths["owt"].valid, k=args.sample_docs, rng=rng
    )

    # (a) compression on in-domain samples
    ts_on_ts = _bytes_per_token(ts_tok, ts_docs)
    owt_on_owt = _bytes_per_token(owt_tok, owt_docs)

    # (b) cross-domain: TS tokenizer on OWT sample
    ts_on_owt = _bytes_per_token(ts_tok, owt_docs)

    print("=== (a)/(b) bytes/token on 10-doc samples (lower is better) ===")
    print(f"TS tokenizer on TinyStories(valid) sample: {ts_on_ts:.6f}")
    print(f"OWT tokenizer on OWT(valid) sample:       {owt_on_owt:.6f}")
    print(f"TS tokenizer on OWT(valid) sample:        {ts_on_owt:.6f}")

    # (c) throughput
    if args.throughput_bytes and args.throughput_bytes > 0:
        bps_ts = _measure_throughput(
            ts_tok,
            corpus_path=corp_paths["tinystories"].valid,
            target_bytes=args.throughput_bytes,
        )
        bps_owt = _measure_throughput(
            owt_tok,
            corpus_path=corp_paths["owt"].valid,
            target_bytes=args.throughput_bytes,
        )
        pile_bytes = 825 * (1024**3)
        pile_seconds = pile_bytes / bps_owt
        pile_hours = pile_seconds / 3600
        print("\n=== (c) throughput estimate ===")
        print(f"TS tokenizer throughput (on TinyStories valid): {bps_ts:,.0f} bytes/s")
        print(f"OWT tokenizer throughput (on OWT valid):       {bps_owt:,.0f} bytes/s")
        print(f"Estimated time for Pile (825GB) at OWT rate:   {pile_hours:,.2f} hours")

    # (d) optional encoding
    if args.encode is not None:
        ds = args.encode
        tok = ts_tok if ds == "tinystories" else owt_tok
        cp = corp_paths[ds]

        print("\n=== (d) encoding datasets to uint16 token ids ===")
        for split in args.splits:
            in_path = getattr(cp, split)
            out_path = os.path.join(args.tokenized_dir, f"{ds}_{split}.npy")
            print(
                f"Encoding {ds}/{split}: {in_path} -> {out_path} (two_pass={args.two_pass})"
            )
            _encode_to_uint16_npy(
                tok, input_path=in_path, output_path=out_path, two_pass=args.two_pass
            )
            print("Done")


if __name__ == "__main__":
    main()
