import os
import time

from bpe import BpeTokenizer


def tinystory(path, jobs):
    tokenizer = BpeTokenizer(special_tokens=["<|endoftext|>"], vocab_size=10000)
    with open(
        path,
        "rb",
    ) as f:
        tokenizer.pretokenize_jobs = jobs
        tokenizer.show_progress = True
        t0 = time.perf_counter()
        tokenizer.train_from_file(f, progress=True)
        elapsed = time.perf_counter() - t0

    mins = int(elapsed // 60)
    secs = elapsed - 60 * mins
    print(f"Training finished in {elapsed:.2f}s ({mins:d}m{secs:05.2f}s)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_merges_txt(os.path.join(out_dir, "tinystories_merges.txt"))
    tokenizer.save_vocab_json(os.path.join(out_dir, "tinystories_vocab.json"))


tinystory(
    path="/home/huang/Documents/learning/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    jobs=16,
)
