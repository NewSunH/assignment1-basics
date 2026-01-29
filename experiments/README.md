# experiments/

这个目录用于放置“可复现的实验脚本”（训练 tokenizer、做对比、跑 profiling 等），避免把一次性脚本混进 `cs336_basics/` 的核心库实现。

## Tokenizer / BPE

- 训练 BPE（输出到 `outputs/`）：
  - `uv run experiments/tokenizer_experiment.py --corpus tinystories --vocab-size 32000 --jobs 8`
  - `uv run experiments/tokenizer_experiment.py --corpus owt --vocab-size 32000 --jobs 8`

- TinyStories vs OWT 对比（bytes/token 2×2）：
  - `uv run experiments/compare_tokenizers.py --lines 2000`
  - 结果默认写到 `outputs/compare_tokenizers.txt`

## Notes

- `cs336_basics/tokenizer_experiment.py` 目前是一个兼容 wrapper，会转发到这里的脚本入口。
