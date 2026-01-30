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

## TinyStories training / learning rate sweep

- 预分词（建议先做一次，训练就不会被 tokenizer 卡住）：
  - 训练集：
    - `uv run experiments/pretokenize_dataset.py --vocab outputs/tinystories_vocab.json --merges outputs/tinystories_merges.txt --input data/TinyStoriesV2-GPT4-train.txt --output outputs/tinystories_train_tokens.uint16 --add-eos`
  - 验证集：
    - `uv run experiments/pretokenize_dataset.py --vocab outputs/tinystories_vocab.json --merges outputs/tinystories_merges.txt --input data/TinyStoriesV2-GPT4-valid.txt --output outputs/tinystories_valid_tokens.uint16 --add-eos`

- 单次训练（默认 batch=128, steps=10000, ctx=256，对应 327,680,000 tokens）：
  - `uv run experiments/train_tinystories_lm.py --train-tokens outputs/tinystories_train_tokens.uint16 --valid-tokens outputs/tinystories_valid_tokens.uint16 --lr 3e-4`

- 学习率 sweep（本地顺序跑或用 submitit/slurm 并行）：
  - 本地顺序：
    - `uv run experiments/learning_rate.py --train-tokens outputs/tinystories_train_tokens.uint16 --valid-tokens outputs/tinystories_valid_tokens.uint16 --lrs 1e-4,2e-4,3e-4,5e-4,8e-4,1e-3 --backend local`
  - Slurm（需要你补 `--slurm-partition` 等参数）：
    - `uv run experiments/learning_rate.py --train-tokens ... --valid-tokens ... --lrs 1e-4,2e-4,3e-4,5e-4 --backend slurm --slurm-partition <partition>`

- 导出 TSV（方便用 pgfplots 画曲线）：
  - `uv run experiments/metrics_to_tsv.py outputs/runs/<run>/metrics.jsonl`

## Slurm / cs336dev（8×5090）建议跑法

你现在的训练脚本是“单卡训练”的（每个 run 申请 1 张 GPU）。集群有 8 张卡时，最推荐的做法是：

- **lr sweep 用数组作业 / 多个单卡作业并行**（一次性把 6–8 个 learning rate 同时跑起来）
- **单次训练**：挑一个最好的 lr，再跑满 steps 拿到 `val loss <= 1.45` 的 checkpoint

仓库里提供了可直接 `sbatch` 的模板：
- `scripts/slurm/pretokenize_tinystories.sbatch`
- `scripts/slurm/train_tinystories.sbatch`
- `scripts/slurm/lr_sweep_array.sbatch`

常用命令：
- 看分区/GPU：`sinfo -o "%P %a %l %D %G"`
- 提交：`sbatch scripts/slurm/lr_sweep_array.sbatch`
- 排队：`squeue -u $USER`
- 取消：`scancel <jobid>`
