# Experiment Log (Assignment 1)

这个文档记录我在 A1 的实验尝试（含参数、耗时、结果与结论），并作为 `experiment_log` 题目的交付物之一。

## 2026-01-29

### Tokenizer
- **train_bpe_tinystories (10K)**：预分词耗时占比最高；最长 token 为 `Ġaccomplishment`。
- **train_bpe_expts_owt (32K)**：最长 token 包含非常长的“乱码样式” byte token 与 64 个 `-`；属于 web 噪声导致的长 byte-level token。
- **tokenizer_experiments**：
  - 10-doc sample bytes/token：TS tok on TS = 3.762821；OWT tok on OWT = 4.308458；TS tok on OWT = 2.955631。
  - 吞吐（50MB valid）：TS tok ≈ 5.76e6 bytes/s；OWT tok ≈ 3.70e6 bytes/s；估算 Pile 825GB ≈ 66.6 小时。

### Optimizer / LR
- **learning_rate_tuning**（SGD toy）：lr=1 慢降；lr=10 快降；lr=100 几步到接近 0；lr=1000 发散。

## Template for future sections

### LM training runs
- run_name:
  - dataset:
  - model:
  - optimizer/lr schedule:
  - batch size / context:
  - max steps:
  - best val loss:
  - wallclock:
  - notes:
