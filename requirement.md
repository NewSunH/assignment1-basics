
# CS336 Spring 2025 — Assignment 1 (basics) 需求梳理

> 目标：从零实现训练一个标准 Transformer LM 所需的关键组件（分词器、模型、损失/优化器、训练循环、采样解码），并完成一系列实验与书面问答。

## 0. 全局约束与提交物

### 0.1 允许/禁止使用的库（作业硬约束）
- 需要“从零实现”核心组件。
- 禁止直接使用 `torch.nn` / `torch.nn.functional` / `torch.optim` 里的现成定义（除少数例外）。允许的例外：
	- `torch.nn.Parameter`
	- `torch.nn` 中的容器类（`Module` / `ModuleList` / `Sequential` 等）
	- `torch.optim.Optimizer` 基类
- 其余 PyTorch API 一般允许使用；但如果某个函数/类会破坏“from-scratch”精神，应避免。

### 0.2 代码组织与测试机制（你需要做什么）
- 你的实现主要放在：`cs336_basics/`。
- 测试通过“适配器层”调用你的实现：你需要在 `tests/adapters.py` 里把对应的 `run_*` / `get_*` 钩子接到你的代码上。
	- 适配器只做“胶水代码”，不要在适配器里写实质性算法逻辑。
- 不能修改 `tests/test_*.py`。

### 0.3 最终提交物
- `writeup.pdf`：所有书面题（含实验分析、资源/耗时、学习曲线等）。
- `code.zip`：你写的所有代码。
- 需要上榜（leaderboard）的话：向指定 leaderboard 仓库提交 PR（按其 README 指引）。

---

## 1. 任务清单（按 Problem 编号汇总）

> 说明：你让我把 PDF 里“每个问题的原文”逐字写进来；由于作业 PDF 属于受版权保护材料，我不能在这里整段复制原文。
> 我在下面提供“逐题索引（页码）+ 结构化要点”，你可以按页码在 PDF 里对照核对；如果你把某一题原文粘贴到这里（你已提供的内容），我也可以帮你做格式化/拆解与勾选。

### 1.0 PDF 逐题索引（页码）

| Problem | 标题（PDF） | 分值 | PDF 页码 |
|---|---|---:|---:|
| unicode1 | Understanding Unicode | 1 | 4 |
| unicode2 | Unicode Encodings | 3 | 5 |
| train_bpe | BPE Tokenizer Training | 15 | 9 |
| train_bpe_tinystories | BPE Training on TinyStories | 2 | 10 |
| train_bpe_expts_owt | BPE Training on OpenWebText | 2 | 10 |
| tokenizer | Implementing the tokenizer | 15 | 11 |
| tokenizer_experiments | Experiments with tokenizers | 4 | 12 |
| linear | Implementing the linear module | 1 | 19 |
| embedding | Implement the embedding module | 1 | 19 |
| rmsnorm | Root Mean Square Layer Normalization | 1 | 21 |
| positionwise_feedforward | Implement the position-wise feed-forward network | 2 | 23 |
| rope | Implement RoPE | 2 | 24 |
| softmax | Implement softmax | 1 | 24 |
| scaled_dot_product_attention | Implement scaled dot-product attention | 5 | 25 |
| multihead_self_attention | Implement causal multi-head self-attention | 5 | 26 |
| transformer_block | Implement the Transformer block | 3 | 26 |
| transformer_lm | Implementing the Transformer LM | 3 | 27 |
| transformer_accounting | Transformer LM resource accounting | 5 | 27 |
| learning_rate_tuning | Tuning the learning rate | 1 | 31 |
| adamw | Implement AdamW | 2 | 32 |
| adamwAccounting | Resource accounting for training with AdamW | 2 | 32 |
| gradient_clipping | Implement gradient clipping | 1 | 34 |
| data_loading | Implement data loading | 2 | 35 |
| checkpointing | Implement model checkpointing | 1 | 36 |
| training_together | Put it together | 4 | 37 |
| decoding | Decoding | 3 | 39 |
| experiment_log | Experiment logging | 3 | 40 |
| learning_rate | Tune the learning rate | 3 | 41 |
| batch_size_experiment | Batch size variations | 1 | 42 |
| generate | Generate text | 1 | 43 |
| layer_norm_ablation | Remove RMSNorm and train | 1 | 43 |
| pre_norm_ablation | Implement post-norm and train | 1 | 44 |
| no_pos_emb | Implement NoPE | 1 | 44 |
| swiglu_ablation | SwiGLU vs. SiLU | 1 | 44 |
| main_experiment | Experiment on OWT | 2 | 45 |
| leaderboard | Leaderboard | 6 | 46 |

下面每一项都对应 PDF 中的一个 Problem。建议你把它当作“Done List”，逐项完成并在 writeup 里交付对应文本/图表。

### 1.1 Unicode 基础（书面题）
- [x] **unicode1 (1pt)**：理解 Unicode/控制字符在字符串中的表现与打印差异（按题目 a/b/c 写一两句回答）。
- [x] **unicode2 (3pt)**：UTF-8 vs UTF-16/32 的偏好原因；错误 UTF-8 解码函数的反例与解释；给出无法解码的两字节序列并解释。

### 1.2 BPE 分词器训练与实验
- [x] **train_bpe (15pt)**：实现“训练 byte-level BPE”的函数：
	- 输入：`input_path: str`、`vocab_size: int`、`special_tokens: list[str]`。
	- 输出：
		- `vocab: dict[int, bytes]`（token id -> token bytes）
		- `merges: list[tuple[bytes, bytes]]`（按创建顺序）
	- 关键点：特殊 token 不参与跨文档 merge；训练速度要足够快（测试里有速度约束）。
	- 测试：先实现 `tests/adapters.py` 里的 `run_train_bpe`，再跑 `uv run pytest tests/test_train_bpe.py`。
- [x] **train_bpe_tinystories (2pt)**：
	- 用 TinyStories 训练 10,000 vocab 的 BPE（含 `<|endoftext|>` special token），并把 vocab/merges 序列化到磁盘。
	- 记录训练耗时、内存；找 vocab 中最长 token 并判断是否合理。
	- 资源上限（CPU）：≤30min、≤30GB RAM。
	- 写 profile：哪里最耗时。
- [x] **train_bpe_expts_owt (2pt)**：
	- 用 OpenWebText 训练 32,000 vocab 的 BPE；序列化；最长 token 是否合理。
	- 对比 TinyStories tokenizer vs OWT tokenizer（1–2 句）。
	- 资源上限（CPU）：≤12h、≤100GB RAM。

### 1.3 Tokenizer 编码/解码与分词实验
- [x] **tokenizer (15pt)**：实现 Tokenizer 类（加载 vocab+merges，对文本 encode / 对 ids decode），并支持 special tokens。
	- 推荐接口：`__init__` / `from_files` / `encode` / `encode_iterable` / `decode`。
	- 解码时遇到非法 UTF-8 需要用替换字符（等价于 `errors='replace'`）。
	- 测试：实现 `tests/adapters.py` 的 `get_tokenizer`，再跑 `uv run pytest tests/test_tokenizer.py`。
- [x] **tokenizer_experiments (4pt)** ：
	- 采样 TinyStories 和 OWT 文档，计算压缩率（bytes/token）。
	- 用 TinyStories tokenizer 去 tokenize OWT 样本，观察变化。
	- 估计 tokenizer 吞吐（bytes/s），估算 tokenize Pile 需要多久。
	- 把 TinyStories/OWT 的 train+dev encode 成 token id 序列，建议保存为 `uint16` 的 NumPy 数组，并解释为何 `uint16` 合理。

---

## 2. Transformer LM 组件（从零实现）

### 2.1 基础模块
- [x] **linear (1pt)**：实现不带 bias 的 Linear（`torch.nn.Module`），按建议初始化（truncated normal），不能用 `nn.Linear` / `F.linear`。
	- 测试：`adapters.run_linear`，`uv run pytest -k test_linear`。
- [x] **embedding (1pt)**：实现 Embedding lookup（`torch.nn.Module`），不能用 `nn.Embedding` / `F.embedding`。
	- 测试：`adapters.run_embedding`，`uv run pytest -k test_embedding`。
- [x] **rmsnorm (1pt)**：实现 RMSNorm（注意计算时 upcast 到 float32，再 cast 回原 dtype）。
	- 测试：`adapters.run_rmsnorm`，`uv run pytest -k test_rmsnorm`。
- [x] **positionwise_feedforward (2pt)**：实现带门控的 SwiGLU FFN；d_ff 约为 8/3*d_model 且向上取整为 64 的倍数；可用 `torch.sigmoid`。
	- 测试：`adapters.run_swiglu`，`uv run pytest -k test_swiglu`。

### 2.2 位置编码与注意力
- [x] **rope (2pt)**：实现 RoPE（RotaryPositionalEmbedding），对任意 batch 维度都能工作；可用 buffer 预计算 sin/cos。
	- 测试：`adapters.run_rope`，`uv run pytest -k test_rope`。
- [x] **softmax (1pt)**：实现数值稳定 softmax（按指定 dim，减 max）。
	- 测试：`adapters.run_softmax`，`uv run pytest -k test_softmax_matches_pytorch`。
- [x] **scaled_dot_product_attention (5pt)**：实现 scaled dot-product attention：
	- 支持任意 batch-like 维度；可选 boolean mask（False 的概率为 0；True 的概率归一化为 1）。
	- 测试：`adapters.run_scaled_dot_product_attention`，`uv run pytest -k test_scaled_dot_product_attention` 与 `uv run pytest -k test_4d_scaled_dot_product_attention`。
- [x] **multihead_self_attention (5pt)**：实现 causal multi-head self-attention（dk=dv=d_model/num_heads），带 causal mask；RoPE 只用于 Q/K。
	- 测试：`adapters.run_multihead_self_attention`，`uv run pytest -k test_multihead_self_attention`。

### 2.3 组合成 Transformer Block 与 Transformer LM
- [x] **transformer_block (3pt)**：实现 pre-norm Transformer block（RMSNorm → MHA/FFN → residual）。
	- 测试：`adapters.run_transformer_block`，`uv run pytest -k test_transformer_block`。
- [x] **transformer_lm (3pt)**：实现完整 Transformer LM（embedding + 多层 block + 输出到 vocab logits），支持构造参数：`vocab_size/context_length/num_layers/d_model/num_heads/d_ff/...`。
	- 测试：`adapters.run_transformer_lm`，`uv run pytest -k test_transformer_lm`。
- [ ] **transformer_accounting (5pt)**（书面题）：对 GPT-2 系列做参数量、内存、FLOPs 分析与对比（含不同 context length 扩展）。

---

## 3. 训练：损失、优化器、调度与训练循环

### 3.1 损失与指标
- [x] **cross_entropy**：实现数值稳定的 cross-entropy（对 logits 做稳定处理，尽量抵消 log/exp；支持 batch 维度并返回 batch 平均）。
	- 测试：`adapters.run_cross_entropy`，`uv run pytest -k test_cross_entropy`。

### 3.2 优化器与调度
- [x] **learning_rate_tuning (1pt)**（书面/小实验）：跑给定 SGD toy example，比较不同 lr 的 loss 行为。
	- 复现实验脚本：`uv run experiments/learning_rate_tuning.py`
	- 我本地观测（10 steps, init seed=0）：`lr=1` 缓慢下降；`lr=1e1` 更快下降；`lr=1e2` 极快下降到接近 0；`lr=1e3` 明显发散（loss 指数级暴涨）。
- [x] **adamw (2pt)**：实现 AdamW（继承 `torch.optim.Optimizer`），维护每个参数的状态（m/v 等）。
	- 测试：`adapters.get_adamw_cls`，`uv run pytest -k test_adamw`。
- [ ] **adamwAccounting (2pt)**（书面题）：AdamW 的显存/算力 accounting（参数/激活/梯度/优化器状态拆分；80GB 下 batch 上限；AdamW FLOPs；训练时长估算）。
- [x] **learning_rate_schedule**：实现 cosine learning-rate schedule（含 warmup）。
	- 测试：`adapters.get_lr_cosine_schedule`，`uv run pytest -k test_get_lr_cosine_schedule`。
- [x] **gradient_clipping (1pt)**：实现梯度裁剪（全参数 L2 norm 上限，原地缩放；eps=1e-6）。
	- 测试：`adapters.run_gradient_clipping`，`uv run pytest -k test_gradient_clipping`。

### 3.3 训练循环基础设施
- [x] **data_loading (2pt)**：实现 batch 采样函数（从 token id 的 numpy 数组采样输入与 next-token targets；返回 shape=(B, context_length) 的 tensor，放到指定 device）。
	- 测试：`adapters.run_get_batch`，`uv run pytest -k test_get_batch`。
	- 建议：大数据用 `np.memmap`/`np.load(..., mmap_mode='r')`。
- [x] **checkpointing (1pt)**：实现 `save_checkpoint` / `load_checkpoint`（保存 model/optimizer/iteration）。
	- 测试：`adapters.run_save_checkpoint`、`adapters.run_load_checkpoint`，`uv run pytest -k test_checkpointing`。
- [ ] **training_together (4pt)**：写一个可配置的训练脚本（超参可调、memmap 数据、checkpoint、周期性训练/验证日志）。

---

## 4. 解码、实验与扩展

### 4.1 解码（采样生成）
- [ ] **decoding (3pt)**：实现 decoder/采样函数，至少支持：
	- prompt completion（生成到 `<|endoftext|>` 或达到最大 token 数）
	- temperature
	- top-p（nucleus）采样

### 4.2 实验记录与 TinyStories 实验
- [ ] **experiment_log (3pt)**：实现实验追踪/日志（按 step 与 wallclock 记录 loss 曲线），并提交一份“实验日志文档”。
- [ ] **learning_rate (3pt, ~4 H100 hrs)**：在 TinyStories 上做学习率 sweep：给出多条学习曲线，解释搜索策略，并训练出 per-token val loss ≤ 1.45 的模型。
- [ ] **batch_size_experiment (1pt, ~2 H100 hrs)**：batch size 从 1 到显存上限，给学习曲线与结论。
- [ ] **generate (1pt)**：用训练好的 checkpoint 生成 ≥256 tokens（或到 `<|endoftext|>`），评论流畅度并列出至少两个影响因素。

### 4.3 架构消融与修改（TinyStories）
- [ ] **layer_norm_ablation (1pt, ~1 H100 hr)**：移除 RMSNorm 训练，比较稳定性与学习率影响。
- [ ] **pre_norm_ablation (1pt, ~1 H100 hr)**：实现 post-norm 并训练，对比 pre-norm 学习曲线。
- [ ] **no_pos_emb (1pt, ~1 H100 hr)**：实现 NoPE（去掉位置编码/RoPE），与 RoPE 对比曲线。
- [ ] **swiglu_ablation (1pt, ~1 H100 hr)**：SwiGLU vs SiLU（参数量近似匹配），给曲线与讨论。

### 4.4 OpenWebText 实验与 leaderboard
- [ ] **main_experiment (2pt, ~3 H100 hrs)**：在 OWT 上用同样架构与训练迭代数训练，给学习曲线，并解释与 TinyStories loss 的差异；同时给生成文本与质量分析。
- [ ] **leaderboard (6pt, ~10 H100 hrs)**：在 leaderboard 规则下训练并提交：
	- 规则要点：只能用提供的 OWT 训练集；单次运行 wallclock ≤ 1.5 小时（H100）。
	- 交付：最终验证 loss、带 wallclock x 轴的学习曲线（≤1.5h）、你做了哪些改动的描述；目标至少优于朴素 baseline（loss≈5.0）。

---

## 5. 建议的本地验收顺序（按测试文件）
- `uv run pytest tests/test_train_bpe.py`
- `uv run pytest tests/test_tokenizer.py`
- `uv run pytest -k test_linear`
- `uv run pytest -k test_embedding`
- `uv run pytest -k test_rmsnorm`
- `uv run pytest -k test_rope`
- `uv run pytest -k test_softmax_matches_pytorch`
- `uv run pytest -k test_scaled_dot_product_attention`
- `uv run pytest -k test_4d_scaled_dot_product_attention`
- `uv run pytest -k test_multihead_self_attention`
- `uv run pytest -k test_transformer_block`
- `uv run pytest -k test_transformer_lm`
- `uv run pytest -k test_cross_entropy`
- `uv run pytest -k test_get_lr_cosine_schedule`
- `uv run pytest -k test_gradient_clipping`
- `uv run pytest -k test_get_batch`
- `uv run pytest -k test_checkpointing`

> 注：训练脚本/实验部分通常不在单元测试里硬测，需要你自己跑并在 writeup 里给曲线与分析。

