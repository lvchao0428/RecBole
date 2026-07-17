# WSDM 三机并行实验规划 (2026-07-11)

## 一、硬件资源盘点

| 机器 | GPU | 显存 | RAM | 特点 |
|------|-----|------|-----|------|
| **5090** | RTX 5090 | 32 GB | ~48 GB | 主力，跑重型 text 模型 |
| **log10** | GTX 1080 Ti | 11 GB | ~32 GB | 跑 ID-only、轻量 TF-IDF |
| **本机 Mac** | M3 Max (40 GPU cores) | 128 GB 统一内存 | 128 GB | MPS 后端，ID-only 辅助 |

## 二、单配置耗时基线（Beauty 数据集）

| 配置 | 5090 耗时 | 5090 显存 | log10 耗时 | Mac MPS 耗时 (est.) |
|------|----------|----------|-----------|-------------------|
| **ID-only** (50ep) | ~26 min | ~3 GB | ~117 min (4.5x) | ~200 min (7.5x) |
| **TF-IDF** (20+50ep) | ~88 min | ~5 GB | ~400 min (est.) | ~660 min (est.) |
| **TF-IDF+LLM** (20+50ep) | ~113 min | ~6 GB | — | ~850 min (est.) |
| **MV-Align** (20+50ep) | ~267 min | ~8 GB | — | ~2000 min (est.) |

> Mac M3 Max MPS 实测：ID-only 2ep = 8min → 50ep ≈ 200min（5090 的 ~7.5x）
> Mac 更适合跑 ID-only 和 TF-IDF，**不建议跑 MV-Align**
> log10 适合跑 ID-only，TF-IDF 勉强可跑但很慢

## 三、实验全景

### 阶段 A：Beauty V2 Pilot（当前进行中）
**目的**：确认 TS-aware 全特征 + min5 过滤 + no-boost 下四层模型层级是否成立

| # | 配置 | 机器 | 状态 |
|---|------|------|------|
| A1 | ID-only seed=2025 | log10（排队中） | ⏳ |
| A2 | TF-IDF seed=2025 | 5090 | 🔄 运行中 |
| A3 | TF-IDF+LLM seed=2025 | 5090（排队中） | ⏳ |
| A4 | MV-Align seed=2025 | 5090（排队中） | ⏳ |

预计完成时间：~20:30 (约 6h from 14:15)

---

### 阶段 B：Beauty 3-seed + Toys/Grocery 扩展（A 完成后启动）
**前提**：阶段 A 层级 ID < TF-IDF < LLM < MV 成立

#### 策略原则
1. **5090 专注高价值**：TF-IDF+LLM、MV-Align（只有 5090 显存够、速度快）
2. **log10 跑 ID-only**：所有数据集的 ID-only 3 seeds
3. **Mac 跑 ID-only 补充**：当 log10 排满时分流 ID-only（比 log10 慢 1.7x 但可以并行）
4. **5090 并行 ID+TF-IDF**：显存 3+5=8 GB << 32 GB，可以同时跑

#### 5090 详细调度（关键路径）

| Slot | 主任务 | 并行任务 | 耗时 | 累计 |
|------|--------|---------|------|------|
| 1 | TF-IDF Beauty s=2024 (88min) | ID-only Beauty s=2024 (26min) | 88min | 1.5h |
| 2 | TF-IDF Beauty s=42 (88min) | ID-only Beauty s=42 (26min) | 88min | 3h |
| 3 | TF-IDF+LLM Beauty s=2024 (113min) | — | 113min | 4.9h |
| 4 | TF-IDF+LLM Beauty s=42 (113min) | — | 113min | 6.8h |
| 5 | MV-Align Beauty s=2024 (267min) | — | 267min | 11.2h |
| 6 | MV-Align Beauty s=42 (267min) | — | 267min | 15.7h |
| 7 | TF-IDF Toys s=2025 (88min) | — | 88min | 17.2h |
| 8 | TF-IDF Toys s=2024 (88min) | — | 88min | 18.7h |
| 9 | TF-IDF Toys s=42 (88min) | — | 88min | 20.1h |
| 10 | TF-IDF+LLM Toys s=2025 (113min) | — | 113min | 22h |
| 11 | TF-IDF+LLM Toys s=2024 (113min) | — | 113min | 23.9h |
| 12 | TF-IDF+LLM Toys s=42 (113min) | — | 113min | 25.8h |
| 13 | MV-Align Toys s=2025 (267min) | — | 267min | 30.2h |
| ... | (Grocery 同理) | | | |

> 5090 关键路径 Beauty 3-seed: ~16h, + Toys/Grocery: 再 ~28h → 总 ~44h

#### log10 详细调度（ID-only 专用）

| # | 任务 | 耗时 | 累计 |
|---|------|------|------|
| 1 | ID-only Beauty s=2024 | 117min | 2h |
| 2 | ID-only Beauty s=42 | 117min | 4h |
| 3 | ID-only Toys s=2025 | ~130min | 6.2h |
| 4 | ID-only Toys s=2024 | ~130min | 8.3h |
| 5 | ID-only Toys s=42 | ~130min | 10.5h |
| 6 | ID-only Grocery s=2025 | ~90min | 12h |
| 7 | ID-only Grocery s=2024 | ~90min | 13.5h |
| 8 | ID-only Grocery s=42 | ~90min | 15h |

> log10 总计 ~15h

#### Mac MPS 调度（辅助分流）

Mac 速度慢(7.5x)，适合跑零散的 ID-only 分流或后处理任务：

| 场景 | 任务 | 耗时(est.) |
|------|------|-----------|
| 辅助 | ID-only Grocery s=42 | ~140min |
| 辅助 | ID-only Beauty s=42 (验证) | ~200min |
| 后处理 | 机制分析脚本 (CPU) | ~30min |

> Mac 主要作用：(1) 分流个别 ID-only 实验 (2) 运行机制分析 CPU 脚本 (3) 代码开发和调试

#### 阶段 B 总耗时（三机并行，瓶颈在 5090）
- **Beauty 3-seed 全配置**：5090 ~16h，log10 同时跑完 ID-only 
- **Toys/Grocery 扩展**：5090 再 ~28h（log10 ID-only 15h，5090 并行 TF-IDF 时搭 ID-only）
- **总计 ~44h (≈1.8 天)**，比单机串行 (~80h+) 节省约 45%

---

### 阶段 C：UniSRec Portability + Ablation（B 完成后）

| 实验 | 配置数 | seeds | 数据集 | 推荐机器 |
|------|--------|-------|--------|----------|
| UniSRec Base / +Align / +MV | 3 | 3 | Beauty | 5090 |
| Ablation: -Whiten / -Cross / -Align | 3 | 3 | Beauty | 5090 + Mac |

预计 ~8h (5090) + ~6h (Mac 辅助)

---

### 阶段 D：机制分析（C 完成后或并行）

纯 CPU/后处理，可在 Mac 上跑：
- rank-transition matrix
- score entropy / Gini / Top-K head-tail
- per-view gate-by-frequency
- component × metric heatmap

---

## 四、数据同步策略

```
Mac ──rsync──→ 5090 (中心节点) ←──rsync── log10
```

- 代码同步：Mac 本地修改 → scp 到 5090 → scp 到 log10
- 结果同步：log10/Mac 跑完 → rsync saved/ 到 5090 → 最终从 5090 同步回 Mac
- 特征文件：在 5090 上生成，rsync 分发到 log10 和 Mac

同步命令模板：
```bash
# Mac → 5090
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
  /Users/lvchao0428/project/ownRecBole/RecBole/ \
  charlie@www.ultrapp.online:/home/charlie/project/RecBole/

# 5090 → Mac (结果)
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/RecBole/saved/ \
  /Users/lvchao0428/project/ownRecBole/RecBole/saved/

# log10 → 5090 (结果)
ssh charlie@www.ultrapp.online "rsync -avz charlie@192.168.0.107:/home/charlie/project/RecBole/saved/ /home/charlie/project/RecBole/saved/"
```

## 五、Bug 修复记录

### `--config_dict` 解析 bug (2026-07-11 发现并修复)
`ast.literal_eval` 无法解析 JSON 风格的 `false`/`true`/`null`（Python 要求 `False`/`True`/`None`）。
导致 `{'freeze_backbone': false}` 解析失败，Phase-A 的 `freeze_backbone: True` 始终生效。

修复：在 `scripts/two_phase_train.py` 中对 `--config_dict` 值做 `false→False`, `true→True`, `null→None` 替换。

### MPS 设备支持 (2026-07-11)
`recbole/config/configurator.py` 新增 MPS fallback：当 CUDA 不可用但 MPS 可用时，自动使用 `torch.device("mps")`。

## 六、总时间线估算

| 阶段 | 耗时 | 累计 |
|------|------|------|
| A: Beauty V2 Pilot (4 config × 1 seed) | ~7h | 7h |
| 特征生成 Toys/Grocery (TS-aware) | ~0.5h | 7.5h |
| B: Beauty 3-seed (5090 主力, log10 ID-only) | ~16h | 23.5h |
| B+: Toys/Grocery 3-seed (5090+log10) | ~28h (5090) | 51.5h |
| C: UniSRec + Ablation (Beauty 3-seed) | ~10h | 61.5h |
| D: 机制分析 (Mac CPU, 可与 C 并行) | ~4h | 61.5h |
| **总计** | | **~62h (≈2.6 天)** |

> 关键路径在 5090，Toys/Grocery 可根据 Beauty 结论决定是否精简配置
> 如果需要加上 boost fallback（3 配置 × 3 seeds × 1 数据集），额外 ~6h
> Mac 主要贡献：开发调试、机制分析后处理、个别 ID-only 分流
