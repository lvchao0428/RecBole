# Experiment Status — 2026-08-03

> 更新时间: 18:50  
> 上一日: [`experiment_status_20260730.md`](experiment_status_20260730.md)

---

## 〇、一句话

收到老师 0803 指导后确认：**当前 GTS 使用的是 Successive-All 模式**（每 user 在 test 期间所有 step 参与评估），老师要求改为 GTS-Last（每 user 只评最后一个 target）。无需重训——已有 checkpoint 可直接用 GTS-Last 重评估。今晚补充实验已部署到 5090。

---

## 一、重要修正（8/2）

### 1.1 UniSRec min5 eval filter 重测

原来 UniSRec 训练时没有 `min_train_interactions=5` 的 eval filter，导致评估指标偏高。用统一 filter 重测后：

| Dataset | 原始 test MRR@10 | **修正 test MRR@10** | valid MRR@10 |
|---------|:----------------:|:-------------------:|:------------:|
| Beauty seed42 | 0.0204 | **0.0122** | 0.0204 |
| Beauty seed2024 | 0.0206 | **0.0123** | 0.0206 |
| Beauty seed2026 | 0.0208 | **0.0124** | 0.0208 |
| Toys seed42 | 0.0162 | **0.0083** | 0.0162 |
| Toys seed2024 | 0.0163 | **0.0083** | 0.0166 |
| Toys seed2026 | 0.0164 | **0.0081** | 0.0164 |

| Dataset | **mean ± std (修正)** |
|---------|:--------------------:|
| Beauty | **0.0123 ± 0.0001** |
| Toys | **0.0082 ± 0.0001** |

**重要发现**：统一 filter 后 UniSRec 反而**弱于** TF-IDF/LLM align 方案（Beauty 0.0123 < 0.0165），说明 UniSRec 的优势来自评估条件差异而非模型能力。

### 1.2 Leave-One-View-Out 正确重跑（zero-mask）

旧方法修改 `text_view_indices` 实际是 no-op（模型读 fixed buffer），用 zero-mask 重跑后有真实差异：

**Beauty（GTS seed42, min5 filter）：**

| Config | MRR@10 | Δ vs Full |
|--------|-------:|----------:|
| Full (4 views) | 0.015906 | — |
| Drop Description | 0.015845 | -0.000061 |
| Drop Function | 0.016018 | +0.000112 |
| Drop Audience | 0.015743 | -0.000163 |
| **Drop Style** | 0.015564 | **-0.000342** |

**Toys（GTS seed42, min5 filter）：**

| Config | MRR@10 | Δ vs Full |
|--------|-------:|----------:|
| Full (4 views) | 0.010175 | — |
| Drop Description | 0.009982 | -0.000193 |
| Drop Function | 0.010067 | -0.000108 |
| Drop Audience | 0.010051 | -0.000124 |
| Drop Style | 0.010025 | -0.000150 |

**结论**：各 view 贡献极小（Δ < 0.0004），最大贡献是 Style view（Beauty）。总体变化在噪声范围内，验证 **view redundancy**——虽不是精确为零，但边际贡献可忽略。

### 1.3 Warm-Only Eval 主表

限制 test targets 为 train catalog 内的 warm items（Beauty unseen=31.9%, Toys unseen=40.8%）：

**Beauty GTS — warm targets only (MRR@10)：**

| 模型 | seed42 | seed2024 | seed2026 | mean |
|------|-------:|---------:|---------:|-----:|
| ID | 0.0164 | 0.0153 | 0.0152 | **0.0156** |
| TF | 0.0235 | 0.0237 | 0.0255 | **0.0242** |
| LLM | 0.0230 | 0.0236 | 0.0242 | **0.0236** |
| MV | 0.0234 | 0.0225 | 0.0249 | **0.0236** |

**Toys GTS — warm targets only (MRR@10)：**

| 模型 | seed42 | seed2024 | seed2026 | mean |
|------|-------:|---------:|---------:|-----:|
| ID | 0.0119 | 0.0119 | 0.0116 | **0.0118** |
| TF | 0.0197 | 0.0190 | 0.0186 | **0.0191** |
| LLM | 0.0188 | 0.0178 | 0.0176 | **0.0181** |
| MV | 0.0172 | 0.0189 | 0.0192 | **0.0184** |

排序不变：**TF ≥ LLM ≈ MV >> ID**，warm-only 下各模型绝对值抬高但排序一致。

---

## 二、修正后完整主表（论文 ready）

### GTS Protocol — MRR@10 test, 3-seed mean ± std (min5 filter)

| Dataset | ID-only | TF-IDF | LLM | MV | UniSRec |
|---------|--------:|-------:|----:|---:|--------:|
| Beauty | 0.0106 | **0.0165** | 0.0161 | 0.0160 | 0.0123 |
| Toys | 0.0070 | **0.0113** | 0.0107 | 0.0109 | 0.0082 |

### LOO Protocol — MRR@10 test, 3-seed mean ± std

| Dataset | ID-only | TF-IDF | LLM | MV |
|---------|--------:|-------:|----:|---:|
| Beauty | 0.0175 | 0.0251 | **0.0254** | 0.0253 |
| Toys | 0.0152 | 0.0247 | **0.0258** | 0.0258 |

### Warm-Only GTS — MRR@10 test, 3-seed mean

| Dataset | ID-only | TF-IDF | LLM | MV |
|---------|--------:|-------:|----:|---:|
| Beauty | 0.0156 | **0.0242** | 0.0236 | 0.0236 |
| Toys | 0.0118 | **0.0191** | 0.0181 | 0.0184 |

---

## 三、关键故事线更新

1. **UniSRec 在公平条件下弱于 align 方案** — 之前看到的 "UniSRec 更强" 是 eval filter 不一致的假象
2. **TF-IDF 在所有 GTS 条件下最优** — all targets / warm-only / 两个 dataset 一致
3. **View redundancy 实锤** — zero-mask LOO-V 各 view Δ < 0.0004，远小于模型间差异
4. **协议决定排序** — GTS: TF > LLM ≈ MV; LOO: LLM ≈ MV ≈ TF

---

## 四、关键发现：GTS Target 模式确认

### 4.1 代码分析结论

**当前 GTS 使用 "Successive-All" 模式：**

- `SequentialDataset.data_augmentation()` 在 split 前执行，将每用户序列展开为所有 next-item step
- `time_based_split()` 按 target item 时间戳划分 → test 中包含每用户在阈值后的**所有** step
- 实证：Beauty test 有 4520 users → 12001 samples (avg 2.65/user)；Toys 4873 users → 13186 samples (avg 2.71/user)

### 4.2 影响

老师明确要求：如果是 All 则必须改为 Last 或 Random。好消息是**不需要重训模型**——训练仍用所有 step（标准 SASRec 训练方式），只在 evaluation 时过滤为每 user 的 last test interaction 即可。

---

## 五、缺口分析（0803 指导 vs 现有数据）

| 老师要求 | 当前状态 | 工作量 |
|---------|---------|--------|
| **GTS-Last eval** | 当前 All | 重评估现有 ckpt，**不重训** |
| CKA 3次子样本 mean±std | 只跑1次全量 | 轻量 CPU，几分钟 |
| LOOV 全量 4 views | ✅ 已完成（8/2 zero-mask） | — |
| 参数/推理成本表 | 未做 | 轻量 GPU 统计 |
| LOO vs GTS 对照表 | ✅ 有数据 | — |
| GTS Main/Early 表 | ✅ 有数据 | 需 Last mode 重算 |
| UniSRec baseline | ✅ 有（min5 修正） | 需 Last mode 重算 |
| CKA heatmap + ER + LOOV 组合图 | 数据有，图未出 | 论文写作时出 |
| Cross → negative appendix | ✅ 有数据 | — |

---

## 六、今晚补充实验（已部署 5090）

### 6.1 GTS-Last 重评估（核心，~60min）

- 评估脚本: `scripts/eval_gts_last.py`
- 覆盖: ID/TF/LLM/MV × Beauty/Toys × 3 seeds + UniSRec × 2 datasets × 3 seeds = **30 组**
- 方法: 加载 checkpoint → rebuild dataset → filter test 为 last-per-user → 重算 MRR/NDCG/HR@10
- 输出: `logs/gts_last_eval.csv`

### 6.2 CKA 子样本分析（轻量，~5min）

- 脚本: `scripts/cka_subsample.py`
- 3 次独立随机 item 子样本 (n=3000)，报告 CKA mean±std
- 输出: `paper_recsys/cka_subsample_beauty.json`, `paper_recsys/cka_subsample_toys.json`

### 6.3 参数/推理成本表（轻量，~10min）

- 脚本: `scripts/model_cost_table.py`
- 统计: 每模型参数量 (M)、text feature dim、推理 ms/sample
- 输出: `logs/model_cost_table.csv`

---

## 七、机器快照

| 机器 | 状态 |
|------|------|
| **5090** | 正在运行 `run_5090_0803_supplement.sh`（GTS-Last + CKA + Cost） |
| **本机** | 文档/脚本已同步；等待 5090 结果 |

**预计明早可看到 GTS-Last 主表结果。**
