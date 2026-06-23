# Food 数据集实验设计

> 日期: 2026-06-21（设计）· **进度更新 2026-06-22**  
> 目标: 第三域非 Amazon 数据集，对齐 Beauty/Toys 四配置主表  
> 5090 流水线: `run_5090_food_serial.sh`  
> **实时状态**: [`experiment_status_20260622.md`](experiment_status_20260622.md) ← 实验主线

---

## 0. 当前训练进展（2026-06-22 17:12）

| Step | 内容 | 状态 | 备注 |
|------|------|------|------|
| 0_setup | 数据集解压 | ✅ | |
| 1_emb | TF-IDF + LLM 嵌入 | ✅ | 6/22 05:13，~5.5h |
| 2_id | ID baseline 50ep | ✅ | test MRR@10=**0.0060**，~45min |
| 3_tfidf | two-phase TF-IDF | 🔄 | Phase-B **epoch 16/40**；valid 最高 0.0059 |
| 4_tfidf_llm | two-phase LLM | ⏸ | 排队，~5h |
| 5_mv | two-phase MV 7B | ⏸ | 排队，~5h |

- **RUN_ID**: `20260622_food` · **SEED**: 2024 · **GPU**: 5090 单卡
- **日志**: `logs/5090_food_20260622_food/` · history: `logs/5090_food_20260622_food_history.log`
- **待办**: `METRIC_BASELINE` 应改为 **0.0052**（ID valid MRR@10），当前占位 0.015

---

## 1. 实验配置

| 代号 | 模型 | 文本 | 脚本 |
|------|------|------|------|
| **ID-only** | SASRecAlign | 无 | `run50epBase_food_stratified.sh` |
| **TF-IDF** | SASRecAlignV3 | TF-IDF (name+nutrition+tags) | `two_phase_run_tfidf_v3_food_stratified.sh` |
| **TF-IDF+LLM** | SASRecAlignV3 | TF-IDF + Qwen2.5-7B 单视角 | `two_phase_run_tfidf_llm_v3_food_stratified.sh` |
| **MV-Align** | SASRecAlignMultiViewV3 | TF-IDF + Qwen2.5-7B 4-view | `two_phase_run_multiview_v3_food_stratified_7b.sh` |

**Seed**: `2024`（从 `{42, 2023, 2024, 2025, 2026}` 随机抽取，本 run 固定）

**超参**（与 Beauty/Toys 主表 V3 Aggressive 一致）:
- `align_weight=0.1`, `tau=0.05` (Phase-A grid)
- `cold_text_boost=3.0`, `infer_boost=0.6`, `cold_threshold=10`
- **two-phase 协议**（与主表 ID-only 50ep 对齐）:
  - **Phase-A**: 20ep — grid/warmup 选 (λ, τ)，**不计入**可比训练量
  - **Phase-B**: **50ep** — 与 ID-only 50ep baseline 对齐的主训练阶段
  - 旧版脚本曾用 Phase-B=40ep（`phaseb40` checkpoint），已废弃

**ID baseline**: 50 epoch 单阶段（`only_phase_a` / `run_recbole`），`cosine_score=false`

---

## 2. 文本特征设计

### 输入文本（全量，统一字段顺序）

```
title = name + nutrition + tags
```

- **name**: 食谱标题
- **nutrition**: 7 维数值，格式化为 `nutrition calories 51.5 total_fat 0.0 sugar 13.0 ...`（100% 覆盖）
- **tags**: 完整 tags 序列（不截取）

由 `tools/prepare_food_item_mapping.py` 写入 `dataset/Food/item_index_mapping.csv` 的 `title` 列。  
TF-IDF / 单视角 LLM / 4-view MV **共用同一 `title`**；MV 各 view prompt 均传入完整 `{text}`，由模型自行提取视角信息。

### TF-IDF

- `build_item_text_emb_base.py --mapping_csv item_index_mapping.csv`
- char n-gram (1,2), SVD 256d, **center + whiten**

### LLM（Qwen2.5-7B）

与 Beauty/Toys **相同 prompt**（`tools/build_item_text_emb_qwen3_hf.py`）:

| View | Prompt |
|------|--------|
| base (单视角) | `[TITLE] {text}` |
| view_0 Identity | `Identify the item: [TITLE] {text}` |
| view_1 Function | `What are the main functions and features of [TITLE] {text}?` |
| view_2 Audience | `Who is the target audience or user group for [TITLE] {text}?` |
| view_3 Category | `Categorize the item [TITLE] {text} and describe its context.` |

`{text}` = 完整 **name + nutrition + tags**；四视角由 LLM 从统一全文本提取，**不按 tag 字段手工拆分**。

### 产出路径

```
dataset/Food/
├── Food.inter / Food.item
├── item_index_mapping.csv
├── item_text_emb.base.npy
├── item_text_emb.qwen2.5_7b.base.npy
└── qwen2.5_7b_4views/view_{0..3}.npy
```

---

## 3. 5090 执行步骤

```bash
# 1. sync 代码到 5090
./sync_recbole.sh

# 2. 停止 book-crossing 队列并快照
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && bash run_5090_stop_book_crossing.sh'

# 3. 启动 Food 流水线（embedding 已就绪时从 2_id 起）
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && mkdir -p logs && \
  START_FROM=2_id SKIP_SETUP=1 SKIP_EMB=1 RUN_ID=20260622_food SEED=2024 \
  nohup bash run_5090_food_serial.sh > logs/exp_5090_food_20260622_food.log 2>&1 &'
```

### 流水线步骤

| Step | 内容 | 预估 |
|------|------|------|
| 0_setup | 从 ProcessedDatasets 解压到 `dataset/Food/` | ~1 min |
| 1_emb | TF-IDF + LLM 1v + 4v (center+whiten) | ~2–4 h |
| 2_id | SASRecAlign 50ep | ~1 h |
| 3_tfidf | two-phase TF-IDF | ~5 h |
| 4_tfidf_llm | two-phase single-view | ~5 h |
| 5_mv | two-phase MV | ~5 h |

**总计**: ~18–20 h

### 断点续跑

```bash
START_FROM=3_tfidf SKIP_SETUP=1 SKIP_EMB=1 RUN_ID=20260621_food bash run_5090_food_serial.sh
```

---

## 4. Checkpoint 命名

| 配置 | 目录 |
|------|------|
| ID | `saved/` (run_recbole 默认) |
| TF-IDF | `saved/two_phase_run_tfidf_v3_food_stratified_phaseb50_seed2024/` |
| TF-IDF+LLM | `saved/two_phase_run_tfidf_llm_v3_food_stratified_phaseb50_seed2024/` |
| MV | `saved/two_phase_run_multiview_v3_food_stratified_7b_phaseb50_seed2024/` |

> 5090 当前在跑 `RUN_ID=20260622_food` 的 TF-IDF 为旧 **phaseb40** checkpoint 目录（脚本更新前启动）；LLM/MV 将走 phaseb50。

---

## 5. 与 book-crossing 的关系

- book-crossing 实验在记录现状后 **暂停**（见 `book_crossing_status_snapshot_*.md`）
- Food 作为新第三域候选：inter 1.13M，freq 9.2%，文本覆盖 100%
- 数据集统计见 `dataset_strata_analysis_20260621.md`、`food_text_analysis_20260621.md`

---

## 6. 相关文件

| 类型 | 路径 |
|------|------|
| Plain config | `sasrec_food_plain.yaml` |
| ID yaml | `sasrec_baseline_50ep_food_stratified.yaml` |
| TF-IDF yaml | `sasrec_align_food_base_stratified_v3.yaml` |
| LLM yaml | `sasrec_align_food_qwen3_stratified_v3.yaml` |
| MV yaml | `sasrec_align_multi_view_v3_food_stratified_7b.yaml` |
| Setup | `tools/setup_food_dataset.sh` |
| Embeddings | `tools/gen_text_emb_food_qwen2.5_7b.sh` |
| Mapping | `tools/prepare_food_item_mapping.py` |
| Master | `run_5090_food_serial.sh` |
