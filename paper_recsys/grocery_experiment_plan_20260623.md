# Amazon Grocery 第三域实验规划

> 日期: 2026-06-23  
> 目标: 复现 Beauty/Toys 主表增益形态（ID → TF-IDF +40~50% MRR → LLM → **MV > single**）  
> 依据: `dataset_strata_compare_20260623.md` · Grocery Δinter=**3.4**（最接近 Beauty）

---

## 1. 为什么选 Grocery

| 维度 | Beauty | Grocery | book-crossing |
|------|--------|---------|---------------|
| AvgSeq | 1.67 | **1.69** | 10.9 |
| coldR% | 25.1 | **26.8** | 53.3 |
| Δinter vs Beauty | 0 | **3.4** | >>40 |
| 域 | Amazon | Amazon | 非 Amazon |
| 已有 BC 实验 MRR 阶梯 | +45% TF-IDF | **未跑** | +14% TF-IDF |

Grocery = **分布最像 Beauty** + **Amazon 管线零改造** + **title/categories 文本**（MV 潜力 high）。

---

## 2. 数据准备（5090 · ~30 min）

### 2.1 源数据（已有）

```
/home/charlie/project/RecSysDatasets/RecBole/ProcessedDatasets/Amazon_ratings/
  Amazon_Grocery_and_Gourmet_Food.zip          # 1.30M inter, 166K items
  Amazon_Grocery_and_Gourmet_Food-example/     # schema 样例
```

**RecBole dataset 名**: `Amazon_Grocery_and_Gourmet_Food`（与 zip 前缀一致，仿 `Amazon_Toys_and_Games`）

**item 字段**: `title`, `categories`（与 Beauty 相同，可直接复用 4-view prompt）

### 2.2 待创建脚本

| 文件 | 仿照 | 作用 |
|------|------|------|
| `tools/setup_grocery_dataset.sh` | `setup_food_dataset.sh` | unzip → `dataset/Amazon_Grocery_and_Gourmet_Food/` |
| `tools/prepare_grocery_item_mapping.py` | Beauty mapping（title+categories） | `item_index_mapping.csv` |
| `tools/gen_text_emb_grocery_qwen2.5_7b.sh` | `gen_text_emb_food_qwen2.5_7b.sh` | TF-IDF + Qwen2.5-7B single + 4-view MV |

```bash
# setup 核心
GROCERY_ZIP=.../Amazon_Grocery_and_Gourmet_Food.zip
DST=dataset/Amazon_Grocery_and_Gourmet_Food
unzip -o -j "$GROCERY_ZIP" \
  "Amazon_Grocery_and_Gourmet_Food.inter" \
  "Amazon_Grocery_and_Gourmet_Food.item" -d "$DST"
```

**embedding 输出**（与 Beauty 同结构）:
```
dataset/Amazon_Grocery_and_Gourmet_Food/
  item_text_emb.base.npy              # TF-IDF SVD 256d, center+whiten
  item_text_emb.qwen2.5_7b.base.npy   # single-view LLM
  qwen2.5_7b_4views/                  # view_0..3.npy + views.json
  item_index_mapping.csv
```

**预估耗时**（166K items, 5090）:
- TF-IDF: ~15–30 min
- Qwen2.5-7B single: ~2–3 h
- 4-view MV: ~3–4 h  
- **合计 ~6–8 h**（可 overnight）

---

## 3. 配置文件（fork Beauty/Toys · 改 dataset 名）

| 用途 | 新文件 | 模板 |
|------|--------|------|
| ID-only 50ep | `sasrec_baseline_50ep_grocery_stratified.yaml` | `sasrec_baseline_50ep_stratified.yaml` |
| TF-IDF V3 | `sasrec_align_grocery_base_stratified_v3.yaml` | `sasrec_align_base_stratified_v3.yaml` |
| TF-IDF+LLM | `sasrec_align_grocery_qwen_stratified_v3.yaml` | `sasrec_align_qwen3_stratified_v3.yaml` → **改用 qwen2.5_7b 路径** |
| MV 4-view | `sasrec_align_multi_view_v3_grocery_stratified_7b.yaml` | `sasrec_align_multi_view_v3_toys_stratified_7b.yaml` |

**yaml 关键改动**:
```yaml
dataset: Amazon_Grocery_and_Gourmet_Food
load_col:
  inter: [user_id, item_id, rating, timestamp]
  item: [item_id, title, categories]
item_text_emb_path_base: dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.npy
# LLM / MV 路径同理
eval_args:
  order: TO
  split: {'LS': 'valid_and_test'}
```

---

## 4. 运行脚本（fork Food 四配置流水线）

| 文件 | 仿照 |
|------|------|
| `run50epBase_grocery_stratified.sh` | `run50epBase_food_stratified.sh` |
| `two_phase_run_tfidf_v3_grocery_stratified.sh` | `two_phase_run_tfidf_v3_food_stratified.sh` |
| `two_phase_run_tfidf_llm_v3_grocery_stratified.sh` | food 同名 |
| `two_phase_run_multiview_v3_grocery_stratified_7b.sh` | toys 同名 |
| **`run_5090_grocery_serial.sh`** | `run_5090_food_serial.sh` |

**超参（与 Beauty/Food 主表一致）**:
```
Phase-A: 20ep grid (align=0.10, tau=0.05)
Phase-B: 50ep (phaseb50)
config_dict: align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10
METRIC_BASELINE: AUTO（从 ID valid MRR@10 解析）
```

---

## 5. 执行顺序 & 时间线

### ✅ 已完成（RUN_ID=`20260623_grocery` · seed=2024 · 6/24 17:59）

| Step | 配置 | 完成时间 | 耗时 | test MRR@10 | test HR@10 |
|------|------|----------|------|-------------|------------|
| setup | dataset | 6/23 18:00 | — | — | — |
| 1_emb | TF-IDF + Qwen 1v/4v | 12:53 | 3.4h | — | — |
| 2_id | SASRecAlign 50ep（5090 fallback） | 13:13 | 20min | **2.08%** | **6.36%** |
| 3_tfidf | two-phase | 14:07 | 54min | **2.93%** | 5.77% |
| 4_llm | two-phase | 15:15 | 68min | **2.95%** | 5.83% |
| 5_mv | two-phase | 17:59 | 2.7h | **3.01%** | 6.02% |

> log10 ID（6/23 19:16）test MRR@10=2.08%；pipeline 中 log10 wait 失败，5090 重跑 ID 结果一致。  
> 日志: `logs/5090_grocery_20260623_grocery/` · `logs/grocery_pipeline_history.log`

### Phase 3 — 趋势判断 → **Go ✅**

| 指标 | Beauty 参考 | Grocery 实测 | 判定 |
|------|-------------|--------------|------|
| TF-IDF / ID (MRR@10) | ~1.45× | **1.41×** (2.93/2.08) | ✅ ≥1.30× |
| MV / TF-IDF+LLM (MRR) | 显著 > | 3.01% > 2.95% | ✅ |
| MV vs ID MRR 增益 | ~+45% | **+45%** | ✅ |
| TF-IDF vs ID (HR@10) | +2.6% (5.56/5.42) | **−9.3%** (5.77/6.36) | ⚠️ trade-off |
| MV vs ID (HR@10) | +6.8% (5.79/5.42) | **−5.3%** (6.02/6.36) | ⚠️ MV 部分收回 |
| HR_few@10 | MV > TF-IDF | 3.19% > 3.12% | ✅ |
| HR_freq@10 | MV ≈ ID | 11.09% vs 11.67% | ✅ 接近 |

→ **建议启动 Phase 4 multiseed**（2024/2025/2026/42 × 四配置）

### Phase 4 — Multiseed（待启动）

4 seeds × 4 configs ≈ **2–3 天**。**当前状态: 待启动**。

---

## 6. 5090 资源调度建议

```
6/24 09:26 ──────── BC phaseb50 四配置 ✅ 完成
6/24 17:59 ──────── Grocery seed=2024 四配置 ✅ 完成（Go）
下一步 ──────────── Grocery multiseed（4 seeds × 4 configs，~2–3 天）
```

**5090 GPU 当前空闲**（2026-06-24 20:18）。BC appendix 不必 multiseed。

---

## 7. 论文叙事定位

| 数据集 | 角色 |
|--------|------|
| Beauty + Toys | 主表（Amazon 两域，4-seed） |
| **Grocery** | **第三域主表**（Amazon 第三域，分布验证泛化） |
| Food | 非 Amazon tail-heavy 对照（overall 低、cold 有增益） |
| book-crossing | 非 Amazon extreme-cold appendix |

---

## 8. 待创建文件 checklist

- [x] `tools/setup_grocery_dataset.sh`
- [x] `tools/prepare_grocery_item_mapping.py`（inline 于 gen script）
- [x] `tools/gen_text_emb_grocery_qwen2.5_7b.sh`
- [x] `sasrec_baseline_50ep_grocery_stratified.yaml`
- [x] `sasrec_align_grocery_base_stratified_v3.yaml`
- [x] `sasrec_align_grocery_qwen_stratified_v3.yaml`
- [x] `sasrec_align_multi_view_v3_grocery_stratified_7b.yaml`
- [x] `run50epBase_grocery_stratified.sh`
- [x] `two_phase_run_tfidf_v3_grocery_stratified.sh`
- [x] `two_phase_run_tfidf_llm_v3_grocery_stratified.sh`
- [x] `two_phase_run_multiview_v3_grocery_stratified_7b.sh`
- [x] `run_5090_grocery_pipeline.sh` + `run_5090_queue_after_bc_grocery.sh`

**实现策略**: 批量 `sed 's/Amazon_Toys_and_Games/Amazon_Grocery_and_Gourmet_Food/g'` 从 Toys yaml/script fork，比从 Food fork 更干净（Grocery 也是 title+categories Amazon 评论域）。

---

## 9. 快速复现命令（5090）

```bash
# 分层审计（已做过，可复查）
python tools/dataset_strata_text_audit.py \
  --zip .../Amazon_Grocery_and_Gourmet_Food.zip \
  --name Amazon_Grocery

# 训练完成后拉 metrics
python tools/pull_bc_metrics.py   # 需改 dataset 过滤 → 或写 pull_grocery_metrics.py
```
