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

### Phase 0 — 等 BC 队列结束（~12–14h，进行中）

当前 5090 GPU 被 BC phaseb50 占用。Grocery **不要并行抢 GPU**。

### Phase 1 — 数据 + Embedding（Day 1 上午，~8h）

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole

# 同步脚本后
bash tools/setup_grocery_dataset.sh
bash tools/gen_text_emb_grocery_qwen2.5_7b.sh   # nohup overnight OK
python tools/verify_whiten.py dataset/Amazon_Grocery_and_Gourmet_Food  # 可选
```

### Phase 2 — 四配置训练（Day 1 晚 ~ Day 2，~12–16h）

```bash
RUN_ID=20260624_grocery SEED=2024 \
  nohup bash run_5090_grocery_serial.sh \
  > logs/exp_5090_grocery_20260624.log 2>&1 &
```

| Step | 配置 | 预估 |
|------|------|------|
| 2_id | SASRecAlign 50ep | ~1 h |
| 3_tfidf | two-phase | ~3 h |
| 4_llm | two-phase | ~3 h |
| 5_mv | two-phase | ~3.5 h |

### Phase 3 — 趋势判断（Day 2）

**Go / No-Go 标准**（对比 Beauty 主表 seed=2024）:

| 指标 | Beauty 参考 | Grocery 目标 |
|------|-------------|--------------|
| TF-IDF / ID (MRR@10) | ~1.45× | **≥1.30×** |
| MV / TF-IDF+LLM | 显著 > | **MV MRR > LLM MRR** |
| new/few MRR 提升 | 明显 | TF-IDF new MRR > 2× ID |

- **Go** → multiseed（2024/2025/2026/42，仿主表 4-seed）
- **No-Go** → 检查 embedding whiten / infer_boost；Grocery 仍比 Food/BC 更接近 Beauty，优先调参而非换域

### Phase 4 — Multiseed（Day 3–5，~48h）

仅 trend OK 后启动；4 seeds × 4 configs ≈ **2–3 天**。

---

## 6. 5090 资源调度建议

```
现在 ────────────── BC phaseb50 四配置 (~14h)
BC 完成 ─────────── Grocery setup + embedding (~8h, 可 nohup)
embedding OK ────── Grocery 四配置 seed=2024 (~14h)
trend OK ────────── Grocery multiseed (~48h)
```

**总日历**: 约 **4–5 天**（单卡 5090，串行）。

BC 队列跑完后不必 multiseed（已定性为 appendix）；GPU 让给 Grocery。

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

- [ ] `tools/setup_grocery_dataset.sh`
- [ ] `tools/prepare_grocery_item_mapping.py`（或 inline 于 gen script）
- [ ] `tools/gen_text_emb_grocery_qwen2.5_7b.sh`
- [ ] `sasrec_baseline_50ep_grocery_stratified.yaml`
- [ ] `sasrec_align_grocery_base_stratified_v3.yaml`
- [ ] `sasrec_align_grocery_qwen_stratified_v3.yaml`
- [ ] `sasrec_align_multi_view_v3_grocery_stratified_7b.yaml`
- [ ] `run50epBase_grocery_stratified.sh`
- [ ] `two_phase_run_tfidf_v3_grocery_stratified.sh`
- [ ] `two_phase_run_tfidf_llm_v3_grocery_stratified.sh`
- [ ] `two_phase_run_multiview_v3_grocery_stratified_7b.sh`
- [ ] `run_5090_grocery_serial.sh`

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
