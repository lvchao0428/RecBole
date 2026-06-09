# Camera-Ready 补充实验 — 任务跟踪

> 最后更新: 2026-06-09 17:30
> 对应审稿意见: R1(缺外部基线) / R2(缺文本基线+单backbone+单领域+视角消融) / R3(能否精简+泛化+外部基线)

---

## 总览

```
代码/配置  ████████████████████████  100% ✅
实验执行    ░░░░░░░░░░░░░░░░░░░░░░░░  0% (5090 暂无算力)
```

| 实验块 | 审稿意见 | 代码 | 配置 | 脚本 | 实验 |
|--------|---------|------|------|------|------|
| P0a: FDSA + S3-Rec | R2-W1 | ☑ | ☑ | ☑ | ☐ |
| P0b: UniSRec | R2-W1 / R3-Q3 | ☐ | ☐ | ☐ | ☐ |
| P1: GRU4Rec backbone | R2-W2 / R3-Q2 | ☑ | ☑ | ☑ | ☐ |
| P2: MV-Align-Lite | R3-Q1 | N/A | ☑ | ☑ | ☐ |
| P4: 单视角消融 | R2-W5 | N/A | ☑ | ☑ | ☐ |
| DS: book-crossing | R2-W3 / R3-Q2 | N/A | ☑ | ☑ | ☐ |

---

## 基础设施 (已完成)

- [x] **修复 `sync_recbole.sh`**: 添加 `--filter "P dataset/"` 和 `--filter "P release/dataset/"` 保护服务器数据文件不被 `--delete` 清除
  - 文件: `sync_recbole.sh` L34-38
- [x] **数据集恢复**: 从 5090 的 `ProcessedDatasets/` 解压/复制 5 个数据集到 `dataset/`
  - Amazon_Beauty: 2.02M inter, 249K items ✓
  - Amazon_Toys_and_Games: 2.25M inter, 328K items ✓
  - book-crossing: 1.15M inter, 341K items ✓
  - yelp: 8.02M inter, 209K items ✓
  - ml-1m: 1.00M inter, 3.7K items ✓
- [x] **数据集统计脚本**: `release/tools/dataset_stats.py` — 计算 #Users/#Items/#Inter/Density/AvgSeq/Strata 等
- [x] **统计分析完成**: 决定 **book-crossing** 为第三数据集 (75.4% new items, 稀疏性强, 有 title)

### 数据集统计结果

| Dataset | #Users | #Items | #Inter | Density | AvgSeq | New% | Few% | Freq% |
|---------|--------|--------|--------|---------|--------|------|------|-------|
| Amazon_Beauty | 1.21M | 249K | 2.02M | 0.0007% | 1.7 | 58.4% | 26.7% | 14.8% |
| Amazon_Toys | 1.34M | 328K | 2.25M | 0.0005% | 1.7 | 60.7% | 26.3% | 13.0% |
| **book-crossing** | 105K | 341K | 1.15M | 0.0032% | 10.9 | **75.4%** | 19.2% | 5.4% |
| yelp (备选) | 1.97M | 209K | 8.02M | 0.0019% | 4.1 | 0.0% | 49.5% | 50.5% |
| ml-1m (备选) | 6.0K | 3.7K | 1.00M | 4.47% | 165.6 | 5.5% | 6.6% | 88.0% |

---

## P2: MV-Align-Lite 精简变体 (代码完成 ☑, 待跑实验)

> 回应 R3-Q1: "能否精简?"

- [x] **配置文件**: `release/configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_lite.yaml`
  - 基于 `sasrec_align_multi_view_v3_toys_stratified_7b.yaml`
  - 修改: `use_text_view_senet: false`, `cold_text_boost: 0.0`, `infer_boost: 0.0`
  - 保留: `use_cross: true`, `use_align: true`, whitening, multi-view embeddings
- [x] **执行脚本**: 集成到 `release/run_scripts/run_per_view_ablation.sh` (P2+P4 共用)
- [ ] **执行实验**: Toys 单种子 (2025), ~1h

---

## P4: 单视角消融 (代码完成 ☑, 待跑实验)

> 回应 R2-W5: "各视角单独效果不清楚"

- [x] **配置文件** (4 个):
  - `release/configs/ablation/sasrec_align_toys_view0_identity.yaml` → `qwen2.5_7b_4views/view_0.npy`
  - `release/configs/ablation/sasrec_align_toys_view1_function.yaml` → `qwen2.5_7b_4views/view_1.npy`
  - `release/configs/ablation/sasrec_align_toys_view2_audience.yaml` → `qwen2.5_7b_4views/view_2.npy`
  - `release/configs/ablation/sasrec_align_toys_view3_category.yaml` → `qwen2.5_7b_4views/view_3.npy`
- [x] **执行脚本**: `release/run_scripts/run_per_view_ablation.sh`
- [ ] **执行实验**: Toys 单种子 (2025), 4 runs, ~4h

---

## P0a: FDSA + S3-Rec 外部基线 (代码完成 ☑, 待跑实验)

> 回应 R2-W1: "缺少文本基线 (FDSA, S3-Rec)"

### 代码移植
- [x] 复制 `fdsa.py` → `release/recbole/model/sequential_recommender/`
- [x] 复制 `s3rec.py` → `release/recbole/model/sequential_recommender/`
- [x] 确认 `release/recbole/model/layers.py` 包含 `FeatureSeqEmbLayer` + `VanillaAttention` ✓
- [x] 更新 `release/recbole/model/sequential_recommender/__init__.py` 添加 FDSA/S3Rec 导入
- [x] 确认 `release/recbole/trainer/trainer.py` 包含 `S3RecTrainer` ✓

### Model Properties
- [x] `release/recbole/properties/model/FDSA.yaml` — `selected_features: ['class']`
- [x] `release/recbole/properties/model/S3Rec.yaml` — `item_attribute: 'categories'`, `pretrain_epochs: 30`

### 实验配置 (4 个)
- [x] `release/configs/beauty/fdsa_beauty_stratified.yaml`
- [x] `release/configs/toys/fdsa_toys_stratified.yaml`
- [x] `release/configs/beauty/s3rec_beauty_stratified.yaml`
- [x] `release/configs/toys/s3rec_toys_stratified.yaml`

### 执行脚本
- [x] `release/run_scripts/run_external_baselines.sh` — FDSA: 标准训练, S3-Rec: pretrain→finetune 两步

### 实验 (4 seeds x 2 datasets x 2 models = 16 runs)
- [ ] FDSA Beauty (4 seeds)
- [ ] FDSA Toys (4 seeds)
- [ ] S3-Rec Beauty (pretrain 30ep → finetune, 4 seeds)
- [ ] S3-Rec Toys (pretrain 30ep → finetune, 4 seeds)

---

## P1: GRU4Rec 第二骨干 (代码完成 ☑, 待跑实验)

> 回应 R2-W2 / R3-Q2: "方法只在 SASRec 上验证, 不知是否泛化"

### 代码
- [x] 复制 `gru4rec.py` → `release/recbole/model/sequential_recommender/`
- [x] 复制 `GRU4Rec.yaml` → `release/recbole/properties/model/`
- [x] **编写 `gru4recalignv3.py`**: GRU backbone + V3 text fusion/cross/align (独立实现, 与 SASRecAlignV3 逻辑对齐)
- [x] **编写 `gru4recalignmultiviewv3.py`**: 继承 GRU4RecAlignV3, 加多视角 SENet/per-view align
- [x] `release/recbole/properties/model/GRU4RecAlignV3.yaml`
- [x] `release/recbole/properties/model/GRU4RecAlignMultiViewV3.yaml`
- [x] 更新 `__init__.py` 添加 GRU4Rec/GRU4RecAlignV3/GRU4RecAlignMultiViewV3 导入

### 实验配置 (6 个)
- [x] `release/configs/beauty/gru4rec_baseline_beauty_stratified.yaml` (ID-only)
- [x] `release/configs/beauty/gru4rec_align_beauty_base_stratified_v3.yaml` (TF-IDF)
- [x] `release/configs/beauty/gru4rec_align_multi_view_v3_beauty_stratified.yaml` (MV-Align)
- [x] `release/configs/toys/gru4rec_baseline_toys_stratified.yaml`
- [x] `release/configs/toys/gru4rec_align_toys_base_stratified_v3.yaml`
- [x] `release/configs/toys/gru4rec_align_multi_view_v3_toys_stratified.yaml`

### 执行脚本
- [x] `release/run_scripts/run_gru4rec_all.sh` — 3 配置 x 2 数据集 x 4 seeds = 24 runs

### 实验
- [ ] GRU4Rec ID-only Beauty + Toys (8 runs)
- [ ] GRU4Rec + TF-IDF Beauty + Toys (8 runs)
- [ ] GRU4Rec + MV-Align Beauty + Toys (8 runs)

---

## DS: Book-Crossing 第三数据集 (数据+配置完成 ☑, 待跑实验)

- [x] **添加 timestamp 列**: `tools/preprocess_book_crossing_inter_add_timestamp.py` 在 5090 执行完成
- [x] **复制到 release/**: `release/dataset/book-crossing/` 包含全部数据和 embedding 文件
- [x] **配置文件** (4 个):
  - `release/configs/book-crossing/sasrec_baseline_50ep_book_crossing_stratified.yaml` — ID-only
  - `release/configs/book-crossing/sasrec_align_book_crossing_base_stratified_v3.yaml` — TF-IDF
  - `release/configs/book-crossing/sasrec_align_book_crossing_qwen3_stratified_v3.yaml` — TF-IDF + Qwen3 LLM
  - `release/configs/book-crossing/sasrec_align_multi_view_v3_book_crossing_stratified.yaml` — 4-view MV-Align
- [x] **执行脚本**: `release/run_scripts/run_book_crossing.sh` — 4 配置 x 4 seeds = 16 runs
- [ ] **同步到 5090**: `./sync_recbole.sh` 推送配置和脚本
- [ ] **执行实验**: `bash run_scripts/run_book_crossing.sh 0` (预计 ~10h)

---

## P0b: UniSRec 外部基线 (待实现, 优先级较低)

> 回应 R2-W1 / R3-Q3: "PLM-based text-enhanced baseline"

- [ ] 从 [RUCAIBox/UniSRec](https://github.com/RUCAIBox/UniSRec) 移植代码
- [ ] 适配 full-ranking 评估协议 (去 sampled evaluation)
- [ ] 配置文件: Beauty + Toys
- [ ] 执行脚本
- [ ] w/o pretrain 实验 (4 seeds x 2 datasets)
- [ ] w/ pretrain 实验 (可选, 参考上限)

---

## 执行时间线 (5090 有算力后)

```
Step 1: sync 到 5090
  ./sync_recbole.sh

Step 2: 快速消融 (~5h)
  bash run_scripts/run_per_view_ablation.sh 0  # P2 Lite + P4 单视角

Step 3: 外部基线 (~8h)
  bash run_scripts/run_external_baselines.sh 0  # FDSA + S3-Rec

Step 4: GRU4Rec 全矩阵 (~12h)
  bash run_scripts/run_gru4rec_all.sh 0

Step 5: Book-Crossing (~10h)
  bash run_scripts/run_book_crossing.sh 0

Step 6: (可选) UniSRec
```

---

## 产出物清单

完成后论文可新增:

| 表格/图 | 内容 | 覆盖审稿意见 | 状态 |
|---------|------|-------------|------|
| Table: External baselines | FDSA / S3-Rec / UniSRec vs MV-Align (Beauty + Toys) | R2-W1 | ☐ |
| Table: Backbone generalization | GRU4Rec ID / TF-IDF / MV-Align | R2-W2 / R3-Q2 | ☐ |
| Table row: MV-Align-Lite | 精简变体 vs Full MV-Align | R3-Q1 | ☐ |
| Table: Per-view analysis | 4 个单视角 HR/NDCG/MRR | R2-W5 | ☐ |
| Table 1 扩展 | book-crossing 数据集统计 | R2-W3 | ☑ |
| Table: book-crossing results | book-crossing 完整实验矩阵 | R2-W3 / R3-Q2 | ☐ |

### 审稿人质疑覆盖矩阵

| 审稿意见 | 对应实验 | 代码 | 实验 |
|----------|---------|------|------|
| R1: Baselines are ablations | P0 (FDSA/S3-Rec/UniSRec) | ☑ | ☐ |
| R2-W1: 缺文本基线 | P0 (FDSA/S3-Rec/UniSRec) | ☑ | ☐ |
| R2-W2: 单 backbone | P1 (GRU4Rec) | ☑ | ☐ |
| R2-W3: 单领域 | DS (book-crossing) | ☑ | ☐ |
| R2-W5: 各视角效果不清 | P4 (单视角消融) | ☑ | ☐ |
| R3-Q1: 能否精简 | P2 (MV-Align-Lite) | ☑ | ☐ |
| R3-Q2: 是否泛化 | P1 + DS | ☑ | ☐ |
| R3-Q3: 外部基线 | P0 | ☑ (部分, UniSRec 待做) | ☐ |

---

## 关键文件索引

### 本次新增/修改的文件

```
# P2 MV-Align-Lite (✅)
release/configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_lite.yaml

# P4 单视角消融 (✅)
release/configs/ablation/sasrec_align_toys_view0_identity.yaml
release/configs/ablation/sasrec_align_toys_view1_function.yaml
release/configs/ablation/sasrec_align_toys_view2_audience.yaml
release/configs/ablation/sasrec_align_toys_view3_category.yaml
release/run_scripts/run_per_view_ablation.sh

# P0a FDSA + S3-Rec (✅)
release/recbole/model/sequential_recommender/fdsa.py
release/recbole/model/sequential_recommender/s3rec.py
release/recbole/properties/model/FDSA.yaml
release/recbole/properties/model/S3Rec.yaml
release/configs/beauty/fdsa_beauty_stratified.yaml
release/configs/toys/fdsa_toys_stratified.yaml
release/configs/beauty/s3rec_beauty_stratified.yaml
release/configs/toys/s3rec_toys_stratified.yaml
release/run_scripts/run_external_baselines.sh

# P1 GRU4Rec (✅)
release/recbole/model/sequential_recommender/gru4rec.py
release/recbole/model/sequential_recommender/gru4recalignv3.py      # 新编写
release/recbole/model/sequential_recommender/gru4recalignmultiviewv3.py  # 新编写
release/recbole/properties/model/GRU4Rec.yaml
release/recbole/properties/model/GRU4RecAlignV3.yaml                # 新增
release/recbole/properties/model/GRU4RecAlignMultiViewV3.yaml       # 新增
release/configs/beauty/gru4rec_baseline_beauty_stratified.yaml
release/configs/beauty/gru4rec_align_beauty_base_stratified_v3.yaml
release/configs/beauty/gru4rec_align_multi_view_v3_beauty_stratified.yaml
release/configs/toys/gru4rec_baseline_toys_stratified.yaml
release/configs/toys/gru4rec_align_toys_base_stratified_v3.yaml
release/configs/toys/gru4rec_align_multi_view_v3_toys_stratified.yaml
release/run_scripts/run_gru4rec_all.sh

# DS Book-Crossing (✅, 上一轮已完成)
release/configs/book-crossing/*.yaml                                # 4 个配置
release/run_scripts/run_book_crossing.sh

# __init__.py 更新 (✅)
release/recbole/model/sequential_recommender/__init__.py            # 新增 FDSA/S3Rec/GRU4Rec 系列导入
```

### 待创建的文件 (仅 P0b UniSRec)
```
release/recbole/model/sequential_recommender/unisrec.py
release/configs/{beauty,toys}/unisrec_*.yaml
release/run_scripts/run_unisrec.sh
```

### 5090 服务器数据状态
```
/home/charlie/project/RecBole/dataset/
  Amazon_Beauty/       .inter ✓  .item ✓  embeddings(npy) ✓  4views ✓
  Amazon_Toys_and_Games/ .inter ✓  .item ✓  embeddings(npy) ✓  4views ✓
  book-crossing/       .inter ✓ (含timestamp)  .item ✓  embeddings ✓  4views ✓
  yelp/                .inter ✓  .item ✓  (无embedding, 暂不使用)
  ml-1m/               .inter ✓  .item ✓  embeddings ✓  4views ✓ (暂不使用)

/home/charlie/project/RecBole/release/dataset/
  book-crossing/       .inter ✓  .item ✓  embeddings ✓  4views ✓
  Amazon_Beauty/       .gitkeep only (数据在 dataset/ 下, 配置用相对路径)
  Amazon_Toys_and_Games/ .gitkeep only
```

---

## 备注

- S3-Rec 预训练 epochs 设为 30 (原论文默认 500, 前 20 epochs 获得大部分收益)
- book-crossing 没有 `categories` 字段, FDSA/S3-Rec 如需在 book-crossing 上跑, 需用 `book_author` 或 `publisher` 作为 `selected_features`
- book-crossing 的 LLM embedding 使用 Qwen3 (非 Qwen2.5-7B), 对应配置中 emb 路径为 `item_text_emb.qwen3.*.npy`
- Beauty/Toys 的 V3 配置用 Qwen2.5-7B, book-crossing 用 Qwen3 — 论文中需说明
- GRU4RecAlignV3 使用 `embedding_size` (而非 `hidden_size`) 作为 item embedding 维度, 对齐原始 GRU4Rec 的设计
