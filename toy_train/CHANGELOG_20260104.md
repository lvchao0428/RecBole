# Toy Train Changelog - 2026-01-04

## 配置文件概览

本目录包含 Amazon_Toys_and_Games 数据集的训练配置。

### 配置文件列表

| 配置文件 | GPU | Model | use_cross | use_multiview_text_cross | use_text_view_senet | 备注 |
|---------|-----|-------|-----------|--------------------------|---------------------|------|
| `sasrec_align_multi_view_v2_toys_stratified_7b.yaml` | 5 | SASRecAlignMultiViewV2 | true | true | 默认(true) | 完整版 |
| `sasrec_align_multi_view_v2_toys_stratified_7b_nocross.yaml` | 3 | SASRecAlignMultiViewV2 | true | true | false | 消融：无cross(命名有误) |
| `sasrec_align_multi_view_v2_toys_stratified_7b_nosenet.yaml` | 4 | SASRecAlignMultiViewV2 | false | false | true | 消融：有SENet无cross |
| `sasrec_align_multi_view_v2_toys_stratified_7b_nosenet_nocross.yaml` | 5 | SASRecAlignMultiViewV2 | false | false | false | 消融：无SENet无cross |
| `sasrec_align_toys_qwen3_stratified_14b_fair.yaml` | - | SASRecAlign | true | - | - | 公平对比基线 |

---

## 当前配置要点

### Multi-View V2 模型

#### 模型结构
- **Model**: SASRecAlignMultiViewV2
- **hidden_size**: 256
- **n_layers**: 2
- **n_heads**: 2
- **num_text_views**: 4

#### 数据集
- **Dataset**: Amazon_Toys_and_Games
- **load_col.item**: [item_id, title, categories]
- **Text Embedding**: TF-IDF base + Qwen2.5-7B 4-view split embeddings
- **路径**: `/home/ubuntu/own/RecBole/dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views`

#### V2 特性配置
- **multiview_align_scale**: 1.0
- **text_view_senet_ratio**: 2
- **per_view_l2_norm**: true
- **cold_start_align_boost**: 0
- **cold_start_align_threshold**: 10

#### 训练参数
- **epochs**: 50
- **batch_size**: 512
- **learning_rate**: 0.0001
- **eval_step**: 5
- **stopping_step**: 20

### 公平对比基线 (SASRecAlign)

#### 模型结构
- **Model**: SASRecAlign (非 MultiView)
- **hidden_size**: 256
- **n_layers**: 2, **n_heads**: 2

#### 与 MultiView 对齐的配置
- **text_use_senet**: true
- **num_text_views**: 1 (单一视角)
- **text_weight**: 1.0
- **text_gate_init**: 0.7
- **text_gate_reg_l2**: 0.01 (更弱的正则化)

#### 正则化 (高 dropout)
- **hidden_dropout_prob**: 0.5
- **attn_dropout_prob**: 0.5
- **cross_dropout_prob**: 0.5
- **cold_start_align_boost**: 3.0

---

## 配置差异总结

### MultiView V2 各版本差异

| 特性 | 完整版 | nocross | nosenet | nosenet_nocross |
|------|--------|---------|---------|-----------------|
| use_cross | true | true | false | false |
| use_multiview_text_cross | true | true | false | false |
| use_text_view_senet | true | false | true | false |
| GPU | 5 | 3 | 4 | 5 |

### MultiView vs 公平对比基线

| 配置项 | MultiView V2 | SASRecAlign 基线 |
|--------|--------------|------------------|
| Model | SASRecAlignMultiViewV2 | SASRecAlign |
| num_text_views | 4 | 1 |
| Text Source | 4-view split | Single LLM embedding |
| dropout | 0.2 | 0.5 |
| cold_start_align_boost | 0 | 3.0 |
| text_gate_reg_l2 | 0.05 | 0.01 |

---

## 评估指标
- 标准指标: Recall, MRR, NDCG, Hit, Precision
- 分档指标: StratifiedRecall, StratifiedNDCG, StratifiedMRR, StratifiedHit
- 统计指标: ItemPopularityStats

---

## 变更历史

### 2026-01-04
- 创建 changelog 文档
- 记录当前所有配置文件的状态
- 注意：`sasrec_align_multi_view_v2_toys_stratified_7b_nocross.yaml` 文件名与实际配置不一致
  - 文件名暗示 no cross，但实际配置 `use_cross=true`, `use_multiview_text_cross=true`
  - 实际效果是 no SENet (`use_text_view_senet` 未设置/默认)

