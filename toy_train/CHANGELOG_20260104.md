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
- **text_gate_init**: 0.8 *(2026-01-05 更新: 0.7→0.8)*
- **text_gate_reg_l2**: 0.02 *(2026-01-05 更新: 0.05→0.02)*

#### 正则化 (标准 dropout)
- **hidden_dropout_prob**: 0.2
- **attn_dropout_prob**: 0.2
- **cross_dropout_prob**: 0.2
- **cold_start_align_boost**: 0 (关闭)

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
| dropout | 0.2 | 0.2 |
| cold_start_align_boost | 0 | 0 |
| text_gate_init | 0.8 | 0.8 |
| text_gate_reg_l2 | 0.02 | 0.02 |

> **注**: 以上为 2026-01-05 更新后的值。旧值: text_gate_init=0.7, text_gate_reg_l2=0.05

---

## 评估指标
- 标准指标: Recall, MRR, NDCG, Hit, Precision
- 分档指标: StratifiedRecall, StratifiedNDCG, StratifiedMRR, StratifiedHit
- 统计指标: ItemPopularityStats

---

## 变更历史

### 2026-01-05: 增强 Text 特征权重实验

#### 问题背景

根据 Toy 数据集消融实验结果发现异常：
- **tfidf + llm + no_whiten** 比 **tfidf + llm** 效果更好 (recall@5: 0.0534 vs 0.0527, +1.33%)
- **multi-view + 7b + no_whiten** 下降温和 (-1.32%)，而 Beauty 数据集类似配置 recall 上升 7.03%

这表明 Toy 数据集上文本特征的影响力不足，白化效果不明显。

#### 修改内容

**YAML 配置文件修改** (增强文本门控初始化和降低正则化):

| 配置项 | 旧值 | 新值 | 说明 |
|--------|------|------|------|
| `text_gate_init` | 0.7 | 0.8 | 提高初始门控权重，增强文本融合 |
| `text_gate_reg_l2` | 0.05 | 0.02 | 降低 L2 正则化，让 gate 更自由学习 |

**受影响的配置文件 (项目根目录)**:
- `sasrec_align_toys_base_stratified.yaml`
- `sasrec_align_toys_qwen3_stratified.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_7b.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_14b.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_32b.yaml`

**受影响的配置文件 (toy_train 目录)**:
- `sasrec_align_toys_qwen3_stratified_no_whiten.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_7b_no_whiten.yaml`

**Shell 脚本修改** (增加 Phase B 的门控正则化参数):

新增参数: `--phase_b_text_gate_reg_l2 0.01`

**受影响的脚本 (项目根目录)**:
- `two_phase_run_tfidf_toys_stratified.sh`
- `two_phase_run_tfidf_llm_toys_stratified.sh`
- `two_phase_run_multiview_v2_toys_stratified_7b.sh`
- `two_phase_run_multiview_v2_toys_stratified_14b.sh`
- `two_phase_run_multiview_v2_toys_stratified_32b.sh`

**受影响的脚本 (toy_train 目录)**:
- `two_phase_run_tfidf_llm_toys_stratified_no_whiten.sh`
- `two_phase_run_multiview_v2_toys_stratified_7b_no_whiten.sh`

#### 预期效果

| 调整项 | 预期影响 |
|--------|----------|
| ↑ text_gate_init (0.7→0.8) | 增强文本对 fused embedding 的初始贡献 |
| ↓ text_gate_reg_l2 (0.05→0.02) | 让 gate 更自由学习最优权重 |
| ↓ phase_b_text_gate_reg_l2 (0.01) | Phase B 微调时进一步降低约束 |

---

### 2026-01-04
- 创建 changelog 文档
- 记录当前所有配置文件的状态
- 注意：`sasrec_align_multi_view_v2_toys_stratified_7b_nocross.yaml` 文件名与实际配置不一致
  - 文件名暗示 no cross，但实际配置 `use_cross=true`, `use_multiview_text_cross=true`
  - 实际效果是 no SENet (`use_text_view_senet` 未设置/默认)

