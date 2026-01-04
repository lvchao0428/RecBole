# Beauty Train Changelog - 2026-01-04

## 配置文件概览

本目录包含 Amazon_Beauty 数据集的 SASRecAlignMultiViewV2 模型训练配置。

### 配置文件列表

| 配置文件 | GPU | use_cross | use_multiview_text_cross | use_text_view_senet | 备注 |
|---------|-----|-----------|--------------------------|---------------------|------|
| `sasrec_align_multi_view_v2_stratified.yaml` | - | false | false | 默认(true) | 基础配置，高dropout |
| `sasrec_align_multi_view_v2_stratified_7b.yaml` | 0 | true | true | 默认(true) | Qwen2.5-7B完整版，低dropout |
| `sasrec_align_multi_view_v2_stratified_no_cross.yaml` | 1 | false | false | 默认(true) | 消融实验：无cross attention |
| `sasrec_align_multi_view_v2_stratified_no_senet.yaml` | 0 | true | true | false | 消融实验：无SENet |
| `sasrec_align_multi_view_v2_stratified_no_cross_and_senet.yaml` | 2 | false | false | false | 消融实验：无cross和SENet |

---

## 当前配置要点

### 模型结构
- **Model**: SASRecAlignMultiViewV2
- **hidden_size**: 256
- **n_layers**: 2
- **n_heads**: 2
- **num_text_views**: 4 (4个视角的文本嵌入)

### 数据集
- **Dataset**: Amazon_Beauty
- **Text Embedding**: TF-IDF base + Qwen2.5-7B 4-view split embeddings
- **路径**: `/home/ubuntu/own/RecBole/dataset/Amazon_Beauty/qwen2.5_7b_4views`

### V2 新增特性
1. **multiview_align_scale**: 1.0-2.0 (Multi-View对齐损失放大系数)
2. **text_view_senet_ratio**: 2 (SENet压缩比)
3. **per_view_l2_norm**: true (每个view独立L2归一化)
4. **cold_start_align_boost**: 0-3.0 (冷启动对齐权重增强)

### 训练参数
- **epochs**: 50
- **batch_size**: 512
- **learning_rate**: 0.0001
- **eval_step**: 5
- **stopping_step**: 20

### 正则化配置

| 配置 | 完整版 (7b) | 基础版/消融版 |
|------|-------------|---------------|
| hidden_dropout_prob | 0.2 | 0.2-0.5 |
| attn_dropout_prob | 0.2 | 0.2-0.5 |
| token_dropout_prob | 0.2 | 0.2 |
| cross_dropout_prob | 0.2 | 0.2-0.5 |
| text_weight | 1.0 | 0.8-1.0 |
| text_gate_init | 0.7 | 0.5-0.7 |

### 评估指标
- 标准指标: Recall, MRR, NDCG, Hit, Precision
- 分档指标: StratifiedRecall, StratifiedNDCG, StratifiedMRR, StratifiedHit
- 统计指标: ItemPopularityStats

---

## 主要配置差异

### 1. 完整版 vs 基础版
- 完整版启用 `use_cross=true` 和 `use_multiview_text_cross=true`
- 完整版使用更低的 dropout (0.2 vs 0.5)

### 2. 消融实验配置
- **no_cross**: 关闭 cross attention，测试纯 align 效果
- **no_senet**: 关闭 SENet，测试无注意力加权的多视角融合
- **no_cross_and_senet**: 同时关闭 cross 和 SENet

---

## 变更历史

### 2026-01-04
- 创建 changelog 文档
- 记录当前所有配置文件的状态

