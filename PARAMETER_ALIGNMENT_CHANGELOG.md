# 参数对齐更改日志

**日期**: 2025-12-31  
**参考脚本**: `two_phase_run_multiview_v2_toys_stratified_14b.sh` 和 `sasrec_align_multi_view_v2_toys_stratified_14b.yaml`

---

## 概述

为了确保所有 SASRec 及其扩展模型在 Beauty 和 Toys 数据集上的实验参数一致性，对以下 Shell 脚本和 YAML 配置文件进行了参数对齐。

---

## Shell 脚本参数更改

### 更改的参数

| 参数 | 原值 | 新值 |
|------|------|------|
| `--lr_text_head` | `1e-3` | `2e-3` |
| `--lr_dnn_cross` | `5e-4` | `1e-3` |
| `--phase_b_alignment_weight` | `0.05` | `0.10` |

### Beauty 数据集脚本

| 脚本文件 | 状态 |
|----------|------|
| `run50epBase_stratified.sh` | 基线脚本，无需两阶段训练参数 |
| `two_phase_run_tfidf_stratified.sh` | ✅ 已更新 |
| `two_phase_run_tfidf_llm_stratified.sh` | ✅ 已更新 |
| `two_phase_run_multiview_v2_stratified.sh` | ✅ 已更新 |
| `two_phase_run_multiview_v2_stratified_32b.sh` | ✅ 已更新 |
| `two_phase_run_multiview_v2_stratified_14b.sh` | ✅ 新创建 |

### Toys 数据集脚本

| 脚本文件 | 状态 |
|----------|------|
| `run50epBase_toys_stratified.sh` | 基线脚本，无需两阶段训练参数 |
| `two_phase_run_tfidf_toys_stratified.sh` | ✅ 已更新 |
| `two_phase_run_tfidf_llm_toys_stratified.sh` | ✅ 已更新 |
| `two_phase_run_multiview_v2_toys_stratified_7b.sh` | ✅ 已更新 |
| `two_phase_run_multiview_v2_toys_stratified_14b.sh` | 参考脚本，无需更改 |
| `two_phase_run_multiview_v2_toys_stratified_32b.sh` | ✅ 已更新 |

---

## YAML 配置文件参数更改

### 通用参数更改

| 参数 | 原值 | 新值 |
|------|------|------|
| `text_weight` | `0.8` | `1.0` |
| `text_gate_init` | `0.5` | `0.7` |
| `text_gate_reg_l2` | `0.05` | `0.01` |

### Multi-View 专属参数更改

| 参数 | 原值 | 新值 |
|------|------|------|
| `multiview_align_scale` | `2.0` | `4.0` |
| `text_view_senet_ratio` | `2` | `1` |

### Beauty 数据集配置

| 配置文件 | 状态 |
|----------|------|
| `sasrec_align_base_stratified.yaml` | ✅ 已更新 |
| `sasrec_align_qwen3_stratified.yaml` | ✅ 已更新 |
| `sasrec_align_multi_view_v2_stratified.yaml` | ✅ 已更新 |
| `sasrec_align_multi_view_v2_stratified_32b.yaml` | ✅ 已更新 |
| `sasrec_align_multi_view_v2_stratified_14b.yaml` | ✅ 已更新 (embedding路径修正为14b) |

### Toys 数据集配置

| 配置文件 | 状态 |
|----------|------|
| `sasrec_align_toys_base_stratified.yaml` | ✅ 已更新 |
| `sasrec_align_toys_qwen3_stratified.yaml` | ✅ 已更新 |
| `sasrec_align_multi_view_v2_toys_stratified_7b.yaml` | ✅ 已更新 |
| `sasrec_align_multi_view_v2_toys_stratified_14b.yaml` | 参考配置，无需更改 |
| `sasrec_align_multi_view_v2_toys_stratified_32b.yaml` | ✅ 已更新 |

---

## 新创建的文件

### `two_phase_run_multiview_v2_stratified_14b.sh`

基于 `two_phase_run_multiview_v2_toys_stratified_14b.sh` 为 Beauty 数据集创建的新脚本。

**主要配置:**
- Dataset: `Amazon_Beauty`
- Model: `SASRecAlignMultiViewV2`
- Config: `sasrec_align_multi_view_v2_stratified_14b.yaml`
- LLM: Qwen2.5-14B-Instruct

---

## 参数说明

### Shell 脚本参数

- **`--lr_text_head`**: 文本头部网络的学习率
- **`--lr_dnn_cross`**: Cross DNN 网络的学习率
- **`--phase_b_alignment_weight`**: Phase B 阶段的对齐损失权重

### YAML 配置参数

- **`text_weight`**: 文本特征的融合权重
- **`text_gate_init`**: 文本门控的初始值
- **`text_gate_reg_l2`**: 文本门控的 L2 正则化系数
- **`multiview_align_scale`**: Multi-View 对齐损失的放大系数
- **`text_view_senet_ratio`**: SENet 压缩比 (hidden_size / ratio)

---

## 备注

1. 基线脚本 (`run50epBase_stratified.sh`, `run50epBase_toys_stratified.sh`) 使用 `run_recbole.py` 直接训练，不涉及两阶段训练参数，因此不需要更新。

2. `sasrec_align_multi_view_v2_stratified_14b.yaml` 的 `item_text_emb_split_dir` 路径已从 `qwen2.5_32b_4views` 修正为 `qwen2.5_14b_4views`。

3. 所有更改保持与参考脚本 `two_phase_run_multiview_v2_toys_stratified_14b.sh` 的参数一致性。

