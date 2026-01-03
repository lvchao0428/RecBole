# Configuration Changelog

本文件记录 SASRec Multi-View 实验配置的修改历史。

---

## [2026-01-03 19:10] 配置迭代优化 - Commit 9ab7924e

### 📋 Git Commit 历史

| Commit | 时间 | 说明 |
|--------|------|------|
| `9ab7924e` | 19:10 | update change log |
| `01a4aa00` | 19:07 | update text l2 |
| `f994c652` | 18:51 | update tau to 0.05 |
| `ecbfa86f` | 18:34 | update yaml |
| `724599bc` | 12:26 | update senet compress ratio |
| `9239be49` | 11:50 | update 0103 train config |

### 🔧 详细变更记录

#### 1. Text Gate L2 正则化调整 (`01a4aa00`)

**Shell 脚本参数变更** (10 个文件):

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `--phase_a_text_gate_reg_l2` | 0.05 | **0.01** | Phase A 放宽门控约束 |
| `--phase_b_text_gate_reg_l2` | 0.05 | **0.03** | Phase B 适度约束 |

**影响文件**:
- `two_phase_run_tfidf_stratified.sh`
- `two_phase_run_tfidf_llm_stratified.sh`
- `two_phase_run_tfidf_toys_stratified.sh`
- `two_phase_run_tfidf_llm_toys_stratified.sh`
- `two_phase_run_multiview_v2_stratified.sh`
- `two_phase_run_multiview_v2_stratified_14b.sh`
- `two_phase_run_multiview_v2_stratified_32b.sh`
- `two_phase_run_multiview_v2_toys_stratified_7b.sh`
- `two_phase_run_multiview_v2_toys_stratified_14b.sh`
- `two_phase_run_multiview_v2_toys_stratified_32b.sh`

#### 2. Temperature 和 Alignment Weight 调整 (`f994c652`)

**Shell 脚本参数变更** (10 个文件):

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `--tau_grid` | 0.12 | **0.05** | 降低温度，对齐更严格 |
| `--phase_b_alignment_weight` | 0.05 | **0.15** | Phase B 增强对齐权重 |

#### 3. SENet 压缩比调整 (`724599bc`)

**YAML 配置变更** (7 个文件):

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `text_view_senet_ratio` | 1 | **2** | 恢复轻度压缩 (256→128→256) |

**影响文件**:
- `sasrec_align_multi_view_v2_stratified.yaml`
- `sasrec_align_multi_view_v2_stratified_14b.yaml`
- `sasrec_align_multi_view_v2_stratified_32b.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_7b.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_14b.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_32b.yaml`
- `recbole/model/sequential_recommender/text_amplifier.py`

#### 4. 正则化参数统一 (`9239be49`)

**YAML 配置变更** (12 个 YAML + 10 个 Shell):

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `text_gate_reg_l2` | 0.01 | **0.05** | 恢复中等门控约束 |
| `label_smoothing` | 0.05 | **0.1** | 恢复标准标签平滑 |

---

### 📊 当前最终配置状态

#### Shell 脚本参数

```bash
# Temperature 和对齐
--tau_grid "0.05"                    # 对齐严格
--align_grid "0.08"
--phase_b_alignment_weight 0.15      # Phase B 增强

# 门控正则化 (渐进式)
--phase_a_text_gate_reg_l2 0.01      # Phase A 宽松
--phase_b_text_gate_reg_l2 0.03      # Phase B 适中

# 学习率
--lr_text_head 2e-3
--lr_dnn_cross 5e-4
--backbone_lr_scale 0.1
```

#### YAML 配置

```yaml
# SENet
text_view_senet_ratio: 2             # 轻度压缩 (256→128→256)

# 门控
text_gate_init: 0.7
text_gate_reg_l2: 0.05               # YAML 中的默认值

# 正则化
label_smoothing: 0.1
hidden_dropout_prob: 0.2
```

---

### ⚠️ 注意事项

1. **Shell 参数会覆盖 YAML**: 两阶段训练脚本中的 `--phase_a_text_gate_reg_l2` 和 `--phase_b_text_gate_reg_l2` 会在运行时覆盖 YAML 中的 `text_gate_reg_l2`

2. **tau 调整方向**: 从 0.12 → 0.05 是大幅降低，可能导致对齐过于严格。如果文本效果不好，考虑调回 0.1

3. **SENet 压缩比**: 从 1 调回 2，增加了信息压缩，但也可能提升泛化能力

---

*Updated: 2026-01-03 19:10*

---

## [2026-01-03] Multi-View V2 配置统一与优化

### 📋 修改概述

本次修改统一了所有 Multi-View V2 配置文件的参数，优化文本特征相关配置。

### 🔧 核心配置变更

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `multiview_align_scale` | 2.0 | **1.0** | 降低对齐损失放大系数 |
| `text_view_senet_ratio` | 2 | **1** | SENet 无压缩 (256/1=256维) |
| `text_gate_init` | 0.5 | **0.7** | 初始偏重多视图特征 |
| `text_gate_reg_l2` | 0.05 | **0.01** (14B) / **0.05** (其他) | 放宽门控约束 |
| `hidden_dropout_prob` | 0.5 | **0.2** | 降低 Dropout |
| `attn_dropout_prob` | 0.5 | **0.2** | 降低 Dropout |
| `cross_dropout_prob` | 0.5 | **0.2** | 降低 Dropout |
| `label_smoothing` | 0.1 | **0.05** (14B) / **0.1** (其他) | 减少标签平滑 |
| `text_weight` | 0.8 | **1.0** | 增强文本权重 |

### 📁 修改的文件列表

#### YAML 配置文件

**Beauty 数据集:**
- `sasrec_baseline_50ep_stratified.yaml` - 纯 SASRec 基线
- `sasrec_align_base_stratified.yaml` - TF-IDF 基线
- `sasrec_align_qwen3_stratified.yaml` - TF-IDF + LLM
- `sasrec_align_multi_view_v2_stratified.yaml` - Multi-View 7B
- `sasrec_align_multi_view_v2_stratified_14b.yaml` - Multi-View 14B
- `sasrec_align_multi_view_v2_stratified_32b.yaml` - Multi-View 32B

**Toys 数据集:**
- `sasrec_baseline_50ep_toys_stratified.yaml` - 纯 SASRec 基线
- `sasrec_align_toys_base_stratified.yaml` - TF-IDF 基线
- `sasrec_align_toys_qwen3_stratified.yaml` - TF-IDF + LLM
- `sasrec_align_multi_view_v2_toys_stratified_7b.yaml` - Multi-View 7B
- `sasrec_align_multi_view_v2_toys_stratified_14b.yaml` - Multi-View 14B
- `sasrec_align_multi_view_v2_toys_stratified_32b.yaml` - Multi-View 32B

#### Shell 脚本文件

**Beauty 数据集:**
- `two_phase_run_tfidf_stratified.sh` - TF-IDF 基线训练
- `two_phase_run_tfidf_llm_stratified.sh` - TF-IDF + LLM 训练
- `two_phase_run_multiview_v2_stratified.sh` - Multi-View 7B 训练
- `two_phase_run_multiview_v2_stratified_14b.sh` - Multi-View 14B 训练
- `two_phase_run_multiview_v2_stratified_32b.sh` - Multi-View 32B 训练

**Toys 数据集:**
- `two_phase_run_tfidf_toys_stratified.sh` - TF-IDF 基线训练
- `two_phase_run_tfidf_llm_toys_stratified.sh` - TF-IDF + LLM 训练
- `two_phase_run_multiview_v2_toys_stratified_7b.sh` - Multi-View 7B 训练
- `two_phase_run_multiview_v2_toys_stratified_14b.sh` - Multi-View 14B 训练
- `two_phase_run_multiview_v2_toys_stratified_32b.sh` - Multi-View 32B 训练

---

### 📊 各配置文件当前参数汇总

#### Multi-View V2 配置 (所有模型规模)

```yaml
# V2 增强配置
multiview_align_scale: 1.0      # 对齐损失放大系数
text_view_senet_ratio: 1        # SENet 无压缩
per_view_l2_norm: true          # 每个视图独立 L2 归一化
text_view_half_precision: true  # 半精度存储嵌入

# 冷启动配置 (已关闭)
cold_start_align_boost: 0
cold_start_align_threshold: 10

# 对齐与门控
alignment_weight: 0.05
temperature: 0.07
text_gate_init: 0.7
text_gate_reg_l2: 0.05          # 14B 为 0.01
text_weight: 1.0

# 正则化
hidden_dropout_prob: 0.2
attn_dropout_prob: 0.2
cross_dropout_prob: 0.2
token_dropout_prob: 0.2
label_smoothing: 0.1            # 14B 为 0.05

# 评分
cosine_score: true
cosine_scale: 10.0
```

#### 基线配置 (TF-IDF / TF-IDF+LLM)

```yaml
# 对齐与门控
alignment_weight: 0.05
temperature: 0.07
text_gate_init: 0.7
text_gate_reg_l2: 0.05
text_weight: 1.0

# 公平比较配置
text_use_senet: true            # 启用 SENet 增强
num_text_views: 1               # 单视图

# 正则化
hidden_dropout_prob: 0.2
attn_dropout_prob: 0.2
cross_dropout_prob: 0.2
label_smoothing: 0.1

# 评分
cosine_score: true
cosine_scale: 10.0
```

#### 纯 SASRec 基线

```yaml
# 文本特征完全禁用
disable_text_feature: true
fuse_text_feature: false
use_align: false
use_cross: false
alignment_weight: 0.0
text_weight: 0.0

# 正则化
hidden_dropout_prob: 0.2
attn_dropout_prob: 0.2
cosine_score: false
```

---

### 📝 Shell 脚本参数汇总

所有两阶段训练脚本使用统一的参数：

```bash
# Phase A 配置
--phase_a_grid
--align_grid "0.05"
--tau_grid "0.07"
--backbone_burnin_epochs 10
--burnin_eval_step 2
--phase_a_epochs 20
--phase_a_eval_step 1
--phase_a_valid_metric "MRR@10"
--metric_baseline 0.0272
--metric_gain_threshold 0.01
--phase_a_text_gate_reg_l2 0.05

# 学习率
--lr_text_head 1e-3
--lr_dnn_cross 5e-4
--backbone_lr_scale 0.1

# Phase B 配置
--phase_a_auto_to_b
--phase_b_epochs 40
--phase_b_alignment_weight 0.05
--phase_b_text_gate_reg_l2 0.05
--phase_b_text_weight 1.0

# 其他
--seed 2025
--watchdog_disable
--save
```

---

### ⚠️ 已知问题

1. **文本特征效果削弱**: 
   - `multiview_align_scale` 从 2.0 降到 1.0 可能削弱对齐信号
   - `temperature` 从 0.1 降到 0.07 可能过于严格
   - 建议后续实验尝试恢复 `temperature: 0.1` 和 `multiview_align_scale: 2.0`

2. **Shell 脚本注释与 YAML 不一致**:
   - Shell 脚本注释说 `alignment_weight: 0.15`，但实际 YAML 中是 `0.05`
   - 建议统一注释与实际配置

---

### 🔄 建议的后续优化方向

如果文本特征效果继续削弱，建议尝试：

```yaml
# 方案 A: 恢复之前有效的配置
temperature: 0.1
multiview_align_scale: 2.0
hidden_dropout_prob: 0.4

# 方案 B: 增强对齐信号
temperature: 0.1
alignment_weight: 0.08
multiview_align_scale: 3.0
```

---

## 历史版本

### [之前版本] 原始配置参考

```yaml
# 原始 V2 配置
multiview_align_scale: 2.0
text_view_senet_ratio: 2
text_gate_init: 0.5
text_gate_reg_l2: 0.05
hidden_dropout_prob: 0.5
attn_dropout_prob: 0.5
cross_dropout_prob: 0.5
label_smoothing: 0.1
text_weight: 0.8
temperature: 0.1
```

---

*Last updated: 2026-01-03*

