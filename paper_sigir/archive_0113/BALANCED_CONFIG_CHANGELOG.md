# Balanced Configuration Changelog

> 实验配置版本管理，用于论文写作参考

---

## v1.0 - Balanced Configuration (2025-01-10)

### 背景与动机

基于 0109 实验数据分析发现：

1. **过强的 text/cold_start 压制高频**: `aggressive` 配置 (`alignment_weight=0.15`, `cold_start_boost=3.0`) 导致 HR_frequent 下降 5.4%
2. **Text 权重不能太高**: `text_weight=1.0` 时 text 通道过于主导
3. **No-whiten 在 Toys 不 work**: 但在 Beauty 上效果与 whiten 相当

### 核心发现 (from 0109toy_aggressive.csv)

| 配置 | HR@10 | MRR@10 | HR_new | HR_few | HR_freq |
|------|-------|--------|--------|--------|---------|
| base | 5.97% | 2.49% | 1.91% | 4.61% | 10.61% |
| tfidf | 6.86% | 3.66% | 2.24% | 5.46% | 12.11% |
| tfidf+llm | 6.83% | 3.64% | 2.26% | 5.31% | 12.10% |
| MV-7B (原) | 6.90% | 3.67% | 2.30% | 5.40% | 12.21% |
| **squeeze** | **6.93%** | **3.67%** | 2.36% | 5.47% | **12.21%** |
| squeeze_nowhiten | 6.79% | 3.63% | 2.28% | 5.40% | 11.96% |
| **aggressive** | 6.68% | 3.62% | **2.48%** | **5.54%** | 11.55% ❌ |

### Balanced V1 配置

| 参数 | 原配置 | Aggressive | **Balanced V1** | 设计原因 |
|------|--------|------------|-----------------|----------|
| `alignment_weight` | 0.10/0.15 | 0.15 | **0.07** | 温和对齐，不过度压制高频 |
| `temperature` | 0.05 | 0.03 | **grid: 0.03,0.05,0.07** | Phase-A 搜索最优 |
| `text_weight` | 0.7/1.0 | 0.6 | **0.8** | 适中，不让 text 主导 |
| `cold_start_boost` | 0/3.0 | 3.0 | **2.0** | 温和加权 (最高 3x) |
| `cross_dropout_prob` | 0.15/0.2 | 0.1 | **0.15** | 稳定正则化 |
| `multiview_align_scale` | 1.0/2.0 | 2.0 | **1.0** | 不放大对齐 loss |

---

## 创建的配置文件

### YAML 配置

#### Beauty Dataset
- `sasrec_align_base_balanced.yaml` - TF-IDF baseline
- `sasrec_align_qwen3_balanced.yaml` - TF-IDF + LLM (Single-View)
- `sasrec_align_multi_view_v2_beauty_balanced.yaml` - Multi-View 7B
- `sasrec_align_multi_view_v2_beauty_balanced_14b.yaml` - Multi-View 14B
- `sasrec_align_multi_view_v2_beauty_balanced_32b.yaml` - Multi-View 32B

#### Toys Dataset
- `sasrec_align_toys_base_balanced.yaml` - TF-IDF baseline
- `sasrec_align_toys_qwen3_balanced.yaml` - TF-IDF + LLM (Single-View)
- `sasrec_align_multi_view_v2_toys_balanced.yaml` - Multi-View 7B
- `sasrec_align_multi_view_v2_toys_balanced_14b.yaml` - Multi-View 14B
- `sasrec_align_multi_view_v2_toys_balanced_32b.yaml` - Multi-View 32B

### Shell 脚本

#### Beauty Dataset
- `two_phase_run_tfidf_balanced.sh`
- `two_phase_run_tfidf_llm_balanced.sh`
- `two_phase_run_multiview_v2_beauty_balanced.sh`
- `two_phase_run_multiview_v2_beauty_balanced_14b.sh`
- `two_phase_run_multiview_v2_beauty_balanced_32b.sh`

#### Toys Dataset
- `two_phase_run_tfidf_toys_balanced.sh`
- `two_phase_run_tfidf_llm_toys_balanced.sh`
- `two_phase_run_multiview_v2_toys_balanced.sh`
- `two_phase_run_multiview_v2_toys_balanced_14b.sh`
- `two_phase_run_multiview_v2_toys_balanced_32b.sh`

---

## 公平对比设计

### 统一参数
所有模型使用相同的核心参数，确保公平对比：

```yaml
alignment_weight: 0.07
temperature: grid search [0.03, 0.05, 0.07]
text_weight: 0.8
cold_start_align_boost: 2.0
cold_start_align_threshold: 10
cross_dropout_prob: 0.15
```

### Two-Phase Training 参数

```bash
--backbone_burnin_epochs 10
--burnin_eval_step 2
--phase_a_epochs 20
--phase_a_eval_step 1
--phase_b_epochs 40
--backbone_lr_scale 0.1
--lr_text_head 2e-3
--lr_dnn_cross 5e-4
--phase_a_text_gate_reg_l2 0.01
--phase_b_text_gate_reg_l2 0.03
```

---

## 预期层级关系

| Rank | Model | 预期优势 |
|------|-------|---------|
| 1 | Multi-View 32B | 最强语义，Scale Law 验证 |
| 2 | Multi-View 14B | 高质量语义 |
| 3 | Multi-View 7B | 基础多视角 |
| 4 | TF-IDF+LLM | 单视角 LLM |
| 5 | TF-IDF | 基础文本特征 |
| 6 | Base (50ep) | 纯 ID 协同 |

---

## 运行命令

### Beauty (GPU:0)
```bash
./two_phase_run_tfidf_balanced.sh                      # TF-IDF
./two_phase_run_tfidf_llm_balanced.sh                  # TF-IDF+LLM
./two_phase_run_multiview_v2_beauty_balanced.sh        # MV-7B
./two_phase_run_multiview_v2_beauty_balanced_14b.sh    # MV-14B
./two_phase_run_multiview_v2_beauty_balanced_32b.sh    # MV-32B
```

### Toys (GPU:1)
```bash
./two_phase_run_tfidf_toys_balanced.sh                 # TF-IDF
./two_phase_run_tfidf_llm_toys_balanced.sh             # TF-IDF+LLM
./two_phase_run_multiview_v2_toys_balanced.sh          # MV-7B
./two_phase_run_multiview_v2_toys_balanced_14b.sh      # MV-14B
./two_phase_run_multiview_v2_toys_balanced_32b.sh      # MV-32B
```

---

## 关键代码变更 (CHANGE-7)

### sasrecalignmultiviewv2.py

在 `_fuse_with_cross_network` 方法中，重新引入 `alignment_weight` 和 `temperature` 对推理时 `effective_text_weight` 的影响：

```python
# CHANGE-7: Re-enable alignment_weight and temperature to influence inference
if not self.training:
    align_scale = 1.0 + self.alignment_weight
    temp_scale = 0.07 / self.temperature
    effective_text_weight = effective_text_weight * align_scale * temp_scale
```

### 推理时 text weight 公式

```
effective = alpha × text_weight × (1 + align_weight) × (0.07 / temp)
          ≈ 0.67 × 0.8 × 1.07 × 1.4 ≈ 0.80
```

---

## 待验证假设

1. **Scale Law**: 7B → 14B → 32B 在 new/few 指标上是否呈现递增趋势
2. **层级关系**: Multi-View > Single-View > TF-IDF > Base 是否在两个数据集上一致
3. **整体提升**: Balanced 配置能否同时提升 HR 和 MRR

---

---

## v1.1 - Bug Fix: Stratified Metrics Threshold (2025-01-10)

### 问题描述

验证指标计算时发现：分层 MRR 加权平均与整体 MRR 不一致

```
加权平均 MRR: 0.0528
实际 MRR@10:  0.0366
差异: 44.36%  ← 这说明有用户被排除了！
```

### 根因分析

`StratifiedMRR._get_item_stratum()` 中：
- `new_threshold = (1, 3)`，要求 `count >= 1`
- 训练集中未出现的 item (`count=0`) 返回 `None`
- 这些用户在整体指标中计入，但在分层指标中被排除

### 修复内容

```python
# Before (BUG)
self.new_threshold = (1, 3)

# After (FIXED)
self.new_threshold = (0, 3)  # Include count=0 (unseen items)
```

**影响的类**：
- `StratifiedRecall`
- `StratifiedNDCG`
- `StratifiedMRR`
- `StratifiedHit`
- `ItemPopularityStats`

### 影响

修复后：
- `new` 分层现在包含 `count=0` 的 unseen items
- 分层指标加权平均将与整体指标一致
- **之前的实验数据中 `*_new@K` 指标偏高**（因为排除了最难的 unseen items）
- 需要重新运行实验以获得正确的分层指标

---

## 相关文件引用

- 实验数据: `paper_sigir/0109toy_aggressive.csv`, `paper_sigir/0109_toy_ipw.csv`
- 之前的实验配置分析: `paper_sigir/0107.csv`, `paper_sigir/0108.csv`
- 代码变更记录: `TOYS_SQUEEZE_CHANGELOG.md`
- 参数对齐记录: `PARAMETER_ALIGNMENT_CHANGELOG.md`
- 配置变更记录: `CONFIG_CHANGELOG.md`
- 指标验证脚本: `scripts/verify_stratified_metrics.py`