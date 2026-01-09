# Toys Squeeze 实验 Changelog

**日期**: 2025-01-09  
**目标**: 压榨 Toys 性能，使 no-whiten 趋势与 Beauty 一致

---

## 背景

### 问题发现

在 0107.csv 数据分析中发现 Beauty 和 Toys 的 no-whiten 趋势不一致：

| 数据集 | Multi-view no whiten | Single-view no whiten |
|--------|---------------------|----------------------|
| **Beauty** | HR↑1.3%, MRR↓1.9% | 几乎无变化 |
| **Toys** | HR↓0.9%, MRR↓1.6% ❌ | HR↑2.5%, MRR↑1.1% |

### 假设

Toys 的 text 特征没有充分学习，原因是配置差异：

| 参数 | Beauty | Toys (原) | 差异影响 |
|------|--------|-----------|---------|
| `alignment_weight` | **0.10** | 0.05 | Toys 对齐强度只有一半 |
| `temperature` | **0.05** | 0.07 | Toys InfoNCE 更松散 |
| `text_weight` | 0.7 | **1.0** | Toys 过度依赖 text |
| `cross_dropout_prob` | **0.15** | 0.2 | Toys cross 信号更弱 |
| 冷启动机制 | `cold_start_boost=3.0` | IPW | 不同机制 |

---

## 创建的实验配置

### 1. Squeeze (基础压榨) - 对齐 Beauty

**文件**:
- `sasrec_align_multi_view_v2_toys_stratified_7b_squeeze.yaml`
- `two_phase_run_multiview_v2_toys_stratified_7b_squeeze.sh`

**核心修改** (对比原 Toys 配置):
```yaml
alignment_weight: 0.10       # 从 0.05 提升
temperature: 0.05            # 从 0.07 降低
text_weight: 0.7             # 从 1.0 降低
cross_dropout_prob: 0.15     # 从 0.2 降低
cold_start_align_boost: 3.0  # 开启（原为 0，使用 IPW）
use_ipw_weighting: false     # 关闭 IPW
```

**目标**: 充分学习 text 特征，预期 HR ≥ 7.0%

---

### 2. Squeeze + No Whiten - 验证趋势

**文件**:
- `sasrec_align_multi_view_v2_toys_stratified_7b_squeeze_nowhiten.yaml`
- `two_phase_run_multiview_v2_toys_stratified_7b_squeeze_nowhiten.sh`

**核心修改** (对比 Squeeze):
```yaml
# 使用 center_only 版本（无 whitening）
item_text_emb_path_base: dataset/Amazon_Toys_and_Games/item_text_emb.base.center_only.npy
item_text_emb_split_dir: dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_center_only
```

**目标**: 验证 no-whiten 趋势是否与 Beauty 一致
**预期**: HR↑, MRR↓ (关键验证点！)

---

### 3. Aggressive (激进压榨) - 探索上限

**文件**:
- `sasrec_align_multi_view_v2_toys_stratified_7b_squeeze_aggressive.yaml`
- `two_phase_run_multiview_v2_toys_stratified_7b_squeeze_aggressive.sh`

**核心修改** (对比 Squeeze):
```yaml
alignment_weight: 0.15       # 更强对齐
temperature: 0.03            # 更紧的 InfoNCE
text_weight: 0.6             # 更低的 text 权重
multiview_align_scale: 2.0   # 开启对齐放大
cross_dropout_prob: 0.1      # 更稳定的 cross
cosine_scale: 12.0           # 略高的 cosine scale
epochs: 60                   # 更长训练
phase_b_epochs: 50           # 更长 phase B
```

**目标**: 探索 Toys 性能上限

---

## 配置对比总表

| 参数 | 原 Toys | Squeeze | Squeeze+NoWhiten | Aggressive |
|------|---------|---------|------------------|------------|
| `alignment_weight` | 0.05 | 0.10 | 0.10 | 0.15 |
| `temperature` | 0.07 | 0.05 | 0.05 | 0.03 |
| `text_weight` | 1.0 | 0.7 | 0.7 | 0.6 |
| `multiview_align_scale` | 1.0 | 1.0 | 1.0 | 2.0 |
| `cross_dropout_prob` | 0.2 | 0.15 | 0.15 | 0.1 |
| `cold_start_align_boost` | 0 (IPW) | 3.0 | 3.0 | 3.0 |
| `phase_b_epochs` | 40 | 40 | 40 | 50 |
| Whiten | ✅ | ✅ | ❌ | ✅ |

---

## 运行命令

```bash
cd /Users/lvchao0428/project/ownRecBole/RecBole

# 实验 1: Squeeze (w/ whiten)
./two_phase_run_multiview_v2_toys_stratified_7b_squeeze.sh

# 实验 2: Squeeze + No Whiten
./two_phase_run_multiview_v2_toys_stratified_7b_squeeze_nowhiten.sh

# 实验 3: Aggressive
./two_phase_run_multiview_v2_toys_stratified_7b_squeeze_aggressive.sh
```

---

## 预期结果验证

| 对比 | 预期 | 如果符合说明 |
|------|------|-------------|
| Squeeze vs 原 Toys | HR↑, MRR↑ | 配置差异是问题根因 |
| Squeeze vs Squeeze+NoWhiten | **HR↑, MRR↓** | Toys 与 Beauty 趋势一致 ✓ |
| Aggressive vs Squeeze | HR↑ 或持平 | 找到 Toys 性能上限 |

---

## 论文影响

如果实验验证成功：

1. **Whitening 消融结论可以统一**: 两个数据集趋势一致
2. **HR--Ranking Trade-off 普遍存在**: 不是数据集特有现象
3. **配置敏感性**: 强调超参数对 text 学习的重要性

---

## 相关文件

### 新增配置文件
- `sasrec_align_multi_view_v2_toys_stratified_7b_squeeze.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_7b_squeeze_nowhiten.yaml`
- `sasrec_align_multi_view_v2_toys_stratified_7b_squeeze_aggressive.yaml`

### 新增脚本
- `two_phase_run_multiview_v2_toys_stratified_7b_squeeze.sh`
- `two_phase_run_multiview_v2_toys_stratified_7b_squeeze_nowhiten.sh`
- `two_phase_run_multiview_v2_toys_stratified_7b_squeeze_aggressive.sh`

### 参考配置
- `sasrec_align_multi_view_v2_stratified.yaml` (Beauty 成功配置)
- `sasrec_align_multi_view_v2_toys_stratified_7b.yaml` (原 Toys 配置)

---

## 后续步骤

1. [ ] 运行 Squeeze 实验，对比原 Toys 结果
2. [ ] 运行 Squeeze + No Whiten，验证趋势
3. [ ] 如果趋势一致，运行 Aggressive 探索上限
4. [ ] 更新论文 Whitening 消融表格
5. [ ] 更新论文 HR--Ranking Trade-off 讨论
