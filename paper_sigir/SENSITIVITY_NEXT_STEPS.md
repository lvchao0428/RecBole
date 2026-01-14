# 敏感性分析实验 - 下一步计划 (2026-01-14)

## ✅ 已完成的实验 (第一批)

| 实验 | GPU | 配置 | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | 状态 |
|------|-----|------|-------|---------|--------|-----------|------|
| **baseline** | - | λ=0.10, τ=0.05, cold=2.5, infer=1.5 | 6.92 | 4.51 | 3.76 | 1.99 | 基线 |
| lambda_005 | GPU3 | λ=0.05 | 6.82 | 4.45 | 3.72 | 1.98 | ✅ |
| lambda_015 | GPU4 | λ=0.15 | 6.93 | 4.48 | 3.72 | 2.04 | ✅ |
| tau_003 | GPU5 | τ=0.03 | 6.94 | 4.50 | 3.75 | 2.01 | ✅ |
| tau_010 | GPU6 | τ=0.10 | 6.90 | 4.50 | 3.76 | 2.01 | ✅ |
| cold_15 | GPU7 | cold=1.5 | 6.90 | 4.46 | 3.71 | 1.95 | ✅ |
| cold_25 | GPU0(5090) | cold=2.5 | - | - | - | - | 🔄 运行中 |

## 📋 待执行实验 (第二批)

### 需要运行的实验

| 实验 | 参数变化 | 预期效果 | 优先级 |
|------|----------|----------|--------|
| **infer_05** | infer_boost=0.5 | MRR↑ HR↓ (低推理放大) | ⭐⭐⭐ 高 |
| **infer_15** | infer_boost=1.5 | 已是baseline | - (跳过) |
| **cold_30** | cold_boost=3.0 | HR_new↑ (验证上限) | ⭐⭐ 中 |

**注意**: baseline 已经是 infer=1.5，所以不需要再跑 infer_15

### 补充实验（可选）

| 实验 | 参数 | 目的 | 优先级 |
|------|------|------|--------|
| infer_20 | infer=2.0 | 验证推理增强上限 | ⭐ 低 |
| lambda_020 | λ=0.20 | 验证对齐权重上限 | ⭐ 低 |

## 🎯 GPU 分配方案（下一批）

### 推荐配置

```bash
# 4090-1: infer_boost=0.5 (推理时低放大)
GPU_ID=1 nohup bash experiments/exp_sensitivity_infer_05.sh > sensitivity_infer_05.log 2>&1 &

# 4090-2: cold_boost=3.0 (冷启动高权重)
GPU_ID=2 nohup bash experiments/exp_sensitivity_cold_30.sh > sensitivity_cold_30.log 2>&1 &
```

### 创建缺失的脚本

需要创建 `exp_sensitivity_cold_30.sh`:

```bash
#!/usr/bin/env bash
# 敏感性分析: cold_start_align_boost = 3.0
# 基线: 2.5, 验证冷启动权重上限

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

python run_recbole.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files sasrec_align_multi_view_v2_toys_stratified_7b.yaml \
  --config_dict "{'cold_start_align_boost': 3.0, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5}" \
  --checkpoint_dir ./saved/sensitivity_cold_30 \
  --variant_features "sasrec,multiview,7b,agg,cold30,toys,stratified"
```

## 📊 初步结论 (基于已完成实验)

### 1. λ (alignment_weight) 敏感性

| λ | HR@10 | MRR@10 | HR_new@10 | 结论 |
|---|-------|--------|-----------|------|
| 0.05 | 6.82 ↓ | 3.72 ↓ | 1.98 ↓ | 对齐不足，所有指标下降 |
| **0.10** | **6.92** | **3.76** | **1.99** | **最优** |
| 0.15 | 6.93 ≈ | 3.72 ↓ | 2.04 ↑ | HR_new提升，但MRR下降 |

**发现**: λ=0.10 是最佳权衡点

### 2. τ (temperature) 敏感性

| τ | HR@10 | MRR@10 | HR_new@10 | 结论 |
|---|-------|--------|-----------|------|
| 0.03 | 6.94 ↑ | 3.75 ↓ | 2.01 ↑ | 锐利对比，HR提升 |
| **0.05** | **6.92** | **3.76** | **1.99** | **平衡** |
| 0.10 | 6.90 ↓ | 3.76 = | 2.01 ↑ | 平滑对比，HR略降 |

**发现**: τ 影响较小，0.03-0.10 范围内都可接受

### 3. cold_boost 敏感性

| cold | HR@10 | MRR@10 | HR_new@10 | 结论 |
|------|-------|--------|-----------|------|
| 1.5 | 6.90 ↓ | 3.71 ↓ | 1.95 ↓ | 冷启动权重不足 |
| **2.5** | **6.92** | **3.76** | **1.99** | **最优** |
| 3.0 | - | - | - | 待验证 |

**发现**: cold_boost 对 HR_new 影响显著（-2.0%）

## 🔬 下一步实验优先级

### 高优先级 (必须完成)

1. ⏳ **cold_25** (5090 GPU0) - 运行中，等待结果
2. ⭐ **infer_05** (4090 GPU1) - 验证推理时低放大的影响

### 中优先级 (建议完成)

3. ⭐ **cold_30** (4090 GPU2) - 验证冷启动权重上限

### 低优先级 (可选)

4. **infer_20** (4090 GPU3) - 如果 infer_05 效果好，验证上限

## 📌 一键启动命令

```bash
# 第二批敏感性实验 (等 GPU 0 的 cold_25 完成后)
GPU_ID=1 nohup bash experiments/exp_sensitivity_infer_05.sh > sensitivity_infer_05.log 2>&1 &
GPU_ID=2 nohup bash experiments/exp_sensitivity_cold_30.sh > sensitivity_cold_30.log 2>&1 &
```

## 📝 待创建脚本

- `experiments/exp_sensitivity_cold_30.sh` (见上方模板)

## 🎯 预期完成时间

- cold_25: ~2小时 (运行中)
- infer_05: ~2小时
- cold_30: ~2小时

**总计**: 约 4-6 小时完成所有敏感性实验

## 📈 论文表格填充进度

**Table 10 (Sensitivity Analysis)**: 3/4 组参数完成 (75%)
- ✅ λ (alignment_weight): 完成
- ✅ τ (temperature): 完成  
- 🔄 cold_boost: 2/3 完成 (等 cold_30)
- ⏳ infer_boost: 0/3 完成 (需要 infer_05)

完成后即可提交论文！
