# 剩余实验规划 (2026-01-16)

## 当前实验状态

### ✅ 运行中 (8个实验)

**批次2: No-Whiten** (GPU 0-3, RUNNING)
- Beauty TF-IDF+LLM no-whiten
- Beauty MV-7B no-whiten
- Toys TF-IDF+LLM no-whiten
- Toys MV-7B no-whiten

**批次3-A: SE-net/Cross** (GPU 4-7, RUNNING)
- Beauty MV no-senet
- Beauty MV no-cross-and-senet
- Toys MV no-senet
- Toys MV no-senet-nocross

**预计完成**: ~3-4小时

---

## 📋 待执行的补充实验

### 优先级1: Infer Boost扩展 (触碰边界)

**动机**: 当前范围infer=0.5-1.5，最优值在1.5，需要扩展找到峰值

**实验**:
```bash
experiments/exp_sensitivity_infer_20_toys.sh  # infer=2.0
experiments/exp_sensitivity_infer_25_toys.sh  # infer=2.5
```

**脚本内容**:
```bash
#!/usr/bin/env bash
# exp_sensitivity_infer_20_toys.sh
# Sensitivity: Inference boost = 2.0 on Toys (Aggressive baseline)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "========================================="
echo "Sensitivity: Infer Boost = 2.0 on Toys"
echo "Using GPU: $GPU_ID"
echo "Baseline: cold=2.5, infer=1.5 → Testing infer=2.0"
echo "========================================="

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --phase_a_epochs 20 \
  --phase_a_valid_metric "MRR@10" \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 2.5, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 2.0}" \
  --checkpoint_dir ./saved/sensitivity_infer_20 \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,agg,infer20" \
  --save

echo "✅ Done!"
```

**GPU分配**:
```bash
# 等待批次2+3完成后
GPU_ID=0 nohup bash experiments/exp_sensitivity_infer_20_toys.sh > logs/sensitivity_infer_20.log 2>&1 &
GPU_ID=1 nohup bash experiments/exp_sensitivity_infer_25_toys.sh > logs/sensitivity_infer_25.log 2>&1 &
```

**预期结果**:
- infer=2.0: 可能略微提升HR_new，但overall HR可能下降
- infer=2.5: 预期性能下降（over-reliance on text）

**论文更新位置**: Table sensitivity, line 1113-1126

---

### 优先级2: Appendix承诺实验

#### 实验2.1: Cold-start Reweighting Ablation

**动机**: Appendix line 1213明确承诺

**脚本**:
```bash
#!/usr/bin/env bash
# exp_ablation_no_cold_reweight_toys.sh
# Ablation: Remove cold-start reweighting in alignment loss

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "Ablation: No Cold-start Reweighting (Toys, Aggressive baseline)"

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --phase_a_epochs 20 \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 0.0, 'cold_start_align_threshold': 10, 'inference_cold_text_boost': 1.5}" \
  --checkpoint_dir ./saved/ablation_no_cold_reweight \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,agg,nocoldboost" \
  --save

echo "✅ Done!"
```

**关键变更**: `cold_start_align_boost': 0.0`

**预期**: HR_new和HR_few显著下降

---

#### 实验2.2: Center-only Normalization

**动机**: Appendix line 1217承诺，对比ZCA vs center-only

**脚本**:
```bash
#!/usr/bin/env bash
# exp_ablation_center_only_toys.sh
# Ablation: Center-only normalization (vs ZCA whitening)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${GPU_ID:-0}
echo "Ablation: Center-only Normalization (Toys, Aggressive)"

python scripts/two_phase_train.py \
  --model SASRecAlignMultiViewV2 \
  --dataset Amazon_Toys_and_Games \
  --config_files "sasrec_align_multi_view_v2_toys_stratified_7b.yaml" \
  --gpu_id $GPU_ID \
  --phase_a_grid \
  --align_grid "0.10" \
  --tau_grid "0.05" \
  --backbone_burnin_epochs 0 \
  --phase_a_epochs 20 \
  --phase_b_epochs 40 \
  --backbone_lr_scale 0.1 \
  --config_dict "{'cold_start_align_boost': 2.5, 'inference_cold_text_boost': 1.5, 'whiten_text': false, 'center_text': true}" \
  --checkpoint_dir ./saved/ablation_center_only \
  --seed 2025 \
  --variant_features "sasrec,multiview_v2,7b,toys,agg,centeronly" \
  --save

echo "✅ Done!"
```

**关键变更**: 
- `whiten_text': false` (禁用ZCA)
- `center_text': true` (仅zero-mean)

**预期**: 性能介于full-whiten和no-whiten之间

---

## 🚀 执行计划

### Phase 1: 等待当前实验 (今晚)
- ⏳ 等待8个实验完成 (~3-4小时)
- ✅ 提取结果到CSV

### Phase 2: 执行扩展实验 (明天)
```bash
# Infer扩展 (2个)
GPU_ID=0 nohup bash experiments/exp_sensitivity_infer_20_toys.sh > logs/infer_20.log 2>&1 &
GPU_ID=1 nohup bash experiments/exp_sensitivity_infer_25_toys.sh > logs/infer_25.log 2>&1 &

# Appendix承诺 (2个)
GPU_ID=2 nohup bash experiments/exp_ablation_no_cold_reweight_toys.sh > logs/no_cold_reweight.log 2>&1 &
GPU_ID=3 nohup bash experiments/exp_ablation_center_only_toys.sh > logs/center_only.log 2>&1 &
```

**预计时间**: ~2-3小时

### Phase 3: 更新论文 (明天晚)
- [ ] 更新Table ablation_whiten
- [ ] 更新Table ablation_senet_cross
- [ ] 更新Table sensitivity (infer扩展)
- [ ] 补充Appendix分析

---

## 📝 创建实验脚本

需要创建以下脚本文件：

### Infer扩展:
1. `experiments/exp_sensitivity_infer_20_toys.sh`
2. `experiments/exp_sensitivity_infer_25_toys.sh`

### Appendix承诺:
3. `experiments/exp_ablation_no_cold_reweight_toys.sh`
4. `experiments/exp_ablation_center_only_toys.sh`

**是否需要我创建这些脚本？**

---

## ✅ 总结

### 已确认的事实:
1. ✅ Seed实验都用aggressive配置 (数据正确)
2. ✅ SE-net/Cross ablation已用aggressive运行中
3. ✅ 配置分布合理 (Scale Law用standard，其他用aggressive)
4. ✅ 论文修复完成 (Method, Table format, Caption)

### 剩余工作:
1. ⏳ 等待当前8个实验完成
2. 📝 创建4个补充实验脚本 (infer扩展 + Appendix)
3. 📊 提取结果并更新论文表格

**论文状态**: ✅ **可投稿!** (核心完成，补充实验可后续提交)

