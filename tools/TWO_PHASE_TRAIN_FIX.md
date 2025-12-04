# Two-Phase Training 脚本修复说明

## 🐛 问题描述

### 问题 1：Phase-A 训练后自动停止
**现象**：Phase-A 完成后没有自动进入 Phase-B，日志显示：
```
INFO  [Two-Phase] Phase-A finished. Not auto-continuing to Phase-B (phase_a_auto_to_b is False).
```

**原因**：`two_phase_run_tfidf.sh` 缺少 `--phase_a_auto_to_b` 参数。

---

### 问题 2：硬编码使用 NDCG@10 而非用户指定的指标
**现象**：虽然设置了 `--phase_a_valid_metric Recall@10`，但日志仍然显示：
```
INFO  [Phase-A:grid] NDCG@10: 0.0287
INFO  [Phase-A:grid] Best combo: ndcg@10=0.028700, alignment_weight=0.01, temperature=0.05
```

**原因**：`scripts/two_phase_train.py` 代码中硬编码使用 NDCG@10 来选择最佳组合，没有遵循 `--phase_a_valid_metric` 参数。

---

## ✅ 修复内容

### 修复 1：`two_phase_run_tfidf.sh` 添加自动进入 Phase-B 参数

```diff
  python scripts/two_phase_train.py \
    --model SASRec_Align \
    --dataset Amazon_Beauty \
    --config_files "sasrec_align_base.yaml" \
    --phase_a_grid \
    --align_grid "0.01,0.03,0.05" \
    --tau_grid "0.05,0.07,0.1" \
    --backbone_burnin_epochs 10 \
    --burnin_eval_step 2 \
    --phase_a_epochs 6 \
    --phase_a_eval_step 1 \
    --phase_a_valid_metric Recall@10 \
    --lr_text_head 1e-3 \
    --lr_dnn_cross 5e-4 \
    --phase_a_text_gate_reg_l2 0.05 \
    --phase_b_alignment_weight 0.05 \
+   --phase_a_auto_to_b \              # ✅ 新增：Phase-A 完成后自动进入 Phase-B
    --phase_b_epochs 40 \
    --backbone_lr_scale 0.1 \
    --checkpoint_dir ./saved/phase_runs \
    --seed 2025 \
    --variant_features "sasrec,tfidf,beauty" \
    --watchdog_interval 20 \
    --watchdog_log "run_metrics/watchdog_tfidf_beauty.log" \
    --watchdog_cpu_gb 40 \
    --watchdog_gpu_gb 28 \
    --save
```

---

### 修复 2：`scripts/two_phase_train.py` 使用动态指标

#### 修改位置 1：提取验证指标值（第868-896行）

**修改前**：
```python
# Extract NDCG@10
valid_dict = res.get("best_valid_result") or {}
ndcg10 = None
for key in ["NDCG@10", "ndcg@10", "NDCG@10(Avg)"]:
    if key in valid_dict:
        ndcg10 = float(valid_dict[key])
        break
# ...
logger_a.info(set_color("[Phase-A:grid] NDCG@10", "yellow") + f": {ndcg10}")
```

**修改后**：
```python
# Extract validation metric (use args.phase_a_valid_metric)
valid_dict = res.get("best_valid_result") or {}
metric_value = _lookup_metric_score(args.phase_a_valid_metric, valid_dict)
# ...
logger_a.info(set_color(f"[Phase-A:grid] {args.phase_a_valid_metric}", "yellow") + f": {metric_value}")
```

---

#### 修改位置 2：比较和选择最佳组合（第898-900行）

**修改前**：
```python
if ndcg10 is not None:
    if best_tuple is None or ndcg10 > best_tuple[0]:
        best_tuple = (ndcg10, aw, tau, res, ckpt_path)
```

**修改后**：
```python
if metric_value is not None:
    if best_tuple is None or metric_value > best_tuple[0]:
        best_tuple = (metric_value, aw, tau, res, ckpt_path)
```

---

#### 修改位置 3：Gate 检查（第898-909行）

**修改前**：
```python
if ndcg_target is not None and ndcg10 >= ndcg_target:
    logger_a.info(set_color("[Phase-A:grid] Gate threshold reached", "green") + f": ndcg@10={ndcg10:.6f} >= target={ndcg_target:.6f}")
    # ...
    phase_a_pass_record = {
        "alignment_weight": aw,
        "temperature": tau,
        "ndcg10": ndcg10,
    }
```

**修改后**：
```python
if ndcg_target is not None and metric_value >= ndcg_target:
    logger_a.info(set_color("[Phase-A:grid] Gate threshold reached", "green") + f": {args.phase_a_valid_metric}={metric_value:.6f} >= target={ndcg_target:.6f}")
    # ...
    phase_a_pass_record = {
        "alignment_weight": aw,
        "temperature": tau,
        "metric_value": metric_value,
        "metric_name": args.phase_a_valid_metric,
    }
```

---

#### 修改位置 4：最佳组合日志输出（第919-926行）

**修改前**：
```python
logger.info(set_color("[Phase-A:grid] Best combo", "blue") + f": ndcg@10={best_tuple[0]:.6f}, alignment_weight={best_tuple[1]}, temperature={best_tuple[2]}")
if ndcg_target is not None:
    logger.info(set_color("[Phase-A:grid] Gate not reached", "red") + f": best_ndcg@10={best_tuple[0]:.6f} < target={ndcg_target:.6f}")
phase_a_pass_record = {
    "alignment_weight": best_tuple[1],
    "temperature": best_tuple[2],
    "ndcg10": best_tuple[0],
}
```

**修改后**：
```python
logger.info(set_color("[Phase-A:grid] Best combo", "blue") + f": {args.phase_a_valid_metric}={best_tuple[0]:.6f}, alignment_weight={best_tuple[1]}, temperature={best_tuple[2]}")
if ndcg_target is not None:
    logger.info(set_color("[Phase-A:grid] Gate not reached", "red") + f": best_{args.phase_a_valid_metric}={best_tuple[0]:.6f} < target={ndcg_target:.6f}")
phase_a_pass_record = {
    "alignment_weight": best_tuple[1],
    "temperature": best_tuple[2],
    "metric_value": best_tuple[0],
    "metric_name": args.phase_a_valid_metric,
}
```

---

#### 修改位置 5：非网格搜索模式的 Gate 检查（第1082-1091行）

**修改前**：
```python
if args.ndcg_baseline is not None:
    ndcg_target = args.ndcg_baseline * (1.0 + float(args.ndcg_gain_threshold))
    ndcg10 = None
    for key in ["NDCG@10", "ndcg@10", "NDCG@10(Avg)"]:
        if key in (res_a.get("best_valid_result") or {}):
            ndcg10 = float(res_a["best_valid_result"][key])
            break
    if ndcg10 is not None and ndcg10 >= ndcg_target:
        phase_a_passed = True
        logger_a.info(set_color("[Phase-A] PASS gate reached", "green") + f": ndcg@10={ndcg10:.6f} >= target={ndcg_target:.6f}")
```

**修改后**：
```python
if args.ndcg_baseline is not None:
    ndcg_target = args.ndcg_baseline * (1.0 + float(args.ndcg_gain_threshold))
    metric_value = _lookup_metric_score(args.phase_a_valid_metric, res_a.get("best_valid_result"))
    if metric_value is not None and metric_value >= ndcg_target:
        phase_a_passed = True
        logger_a.info(set_color("[Phase-A] PASS gate reached", "green") + f": {args.phase_a_valid_metric}={metric_value:.6f} >= target={ndcg_target:.6f}")
```

---

## 📋 修复后的行为

### Phase-A 网格搜索日志示例（使用 Recall@10）

```
INFO  [Phase-A:grid] Saved checkpoint: ./saved/phase_runs/SASRec_Align-Dec-04-2025_21-36-21.pth
INFO  [Phase-A:grid] Recall@10: 0.0581
INFO  [Phase-A:grid] Best combo: Recall@10=0.058100, alignment_weight=0.03, temperature=0.07
INFO  [Two-Phase] Phase-A finished. Auto-continuing to Phase-B...
```

### 支持的指标名称（大小写不敏感）

- `Recall@5`, `Recall@10`, `Recall@20`
- `NDCG@5`, `NDCG@10`, `NDCG@20`
- `MRR@5`, `MRR@10`, `MRR@20`
- `Hit@5`, `Hit@10`, `Hit@20`
- `Precision@5`, `Precision@10`, `Precision@20`

---

## 🎯 参数说明

### `--phase_a_valid_metric`（现已正确支持）
**类型**：字符串  
**默认值**：`"NDCG@10"`  
**作用**：
1. ✅ 用于 RecBole Trainer 的 early stopping
2. ✅ **用于网格搜索选择最佳组合**（现已修复）
3. ✅ 用于 gate 检查（与 `--ndcg_baseline` 配合）

**示例**：
```bash
--phase_a_valid_metric Recall@10   # 使用 Recall@10 选择最佳组合
--phase_a_valid_metric NDCG@10     # 使用 NDCG@10（默认）
--phase_a_valid_metric MRR@10      # 使用 MRR@10
```

---

### `--ndcg_baseline` 和 `--ndcg_gain_threshold`（保持向后兼容）

**注意**：虽然参数名仍然是 `ndcg_baseline`，但现在它会与 `--phase_a_valid_metric` 指定的指标配合使用。

**示例**：
```bash
# 使用 Recall@10 作为指标，基线值 0.05，要求提升 1%
--phase_a_valid_metric Recall@10 \
--ndcg_baseline 0.05 \             # 实际是 Recall@10 的基线
--ndcg_gain_threshold 0.01         # 要求达到 0.05 * (1 + 0.01) = 0.0505
```

**目标值计算**：
```python
target = baseline * (1.0 + gain_threshold)
# 例如：0.05 * (1 + 0.01) = 0.0505
```

---

### `--phase_a_auto_to_b`（新增到脚本）

**类型**：布尔标志（action="store_true"）  
**默认值**：`False`  
**作用**：Phase-A 完成后自动进入 Phase-B

**行为差异**：

| 参数设置 | Gate 状态 | 行为 |
|---------|----------|------|
| `--phase_a_auto_to_b` | Gate PASS | ✅ 进入 Phase-B |
| `--phase_a_auto_to_b` | Gate NOT PASS | ✅ 仍进入 Phase-B（选最佳组合） |
| 不设置 | Gate PASS | ❌ 停止（不进入 Phase-B） |
| 不设置 | Gate NOT PASS | ❌ 停止 |

---

## 🚀 完整示例

### 示例 1：使用 Recall@10，自动进入 Phase-B

```bash
python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.03,0.05" \
  --tau_grid "0.05,0.07,0.1" \
  --phase_a_valid_metric Recall@10 \   # ✅ 使用 Recall@10 选择最佳组合
  --phase_a_auto_to_b \                # ✅ 自动进入 Phase-B
  --phase_a_epochs 6 \
  --phase_b_epochs 40 \
  --save
```

**预期日志**：
```
INFO  [Phase-A:grid] Recall@10: 0.0581
INFO  [Phase-A:grid] Best combo: Recall@10=0.058100, alignment_weight=0.03, temperature=0.07
INFO  [Two-Phase] Phase-A finished. Auto-continuing to Phase-B...
INFO  [Phase-B] Unfreezing backbone and starting full fine-tuning...
```

---

### 示例 2：使用 NDCG@10（默认），带 gate

```bash
python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.02,0.05,0.08" \
  --tau_grid "0.03,0.05,0.07" \
  --phase_a_valid_metric NDCG@10 \     # 默认
  --ndcg_baseline 0.0272 \             # 基线 NDCG@10
  --ndcg_gain_threshold 0.01 \         # 要求提升 1%
  --phase_a_auto_to_b \
  --save
```

**预期行为**：
- 如果任意组合达到 0.0272 × 1.01 = 0.0275，Gate PASS
- 继续搜索所有组合，选择最佳的
- 自动进入 Phase-B

---

## 🔍 验证修复

重新运行你的脚本：

```bash
cd /home/charlie/project/RecBole
bash two_phase_run_tfidf.sh
```

**检查日志中的关键行**：

✅ **正确的日志**（Recall@10）：
```
INFO  [Phase-A:grid] Recall@10: 0.0581
INFO  [Phase-A:grid] Best combo: Recall@10=0.058100, alignment_weight=0.03, temperature=0.07
INFO  [Two-Phase] Phase-A finished. Auto-continuing to Phase-B...
```

❌ **修复前的错误日志**（硬编码 NDCG@10）：
```
INFO  [Phase-A:grid] NDCG@10: 0.0287
INFO  [Phase-A:grid] Best combo: ndcg@10=0.028700, alignment_weight=0.01, temperature=0.05
INFO  [Two-Phase] Phase-A finished. Not auto-continuing to Phase-B (phase_a_auto_to_b is False).
```

---

## 📝 总结

### 修复的文件
1. ✅ `two_phase_run_tfidf.sh` - 添加 `--phase_a_auto_to_b`
2. ✅ `scripts/two_phase_train.py` - 修复硬编码 NDCG@10 的 bug

### 修复的问题
1. ✅ Phase-A 完成后自动进入 Phase-B
2. ✅ 正确使用 `--phase_a_valid_metric` 选择最佳组合
3. ✅ 日志输出显示正确的指标名称

### 向后兼容性
- ✅ 默认仍然是 NDCG@10（`--phase_a_valid_metric` 默认值）
- ✅ 保留 `--ndcg_baseline` 和 `--ndcg_gain_threshold` 参数名（但现在支持任意指标）
- ✅ 不影响现有脚本（未设置 `--phase_a_auto_to_b` 时行为不变）

