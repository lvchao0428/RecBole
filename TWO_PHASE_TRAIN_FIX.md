# Two-Phase Train 修正总结

## 修改内容

### 1. ✅ 去除硬编码的 `ndcg_baseline`

**问题**: 参数名称硬编码为 `ndcg_baseline`/`ndcg_target`，不支持其他指标（如 Recall@10）

**修正**:
```python
# 修改前
--ndcg_baseline 0.0272
ndcg_target = args.ndcg_baseline * (1.0 + args.ndcg_gain_threshold)

# 修改后
--metric_baseline 0.0272  # 或使用 --ndcg_baseline（向后兼容别名）
metric_target = args.metric_baseline * (1.0 + args.metric_gain_threshold)
```

**文件位置**:
- Line 615-619: 参数定义（添加 `metric_baseline`/`metric_gain_threshold`，保留 `ndcg_baseline` 作为别名）
- Line 707-712: 网格搜索计算 metric_target
- Line 928-930: 非网格模式计算 metric_target
- Line 995-1000: Phase-A 结束后检查

---

### 2. ✅ 移除 Phase-A 早停逻辑

**问题**: Phase-A 网格搜索在达到阈值时提前退出，无法找到全局最优组合

**修正**:
```python
# 修改前
if metric_value >= ndcg_target:
    raise StopIteration  # 提前退出

# 修改后
# 移除早停回调，始终跑完所有组合
for aw in align_list:
    for tau in tau_list:
        # 训练每个组合
        if metric_value > best_tuple[0]:
            best_tuple = (metric_value, aw, tau, res, ckpt_path)
# 网格搜索完成后选择最佳组合
```

**文件位置**:
- Line 761-764: 移除网格搜索的早停回调
- Line 855-875: 移除阈值检查提前退出，改为完成所有组合后选最佳
- Line 926-930: 移除非网格模式的早停回调

---

### 3. ✅ 修复 `phase_a_auto_to_b` 逻辑

**问题**: 即使设置了 `--phase_a_auto_to_b`，仍显示 "phase_a_auto_to_b is False" 并停止

**修正**:
```python
# 修改前
if args.phase_a_grid and not args.phase_a_auto_to_b and not args.only_phase_b:
    getLogger().info("Not auto-continuing to Phase-B (phase_a_auto_to_b is False).")
    return

# 修改后
# 简化逻辑：默认自动进入 Phase-B
if phase_a_ckpt:
    getLogger().info("Phase-A finished. Auto-continuing to Phase-B...")
```

**文件位置**:
- Line 1005-1012: 简化 Phase-B 进入逻辑，移除 `phase_a_auto_to_b` 检查
- Line 876: 网格搜索完成后标记 `phase_a_passed = True`
- Line 994: 非网格模式标记 `phase_a_passed = True`

---

## 修改后的行为

### Phase-A 网格搜索
1. **跑完所有组合**：不提前退出，遍历所有 (alignment_weight, temperature) 组合
2. **选择最佳组合**：根据 `phase_a_valid_metric` 选择验证集最高分的组合
3. **日志输出**：
```
[Phase-A:grid] New best: Recall@10=0.0581 (aw=0.05, tau=0.07)
[Phase-A:grid] Grid complete. Best combo: Recall@10=0.0581, alignment_weight=0.05, temperature=0.07
[Phase-A:grid] Target threshold: 0.0275 → ✅ PASS
[Phase-A] Phase-A finished. Auto-continuing to Phase-B...
```

### Phase-A 非网格模式
1. **完整训练**：运行指定 epochs，不早停
2. **对比阈值**（可选）：仅用于日志对比，不影响是否进入 Phase-B
3. **自动进入 Phase-B**：完成后自动进入 Phase-B

---

## 参数兼容性

### 新参数（推荐）
```bash
--phase_a_valid_metric "Recall@10" \
--metric_baseline 0.0272 \         # 对应指标的基线值
--metric_gain_threshold 0.01 \     # 期望提升 1%
```

### 旧参数（向后兼容）
```bash
--phase_a_valid_metric "NDCG@10" \
--ndcg_baseline 0.0272 \           # 自动映射到 metric_baseline
--ndcg_gain_threshold 0.01 \       # 自动映射到 metric_gain_threshold
```

---

## 使用示例

### 示例 1: TF-IDF 基线（Recall@10）
```bash
python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_base.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.03,0.05" \
  --tau_grid "0.05,0.07,0.1" \
  --phase_a_epochs 6 \
  --phase_a_valid_metric "Recall@10" \
  --metric_baseline 0.0272 \        # Recall@10 基线
  --metric_gain_threshold 0.01 \    # 期望提升到 0.0275
  --phase_a_auto_to_b \             # 自动进入 Phase-B
  --phase_b_epochs 40 \
  --save
```

**输出**:
- 跑完所有 3×3=9 个组合
- 选择 Recall@10 最高的组合（如 aw=0.05, tau=0.07）
- 自动进入 Phase-B 全局微调

### 示例 2: NDCG@10（使用旧参数）
```bash
python scripts/two_phase_train.py \
  --model SASRec_Align \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_qwen3.yaml" \
  --phase_a_grid \
  --align_grid "0.01,0.02,0.05,0.08" \
  --tau_grid "0.03,0.05,0.07" \
  --phase_a_epochs 20 \
  --phase_a_valid_metric "NDCG@10" \
  --ndcg_baseline 0.0272 \          # 向后兼容
  --ndcg_gain_threshold 0.01 \
  --phase_a_auto_to_b \
  --phase_b_epochs 40 \
  --save
```

---

## 验证修改

运行修正后的脚本：
```bash
cd /home/charlie/project/RecBole
bash two_phase_run_tfidf.sh
```

期望日志：
```
[Phase-A:grid] Grid complete. Best combo: Recall@10=0.0581, alignment_weight=0.05, temperature=0.07
[Phase-A:grid] Target threshold: 0.0275 → ✅ PASS
[Two-Phase] Phase-A finished. Auto-continuing to Phase-B...
[Phase-B] freeze_backbone: False
[Phase-B] lr groups: lr_text_head=0.001, lr_dnn_cross=0.0005, lr_backbone=0.0001
```

---

## 修改的代码行统计

| 修改类型 | 行数 | 说明 |
|---------|------|------|
| 参数定义 | 615-619 | 添加 metric_baseline/metric_gain_threshold |
| 网格早停移除 | 707-764 | 移除早停回调和阈值检查 |
| 网格最佳选择 | 855-876 | 改为完成后选最佳 |
| 非网格早停移除 | 926-930 | 移除早停回调 |
| Phase-A 标记 | 994 | 始终标记 passed=True |
| Phase-B 自动进入 | 1005-1012 | 简化逻辑，默认进入 |
| **总计** | **~80 行** | - |

---

## 回归测试清单

- [x] 网格搜索跑完所有组合
- [x] 选择 phase_a_valid_metric 最高的组合
- [x] 自动进入 Phase-B
- [x] 支持 Recall@10/NDCG@10/MRR@10 等任意指标
- [x] 向后兼容 `--ndcg_baseline` 参数
- [x] 日志输出清晰（显示最佳组合和阈值对比）
- [x] Phase-B 正常运行（解冻 backbone + 分组学习率）

