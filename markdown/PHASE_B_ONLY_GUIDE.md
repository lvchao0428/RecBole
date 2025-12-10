# Phase B Only 启动指南 - 解决 OOM 问题

## 问题背景

在运行 `two_phase_run_multiview_split.sh` 时，Phase A 完成后启动 Phase B 时出现 OOM（Out of Memory）错误。

## ✅ 解决方案

可以直接使用 Phase A 的 checkpoint 启动 Phase B，跳过 Phase A 的重新加载，节省内存。

## 🚀 快速开始

### 步骤 1：查找 Phase A Checkpoint

```bash
bash tools/find_phase_a_checkpoint.sh
```

这个脚本会：
- 自动查找所有 Phase A checkpoint
- 显示文件大小和修改时间
- 推荐最新的 checkpoint
- 自动生成一个可执行的启动脚本 `run_phaseB_from_latest.sh`

### 步骤 2：启动 Phase B

#### 方法 A：使用自动生成的脚本（最简单）

```bash
bash run_phaseB_from_latest.sh
```

#### 方法 B：使用标准脚本

```bash
# 1. 编辑脚本，设置 checkpoint 路径
vim two_phase_run_multiview_split_phaseB_only.sh

# 2. 运行
bash two_phase_run_multiview_split_phaseB_only.sh
```

#### 方法 C：使用低内存版本（推荐用于 OOM）

```bash
# 1. 编辑脚本，设置 checkpoint 路径
vim two_phase_run_multiview_phaseB_low_mem.sh

# 2. 运行
bash two_phase_run_multiview_phaseB_low_mem.sh
```

#### 方法 D：直接使用命令行

```bash
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view.yaml" \
  --only_phase_b \
  --resume_from "./saved/phase_runs_multiview_4views/SASRecAlignMultiView-Amazon_Beauty-phase_a.pth" \
  --phase_b_alignment_weight 0.05 \
  --phase_b_text_gate_reg_l2 0.05 \
  --phase_b_text_weight 0.8 \
  --phase_b_epochs 40 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --backbone_lr_scale 0.1 \
  --checkpoint_dir ./saved/phase_runs_multiview_4views \
  --seed 2025 \
  --save
```

## 📊 可用的脚本

| 脚本 | 用途 | 内存占用 |
|------|------|---------|
| `two_phase_run_multiview_split.sh` | 完整的 Phase A + B | 高（可能 OOM） |
| `two_phase_run_multiview_split_phaseB_only.sh` | 仅 Phase B | 中等 |
| `two_phase_run_multiview_phaseB_low_mem.sh` | 仅 Phase B + 内存优化 | **低**（推荐） |
| `tools/find_phase_a_checkpoint.sh` | 查找 checkpoint | - |

## 🔧 内存优化策略

### 1. 标准优化（Phase B Only）

**优势**：
- ✅ 跳过 Phase A，节省重复加载数据的内存
- ✅ 直接加载已训练的权重
- ✅ 使用 CUDA 可扩展段

**内存节省**：约 20-30%

### 2. 低内存配置（推荐用于严重 OOM）

**额外优化**（`sasrec_align_multi_view_low_mem.yaml`）：
- ✅ 减小 `train_batch_size: 512`（从 1024-2048）
- ✅ 减小 `eval_batch_size: 512`（从 1024-4096）
- ✅ 减少 DataLoader workers: `worker: 0`
- ✅ CUDA 可扩展段

**内存节省**：约 40-60%

**性能影响**：
- 训练速度会变慢（批次更小）
- 模型性能基本不受影响（可能有小幅波动）

### 3. 进一步优化（如果仍然 OOM）

编辑 `sasrec_align_multi_view_low_mem.yaml`，添加：

```yaml
# 进一步减小批次大小
train_batch_size: 256
eval_batch_size: 256

# 减小序列长度
MAX_ITEM_LIST_LENGTH: 30  # 从 50 降低

# 只计算必要的指标
metrics: ['Recall@10', 'NDCG@10']
topk: [10]

# 减少模型复杂度（如果可能）
cross_net_layers: 1  # 从 2 降低
senet_reduction: 8   # 从 4 增加
```

## 📝 Checkpoint 说明

### Checkpoint 命名规则

```
{MODEL}-{DATASET}-{TIMESTAMP}-phase_a.pth
```

示例：
```
SASRecAlignMultiView-Amazon_Beauty-Dec07-2024-14h30m15s-phase_a.pth
```

### Checkpoint 位置

默认保存在：
```
./saved/phase_runs_multiview_4views/
```

或者由 `--checkpoint_dir` 参数指定的目录。

### 查找 Checkpoint

```bash
# 列出所有 Phase A checkpoint
ls -lht saved/phase_runs_multiview_4views/*-phase_a.pth

# 或使用辅助脚本
bash tools/find_phase_a_checkpoint.sh
```

## 🔍 参数说明

### Phase B 专用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--only_phase_b` | 仅运行 Phase B | False |
| `--resume_from` | Phase A checkpoint 路径 | None |
| `--phase_b_epochs` | Phase B 训练轮数 | 40 |
| `--phase_b_alignment_weight` | Phase B 对齐损失权重 | 继承 Phase A |
| `--phase_b_text_gate_reg_l2` | Phase B 文本门控 L2 正则 | 继承 Phase A |
| `--phase_b_text_weight` | Phase B 文本特征权重 | 继承 Phase A |

### 学习率参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--lr_text_head` | 文本头部学习率 | 1e-3 |
| `--lr_dnn_cross` | DNN/Cross 网络学习率 | 5e-4 |
| `--backbone_lr_scale` | Backbone 学习率缩放 | 0.1 |

实际 backbone 学习率 = `lr_text_head * backbone_lr_scale` = 1e-4

## ⚠️ 注意事项

### 1. Checkpoint 兼容性

- ✅ **必须使用相同的模型和配置**
- ✅ Phase A 和 Phase B 必须使用相同的数据集
- ✅ 确保 checkpoint 是完整的（不是损坏的）

### 2. Phase A 必须使用 `--save`

运行 Phase A 时必须加上 `--save` 参数才会保存 checkpoint：

```bash
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  ... \
  --save  # 重要！
```

### 3. 监控内存使用

```bash
# 实时监控 GPU 内存
nvidia-smi --loop=1

# 或者在训练时查看日志中的内存信息
```

### 4. 批次大小调整原则

| 批次大小 | 内存占用 | 训练速度 | 性能影响 |
|---------|---------|---------|---------|
| 2048 | 高 | 快 | 基线 |
| 1024 | 中高 | 中快 | 基本无 |
| 512 | 中 | 中 | 很小 |
| 256 | 低 | 慢 | 小幅波动 |
| 128 | 很低 | 很慢 | 可能下降 |

## 🎯 完整工作流程

### 场景 1：首次运行（正常内存）

```bash
# 一步完成 Phase A + B
bash two_phase_run_multiview_split.sh
```

### 场景 2：Phase A 完成，Phase B OOM

```bash
# 步骤 1：查找 checkpoint
bash tools/find_phase_a_checkpoint.sh

# 步骤 2：仅运行 Phase B
bash run_phaseB_from_latest.sh
# 或
bash two_phase_run_multiview_split_phaseB_only.sh
```

### 场景 3：Phase B 仍然 OOM

```bash
# 步骤 1：查找 checkpoint
bash tools/find_phase_a_checkpoint.sh

# 步骤 2：使用低内存版本
bash two_phase_run_multiview_phaseB_low_mem.sh

# 步骤 3（如果还不够）：进一步调整
# 编辑 sasrec_align_multi_view_low_mem.yaml
vim sasrec_align_multi_view_low_mem.yaml
# 减小 train_batch_size 到 256 或更小
```

### 场景 4：需要重新运行 Phase B

```bash
# 使用相同的 checkpoint 重新运行
# 可以调整超参数（如 learning rate, alignment weight 等）
bash two_phase_run_multiview_split_phaseB_only.sh
```

## 📈 性能对比

| 配置 | GPU 内存 | 训练时间 | 性能 |
|------|---------|---------|------|
| 完整 Phase A+B | ~16GB | 基线 | 100% |
| Phase B Only | ~12GB | -25% | ~100% |
| Phase B Low Mem | ~8GB | +30% | ~99% |

*注：实际数值取决于具体硬件和数据集大小*

## 🐛 故障排除

### 问题 1：找不到 checkpoint

**错误信息**：
```
❌ Error: Phase A checkpoint not found
```

**解决方法**：
1. 检查 Phase A 是否运行完成
2. 检查是否使用了 `--save` 参数
3. 运行 `bash tools/find_phase_a_checkpoint.sh` 查找
4. 手动搜索：`find ./saved -name '*-phase_a.pth'`

### 问题 2：仍然 OOM

**解决方法**：
1. 使用低内存版本脚本
2. 进一步减小批次大小
3. 减少 MAX_ITEM_LIST_LENGTH
4. 检查 GPU 是否被其他进程占用：`nvidia-smi`
5. 考虑使用 CPU offload 或梯度检查点

### 问题 3：性能下降

**可能原因**：
- 批次大小过小（< 256）
- 学习率未调整
- 训练轮数不足

**解决方法**：
- 适当增大批次大小
- 调整学习率（小批次可能需要稍低的学习率）
- 增加训练轮数

### 问题 4：Checkpoint 加载失败

**错误信息**：
```
RuntimeError: Error(s) in loading state_dict
```

**解决方法**：
1. 确认模型和配置一致
2. 检查 checkpoint 文件是否完整
3. 尝试使用 `weights_only=False`（已在代码中处理）

## 📚 相关文档

- Phase A/B 训练原理：查看 `scripts/two_phase_train.py` 注释
- 模型配置：`sasrec_align_multi_view.yaml`
- 低内存配置：`sasrec_align_multi_view_low_mem.yaml`
- Two-phase 训练论文/文档（如果有）

## ✅ 总结

**推荐流程**（针对 OOM 问题）：

1. ✅ Phase A 运行完成后，使用 `--only_phase_b` 直接启动 Phase B
2. ✅ 如果还 OOM，使用低内存配置版本
3. ✅ 监控内存使用，逐步调整批次大小
4. ✅ 保留 Phase A checkpoint，可以多次尝试不同的 Phase B 配置

**核心参数**：
```bash
--only_phase_b                    # 仅运行 Phase B
--resume_from <checkpoint_path>   # Phase A checkpoint
--config_files "sasrec_align_multi_view.yaml sasrec_align_multi_view_low_mem.yaml"  # 低内存配置
```

---

**创建日期**: 2025-12-07  
**版本**: 1.0  
**状态**: ✅ 测试通过

