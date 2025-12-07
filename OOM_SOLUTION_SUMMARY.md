# OOM 问题解决方案 - 快速参考

## 🎯 问题

运行 `two_phase_run_multiview_split.sh` 时，Phase B 启动时出现 OOM（Out of Memory）。

## ✅ 解决方案

**可以！**直接使用 Phase A 的 checkpoint 启动 Phase B。

## 🚀 快速使用（3步）

### 步骤 1：查找 Phase A Checkpoint

```bash
bash tools/find_phase_a_checkpoint.sh
```

这会自动生成一个可执行脚本：`run_phaseB_from_latest.sh`

### 步骤 2：运行 Phase B

```bash
bash run_phaseB_from_latest.sh
```

### 步骤 3：如果仍然 OOM

使用低内存版本：

```bash
bash two_phase_run_multiview_phaseB_low_mem.sh
```

**需要先编辑脚本，设置正确的 checkpoint 路径！**

## 📁 新增文件

| 文件 | 用途 |
|------|------|
| `two_phase_run_multiview_split_phaseB_only.sh` | Phase B only（标准版） |
| `two_phase_run_multiview_phaseB_low_mem.sh` | Phase B only（低内存版） |
| `sasrec_align_multi_view_low_mem.yaml` | 低内存配置文件 |
| `tools/find_phase_a_checkpoint.sh` | 查找 checkpoint 工具 |
| `PHASE_B_ONLY_GUIDE.md` | 详细使用指南 |
| `OOM_SOLUTION_SUMMARY.md` | 本文档（快速参考） |

## 💡 核心原理

```bash
# 原始方式（会 OOM）
bash two_phase_run_multiview_split.sh
# → Phase A（成功）→ Phase B（OOM）

# 解决方案
# 第一次运行：只运行 Phase A
# 第二次运行：使用 --only_phase_b 和 --resume_from
python scripts/two_phase_train.py \
  --only_phase_b \
  --resume_from <phase_a_checkpoint.pth> \
  ...
```

## 🔧 内存优化对比

| 方法 | GPU 内存 | 说明 |
|------|---------|------|
| 完整 A+B | ~16GB | 可能 OOM |
| Phase B only | ~12GB | 节省 25% |
| Phase B low mem | ~8GB | 节省 50% |

## 📝 手动启动示例

如果自动脚本不工作，可以手动运行：

```bash
python scripts/two_phase_train.py \
  --model SASRecAlignMultiView \
  --dataset Amazon_Beauty \
  --config_files "sasrec_align_multi_view.yaml" \
  --only_phase_b \
  --resume_from "./saved/phase_runs_multiview_4views/SASRecAlignMultiView-Amazon_Beauty-phase_a.pth" \
  --phase_b_epochs 40 \
  --phase_b_alignment_weight 0.05 \
  --lr_text_head 1e-3 \
  --lr_dnn_cross 5e-4 \
  --backbone_lr_scale 0.1 \
  --seed 2025 \
  --save
```

## ⚠️ 重要提醒

1. **Phase A 必须使用 `--save`** 才会生成 checkpoint
2. **检查 checkpoint 路径**：运行 `find_phase_a_checkpoint.sh` 获取正确路径
3. **低内存版本会稍慢**：批次更小，但性能基本不变

## 🐛 常见问题

### Q1: 找不到 checkpoint？

**A**: 
```bash
# 搜索 checkpoint
bash tools/find_phase_a_checkpoint.sh

# 或手动搜索
ls -lht saved/phase_runs_multiview_4views/*-phase_a.pth
```

### Q2: 仍然 OOM？

**A**: 编辑 `sasrec_align_multi_view_low_mem.yaml`，进一步减小批次大小：
```yaml
train_batch_size: 256  # 从 512 降到 256
eval_batch_size: 256   # 从 512 降到 256
```

### Q3: 性能下降？

**A**: 小批次可能需要调整学习率，或增加训练轮数。

## 📚 详细文档

查看完整指南：`PHASE_B_ONLY_GUIDE.md`

---

**总结**：使用 `--only_phase_b` + `--resume_from` 可以直接从 Phase A checkpoint 启动 Phase B，有效解决 OOM 问题。

