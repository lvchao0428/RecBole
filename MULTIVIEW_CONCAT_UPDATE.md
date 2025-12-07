# Multi-View Concat 配置更新说明

## 更新日期
2025-12-07

## 更新内容

已更新 `two_phase_run_multiview_concat.sh` 和 `sasrec_align_multiview_concat.yaml`，使其使用最新的文本特征结构。

## 主要变化

### 1. 特征文件路径更新 ⭐

**旧版本**：
```yaml
item_text_emb_path_llm: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_amplified.npy
num_text_views: 5
```

**新版本**：
```yaml
item_text_emb_path_llm: /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
num_text_views: 4  # 改为 4 个视图
```

### 2. 特征结构变化

#### 旧版特征（已废弃）
- 文件：`item_text_emb_amplified.npy`
- 视图数：5 个
- 生成方式：旧的 amplified 流程

#### 新版特征（当前使用）✅
- 文件：`item_text_emb.qwen3.multiview.npy`
- 视图数：4 个（Identity, Function, Audience, Category）
- 维度：256 维（4 × 64）
- 生成脚本：`tools/gen_text_emb_beauty_full_fast.sh`
- Whiten 统计：`item_text_emb.qwen3.multiview_whiten_stats.npz`

### 3. 训练参数对齐

参考 `two_phase_run_multiview_split.sh` 的参数：

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `--align_grid` | `"0.05"` | `"0.01,0.03,0.05"` | 扩展网格搜索 |
| `--tau_grid` | `"0.05,0.07"` | `"0.05,0.07,0.1"` | 扩展温度范围 |
| `--phase_a_epochs` | `20` | `6` | Phase A 轮数 |
| `--phase_a_eval_step` | `2` | `1` | 评估间隔 |
| `--phase_a_valid_metric` | `"NDCG@10"` | `"Recall@10"` | 验证指标 |
| `--checkpoint_dir` | `./saved/phase_runs` | `./saved/phase_runs_multiview_concat` | 独立目录 |
| `--variant_features` | 无 | `"sasrec,multiview,4views,concat,qwen3"` | 特征标签 |
| `--watchdog_disable` | 无 | ✅ | 禁用 watchdog |
| `--phase_b_text_gate_reg_l2` | 无 | `0.05` | Phase B 门控正则 |
| `--phase_b_text_weight` | 无 | `0.8` | Phase B 文本权重 |

### 4. 配置文件更新

**sasrec_align_multiview_concat.yaml** 主要变化：
- ✅ 更新特征路径为 `item_text_emb.qwen3.multiview.npy`
- ✅ 视图数改为 4（从 5）
- ✅ 添加 `fuse_text_feature: true` 确保融合启用
- ✅ 添加注释说明特征结构

## 特征文件对应关系

### 当前目录结构（Amazon_Beauty）

```
dataset/Amazon_Beauty/
├── item_text_emb.base.npy                      # TF-IDF 基线 (256维)
├── item_text_emb.base_whiten_stats.npz         # TF-IDF whiten 统计
├── item_text_emb.qwen3.base.npy                # Qwen3 单视图 (256维)
├── item_text_emb.qwen3.base_whiten_stats.npz   # Qwen3 单视图 whiten 统计
├── item_text_emb.qwen3.multiview.npy           # Qwen3 多视图拼接 (256维) ⭐ Concat 使用
├── item_text_emb.qwen3.multiview_whiten_stats.npz  # 多视图 whiten 统计
└── qwen3_4views/                                # 多视图分视图目录 ⭐ Split 使用
    ├── view_0.npy                               # Identity (64维)
    ├── view_1.npy                               # Function (64维)
    ├── view_2.npy                               # Audience (64维)
    ├── view_3.npy                               # Category (64维)
    └── views.json                               # 视图元数据
```

### 两种模式对比

| 特性 | Concat 模式 | Split 模式 |
|------|------------|-----------|
| **脚本** | `two_phase_run_multiview_concat.sh` | `two_phase_run_multiview_split.sh` |
| **模型** | `SASRec_Align` | `SASRecAlignMultiView` |
| **配置** | `sasrec_align_multiview_concat.yaml` | `sasrec_align_multi_view.yaml` |
| **特征文件** | `item_text_emb.qwen3.multiview.npy` | `qwen3_4views/` 目录 |
| **特征形式** | 拼接向量 (256维) | 分离视图 (4×64维) |
| **处理方式** | 整体处理 | 每个视图单独处理 + SENet |
| **优势** | 简单直接 | 灵活，可学习视图权重 |

## 使用方法

### 1. 确认特征文件存在

```bash
cd /home/charlie/project/RecBole

# 检查必需的特征文件
ls -lh dataset/Amazon_Beauty/item_text_emb.base.npy
ls -lh dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
ls -lh dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz
```

如果不存在，运行：
```bash
bash tools/gen_text_emb_beauty_full_fast.sh
```

### 2. 运行训练

```bash
cd /home/charlie/project/RecBole
bash two_phase_run_multiview_concat.sh
```

### 3. 监控训练

```bash
# 查看日志
tail -f log/*.log

# 查看 checkpoint
ls -lh saved/phase_runs_multiview_concat/

# 监控 GPU
nvidia-smi --loop=1
```

## 预期输出

### Phase A Grid Search

```
[Phase-A:grid] Grid search: 3 × 3 = 9 combinations
  alignment_weight: [0.01, 0.03, 0.05]
  temperature: [0.05, 0.07, 0.1]

[Phase-A:grid] Testing: alignment_weight=0.01, temperature=0.05
...
[Phase-A:grid] Testing: alignment_weight=0.05, temperature=0.1

[Phase-A:grid] Grid complete. Best combo: Recall@10=0.030xxx, 
alignment_weight=0.0x, temperature=0.0x
```

### Phase B

```
[Phase-B] Loaded checkpoint: ./saved/phase_runs_multiview_concat/...
[Phase-B] freeze_backbone: False
[Phase-B] lr groups: lr_text_head=0.001, lr_dnn_cross=0.0005, lr_backbone=0.0001
...
[Phase-B] Best valid result: Recall@10=0.03xxxx, NDCG@10=0.02xxxx
```

## 对比实验建议

### 对比 Concat vs Split

```bash
# 运行 Concat 版本
bash two_phase_run_multiview_concat.sh

# 运行 Split 版本（参考）
bash two_phase_run_multiview_split.sh

# 对比结果
# - Concat: saved/phase_runs_multiview_concat/
# - Split:  saved/phase_runs_multiview_4views/
```

### 预期差异

- **Concat 模式**：
  - 更简单直接
  - 计算效率可能更高
  - 视图之间的交互较少

- **Split 模式**：
  - 更灵活
  - 可学习视图权重（SENet）
  - 每个视图独立对齐
  - 可能性能更好（但计算更复杂）

## 故障排除

### 问题 1：特征文件不存在

**错误**：
```
FileNotFoundError: item_text_emb.qwen3.multiview.npy not found
```

**解决**：
```bash
bash tools/gen_text_emb_beauty_full_fast.sh
```

### 问题 2：维度不匹配

**错误**：
```
RuntimeError: Expected 256-dim features, got XXX
```

**原因**：使用了旧版本的特征文件

**解决**：
```bash
# 备份旧文件
mv dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy{,.old}

# 重新生成
bash tools/gen_text_emb_beauty_full_fast.sh
```

### 问题 3：num_text_views 不匹配

**错误**：
```
AssertionError: num_text_views mismatch
```

**解决**：
确保配置文件中 `num_text_views: 4`（不是 5）

## 关键文件清单

### 脚本文件
- ✅ `two_phase_run_multiview_concat.sh` - 训练启动脚本
- ✅ `sasrec_align_multiview_concat.yaml` - 配置文件
- ✅ `tools/gen_text_emb_beauty_full_fast.sh` - 特征生成脚本

### 特征文件（需要存在）
- ✅ `dataset/Amazon_Beauty/item_text_emb.base.npy`
- ✅ `dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy`
- ✅ `dataset/Amazon_Beauty/item_text_emb.qwen3.multiview_whiten_stats.npz`

### 输出目录
- ✅ `saved/phase_runs_multiview_concat/` - Checkpoint 保存位置
- ✅ `log/` - 训练日志

## 性能预期

基于 Beauty 数据集：
- **Baseline (ID-only)**: Recall@10 ≈ 0.027
- **TF-IDF**: Recall@10 ≈ 0.029
- **Qwen3 Multi-view Concat**: Recall@10 ≈ 0.030-0.032（预期）
- **Qwen3 Multi-view Split**: Recall@10 ≈ 0.030-0.033（预期，可能更好）

## 总结

✅ **已更新**：
- 特征路径指向最新的 `item_text_emb.qwen3.multiview.npy`
- 视图数更新为 4（对齐最新特征结构）
- 训练参数对齐 `two_phase_run_multiview_split.sh`
- 添加 center+whiten 支持

✅ **可以使用**：
- 运行 `bash two_phase_run_multiview_concat.sh` 即可
- 特征文件已存在于 `dataset/Amazon_Beauty/`
- 输出保存到独立目录 `saved/phase_runs_multiview_concat/`

---

**更新人**：AI Assistant  
**审核**：请确认特征文件存在后再运行训练

