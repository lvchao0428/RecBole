# 文本特征生成脚本使用指南

## 概述

本指南介绍了对齐 Beauty 最新 center+whiten 格式的文本特征生成脚本。

## 新版脚本（推荐使用）

所有数据集均已对齐 Beauty 最新格式，确保指标一致性：

- **Beauty**: `gen_text_emb_beauty_full_fast.sh`
- **Toys**: `gen_text_emb_toys_full_fast.sh` ✨ **新创建**
- **VideoGames**: `gen_text_emb_videogames_full_fast.sh` ✨ **新创建**
- **Yelp**: `gen_text_emb_yelp_full_fast.sh` ✨ **新创建**

## 主要改进（对齐 Beauty 格式）

### 1. TF-IDF 特征生成

新增参数：
- `--ngram_min 1 --ngram_max 2`: 使用 1-gram 和 2-gram
- `--min_df 2`: 最小文档频率为 2
- `--svd_random_state 42`: 确保可复现性
- 自动生成 center+whiten 统计文件：`item_text_emb.base_whiten_stats.npz`

### 2. Qwen3 单视图特征

关键改进：
- 输出文件名：`item_text_emb.qwen3.base.npy`（统一命名）
- 添加 `--use_chat_template`: 使用 Qwen3 的聊天模板
- 添加 `--svd_random_state 42`: 确保可复现性
- 添加 `--device cuda:0`: 明确指定 GPU
- 自动生成 center+whiten 统计文件：`item_text_emb.qwen3.base_whiten_stats.npz`

### 3. Qwen3 多视图特征（4 views）

**关键变更**：
- **view_project_dim**: 从 128 降为 **64**（每个视图 64 维）
- **总维度**: 4 视图 × 64 = **256 维**（与 Beauty 对齐）
- **输出目录**: 统一为 `qwen3_4views/`（之前是 `item_text_emb_qwen3_4views_split/`）
- 添加 `--use_chat_template`: 使用 Qwen3 的聊天模板
- 添加 `--svd_random_state 42`: 确保可复现性
- 自动生成 center+whiten 统计文件：`item_text_emb.qwen3.multiview_whiten_stats.npz`

### 4. 配置文件统一

- Amazon 数据集：使用 `sasrec_base_plain.yaml`
- Yelp 数据集：使用 `yelp_config/yelp_sasrec_base_plain.yaml`

### 5. 性能优化

添加了环境变量优化：
```bash
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
```

## 使用方法

### Toys 数据集

```bash
# 生成所有特征（TF-IDF + Qwen3 单视图 + Qwen3 多视图）
bash tools/gen_text_emb_toys_full_fast.sh
```

### VideoGames 数据集

```bash
# 生成所有特征
bash tools/gen_text_emb_videogames_full_fast.sh
```

### Yelp 数据集

```bash
# 生成所有特征
bash tools/gen_text_emb_yelp_full_fast.sh
```

## 生成的文件结构

以 Toys 为例，生成的文件包括：

```
dataset/Amazon_Toys_and_Games/
├── item_index_mapping.csv                      # Item ID 映射表
├── item_text_emb.base.npy                      # TF-IDF 特征
├── item_text_emb.base_whiten_stats.npz         # TF-IDF center+whiten 统计
├── item_text_emb.qwen3.base.npy                # Qwen3 单视图特征
├── item_text_emb.qwen3.base_whiten_stats.npz   # Qwen3 单视图 center+whiten 统计
├── item_text_emb.qwen3.multiview.npy           # Qwen3 多视图拼接特征 (256维)
├── item_text_emb.qwen3.multiview_whiten_stats.npz  # 多视图 center+whiten 统计
└── qwen3_4views/                                # 多视图分视图目录
    ├── view_0.npy                               # Identity 视图 (64维)
    ├── view_1.npy                               # Function 视图 (64维)
    ├── view_2.npy                               # Audience 视图 (64维)
    ├── view_3.npy                               # Category 视图 (64维)
    └── views.json                               # 视图元数据
```

## 与旧版本的对比

### Toys 旧版脚本的问题

1. **gen_text_emb_toys.sh**:
   - ❌ 缺少 `ngram_min`, `ngram_max`, `min_df` 参数
   - ❌ 缺少 `svd_random_state`（不可复现）
   - ❌ 缺少 `--use_chat_template`
   - ❌ 使用 `sasrec_align_base.yaml` 配置

2. **gen_multiview_4views_toys.sh**:
   - ❌ `view_project_dim 128`（总计 512 维，与 Beauty 不对齐）
   - ❌ 输出目录 `item_text_emb_qwen3_4views_split/`（命名不一致）
   - ❌ 缺少性能优化环境变量
   - ❌ 缺少详细的时间戳和进度提示

### 新版脚本的改进

✅ 所有参数完全对齐 Beauty
✅ 多视图维度：4×64=256（与 Beauty 一致）
✅ 统一的输出目录命名：`qwen3_4views/`
✅ 添加 center+whiten 统计文件
✅ 添加性能优化
✅ 添加详细的进度显示和时间戳
✅ 确保可复现性（`svd_random_state 42`）

## 验证特征

生成完成后，可以验证特征文件：

```bash
# 查看特征维度
python -c "import numpy as np; print('TF-IDF:', np.load('dataset/Amazon_Toys_and_Games/item_text_emb.base.npy').shape)"
python -c "import numpy as np; print('Qwen3 单视图:', np.load('dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.base.npy').shape)"
python -c "import numpy as np; print('Qwen3 多视图:', np.load('dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.multiview.npy').shape)"

# 查看 center+whiten 统计信息
python -c "import numpy as np; stats=np.load('dataset/Amazon_Toys_and_Games/item_text_emb.base_whiten_stats.npz'); print('Center shape:', stats['center'].shape, 'Whiten shape:', stats['whiten'].shape)"
```

预期输出（以 Toys 为例）：
```
TF-IDF: (N, 256)
Qwen3 单视图: (N, 256)
Qwen3 多视图: (N, 256)
Center shape: (256,) Whiten shape: (256, 256)
```

## 下一步实验

1. **TF-IDF 实验**:
   ```bash
   bash two_phase_run_tfidf_toys.sh
   bash two_phase_run_tfidf_videogames.sh
   bash two_phase_run_tfidf_yelp.sh
   ```

2. **多视图实验**:
   ```bash
   bash two_phase_run_multiview_split_toys.sh
   bash two_phase_run_multiview_split_videogames.sh
   bash two_phase_run_multiview_split_yelp.sh
   ```

## 注意事项

1. **GPU 内存**: Qwen3 特征生成需要足够的 GPU 内存（建议 16GB+）
2. **批次大小**: 如果遇到 OOM，可以减小 `--batch_size`（默认 16）
3. **运行时间**: 完整生成所有特征可能需要较长时间（取决于数据集大小）
4. **可复现性**: 所有脚本都使用 `svd_random_state 42` 确保结果可复现

## 问题排查

### 1. CUDA OOM 错误
```bash
# 减小批次大小
--batch_size 8  # 或更小
```

### 2. CPU 线程过多
```bash
# 手动设置线程数
export OMP_NUM_THREADS=8
```

### 3. 统计文件缺失
确保运行的是新版脚本（`*_full_fast.sh`），旧版脚本不会生成统计文件。

## 总结

新版脚本确保了所有数据集的特征生成流程与 Beauty 完全对齐，主要改进包括：

1. ✅ **维度对齐**: 多视图统一为 256 维（4×64）
2. ✅ **命名统一**: 输出目录统一为 `qwen3_4views/`
3. ✅ **可复现性**: 添加 `svd_random_state 42`
4. ✅ **Center+Whiten**: 自动生成统计文件用于归一化
5. ✅ **性能优化**: 添加多线程环境变量
6. ✅ **详细日志**: 添加时间戳和进度提示

这确保了在不同数据集上的实验结果具有可比性。
