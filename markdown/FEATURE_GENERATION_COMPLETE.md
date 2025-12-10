# ✅ Toys/VideoGames/Yelp 特征生成脚本对齐完成

## 📋 任务概述

按照 Beauty 最新的 center+whiten 格式，为 Toys、VideoGames、Yelp 数据集创建了完整的文本特征生成脚本，确保所有指标对齐 Beauty。

## ✨ 完成的工作

### 1. 创建的新脚本（3个）

| 脚本 | 位置 | 说明 |
|------|------|------|
| **Toys** | `tools/gen_text_emb_toys_full_fast.sh` | 对齐 Beauty 的完整版本 |
| **VideoGames** | `tools/gen_text_emb_videogames_full_fast.sh` | 新创建，对齐 Beauty |
| **Yelp** | `tools/gen_text_emb_yelp_full_fast.sh` | 更新版本，对齐 Beauty |

### 2. 工具脚本（2个）

| 脚本 | 位置 | 说明 |
|------|------|------|
| **批量生成** | `tools/gen_all_datasets_full_fast.sh` | 一键生成所有数据集特征 |
| **验证工具** | `tools/verify_all_embeddings.py` | 自动验证特征文件 |

### 3. 文档（3个）

| 文档 | 位置 | 内容 |
|------|------|------|
| **使用指南** | `tools/TEXT_EMB_GENERATION_GUIDE.md` | 详细的使用说明 |
| **迁移总结** | `tools/TOYS_MIGRATION_SUMMARY.md` | Toys 新旧版本对比 |
| **快速参考** | `tools/README_FEATURE_GENERATION.md` | 快速上手指南 |

## 🎯 关键改进（对齐 Beauty）

### 1. 多视图维度调整 ⭐ **最重要**

```bash
# 旧版 Toys
--view_project_dim 128  # 4 × 128 = 512 维 ❌

# 新版 Toys（对齐 Beauty）
--view_project_dim 64   # 4 × 64 = 256 维 ✅
```

### 2. TF-IDF 参数对齐

新增参数：
- `--ngram_min 1 --ngram_max 2`（使用 1-gram 和 2-gram）
- `--min_df 2`（最小文档频率）
- `--svd_random_state 42`（确保可复现）

### 3. Qwen3 参数对齐

- ✅ `--use_chat_template`（使用 Qwen3 聊天模板）
- ✅ `--svd_random_state 42`（确保可复现）
- ✅ `--device cuda:0`（明确指定 GPU）

### 4. Center+Whiten 统计文件

自动生成 3 个统计文件：
- `item_text_emb.base_whiten_stats.npz`（TF-IDF）
- `item_text_emb.qwen3.base_whiten_stats.npz`（Qwen3 单视图）
- `item_text_emb.qwen3.multiview_whiten_stats.npz`（Qwen3 多视图）

每个包含：
- `center`: 中心化向量 (256,)
- `whiten`: 白化矩阵 (256, 256)

### 5. 统一命名规范

| 项目 | 旧版 | 新版 |
|------|------|------|
| 配置文件 | `sasrec_align_base.yaml` | `sasrec_base_plain.yaml` |
| 单视图特征 | `item_text_emb.qwen3.npy` | `item_text_emb.qwen3.base.npy` |
| 多视图目录 | `item_text_emb_qwen3_4views_split/` | `qwen3_4views/` |

### 6. 性能优化

添加多线程环境变量：
```bash
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export VECLIB_MAXIMUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
```

## 📊 生成的特征文件结构

以 **Toys** 为例：

```
dataset/Amazon_Toys_and_Games/
├── item_index_mapping.csv                          # ID 映射表
│
├── item_text_emb.base.npy                          # TF-IDF (256维)
├── item_text_emb.base_whiten_stats.npz             # TF-IDF 统计 ✨
│
├── item_text_emb.qwen3.base.npy                    # Qwen3 单视图 (256维)
├── item_text_emb.qwen3.base_whiten_stats.npz       # 单视图统计 ✨
│
├── item_text_emb.qwen3.multiview.npy               # 多视图拼接 (256维)
├── item_text_emb.qwen3.multiview_whiten_stats.npz  # 多视图统计 ✨
│
└── qwen3_4views/                                   # 分视图目录
    ├── view_0.npy                                  # Identity (64维)
    ├── view_1.npy                                  # Function (64维)
    ├── view_2.npy                                  # Audience (64维)
    ├── view_3.npy                                  # Category (64维)
    └── views.json                                  # 元数据
```

## 🚀 使用方法

### 方法 1: 单独生成

```bash
# Toys
bash tools/gen_text_emb_toys_full_fast.sh

# VideoGames
bash tools/gen_text_emb_videogames_full_fast.sh

# Yelp
bash tools/gen_text_emb_yelp_full_fast.sh
```

### 方法 2: 一键生成所有数据集

```bash
bash tools/gen_all_datasets_full_fast.sh
```

### 验证生成结果

```bash
python tools/verify_all_embeddings.py
```

预期输出：
```
📊 检查数据集: Amazon_Toys_and_Games
🔍 TF-IDF:
  ✅ 特征文件：item_text_emb.base.npy - shape=(N, 256)
  ✅ 统计文件：item_text_emb.base_whiten_stats.npz - center(256,), whiten(256, 256)

🔍 Qwen3 单视图:
  ✅ 特征文件：item_text_emb.qwen3.base.npy - shape=(N, 256)
  ✅ 统计文件：item_text_emb.qwen3.base_whiten_stats.npz - center(256,), whiten(256, 256)

🔍 Qwen3 多视图:
  ✅ 特征文件：item_text_emb.qwen3.multiview.npy - shape=(N, 256)
  ✅ 统计文件：item_text_emb.qwen3.multiview_whiten_stats.npz - center(256,), whiten(256, 256)

🔍 多视图分视图:
  ✅ views.json: 4 个视图
  ✅ view_0.npy (Identity): (N, 64)
  ✅ view_1.npy (Function): (N, 64)
  ✅ view_2.npy (Audience): (N, 64)
  ✅ view_3.npy (Category): (N, 64)

✅ Amazon_Toys_and_Games 所有检查通过！
```

## 📈 运行实验

### TF-IDF 实验

```bash
# Toys
bash two_phase_run_tfidf_toys.sh

# VideoGames
bash two_phase_run_tfidf_videogames.sh

# Yelp
bash two_phase_run_tfidf_yelp.sh
```

### 多视图实验

```bash
# Toys
bash two_phase_run_multiview_split_toys.sh

# VideoGames
bash two_phase_run_multiview_split_videogames.sh

# Yelp
bash two_phase_run_multiview_split_yelp.sh
```

## 🔄 与旧版本的对比

### Toys 旧版脚本

1. **gen_text_emb_toys.sh**:
   - ❌ 缺少 ngram、min_df 参数
   - ❌ 缺少 svd_random_state
   - ❌ 缺少 use_chat_template
   - ❌ 使用 sasrec_align_base.yaml

2. **gen_multiview_4views_toys.sh**:
   - ❌ view_project_dim=128（总计 512 维）
   - ❌ 输出目录命名不一致
   - ❌ 缺少统计文件
   - ❌ 缺少性能优化

### Toys 新版脚本

✅ 所有参数完全对齐 Beauty  
✅ 多视图维度：4×64=256  
✅ 统一的命名规范  
✅ 生成 center+whiten 统计文件  
✅ 性能优化（多线程）  
✅ 可复现性（svd_random_state=42）  
✅ 详细的进度显示

## 📊 对齐结果总结

| 特性 | Beauty | Toys (新) | VideoGames (新) | Yelp (新) | 状态 |
|------|--------|-----------|----------------|----------|------|
| 多视图维度 | 256 | 256 | 256 | 256 | ✅ 对齐 |
| 配置文件 | base_plain | base_plain | base_plain | yelp_base_plain | ✅ 对齐 |
| SVD 种子 | 42 | 42 | 42 | 42 | ✅ 对齐 |
| Chat Template | ✅ | ✅ | ✅ | ✅ | ✅ 对齐 |
| 统计文件 | ✅ | ✅ | ✅ | ✅ | ✅ 对齐 |
| 输出目录 | qwen3_4views | qwen3_4views | qwen3_4views | qwen3_4views | ✅ 对齐 |
| 性能优化 | ✅ | ✅ | ✅ | ✅ | ✅ 对齐 |

## 📚 文档索引

1. **快速开始**: `tools/README_FEATURE_GENERATION.md`
2. **详细指南**: `tools/TEXT_EMB_GENERATION_GUIDE.md`
3. **迁移说明**: `tools/TOYS_MIGRATION_SUMMARY.md`
4. **本总结**: `FEATURE_GENERATION_COMPLETE.md`

## ⚠️ 重要提示

1. **不兼容性**: 新版特征与旧版不兼容（维度不同）
   - 需要重新生成所有特征文件
   - 不要混用新旧版本的特征

2. **GPU 要求**: Qwen3 特征生成需要足够的 GPU 内存
   - 建议 16GB+ 显存
   - 可调整 `--batch_size` 参数

3. **运行时间**: 完整生成可能需要较长时间
   - 取决于数据集大小
   - 建议使用 `gen_all_datasets_full_fast.sh` 批量运行

## ✅ 下一步行动

1. **生成特征**:
   ```bash
   bash tools/gen_text_emb_toys_full_fast.sh
   # 或
   bash tools/gen_all_datasets_full_fast.sh
   ```

2. **验证特征**:
   ```bash
   python tools/verify_all_embeddings.py
   ```

3. **运行实验**:
   ```bash
   # TF-IDF
   bash two_phase_run_tfidf_toys.sh
   
   # 多视图
   bash two_phase_run_multiview_split_toys.sh
   ```

4. **对比结果**: 确保 Toys/VideoGames/Yelp 的性能指标与 Beauty 在相同范围内

## 🎉 总结

✅ 已完成 Toys、VideoGames、Yelp 数据集的特征生成脚本对齐  
✅ 所有指标与 Beauty 完全一致  
✅ 提供了完整的工具和文档  
✅ 确保了可复现性和一致性  

---

**完成日期**: 2025-12-06  
**版本**: 1.0  
**状态**: ✅ 已完成，可以开始生成特征和运行实验

