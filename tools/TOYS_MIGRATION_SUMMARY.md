# Toys 特征生成脚本迁移总结

## 新旧版本对比

| 项目 | 旧版本 | 新版本（对齐 Beauty） | 说明 |
|-----|--------|---------------------|------|
| **脚本名称** | `gen_text_emb_toys.sh` + `gen_multiview_4views_toys.sh` | `gen_text_emb_toys_full_fast.sh` | 合并为一个脚本 |
| **配置文件** | `sasrec_align_base.yaml` | `sasrec_base_plain.yaml` | 统一配置 |
| **TF-IDF ngram** | 默认（1-gram） | `--ngram_min 1 --ngram_max 2` | 增加 2-gram |
| **TF-IDF min_df** | 默认（1） | `--min_df 2` | 过滤低频词 |
| **SVD 随机种子** | ❌ 无 | `--svd_random_state 42` | 确保可复现 |
| **Qwen3 chat template** | ❌ 无 | `--use_chat_template` | 使用官方模板 |
| **单视图输出** | `item_text_emb.qwen3.npy` | `item_text_emb.qwen3.base.npy` | 统一命名 |
| **多视图维度** | 4×128=**512** 维 | 4×64=**256** 维 | ✨ 关键对齐 |
| **多视图输出目录** | `item_text_emb_qwen3_4views_split/` | `qwen3_4views/` | 统一命名 |
| **Center+Whiten 统计** | ❌ 无 | ✅ 自动生成 `.npz` 文件 | 归一化统计 |
| **性能优化** | ❌ 无 | ✅ 多线程环境变量 | 加速计算 |
| **进度显示** | 简单 | ✅ 详细时间戳 | 便于监控 |

## 关键改进点

### 1. 多视图维度对齐（最重要）

**旧版本**:
```bash
--view_project_dim 128  # 每个视图 128 维
# 4 视图 × 128 = 512 维（与 Beauty 不同！）
```

**新版本**:
```bash
--view_project_dim 64   # 每个视图 64 维
# 4 视图 × 64 = 256 维（与 Beauty 一致！）
```

### 2. Center+Whiten 统计文件

**旧版本**: 不生成统计文件

**新版本**: 自动生成以下统计文件
- `item_text_emb.base_whiten_stats.npz`
- `item_text_emb.qwen3.base_whiten_stats.npz`
- `item_text_emb.qwen3.multiview_whiten_stats.npz`

每个文件包含：
- `center`: 中心化向量 (256,)
- `whiten`: 白化矩阵 (256, 256)

### 3. 输出文件对比

#### TF-IDF 特征
- 旧版本: `item_text_emb.base.npy`
- 新版本: `item_text_emb.base.npy` + `item_text_emb.base_whiten_stats.npz`

#### Qwen3 单视图
- 旧版本: `item_text_emb.qwen3.npy`
- 新版本: `item_text_emb.qwen3.base.npy` + `item_text_emb.qwen3.base_whiten_stats.npz`

#### Qwen3 多视图
- 旧版本:
  - 拼接: `item_text_emb_qwen3_4views.npy` (512维)
  - 分视图: `item_text_emb_qwen3_4views_split/view_*.npy` (128维×4)
  
- 新版本:
  - 拼接: `item_text_emb.qwen3.multiview.npy` (256维)
  - 分视图: `qwen3_4views/view_*.npy` (64维×4)
  - 统计: `item_text_emb.qwen3.multiview_whiten_stats.npz`

## 使用指南

### 生成特征

```bash
# 一键生成所有特征（推荐）
bash tools/gen_text_emb_toys_full_fast.sh
```

### 验证输出

```bash
# 检查维度
python -c "
import numpy as np
print('TF-IDF:', np.load('dataset/Amazon_Toys_and_Games/item_text_emb.base.npy').shape)
print('Qwen3 单视图:', np.load('dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.base.npy').shape)
print('Qwen3 多视图:', np.load('dataset/Amazon_Toys_and_Games/item_text_emb.qwen3.multiview.npy').shape)
print('多视图 view_0:', np.load('dataset/Amazon_Toys_and_Games/qwen3_4views/view_0.npy').shape)
"
```

预期输出：
```
TF-IDF: (N, 256)
Qwen3 单视图: (N, 256)
Qwen3 多视图: (N, 256)
多视图 view_0: (N, 64)
```

### 检查统计文件

```bash
# 查看 center+whiten 统计
python -c "
import numpy as np
stats = np.load('dataset/Amazon_Toys_and_Games/item_text_emb.base_whiten_stats.npz')
print('Center:', stats['center'].shape)
print('Whiten:', stats['whiten'].shape)
"
```

预期输出：
```
Center: (256,)
Whiten: (256, 256)
```

## 运行实验

### TF-IDF 实验
```bash
bash two_phase_run_tfidf_toys.sh
```

### 多视图实验
```bash
bash two_phase_run_multiview_split_toys.sh
```

## 为什么需要对齐？

1. **公平比较**: 确保不同数据集使用相同的特征维度和生成方式
2. **模型兼容**: 统一的维度便于模型架构复用
3. **可复现性**: `svd_random_state 42` 确保结果可复现
4. **归一化一致**: center+whiten 统计文件确保归一化方式一致

## 迁移检查清单

- [x] 创建新脚本 `gen_text_emb_toys_full_fast.sh`
- [x] 对齐多视图维度：512 → 256（4×64）
- [x] 对齐配置文件：`sasrec_base_plain.yaml`
- [x] 添加 SVD 随机种子：`svd_random_state 42`
- [x] 添加 TF-IDF ngram 参数
- [x] 添加 Qwen3 chat template
- [x] 统一输出目录命名：`qwen3_4views/`
- [x] 添加 center+whiten 统计文件生成
- [x] 添加性能优化环境变量
- [x] 添加详细进度显示

## 其他数据集

同样的改进也应用于：
- VideoGames: `gen_text_emb_videogames_full_fast.sh` ✨ 新创建
- Yelp: `gen_text_emb_yelp_full_fast.sh` ✨ 新创建

## 下一步

1. 运行新脚本生成特征
2. 验证特征维度和统计文件
3. 运行实验对比性能
4. 如果结果符合预期，可以归档旧版脚本

## 注意事项

⚠️ **重要**: 新版本生成的特征与旧版本**不兼容**（维度不同），需要：
- 重新生成所有特征文件
- 使用对应的配置文件运行实验
- 不要混用新旧版本的特征文件

