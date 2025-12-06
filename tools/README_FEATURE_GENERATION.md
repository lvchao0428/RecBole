# 文本特征生成工具 - 快速参考

## 📋 新创建的文件

### 1. 特征生成脚本（对齐 Beauty 格式）

| 数据集 | 脚本文件 | 状态 |
|--------|---------|------|
| Amazon Beauty | `gen_text_emb_beauty_full_fast.sh` | ✅ 已存在（参考） |
| Amazon Toys | `gen_text_emb_toys_full_fast.sh` | ✨ **新创建** |
| Amazon VideoGames | `gen_text_emb_videogames_full_fast.sh` | ✨ **新创建** |
| Yelp | `gen_text_emb_yelp_full_fast.sh` | ✨ **新创建** |

### 2. 工具脚本

| 文件 | 用途 |
|------|------|
| `gen_all_datasets_full_fast.sh` | 一键生成所有数据集特征 |
| `verify_all_embeddings.py` | 验证生成的特征文件 |

### 3. 文档

| 文件 | 内容 |
|------|------|
| `TEXT_EMB_GENERATION_GUIDE.md` | 详细使用指南 |
| `TOYS_MIGRATION_SUMMARY.md` | Toys 迁移总结和对比 |
| `README_FEATURE_GENERATION.md` | 本文档（快速参考） |

## 🚀 快速开始

### 单个数据集

```bash
# Toys
bash tools/gen_text_emb_toys_full_fast.sh

# VideoGames
bash tools/gen_text_emb_videogames_full_fast.sh

# Yelp
bash tools/gen_text_emb_yelp_full_fast.sh

# Beauty（参考）
bash tools/gen_text_emb_beauty_full_fast.sh
```

### 所有数据集（一键生成）

```bash
bash tools/gen_all_datasets_full_fast.sh
```

### 验证生成结果

```bash
python tools/verify_all_embeddings.py
```

## 📊 生成的特征

每个数据集生成以下文件：

```
dataset/{DATASET_NAME}/
├── item_index_mapping.csv                       # Item ID 映射
├── item_text_emb.base.npy                       # TF-IDF (256维)
├── item_text_emb.base_whiten_stats.npz          # TF-IDF 统计
├── item_text_emb.qwen3.base.npy                 # Qwen3 单视图 (256维)
├── item_text_emb.qwen3.base_whiten_stats.npz    # Qwen3 单视图统计
├── item_text_emb.qwen3.multiview.npy            # Qwen3 多视图拼接 (256维)
├── item_text_emb.qwen3.multiview_whiten_stats.npz # 多视图统计
└── qwen3_4views/                                # 多视图分视图目录
    ├── view_0.npy                               # Identity (64维)
    ├── view_1.npy                               # Function (64维)
    ├── view_2.npy                               # Audience (64维)
    ├── view_3.npy                               # Category (64维)
    └── views.json                               # 元数据
```

## ✨ 关键改进点

### 1. 多视图维度对齐
- **旧版**: 4 × 128 = 512 维
- **新版**: 4 × 64 = **256 维** ✅

### 2. Center+Whiten 归一化
- 自动生成统计文件（`.npz`）
- 包含 `center` 和 `whiten` 矩阵
- 确保归一化一致性

### 3. 可复现性
- TF-IDF: `svd_random_state=42`
- Qwen3: `svd_random_state=42`
- 确保结果可复现

### 4. 统一配置
- Amazon 数据集: `sasrec_base_plain.yaml`
- Yelp 数据集: `yelp_config/yelp_sasrec_base_plain.yaml`

## 🔍 验证检查项

`verify_all_embeddings.py` 会检查：

1. ✅ 特征文件是否存在
2. ✅ 特征维度是否正确（256维）
3. ✅ Center+Whiten 统计文件
4. ✅ 多视图分视图文件（4个视图，每个64维）
5. ✅ 元数据文件（views.json）

## 📈 运行实验

### TF-IDF 实验

```bash
bash two_phase_run_tfidf_toys.sh
bash two_phase_run_tfidf_videogames.sh
bash two_phase_run_tfidf_yelp.sh
```

### 多视图实验

```bash
bash two_phase_run_multiview_split_toys.sh
bash two_phase_run_multiview_split_videogames.sh
bash two_phase_run_multiview_split_yelp.sh
```

## ⚙️ 配置调整

如果遇到问题，可以调整以下参数：

### GPU 内存不足（CUDA OOM）

编辑脚本中的 `--batch_size` 参数：
```bash
--batch_size 8  # 从 16 降为 8
```

### CPU 线程过多

编辑脚本中的环境变量：
```bash
export OMP_NUM_THREADS=8  # 限制为 8 线程
```

### 跳过某个步骤

注释掉对应的生成命令（在脚本中用 `#` 注释）

## 📝 对比表格

| 特性 | 旧版 Toys | 新版 Toys | Beauty |
|------|-----------|-----------|--------|
| 多视图维度 | 512 | 256 ✅ | 256 |
| 统计文件 | ❌ | ✅ | ✅ |
| SVD 种子 | ❌ | 42 ✅ | 42 |
| Chat Template | ❌ | ✅ | ✅ |
| 配置文件 | align_base | base_plain ✅ | base_plain |
| 输出目录 | 4views_split | qwen3_4views ✅ | qwen3_4views |

## 🎯 要点总结

1. ✅ **维度对齐**: 所有数据集统一为 256 维
2. ✅ **命名统一**: 文件和目录命名一致
3. ✅ **可复现性**: 添加随机种子
4. ✅ **归一化**: center+whiten 统计文件
5. ✅ **性能优化**: 多线程加速
6. ✅ **详细日志**: 时间戳和进度显示

## 📞 故障排除

### 问题 1: 脚本找不到 Python 工具

**解决**: 确保在 RecBole 根目录运行
```bash
cd /home/charlie/project/RecBole
bash tools/gen_text_emb_toys_full_fast.sh
```

### 问题 2: 统计文件未生成

**原因**: 可能是旧版本的 Python 工具
**解决**: 确保使用最新的 `build_item_text_emb_*.py` 脚本

### 问题 3: 维度不匹配

**原因**: 可能混用了新旧版本的特征
**解决**: 删除旧特征，重新生成
```bash
rm -rf dataset/Amazon_Toys_and_Games/*.npy
rm -rf dataset/Amazon_Toys_and_Games/*.npz
rm -rf dataset/Amazon_Toys_and_Games/qwen3_4views/
bash tools/gen_text_emb_toys_full_fast.sh
```

## 📚 更多信息

- 详细文档: `TEXT_EMB_GENERATION_GUIDE.md`
- Toys 迁移说明: `TOYS_MIGRATION_SUMMARY.md`
- 原始 Beauty 脚本: `gen_text_emb_beauty_full_fast.sh`

## ✅ 迁移检查清单

- [x] 创建 Toys 新版脚本
- [x] 创建 VideoGames 新版脚本
- [x] 创建 Yelp 新版脚本
- [x] 创建批量生成脚本
- [x] 创建验证工具
- [x] 编写文档
- [ ] 运行脚本生成特征
- [ ] 验证特征文件
- [ ] 运行实验对比

---

**更新时间**: 2025-12-06  
**版本**: 1.0  
**状态**: ✅ 已完成

