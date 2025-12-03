# 文本特征生成指南（全功能版）

本指南说明如何使用新版特征生成脚本，所有功能已全部启用。

## 🎯 核心改进

### ✅ 已启用的功能

1. **Center + Whiten 标准化**
   - 仅在训练集上计算统计量（避免信息泄漏）
   - 保存 `_whiten_stats.npz` 用于推理复用
   - 可通过 `--no_whiten` 禁用

2. **训练集拟合**
   - TF-IDF词汇表仅从训练集构建
   - SVD降维矩阵仅在训练集上拟合
   - 白化矩阵仅从训练集特征计算

3. **多视图支持（Qwen3）**
   - 4个视图提示词（Identity, Function, Audience, Category）
   - 每个视图独立SVD降维
   - 支持拼接/平均/堆叠三种输出模式

---

## 📝 脚本说明

### 1️⃣ `gen_text_emb_beauty_full.sh` - 完整版（推荐）

生成所有类型的文本特征：
- TF-IDF基线
- Qwen3单视图
- Qwen3多视图（4个视图）

**运行方式：**
```bash
cd /home/charlie/project/RecBole
bash tools/gen_text_emb_beauty_full.sh
```

**输出文件：**
```
dataset/Amazon_Beauty/
├── item_text_emb.base.npy                      # TF-IDF特征 [N, 256]
├── item_text_emb.base_whiten_stats.npz         # TF-IDF统计量
├── item_text_emb.qwen3.base.npy                # Qwen3单视图 [N, D]
├── item_text_emb.qwen3.base_whiten_stats.npz   # Qwen3单视图统计量
├── item_text_emb.qwen3.multiview.npy           # Qwen3多视图拼接 [N, 256]
├── item_text_emb.qwen3.multiview_whiten_stats.npz  # 多视图统计量
├── item_index_mapping.csv                       # Item映射表
└── qwen3_4views/                                # 分视图特征
    ├── view_0.npy                               # 视图1: Identity [N, 64]
    ├── view_1.npy                               # 视图2: Function [N, 64]
    ├── view_2.npy                               # 视图3: Audience [N, 64]
    ├── view_3.npy                               # 视图4: Category [N, 64]
    └── views.json                               # 元数据
```

**预计耗时：**
- TF-IDF: 1-5分钟
- Qwen3单视图: 10-30分钟（取决于GPU和数据量）
- Qwen3多视图: 30-60分钟

---

### 2️⃣ `gen_tfidf_only.sh` - 仅TF-IDF（快速验证）

快速生成TF-IDF基线特征，用于验证流程。

**运行方式：**
```bash
bash tools/gen_tfidf_only.sh
```

**输出：**
- `dataset/Amazon_Beauty/item_text_emb.base.npy`
- `dataset/Amazon_Beauty/item_text_emb.base_whiten_stats.npz`

---

### 3️⃣ `gen_qwen3_multiview_only.sh` - 仅Qwen3多视图

仅生成Qwen3多视图特征，跳过TF-IDF和单视图。

**运行方式：**
```bash
bash tools/gen_qwen3_multiview_only.sh
```

**输出：**
- `dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy`
- `dataset/Amazon_Beauty/qwen3_4views/` 目录

---

## 🔧 参数说明

### TF-IDF参数（`build_item_text_emb_base.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | **必填** | 数据集名称（如 Amazon_Beauty） |
| `--config` | `[]` | YAML配置文件 |
| `--output` | **必填** | 输出.npy路径 |
| `--svd_dim` | `256` | SVD降维目标维度 |
| `--svd_random_state` | `42` | 随机种子（可复现） |
| `--ngram_min` | `1` | 字符n-gram最小长度 |
| `--ngram_max` | `2` | 字符n-gram最大长度 |
| `--min_df` | `2` | 最小文档频率 |
| `--dtype` | `float16` | 输出数据类型 |
| `--no_whiten` | `False` | 禁用白化（默认启用） |

### Qwen3参数（`build_item_text_emb_qwen3_hf.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mapping` | **必填** | item_index_mapping.csv路径 |
| `--model_name_or_path` | **必填** | Qwen模型路径 |
| `--output` | **必填** | 输出.npy路径 |
| `--prompt_preset` | `base` | 提示词预设（base/multiview/description） |
| `--output_mode` | `mean` | 输出模式（mean/concat/stack） |
| `--batch_size` | `16` | 批次大小 |
| `--max_length` | `0` | 最大序列长度（0=无限制） |
| `--dtype` | `float16` | 输出数据类型 |
| `--device` | `cuda:0` | 计算设备 |
| `--view_project_dim` | `None` | 每视图SVD降维目标维度 |
| `--project_dim` | `None` | 最终SVD降维目标维度 |
| `--split_output_dir` | `None` | 分视图输出目录 |
| `--dataset` | `None` | RecBole数据集（用于训练集拆分） |
| `--config` | `[]` | YAML配置文件 |
| `--no_whiten` | `False` | 禁用白化（默认启用） |

---

## 🎨 多视图提示词设计

4个视图对应不同的语义角度：

### 视图1: Identity（身份识别）
```
"Identify the item: [TITLE] {text}"
```
捕捉物品的核心身份信息。

### 视图2: Function（功能特性）
```
"What are the main functions and features of [TITLE] {text}?"
```
提取功能性和实用性描述。

### 视图3: Audience（目标用户）
```
"Who is the target audience or user group for [TITLE] {text}?"
```
理解目标受众和使用场景。

### 视图4: Category（类别上下文）
```
"Categorize the item [TITLE] {text} and describe its context."
```
获取类别归属和上下文信息。

---

## 🚀 推荐工作流程

### 方案A：快速验证（TF-IDF基线）

```bash
# 1. 生成TF-IDF特征
bash tools/gen_tfidf_only.sh

# 2. 运行TF-IDF实验
bash two_phase_run_tfidf.sh

# 3. 检查结果
ls -lh dataset/Amazon_Beauty/item_text_emb.base*
```

### 方案B：完整实验（所有特征）

```bash
# 1. 生成所有特征（可能需要1-2小时）
bash tools/gen_text_emb_beauty_full.sh

# 2. 运行TF-IDF基线实验
bash two_phase_run_tfidf.sh

# 3. 运行Qwen3多视图实验
bash two_phase_run_multiview_split.sh

# 4. 对比结果
tensorboard --logdir saved/
```

### 方案C：仅多视图实验

```bash
# 1. 生成Qwen3多视图特征
bash tools/gen_qwen3_multiview_only.sh

# 2. 运行多视图实验
bash two_phase_run_multiview_split.sh
```

---

## 📊 输出文件格式

### `.npy` 特征矩阵

- **形状**: `[n_items, d_text]`
- **第0行**: PAD向量（全零）
- **第1~N行**: 对应RecBole内部item_id 1~N
- **数据类型**: `float16` (节省内存) 或 `float32`

### `_whiten_stats.npz` 统计量文件

包含两个数组：
- `mean`: 训练集均值向量 `[1, d_text]`
- `whiten_matrix`: 白化矩阵 `[d_text, d_text]` (如果启用)

用于推理时应用相同的标准化。

### `views.json` 元数据

```json
{
  "num_items": 12345,
  "num_prompts": 4,
  "dtype": "float16",
  "prompts": [
    {
      "index": 0,
      "prompt": "Identify the item: [TITLE] {text}",
      "file": "view_0.npy",
      "vector_dim": 64
    },
    ...
  ]
}
```

---

## ⚠️ 注意事项

### 1. 路径配置

- ✅ 配置文件: `sasrec_base_plain.yaml` (根目录)
- ✅ 数据集路径: `dataset/Amazon_Beauty/` (不是 `data/`)
- ✅ Qwen模型路径: `/home/charlie/project/qwen/Model`

### 2. GPU内存管理

如果遇到CUDA OOM错误：
```bash
# 方法1: 减小batch_size
--batch_size 8

# 方法2: 启用内存扩展（已在脚本中启用）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 方法3: 使用float32（如果GPU支持）
--dtype float32
```

### 3. 可复现性保证

所有脚本已固定：
- SVD随机种子: `--svd_random_state 42`
- TF-IDF参数完全确定
- 白化统计量保存用于推理

### 4. 验证生成结果

```bash
# 检查文件大小和形状
python -c "
import numpy as np
emb = np.load('dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy')
print(f'Shape: {emb.shape}')
print(f'Dtype: {emb.dtype}')
print(f'PAD row (should be zeros): {emb[0][:10]}')
print(f'Item 1 norm: {np.linalg.norm(emb[1])}')
"
```

---

## 🔍 特征验证

生成特征后，使用验证工具检查质量：

```bash
# 批量验证所有生成的文件
bash tools/verify_all_embeddings.sh

# 验证单个文件
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

# 验证多个文件
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.base.npy \
  dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
```

验证内容：
- ✅ PAD向量为零
- ✅ L2归一化
- ✅ 中心化（center）
- ✅ 白化效果（whiten）

详细说明：[VERIFICATION_GUIDE.md](./VERIFICATION_GUIDE.md)

---

## 📚 相关文档

- **验证工具**:
  - [VERIFICATION_GUIDE.md](./VERIFICATION_GUIDE.md) - 完整验证指南
  - `tools/verify_whiten.py` - 验证脚本
  - `tools/verify_all_embeddings.sh` - 批量验证
- **实验脚本**: `two_phase_run_tfidf.sh`, `two_phase_run_multiview_split.sh`
- **旧版生成器**: `tools/gen_text_emb_beauty.sh` (参考用)
- **源码**:
  - `tools/build_item_text_emb_base.py` (TF-IDF)
  - `tools/build_item_text_emb_qwen3_hf.py` (Qwen3)
  - `tools/export_internal_item_mapping.py` (Mapping导出)

---

## 🐛 常见问题

### Q1: 找不到 `sasrec_base_plain.yaml`

检查文件是否在项目根目录：
```bash
ls -l sasrec_base_plain.yaml
```

### Q2: Qwen模型加载失败

确认模型路径存在：
```bash
ls -l /home/charlie/project/qwen/Model
```

### Q3: 白化后性能下降

尝试禁用白化，仅使用中心化：
```bash
python tools/build_item_text_emb_base.py \
  --no_whiten \
  ...
```

### Q4: 多视图维度不匹配

确认配置：
- `--view_project_dim 64` × 4视图 = 256维
- `--output_mode concat` 拼接模式

---

**创建日期**: 2025-12-03  
**版本**: v2.0 (全功能版)

