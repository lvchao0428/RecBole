# 文本嵌入生成参数修正说明

## 问题诊断

### ❌ 修正前（导致 2GB 文件）
```bash
# Qwen3 单视图生成（缺少降维）
python tools/build_item_text_emb_qwen3_hf.py \
  --prompt_preset base \
  --output_mode mean \
  # 缺少 --project_dim 256 ❌
  --max_length 0 \
  --dtype float16
```

**结果**：
- 输出维度：**3584d**（Qwen2.5-7B hidden_size，未降维）
- 文件大小：14K items × 3584d × 2 bytes ≈ **98 MB**

但如果误用了 `--prompt_preset multiview` + `--output_mode concat` 且未降维：
- 输出维度：**14,336d** (4 prompts × 3584d)
- 文件大小：14K items × 14,336d × 2 bytes ≈ **392 MB**

如果使用了更多 prompts 或 float32：
- 可能达到 **2GB**

---

## ✅ 修正后（符合最早版本标准）

### 统一降维策略

| 特征类型 | 降维参数 | 最终维度 | 文件大小 (float16) |
|---------|---------|---------|-------------------|
| **TF-IDF (base)** | `--svd_dim 256` | **256d** | ~7 MB |
| **Qwen3 单视图** | `--project_dim 256` | **256d** | ~7 MB |
| **Qwen3 多视图 concat** | `--view_project_dim 64` × 4 | **256d** | ~7 MB |
| **Qwen3 多视图 (per-view)** | `--view_project_dim 64` | **64d** × 4 views | ~7 MB (总) |

---

## 📝 修正后的生成命令

### 1. TF-IDF 基线特征
```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \                    # ✅ 降维到 256d
  --dtype float16
```

**输出**：
- 文件：`item_text_emb.base.npy` [14K, 256]
- 统计：`item_text_emb.base_whiten_stats.npz`

---

### 2. Qwen3 单视图特征（对比基线）

```bash
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --prompt_preset base \             # 单提示词
  --output_mode mean \               # 平均模式（虽然只有1个prompt）
  --project_dim 256 \                # ✅ 关键：降维到 256d（与TF-IDF一致）
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \                   # ✅ 不截断（保留完整语义）
  --dtype float16 \
  --svd_random_state 42 \
  --use_chat_template
```

**输出**：
- 文件：`item_text_emb.qwen3.base.npy` [14K, 256]
- 统计：`item_text_emb.qwen3.base_whiten_stats.npz`

**关键参数说明**：
- `--project_dim 256`：全局 SVD 降维，确保与 TF-IDF 维度一致
- `--max_length 0`：不截断输入，与最早版本保持一致
- `--prompt_preset base`：使用单个基础提示词 `[TITLE] {text}`

---

### 3. Qwen3 多视图特征（4个视角）

```bash
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --prompt_preset multiview \        # 4个提示词
  --output_mode concat \             # 拼接模式
  --split_output_dir dataset/Amazon_Beauty/qwen3_4views \
  --view_project_dim 64 \            # ✅ 每个视图降维到 64d（4×64=256d）
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \                   # ✅ 不截断
  --dtype float16 \
  --svd_random_state 42 \
  --use_chat_template
```

**输出**：
- 拼接文件：`item_text_emb.qwen3.multiview.npy` [14K, 256]
- 分视图目录：`qwen3_4views/`
  - `view_0.npy` [14K, 64] - Identity/Title
  - `view_1.npy` [14K, 64] - Function/Features
  - `view_2.npy` [14K, 64] - Target Audience
  - `view_3.npy` [14K, 64] - Category/Context
  - `views.json` - 元数据
  - `view_{0,1,2,3}_whiten_stats.npz` - 每个视图的白化统计
- 主统计：`item_text_emb.qwen3.multiview_whiten_stats.npz`

**关键参数说明**：
- `--view_project_dim 64`：每个视图独立降维到 64d
- `--split_output_dir`：保存分视图文件，供高级模型使用
- **最终 concat 维度**：4 views × 64d = 256d（与单视图、TF-IDF 一致）

---

## 🎯 统一降维尺度的原则

### 原则 1：所有主特征文件统一为 256d

无论是 TF-IDF、Qwen3 单视图还是多视图 concat，最终用于训练的主特征文件都是 **256d**：

```python
# 训练时加载
item_text_emb_base = np.load('item_text_emb.base.npy')           # [N, 256]
item_text_emb_qwen = np.load('item_text_emb.qwen3.base.npy')    # [N, 256]
item_text_emb_multi = np.load('item_text_emb.qwen3.multiview.npy') # [N, 256]
```

### 原则 2：多视图分视角文件统一降维

每个视图独立降维到相同维度（64d），确保公平对比：

```python
# 分视角加载
view_0 = np.load('qwen3_4views/view_0.npy')  # [N, 64]
view_1 = np.load('qwen3_4views/view_1.npy')  # [N, 64]
view_2 = np.load('qwen3_4views/view_2.npy')  # [N, 64]
view_3 = np.load('qwen3_4views/view_3.npy')  # [N, 64]

# Concat后恢复到256d
concat = np.concatenate([view_0, view_1, view_2, view_3], axis=1)  # [N, 256]
```

### 原则 3：与 TF-IDF 基线对齐

所有 Qwen3 特征的最终维度必须与 TF-IDF 基线一致（256d），确保：
1. 公平对比（相同参数量）
2. 可替换性（配置文件无需修改维度）
3. 内存效率（避免超大文件）

---

## 📊 文件大小对比表

| 文件路径 | 维度 | 大小 (float16) | 大小 (float32) |
|---------|------|---------------|---------------|
| `item_text_emb.base.npy` | 256d | ~7 MB | ~14 MB |
| `item_text_emb.qwen3.base.npy` ✅ | 256d | ~7 MB | ~14 MB |
| `item_text_emb.qwen3.multiview.npy` ✅ | 256d | ~7 MB | ~14 MB |
| `qwen3_4views/view_0.npy` | 64d | ~1.8 MB | ~3.5 MB |
| `qwen3_4views/view_1.npy` | 64d | ~1.8 MB | ~3.5 MB |
| `qwen3_4views/view_2.npy` | 64d | ~1.8 MB | ~3.5 MB |
| `qwen3_4views/view_3.npy` | 64d | ~1.8 MB | ~3.5 MB |
| **总计** | - | **~14 MB** | **~28 MB** |

### 对比错误生成的文件

| 文件 | 维度 | 大小 | 问题 |
|------|------|------|------|
| `item_text_emb.qwen3.base.npy` ❌ (旧) | 3584d | 98 MB | 未降维 |
| `item_text_emb.qwen3.base.npy` ❌ (更糟) | 14,336d | 392 MB | 多视角未降维 |
| `item_text_emb.qwen3.base.npy` ❌ (最糟) | 35,840d | **2 GB** | 10个prompt未降维 |

---

## ✅ 验证命令

重新生成后，运行以下命令验证：

```bash
cd /home/charlie/project/RecBole

# 快速验证脚本
python3 << 'EOF'
import numpy as np
import os

print("=" * 60)
print("文本嵌入特征验证")
print("=" * 60)

files = {
    'TF-IDF基线': 'dataset/Amazon_Beauty/item_text_emb.base.npy',
    'Qwen3单视图': 'dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy',
    'Qwen3多视图concat': 'dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy',
}

view_files = {
    f'View {i}': f'dataset/Amazon_Beauty/qwen3_4views/view_{i}.npy'
    for i in range(4)
}

def check_file(name, path):
    if not os.path.exists(path):
        print(f"❌ {name}: 文件不存在")
        return False
    
    arr = np.load(path, mmap_mode='r')
    size_mb = arr.nbytes / 1024**2
    expected_dim = 256 if 'View' not in name else 64
    
    status = "✅" if arr.shape[1] == expected_dim else "❌"
    print(f"{status} {name}:")
    print(f"   Shape: {arr.shape}")
    print(f"   Dtype: {arr.dtype}")
    print(f"   Size:  {size_mb:.1f} MB")
    print(f"   Expected dim: {expected_dim}d")
    print()
    
    return arr.shape[1] == expected_dim

print("主特征文件（应为 256d）:")
print("-" * 60)
all_ok = all(check_file(name, path) for name, path in files.items())

print("分视图文件（应为 64d）:")
print("-" * 60)
views_ok = all(check_file(name, path) for name, path in view_files.items())

if all_ok and views_ok:
    print("=" * 60)
    print("✅ 所有文件维度正确！")
    print("=" * 60)
else:
    print("=" * 60)
    print("❌ 部分文件维度不正确，请重新生成")
    print("=" * 60)
EOF
```

**期望输出**：
```
============================================================
文本嵌入特征验证
============================================================
主特征文件（应为 256d）:
------------------------------------------------------------
✅ TF-IDF基线:
   Shape: (14000, 256)
   Dtype: float16
   Size:  7.0 MB
   Expected dim: 256d

✅ Qwen3单视图:
   Shape: (14000, 256)
   Dtype: float16
   Size:  7.0 MB
   Expected dim: 256d

✅ Qwen3多视图concat:
   Shape: (14000, 256)
   Dtype: float16
   Size:  7.0 MB
   Expected dim: 256d

分视图文件（应为 64d）:
------------------------------------------------------------
✅ View 0:
   Shape: (14000, 64)
   Dtype: float16
   Size:  1.8 MB
   Expected dim: 64d

✅ View 1:
   Shape: (14000, 64)
   Dtype: float16
   Size:  1.8 MB
   Expected dim: 64d

✅ View 2:
   Shape: (14000, 64)
   Dtype: float16
   Size:  1.8 MB
   Expected dim: 64d

✅ View 3:
   Shape: (14000, 64)
   Dtype: float16
   Size:  1.8 MB
   Expected dim: 64d

============================================================
✅ 所有文件维度正确！
============================================================
```

---

## 🚀 重新生成步骤

### 删除旧的错误文件

```bash
cd /home/charlie/project/RecBole

# 备份到时间戳目录
BACKUP_DIR="backup/embedding_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 移动错误的文件
mv dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy "$BACKUP_DIR/" 2>/dev/null || true
mv dataset/Amazon_Beauty/item_text_emb.qwen3.base_whiten_stats.npz "$BACKUP_DIR/" 2>/dev/null || true

echo "✅ 旧文件已备份到: $BACKUP_DIR"
```

### 重新生成所有特征

```bash
cd /home/charlie/project/RecBole

# 运行统一生成脚本（已修正参数）
bash tools/gen_text_emb_beauty_full_fast.sh
```

---

## 📌 总结

### 修正的核心变更
1. **Qwen3 单视图**：添加 `--project_dim 256`
2. **Qwen3 多视图**：确认 `--view_project_dim 64` × 4 = 256d
3. **统一降维尺度**：所有主特征 256d，分视图 64d

### 保持不变的参数
- `--max_length 0`：不截断（与最早版本一致）
- `--svd_random_state 42`：随机种子保持一致
- `--dtype float16`：内存优化
- `--use_chat_template`：使用 Qwen 对话模板

### 适用场景
- `item_text_emb.base.npy`：所有实验的 TF-IDF 基线
- `item_text_emb.qwen3.base.npy`：单视图 Qwen3 对比实验
- `item_text_emb.qwen3.multiview.npy`：多视图 concat 实验
- `qwen3_4views/view_*.npy`：高级多视图融合实验

