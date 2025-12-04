# 文本嵌入生成快速参考

## 🎯 统一降维标准

**所有主特征文件统一为 256d** （与最早版本保持一致）

| 特征类型 | 命令关键参数 | 输出维度 | 文件大小 |
|---------|-------------|---------|---------|
| **TF-IDF** | `--svd_dim 256` | 256d | ~7 MB |
| **Qwen3 单视图** | `--project_dim 256` | 256d | ~7 MB |
| **Qwen3 多视图** | `--view_project_dim 64` × 4 | 256d | ~7 MB |

---

## ⚡ 一键生成所有特征

```bash
cd /home/charlie/project/RecBole
bash tools/gen_text_emb_beauty_full_fast.sh
```

生成文件：
```
dataset/Amazon_Beauty/
├── item_text_emb.base.npy               [14K, 256]  # TF-IDF
├── item_text_emb.base_whiten_stats.npz
├── item_text_emb.qwen3.base.npy         [14K, 256]  # Qwen3 单视图
├── item_text_emb.qwen3.base_whiten_stats.npz
├── item_text_emb.qwen3.multiview.npy    [14K, 256]  # Qwen3 多视图 concat
├── item_text_emb.qwen3.multiview_whiten_stats.npz
└── qwen3_4views/
    ├── view_0.npy                       [14K, 64]   # Identity
    ├── view_1.npy                       [14K, 64]   # Function
    ├── view_2.npy                       [14K, 64]   # Audience
    ├── view_3.npy                       [14K, 64]   # Category
    ├── view_{0,1,2,3}_whiten_stats.npz
    └── views.json
```

---

## 🔍 快速验证

```bash
cd /home/charlie/project/RecBole

python3 << 'EOF'
import numpy as np
files = [
    ('TF-IDF', 'dataset/Amazon_Beauty/item_text_emb.base.npy', 256),
    ('Qwen3单视图', 'dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy', 256),
    ('Qwen3多视图', 'dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy', 256),
    ('View0', 'dataset/Amazon_Beauty/qwen3_4views/view_0.npy', 64),
]

for name, path, expected in files:
    try:
        arr = np.load(path, mmap_mode='r')
        ok = "✅" if arr.shape[1] == expected else "❌"
        print(f"{ok} {name}: {arr.shape} ({arr.nbytes/1024**2:.1f}MB)")
    except:
        print(f"❌ {name}: 文件不存在")
EOF
```

**期望输出**：
```
✅ TF-IDF: (14000, 256) (7.0MB)
✅ Qwen3单视图: (14000, 256) (7.0MB)
✅ Qwen3多视图: (14000, 256) (7.0MB)
✅ View0: (14000, 64) (1.8MB)
```

---

## 🚨 常见问题排查

### 问题1：生成的文件超过 100MB

**原因**：缺少 `--project_dim` 参数

**解决**：
```bash
# 检查是否遗漏降维参数
grep -n "project_dim" tools/gen_text_emb_beauty_full_fast.sh

# 应该看到：
# 83:  --project_dim 256 \        # Qwen3单视图
# 113: --view_project_dim 64 \    # Qwen3多视图
```

### 问题2：多视图 concat 后维度不是 256d

**原因**：`--view_project_dim` 设置错误

**正确配置**：
- 4个视图：`--view_project_dim 64` → 4×64=256d ✅
- 5个视图：`--view_project_dim 51` → 5×51=255d ≈ 256d
- N个视图：`--view_project_dim floor(256/N)`

### 问题3：whitening 统计文件缺失

**原因**：旧版本脚本未启用 whitening

**检查**：
```bash
ls -lh dataset/Amazon_Beauty/*_whiten_stats.npz
```

应该包含：
- `item_text_emb.base_whiten_stats.npz`
- `item_text_emb.qwen3.base_whiten_stats.npz`
- `item_text_emb.qwen3.multiview_whiten_stats.npz`
- `qwen3_4views/view_{0,1,2,3}_whiten_stats.npz`

---

## 📋 实验配置映射

### 实验 1：TF-IDF 基线
```yaml
# two_phase_run_tfidf.sh
item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.base.npy  # 256d
use_llm: false
```

### 实验 2：TF-IDF + Qwen3 单视图
```yaml
# two_phase_run_tfidf_llm.sh
item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.base.npy       # 256d
item_text_emb_path_llm: dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy  # 256d
use_llm: true
# Concat后：256d + 256d = 512d
```

### 实验 3：TF-IDF + Qwen3 多视图 concat
```yaml
# two_phase_run_multiview_concat.sh
item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.base.npy           # 256d
item_text_emb_path_llm: dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy # 256d
use_llm: true
num_text_views: 4
# Concat后：256d + 256d = 512d
```

### 实验 4：TF-IDF + Qwen3 多视图分离
```yaml
# two_phase_run_multiview_split.sh
item_text_emb_path_base: dataset/Amazon_Beauty/item_text_emb.base.npy  # 256d
item_text_multiview_dir: dataset/Amazon_Beauty/qwen3_4views            # 4×64d
use_text_view_cross: true
text_view_residual_weight: 1.0
# 每个view独立融合后再组合
```

---

## 🔧 手动生成单个特征（调试用）

### TF-IDF（如果需要重新生成）
```bash
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --dtype float16
```

### Qwen3 单视图（修正版）
```bash
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  --prompt_preset base \
  --output_mode mean \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --use_chat_template
```

### Qwen3 多视图（修正版）
```bash
python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --prompt_preset multiview \
  --output_mode concat \
  --split_output_dir dataset/Amazon_Beauty/qwen3_4views \
  --view_project_dim 64 \
  --dataset Amazon_Beauty \
  --config sasrec_base_plain.yaml recbole/properties/overall.yaml \
  --batch_size 16 \
  --max_length 0 \
  --dtype float16 \
  --use_chat_template
```

---

## ⏱️ 预计生成时间（RTX 3090）

| 步骤 | 时间 | GPU使用 |
|------|------|---------|
| TF-IDF基线 | ~2分钟 | CPU |
| Qwen3单视图 | ~10分钟 | ~8GB |
| Qwen3多视图（4个） | ~40分钟 | ~8GB |
| **总计** | **~50分钟** | - |

---

## 📝 修正总结

### ✅ 已修正
1. `gen_text_emb_beauty_full_fast.sh` 第83行添加 `--project_dim 256`
2. 统一所有主特征文件为 256d
3. 多视图每个 view 降维到 64d（4×64=256d）

### 🎯 核心原则
- **公平对比**：所有特征相同维度（256d）
- **一致性**：与最早版本参数对齐（`--max_length 0`, `--project_dim 256`）
- **可复现**：固定随机种子（`--svd_random_state 42`）

### 🚀 下一步
```bash
# 1. 删除旧的错误文件
mv dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy backup/

# 2. 重新生成
bash tools/gen_text_emb_beauty_full_fast.sh

# 3. 验证
python3 tools/verify_all_embeddings.sh  # 如果有
```

