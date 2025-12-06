# ⚡ 快速修复：TF-IDF生成太慢（CPU使用率7%）

## 🚨 问题

您的TF-IDF生成脚本CPU使用率只有7%，运行非常慢。

**原因：** NumPy/Scipy 默认单线程，未启用多核加速。

---

## ✅ 立即修复

### 步骤1：停止旧进程

```bash
# 找到并停止旧进程
pkill -f gen_text_emb_beauty_full.sh
```

### 步骤2：启动优化版脚本

```bash
cd /home/charlie/project/RecBole

# 优化版TF-IDF生成（多核加速）
nohup bash tools/gen_tfidf_only_fast.sh > gen_tfidf_fast.log 2>&1 &

# 查看日志
tail -f gen_tfidf_fast.log
```

### 步骤3：监控CPU使用率

```bash
# 另一个终端
htop
```

**期望看到：**
- ✅ CPU使用率：80-95%
- ✅ 多核心同时工作
- ✅ 详细的进度输出
- ✅ 2-5分钟完成（而不是10-20分钟）

---

## 📊 优化内容

### 1. 启用多核

```bash
# 优化版脚本自动设置
export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
```

### 2. 算法优化

- 使用 `randomized` SVD算法（更快）
- 添加详细进度输出
- 禁用TensorFlow警告

### 3. 预期加速比

| 任务 | 优化前 | 优化后 | 加速 |
|------|--------|--------|------|
| TF-IDF | 10-20 min | 2-5 min | **3-4x** |
| CPU使用率 | 7% | 80-95% | **12x** |

---

## 📂 新增文件

- ✅ `tools/gen_tfidf_only_fast.sh` - TF-IDF优化版
- ✅ `tools/gen_text_emb_beauty_full_fast.sh` - 完整版优化
- ✅ `tools/PERFORMANCE_OPTIMIZATION.md` - 详细优化指南
- ✅ `tools/build_item_text_emb_base.py` - 添加进度输出

---

## 🔍 验证优化效果

### 查看日志输出

```bash
tail -f gen_tfidf_fast.log
```

**应该看到：**

```
[Step 1/6] Loading dataset: Amazon_Beauty...
  → Dataset loaded: 259205 items

[Step 2/6] Reading item metadata...
  → Loaded 54540 item records
  → Using title field: 'title'

[Step 4/6] Building TF-IDF + SVD features...
  [SVD] Fitting TruncatedSVD: (178943, 45678) -> 256 components...
  [SVD] Explained variance ratio: 0.8234

✅ Saved item_text_emb to: dataset/Amazon_Beauty/item_text_emb.base.npy
   - Total time: 180.5s (3.0 min)
```

### 使用htop监控

```bash
htop
```

**应该看到：**
- 多个CPU核心都显示绿色（使用中）
- 总CPU使用率：80-95%
- Python进程CPU%：高

---

## 📚 详细文档

- **性能优化详解：** `tools/PERFORMANCE_OPTIMIZATION.md`
- **白化修复说明：** `tools/FIX_WHITENING_ORDER.md`
- **生成指南：** `tools/TEXT_EMB_GENERATION_GUIDE.md`

---

## 🐛 如果仍然慢

### 检查环境变量

```bash
# 在Python中检查
python -c "
import os
print('OMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS', 'not set'))
import numpy as np
np.show_config()
"
```

### 手动设置环境变量

```bash
# 设置后再运行
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

bash tools/gen_tfidf_only_fast.sh
```

---

**更新日期**: 2025-12-03  
**预期加速**: 3-4x  
**优先级**: 🟡 MEDIUM（已提供优化版）

