# Multi-view 白化问题确认与修复

## 问题确认

验证 Multi-view 各视图时发现白化失败：

```bash
python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_0.npy
```

**结果**:
```
✅ Embedding已L2归一化 (mean=1.0000, std=0.0001)
❌ 对角线均值: 0.0156 (期望≈1.0)
```

对于 64 维：1/64 = 0.015625 ≈ **0.0156** ✗

## 同样的问题

Multi-view 的每个视图都受到**同样的白化后 L2 归一化**问题影响：

- `view_0.npy` (Identity, 64维) → 对角线 ≈ 0.0156 ✗
- `view_1.npy` (Function, 64维) → 对角线 ≈ 0.0156 ✗
- `view_2.npy` (Audience, 64维) → 对角线 ≈ 0.0156 ✗
- `view_3.npy` (Category, 64维) → 对角线 ≈ 0.0156 ✗

拼接后的特征也会受影响：
- `item_text_emb.qwen3.multiview.npy` (256维) → 对角线 ≈ 0.0039 ✗

## Multi-view 的双层白化

Multi-view 模式有**两层白化**，都受此问题影响：

### 第1层: 每个视图单独白化
```python
# 代码位置: build_item_text_emb_qwen3_hf.py L619-630
for view_idx in range(4):
    view_mat = _center_whiten_and_normalize(  # ← 这里白化
        view_mat,
        train_ids_cache,
        output_stats_path=f"view_{view_idx}_whiten_stats.npz",
        enable_whiten=True,
    )
    # 保存: view_0.npy, view_1.npy, view_2.npy, view_3.npy
```

### 第2层: 拼接后整体白化
```python
# 代码位置: build_item_text_emb_qwen3_hf.py L658, L684-691
mat = np.concatenate(view_mats_for_concat, axis=1)  # 拼接: 4×64 → 256
mat = _center_whiten_and_normalize(  # ← 再次白化
    mat,
    train_ids_cache,
    output_stats_path='item_text_emb.qwen3.multiview_whiten_stats.npz',
    enable_whiten=True,
)
# 保存: item_text_emb.qwen3.multiview.npy
```

## 代码修复状态

✅ **已修复**：`_center_whiten_and_normalize` 函数已移除白化后的 L2 归一化
- 第1层白化（每个视图）✅
- 第2层白化（拼接后）✅

验证修复：
```bash
python3 tools/verify_whiten_fix.py
```

## 需要重新生成

### 快速方案：仅重新生成 Multi-view

```bash
cd /home/charlie/project/RecBole
bash tools/regen_multiview_beauty.sh
```

这会：
1. 备份旧的 `qwen3_4views/` 目录
2. 重新生成 4 个视图（每个 64 维）
3. 重新生成拼接特征（256 维）
4. 自动验证所有视图的白化效果

### 完整方案：重新生成单视图 + Multi-view

```bash
cd /home/charlie/project/RecBole
bash tools/regen_all_qwen3_beauty.sh
```

这会同时重新生成：
- 单视图：`item_text_emb.qwen3.base.npy` (256维)
- Multi-view：`qwen3_4views/view_*.npy` (4×64维)
- 拼接：`item_text_emb.qwen3.multiview.npy` (256维)

## 验证期望结果

### 验证各视图 (64维)

```bash
for i in 0 1 2 3; do
    python tools/verify_whiten.py \
      dataset/Amazon_Beauty/qwen3_4views/view_${i}.npy \
      --dataset Amazon_Beauty
done
```

**期望输出**:
```
ℹ️  Embedding未L2归一化 (mean=8.xxxx, std=x.xxxx)  ← 正常！
   提示: sqrt(64) = 8，这是正常的

✅ Embedding已中心化 (mean_abs<0.2)

📊 协方差矩阵分析:
   - 对角线均值: 0.95~1.05 (期望≈1.0)  ← 修复成功！
   - 非对角线平均值: <0.05 (期望≈0)

✅ 白化效果优秀
```

### 验证拼接特征 (256维)

```bash
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy \
  --dataset Amazon_Beauty
```

**期望输出**:
```
ℹ️  Embedding未L2归一化 (mean=16.xxxx, std=x.xxxx)  ← 正常！
   提示: sqrt(256) = 16，这是正常的

✅ Embedding已中心化 (mean_abs<0.2)

📊 协方差矩阵分析:
   - 对角线均值: 0.95~1.05 (期望≈1.0)  ← 修复成功！
   - 非对角线平均值: <0.05 (期望≈0)

✅ 白化效果优秀
```

## 修复前后对比

### 修复前（错误）

| 文件 | 维度 | L2范数 | 对角线均值 | 状态 |
|------|------|--------|-----------|------|
| view_0.npy | 64 | 1.0 | 0.0156 (1/64) | ❌ 失败 |
| view_1.npy | 64 | 1.0 | 0.0156 (1/64) | ❌ 失败 |
| view_2.npy | 64 | 1.0 | 0.0156 (1/64) | ❌ 失败 |
| view_3.npy | 64 | 1.0 | 0.0156 (1/64) | ❌ 失败 |
| multiview.npy | 256 | 1.0 | 0.0039 (1/256) | ❌ 失败 |

### 修复后（正确）

| 文件 | 维度 | L2范数 | 对角线均值 | 状态 |
|------|------|--------|-----------|------|
| view_0.npy | 64 | ~8 | ~1.0 | ✅ 通过 |
| view_1.npy | 64 | ~8 | ~1.0 | ✅ 通过 |
| view_2.npy | 64 | ~8 | ~1.0 | ✅ 通过 |
| view_3.npy | 64 | ~8 | ~1.0 | ✅ 通过 |
| multiview.npy | 256 | ~16 | ~1.0 | ✅ 通过 |

## 下一步

1. **在服务器上运行**（需要 GPU）:
   ```bash
   bash tools/regen_multiview_beauty.sh
   ```

2. **检查验证输出**，确认所有视图都通过

3. **如果其他数据集也生成了 Multi-view**，也需要重新生成

## 相关文件

- `tools/regen_multiview_beauty.sh` - Multi-view 重新生成脚本
- `tools/regen_all_qwen3_beauty.sh` - 完整重新生成脚本（单视图+多视图）
- `WHITEN_FIX_SUMMARY.md` - 总体修复说明
- `WHITEN_FIX_QWEN3.md` - 详细技术说明

