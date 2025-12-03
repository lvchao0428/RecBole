# 改进实施清单

## ✅ 已完成的改动

### 1. 代码修改
- [x] `recbole/model/sequential_recommender/sasrecalignmultiview.py`
  - [x] 投影层支持Base特征（Line 93-102）
  - [x] 添加Base特征拼接逻辑（Line 219-230）
  - [x] 重新初始化融合网络768维（Line 107-127）
  - [x] 更新注释和日志输出

### 2. 配置文件
- [x] `sasrec_align_multi_view.yaml`
  - [x] 更新注释说明新架构
  - [x] 修正split_dir路径为 `qwen3_4views`

### 3. 文档创建
- [x] `MULTIVIEW_IMPROVED_ARCHITECTURE.md` - 改进架构详解
- [x] `MULTIVIEW_IMPROVEMENT_SUMMARY.md` - 改进总结
- [x] `ARCHITECTURE_COMPARISON_VISUAL.md` - 可视化对比
- [x] `FUSION_DIMENSION_ANALYSIS.md` - 维度分析
- [x] `FUSION_BOTTLENECK_FIX.md` - 修复说明
- [x] `MULTIVIEW_IMPROVEMENT_README.md` - 使用指南
- [x] `verify_multiview_improvement.py` - 验证脚本

---

## 🚀 使用前检查

### 必需的特征文件

```bash
cd /home/charlie/project/RecBole

# 检查Base特征
ls -lh dataset/Amazon_Beauty/item_text_emb.base.npy

# 检查多视图特征
ls -lh dataset/Amazon_Beauty/qwen3_4views/view_*.npy
ls -lh dataset/Amazon_Beauty/qwen3_4views/views.json
```

**如果文件不存在：**
```bash
# 生成所有特征
bash tools/gen_text_emb_beauty_full.sh

# 或分别生成
bash tools/gen_tfidf_only_fast.sh
bash tools/gen_qwen3_multiview_only.sh
```

---

## 🔍 验证改进

### 运行验证脚本

```bash
python verify_multiview_improvement.py
```

### 期望输出

```
======================================================================
验证多视图模型改进
======================================================================

[1/5] 加载配置...
✅ 配置加载成功

[2/5] 加载数据集...
✅ 数据集加载成功: 259205 items

[3/5] 初始化模型...
✅ 模型初始化成功

[4/5] 检查特征加载...
✅ Base特征已加载: torch.Size([259205, 256])
✅ 多视图特征已加载: 4 views, shape=torch.Size([259205, 64])

[5/5] 检查网络维度...
✅ 投影层维度正确: [1280 → 512]
✅ 融合网络维度正确: 768
✅ Cross Network参数量: 1.18M

======================================================================
✅ 验证通过！改进已生效
======================================================================

架构总结:
  - Base特征: 启用
  - 多视图数量: 4
  - 投影维度: 1280 → 512
  - 融合维度: 768
  - 与双路模型对齐: 是
```

**关键确认项：**
- ✅ `[1280 → 512]` - 正确（有base时）
- ✅ `融合维度: 768` - 正确
- ✅ `与双路模型对齐: 是` - 正确

---

## 🎯 训练模型

### 直接运行

```bash
bash two_phase_run_multiview_split.sh
```

### 观察日志

**初始化阶段应该看到：**
```
[INFO] SASRecAlignMultiView initialized: 4 views, SENet ratio=4, per-view alignment enabled
[INFO] Multi-view projection: [1280 → 512] | Base features: enabled | Fusion input dim: 768
```

**训练阶段应该看到：**
```
[INFO] SASRecAlignMultiView: per-view alignment enabled | 
       total_align_loss=0.XXXX | 
       weights=[w0=0.25, w1=0.28, w2=0.23, w3=0.24] | 
       losses=[L0=0.XXX, L1=0.XXX, L2=0.XXX, L3=0.XXX]
```

---

## 🐛 故障排查

### 如果验证失败

#### 问题1：Base特征未加载

**错误信息：**
```
⚠️  Base特征未加载（可能未配置路径）
```

**检查：**
```bash
# 1. 检查配置文件路径
grep item_text_emb_path_base sasrec_align_multi_view.yaml

# 2. 检查文件是否存在
ls -lh /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb.base.npy

# 3. 如果不存在，生成
bash tools/gen_tfidf_only_fast.sh
```

---

#### 问题2：投影维度错误

**错误信息：**
```
❌ 投影层维度错误: [1024 → 256]
   期望: [1280 → 512]
```

**原因：** Base特征未加载，模型降级到仅多视图模式

**解决：** 确保base特征文件存在并正确配置

---

#### 问题3：融合维度错误

**错误信息：**
```
❌ 融合网络维度错误: 512
   期望: 768
```

**原因：** 融合网络未正确初始化

**解决：** 
- 检查代码修改是否正确应用
- 重新加载模块或重启Python

---

### 如果训练出错

#### 错误1：维度不匹配

**错误信息：**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**调试：**
```python
# 在模型代码中添加调试输出
print(f"DEBUG: text_concat shape = {text_concat.shape}")
print(f"DEBUG: text_proj shape = {text_proj.shape}")
print(f"DEBUG: fusion_input shape = {fusion_input.shape}")
```

**期望：**
- `text_concat: [B, 1280]` (有base) 或 `[B, 1024]` (无base)
- `text_proj: [B, 512]`
- `fusion_input: [B, 768]`

---

#### 错误2：OOM

**错误信息：**
```
CUDA out of memory
```

**解决：**
```yaml
# 减小batch_size
train_batch_size: 256  # 从512

# 启用chunking
fusion_chunk_size: 32768
```

---

## 📊 性能验证

### 对比实验

建议运行以下对比：

```bash
# 实验1: 双路模型（baseline）
bash two_phase_run_tfidf_llm.sh
# 记录: NDCG@10, Recall@10, MRR@10

# 实验2: 改进后的多视图模型
bash two_phase_run_multiview_split.sh
# 记录: NDCG@10, Recall@10, MRR@10

# 对比：多视图应该 ≥ 双路 + 1-3%
```

---

### 分析可学习参数

**训练后查看：**

```python
# 视图融合权重
view_weights = model.text_view_gate_params
print("View fusion weights:", torch.sigmoid(view_weights) / torch.sigmoid(view_weights).sum())

# 视图对齐权重
align_weights = model.text_view_align_weights
print("View alignment weights:", F.softmax(align_weights, dim=0))
```

**示例输出：**
```
View fusion weights: [0.22, 0.28, 0.25, 0.25]
  → View_1 (Function) 最重要

View alignment weights: [0.30, 0.25, 0.20, 0.25]
  → View_0 (Identity) 对齐最重要
```

---

## 🎓 学习要点

### 设计原则

1. **特征互补性**
   - 统计特征（TF-IDF）+ 语义特征（Qwen3）
   - 不同视角互补（Identity, Function, Audience, Category）

2. **信息容量匹配**
   - 文本特征维度应与ID特征维度平衡
   - 融合网络容量应充足（768维 vs 512维）

3. **渐进式压缩**
   - 避免激进压缩（1024→256太激进）
   - 渐进压缩（1280→512→256）更好

4. **向后兼容**
   - 动态适配配置
   - 支持多种特征组合

---

## ✅ 最终检查清单

### 代码层面
- [x] 模型代码已修改
- [x] 配置文件已更新
- [x] 路径已修正
- [x] 无lint错误

### 特征层面
- [ ] Base特征已生成
- [ ] 多视图特征已生成
- [ ] 特征验证通过（可选）

### 验证层面
- [ ] 运行 `verify_multiview_improvement.py`
- [ ] 确认投影维度为1280→512
- [ ] 确认融合维度为768
- [ ] 确认Base特征已启用

### 训练层面
- [ ] 删除旧checkpoint
- [ ] 运行训练脚本
- [ ] 观察日志确认维度
- [ ] 记录性能指标

---

## 📞 获取帮助

### 查看文档
- 快速理解：`MULTIVIEW_IMPROVEMENT_README.md`
- 详细架构：`MULTIVIEW_IMPROVED_ARCHITECTURE.md`
- 可视化对比：`ARCHITECTURE_COMPARISON_VISUAL.md`

### 运行验证
```bash
python verify_multiview_improvement.py
```

### 检查日志
训练时观察是否有：
```
Multi-view projection: [1280 → 512] | Base features: enabled | Fusion input dim: 768
```

---

**更新日期**: 2025-12-03  
**状态**: ✅ 改进完成，ready to use  
**下一步**: 运行验证脚本 → 训练模型 → 对比性能

