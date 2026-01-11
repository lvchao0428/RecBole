# Changelog 2025-01-11

## 核心改动：CHANGE-9 推理时冷启动文本权重增强

### 背景问题

分析 `0111.csv` 数据发现 **exp_burn0** (无 burn-in) 配置存在矛盾现象：

| 指标 | exp_burn0 vs baseline | 问题 |
|------|----------------------|------|
| MRR_new@10 | +39.24% ✅ | 排序精度高 |
| MRR_frequent@10 | +61.40% ✅ | 高频商品排序好 |
| **HR_new@10** | **-2.34%** ❌ | 新品曝光机会减少 |
| HR_few@10 | -1.20% ❌ | 少量商品曝光也下降 |

**根因分析**：
- 训练侧的 `cold_start_align_boost` 只影响对齐损失，帮助模型学习新品表示
- 但推理侧的 `effective_text_weight` 对所有商品一视同仁
- 导致模型"有能力推新品但不敢推" —— MRR 高但 HR 低

### 解决方案：CHANGE-9

在推理时给低频商品更强的文本信号权重：

```python
# sasrecalignmultiviewv2.py - _fuse_with_cross_network()

# [CHANGE-9] 推理时冷启动文本权重增强
if self.inference_cold_text_boost > 0 and item_ids is not None:
    item_pop = self.item_popularity[item_ids].float()
    threshold = float(self.cold_start_align_threshold)
    cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold
    cold_boost = 1.0 + self.inference_cold_text_boost * cold_factor
    effective_text_weight = effective_text_weight * cold_boost.unsqueeze(1)
```

**公式**：
```
effective_text_weight *= 1.0 + inference_cold_text_boost × max(0, threshold - pop) / threshold

示例 (inference_cold_text_boost=1.0, threshold=10):
- pop=0 (新品):   factor = 2.0 (文本权重翻倍)
- pop=5 (少量):   factor = 1.5 (文本权重+50%)
- pop=10+ (高频): factor = 1.0 (无变化)
```

---

## 新增文件

### 1. 模型代码
- **`recbole/model/sequential_recommender/sasrecalignmultiviewv2.py`**
  - 新增配置项：`inference_cold_text_boost` (默认 0.0)
  - 修改方法：`_fuse_with_cross_network()`
  - 新增日志：记录 inference cold text boost 状态

### 2. 配置文件
- **`sasrec_align_multi_view_v2_inference_boost.yaml`**
  ```yaml
  # 训练侧 (温和)
  cold_start_align_boost: 2.0
  cold_start_align_threshold: 10
  
  # 推理侧 (新增)
  inference_cold_text_boost: 1.0
  
  # 其他关键参数
  text_weight: 1.0
  alignment_weight: 0.10
  temperature: 0.05
  ```

### 3. 实验脚本
- **`experiments/exp_inference_boost.sh`**
  - 测试 CHANGE-9 效果
  - 使用 Beauty 数据集 + Multi-View 7B
  - 对比基线：exp_burn0 (无 inference boost)

---

## 实验设计

### 当前运行中的实验 (baseline)

| GPU | 实验 | 配置 | 用途 |
|-----|------|------|------|
| 0 | tfidf_beauty | align=0.10, burn=0 | TF-IDF 基线 |
| 0 | tfidf_toys | align=0.10, burn=0 | TF-IDF 基线 |
| 0 | tfidf_llm_beauty | align=0.10, burn=0 | TF-IDF+LLM 基线 |
| 1 | mv_7b_beauty | cold=0, burn=0 | Multi-View 基线 |
| 2 | mv_14b_beauty | cold=0, burn=0 | Scale Law |
| 3 | mv_32b_beauty | cold=0, burn=0 | Scale Law |
| 4 | tfidf_llm_toys | align=0.10, burn=0 | Toys 基线 |
| 5 | mv_7b_toys | cold=0, burn=0 | Toys Multi-View |
| 6 | mv_14b_toys | cold=0, burn=0 | Toys Scale Law |
| 7 | mv_32b_toys | cold=0, burn=0 | Toys Scale Law |

### 新增实验

| GPU | 实验 | 配置 | 用途 |
|-----|------|------|------|
| ? | **exp_inference_boost** | cold=2.0, **infer_boost=1.0** | 验证 CHANGE-9 |

---

## 预期效果

| 指标 | mv_7b (cold=0) | inference_boost (预期) |
|------|----------------|----------------------|
| HR_new@10 | -2.34% | **+10~15%** |
| HR_few@10 | -1.20% | **+5~10%** |
| HR_freq@10 | +34.86% | +28~32% (略降) |
| MRR_new@10 | +39.24% | +30~40% (保持) |
| MRR_freq@10 | +61.40% | +55~60% (保持) |

**目标**：所有指标相对 baseline 都有提升，不牺牲任何一类商品。

---

## 运行命令

```bash
# 等任意 GPU 空闲后执行
GPU_ID=0 nohup bash experiments/exp_inference_boost.sh > inference_boost.log 2>&1 &
```

---

## 后续计划

1. **验证 CHANGE-9 效果**
   - 对比 exp_inference_boost vs mv_7b_beauty
   - 如果 HR_new 提升且 MRR_freq 不降，则方案有效

2. **参数调优**
   - `inference_cold_text_boost`: 0.5, 1.0, 1.5, 2.0
   - `cold_start_align_boost`: 1.5, 2.0, 2.5

3. **推广到其他模型**
   - 如果有效，更新 14B/32B 和 Toys 配置
   - 统一使用 CHANGE-9 配置

---

## 相关文件引用

- 模型代码：`recbole/model/sequential_recommender/sasrecalignmultiviewv2.py`
- 实验脚本：`experiments/exp_inference_boost.sh`
- 配置文件：`sasrec_align_multi_view_v2_inference_boost.yaml`
- 数据分析：`paper_sigir/0111.csv`
- 前一版 changelog：`paper_sigir/EXPERIMENT_CHANGELOG_0110.md`
