# 论文改进计划 (基于导师反馈 2026-01-15)

## 优先级 P0: 数据一致性 ✅ 已完成
- [x] 统一使用 Aggressive 配置作为主表数据
- [x] 更新 Abstract/Introduction/Main Results/Conclusion 所有数据点
- [x] 确保所有百分比计算一致

## 优先级 P1: 公式与符号修正

### ✅ 已完成
- [x] P1-1: 修复符号冲突 - 将 Eq.(8) 的 $T$ 改为 $P_0$
- [x] P1-2: 添加 ZCA Whitening 数学定义 (Method §4.2)
- [x] P1-3: 澄清 SENet 为 SE-style channel gating
- [x] P1-5: 添加加权对齐损失显式公式 Eq.(9)

### ✅ 已完成 (2026-01-15 continued)
- [x] **P1-4: 确认 DCN-V2 实现并添加说明**
  - ✅ 检查了 `sasrec_align.py` 中的 `DCNV2Cross` 实现
  - ✅ 确认: **DCN-V2 full-rank (non-mix) 版本**
  - ✅ 使用全秩矩阵 $W_l \in \mathbb{R}^{d \times d}$
  - ✅ 在论文中添加了详细说明和动机

- [x] **P1-6: 统一 $\lambda$ 和 $s_{mv}$ 为单一可调权重**
  - ✅ 移除了 Eq.(7) 中的 $s_{mv}$ 缩放
  - ✅ 只保留 $\lambda$ 作为唯一可调权重
  - ✅ 更新了 Method §4.4

## 优先级 P2: 写作结构调整

### 待完成
- [ ] **P2-1: 移除 Baselines 中的脚本名**
  - 位置: §5.2 Baselines
  - 改为: 只写模型配置 + 关键超参
  - 脚本名移到 Appendix E (Reproducibility)

- [ ] **P2-2: 拆分 Table 3**
  - 正文: 只保留 Overall + new stratum
  - Appendix: few/frequent strata 移到 Table "Extended Strata Results"
  - 或者: 添加 $\Delta$ 列突出改进

- [ ] **P2-3: 添加 HR-MRR trade-off 散点图**
  - 横轴: HR@10
  - 纵轴: MRR@10
  - 标注: with/without cross, with/without whitening
  - 位置: §7 Analysis 或 Figure 2

- [ ] **P2-4: 重新设计 Figure 1**
  - 聚焦: Normalize/Whiten → Cross → Align 三步
  - 移除 placeholder
  - 添加 embedding budget 说明 (单视角 256 vs 4×64)

- [ ] **P2-5: 设计 Cover Image (可选)**
  - 第一页吸引眼球的概览图
  - 展示核心贡献或关键发现

## 优先级 P3: 统计显著性

### 待完成
- [ ] **P3-1: 添加多 seed 实验**
  - Seeds: 2025 (当前), 42, 2024
  - 报告: mean ± std
  - 位置: Table 3 添加列或 Appendix F

- [ ] **P3-2: 统计显著性检验**
  - 方法: paired t-test 或 bootstrap
  - 针对: Toys single-view 6.61% vs multi-view 6.92%
  - 位置: Appendix F

## 优先级 P4: Appendix 整理

### 待完成
- [ ] **Appendix A: Sampled Evaluation (uni100)** ✅ 已有
- [ ] **Appendix B: Hyperparameter Configuration**
  - Standard vs Aggressive 配置对比
  - Scale Law 验证结果 (14B/32B)
  
- [ ] **Appendix C: Extended Strata Results**
  - few/frequent 详细结果
  - 从主表移过来

- [ ] **Appendix D: DCN-V2 Implementation Details**
  - 具体实现版本说明
  - 与标准 DCN-V2 的差异
  - 复杂度分析

- [ ] **Appendix E: Reproducibility**
  - 脚本名称和配置文件
  - 环境依赖
  - 完整超参数列表

- [ ] **Appendix F: Statistical Significance**
  - 多 seed 结果
  - 显著性检验

## DCN-V2 实现确认 (P1-4 详细)

### 需要检查的内容
1. **代码位置**: `recbole/model/sequential_recommender/sasrecalignmultiviewv2.py`
2. **关键问题**:
   - Line 243: `from recbole.model.sequential_recommender.sasrec_align import DCNV2Cross`
   - 实现是否是完整的 DCN-V2？
   - 是否有 low-rank/MoE/matrix decomposition？
   - 还是简化的 cross layer？

3. **论文中的描述** (当前 Line 303-305):
   ```latex
   \mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \mathbf{x}^{(0)} \odot (\mathbf{W}_l\mathbf{x}^{(l)} + \mathbf{b}_l)
   ```
   - 这是标准 DCN 的公式
   - DCN-V2 应该有更复杂的参数化

### 行动计划
1. 检查 `sasrec_align.py` 中 `DCNV2Cross` 的实现
2. 如果是简化版: 
   - 论文中改为 "DCN-style cross network"
   - 或说明 "we use a simplified variant"
3. 如果是完整 DCN-V2:
   - 在 Appendix D 添加详细说明
   - 说明使用的具体变体 (low-rank/full-rank)

## 写作改进要点

### Method 部分需要强化的描述
1. **Whitening 动机** (§4.2):
   - 明确写: decorrelate multi-view channels
   - 解决的问题: scale mismatch, view correlation, cross-view redundancy

2. **Cross Network 动机** (§4.3):
   - 为什么 self-attention 不够
   - self-attention: 序列内 item-item 依赖
   - DCN: ID embedding × text embedding 高阶 feature crosses

3. **Alignment 细节** (§4.4):
   - 对齐对象: text view → ID space
   - 正负样本: batch 内 InfoNCE
   - cold-start reweighting: 避免被高频物品主导

## 时间安排建议

### 第一阶段 (2h)
- [x] P1-1 to P1-3, P1-5 ✅ 已完成
- [ ] P1-4: DCN-V2 实现确认
- [ ] P1-6: 统一 $\lambda$ 和 $s_{mv}$

### 第二阶段 (3h)
- [ ] P2-1: 移除脚本名
- [ ] P2-2: 拆分 Table 3
- [ ] P2-3: 添加 trade-off 散点图描述

### 第三阶段 (按需)
- [ ] P2-4: 重新设计 Figure 1
- [ ] P3-1: 多 seed 实验 (需要跑实验)
- [ ] P4: Appendix 整理

## 导师反馈关键点摘要

1. **数字不一致是最严重问题** ✅ 已修复
2. **公式符号冲突** (T 用于序列长度和阈值) ✅ 已修复
3. **Whitening 缺数学定义** ✅ 已添加
4. **DCN vs DCN-V2 需要对齐** 🔄 进行中
5. **需要多 seed 实验验证显著性** (改进幅度小)
6. **HR-Ranking trade-off 需要可视化**
7. **不要在正文写脚本名**

## 下一步行动

1. ✅ 创建此 changelog
2. 🔄 检查 DCN-V2 实现
3. 根据实现完善论文描述
4. 继续 P1-6 和 P2 任务
