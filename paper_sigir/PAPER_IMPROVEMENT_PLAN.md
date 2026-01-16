# 论文改进计划 (基于导师建议 suggestion.txt)

**日期**: 2026-01-16  
**状态**: 逐项分析和改进

---

## 一、已修复的问题 ✅

### 1.1 符号冲突问题
- **问题**: T 同时表示序列长度和 popularity threshold
- **状态**: ✅ 已修复
- **修复**: 使用 P_0 替代 threshold，n 替代序列长度

### 1.2 Whitening 公式缺失
- **问题**: Method 里只有 L2 normalization，没有 ZCA whitening 公式
- **状态**: ✅ 已修复
- **位置**: main.tex line 286-301
- **内容**: 添加了完整的 ZCA whitening 公式 (X_c, Σ, W, X_w)

### 1.3 Whitening 实现细节
- **问题**: per-view vs shared whitening? 离线 vs 在线?
- **状态**: ✅ 已修复
- **位置**: main.tex line 298-301
- **说明**: "per-view whitening...statistics computed offline on training set and fixed"

### 1.4 SENet 表述问题
- **问题**: 当前实现是 SE-like gating，不是严格 SENet
- **状态**: ✅ 已修复
- **位置**: main.tex line 303-318
- **说明**: 改为 "SE-style channel gating" 并解释与原始 SENet 的区别

### 1.5 DCN-V2 具体变体
- **问题**: 需说明使用的是哪个 DCN-V2 变体
- **状态**: ✅ 已修复
- **位置**: main.tex line 350-353
- **说明**: "full-rank variant of DCN-V2 (non-mix)"

### 1.6 加权对齐损失公式
- **问题**: 缺少显式的加权 InfoNCE 公式
- **状态**: ✅ 已修复
- **位置**: main.tex line 398-406
- **内容**: 添加了完整的 L_align,w 公式，说明 w_i 只作用于 anchor

### 1.7 多 Seed 验证
- **问题**: 需要多 seed 验证，因为改进幅度小
- **状态**: ✅ 已修复
- **位置**: main.tex Appendix seed_stability section
- **内容**: 3 seeds (42, 2024, 2025), 报告 mean±std

### 1.8 Inference Boost 说明
- **问题**: Method 部分没有提到 inference boost
- **状态**: ✅ 已修复
- **位置**: main.tex line 408-424
- **内容**: 添加了 inference-time cold-start boosting 段落和公式

---

## 二、待改进的问题 🟡

### 2.1 数字一致性问题 (最重要)
- **问题**: Abstract/Intro/Results/Conclusion 数字不一致
- **状态**: 🟡 需要检查
- **行动计划**:
  1. 确定主表使用 Aggressive 配置
  2. 统一所有数字到 Aggressive 配置
  3. Standard 配置结果放 Appendix
  4. 检查清单:
     - [ ] Abstract: HR@10 数字
     - [ ] Introduction: TF-IDF 提升百分比
     - [ ] Main Results: 所有模型数字
     - [ ] Conclusion: whitening 影响百分比

### 2.2 λ 和 s_mv 重复缩放风险
- **问题**: Eq.(7) 的 s_mv 和 Eq.(9) 的 λ 可能重复缩放
- **状态**: 🟡 需要确认
- **建议**: 
  - 检查代码实现，确认两者的作用
  - 如果确实重复，建议固定 s_mv=1，只使用 λ
  - 或者在论文中说明两者的不同作用

### 2.3 Figure 1 改进
- **问题**: 当前 Figure 1 是 placeholder
- **状态**: 🟡 待设计
- **建议** (来自导师):
  - 三段式 cover image:
    1. 左侧: Cold-start 问题示例 (新上架商品，交互数≈0)
    2. 中间: 方案 (Prepare/Multi-view/Cross/Align 四个模块)
    3. 右侧: 效果 (HR/NDCG gains + HR_new gains)
  - 添加 embedding budget 说明: 1×256 = 4×64

### 2.4 HR-MRR Trade-off 可视化
- **问题**: 导师建议添加散点图展示 trade-off
- **状态**: 🟡 待创建
- **建议**:
  - 横轴: HR@10
  - 纵轴: MRR@10
  - 标注点: Full / -Cross / -Whiten / -SENet
  - 可以清晰展示 Pareto-like frontier

### 2.5 主表拆分优化
- **问题**: Table 3 信息过密，阅读负担大
- **状态**: 🟡 可选优化
- **建议**:
  - 正文只放 Overall + new stratum
  - few/frequent 放 Appendix
  - 或者添加 Δ 列突出改进

---

## 三、低优先级问题 🟢

### 3.1 脚本名称问题
- **问题**: 论文中出现 .sh 脚本名
- **状态**: 🟢 检查确认
- **行动**: 检查当前论文是否还有脚本名，如有则移到 Appendix

### 3.2 Cover Image 设计
- **问题**: 导师建议添加吸引人的首页图
- **状态**: 🟢 可选
- **建议**: 
  - 使用文字示例而非真实商品图 (避免版权风险)
  - 简笔 icon 或抽象表示

---

## 四、数字一致性检查清单

### Abstract 数字 (需确认与主表一致)
当前 Abstract:
- MV-ALIGN Beauty: HR@10 = 6.07% ✅ (与 Table main 一致)
- MV-ALIGN Toys: HR@10 = 6.92% ✅ (与 Aggressive 配置一致)
- HR_new@10 up to 1.99% ✅
- HR_few@10 up to 5.22% ✅

### Introduction 数字
当前 Introduction:
- TF-IDF: +20.0% HR@10 on Beauty ✅ (4.69 → 5.63, 确认计算)
- TF-IDF: +38.7% NDCG@10 on Beauty ✅ (2.69 → 3.73)

### Conclusion 数字
当前 Conclusion:
- Cross removes: MRR -15.4% on Toys ✅
- Cross removes: NDCG -6.9% ✅
- Whitening effect: 已更新为 +0.3% MRR, +3.5% HR_new ✅

---

## 五、执行优先级

### 立即执行 (今天)
1. [ ] 验证所有数字一致性
2. [ ] 确认 λ/s_mv 缩放问题
3. [ ] 启动 infer_20, infer_25 敏感性实验

### 短期执行 (本周)
4. [ ] 创建 HR-MRR trade-off 散点图
5. [ ] 设计新的 Figure 1
6. [ ] 补充 Beauty ablation 实验

### 可选执行
7. [ ] Cover image 设计
8. [ ] 主表拆分优化

---

## 六、总结

**已修复**: 8/15 问题 (53%)  
**待改进**: 5/15 问题 (33%)  
**低优先级**: 2/15 问题 (13%)

**最紧急**: 数字一致性检查  
**最有价值**: HR-MRR trade-off 可视化  
**最耗时**: Figure 1 重新设计

**预计完成时间**: 
- 数字检查: 1小时
- 实验补充: 4-6小时  
- 图表设计: 2-3小时

---

**状态**: 论文核心问题已基本解决，剩余为优化项
