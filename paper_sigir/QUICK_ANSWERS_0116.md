# 快速问答 (2026-01-16晚)

## ✅ 问题1: 敏感性分析是否需要所有数据集？

### 会议惯例

**✅ 单数据集sensitivity是常见做法**

**SIGIR/RecSys惯例**:
- 大多数论文只在**一个主要数据集**上做sensitivity
- 通常选择更challenging或更大的数据集
- Ablation和main results覆盖多数据集即可

### 当前论文状态

**已有的多数据集验证** ✅:
- Main table: Beauty + Toys
- Ablation: Beauty + Toys (补充中)
- Seed stability: Beauty + Toys
- Scale Law: Toys

**Sensitivity**: 仅Toys

### 建议

**✅ 保持仅Toys** (推荐)

**理由**:
1. Toys更challenging (更大catalog，更稀疏)
2. 已有足够的多数据集验证
3. 篇幅和实验资源有限

**可选增强**:
在sensitivity段落开头添加一句：
```latex
\paragraph{Sensitivity analysis.}
We conduct sensitivity analysis on Toys 7B (our more challenging 
dataset). Performance trends are consistent with Beauty as shown 
in main results and ablations.
```

**如果reviewer要求**: 可在rebuttal时补充Beauty sensitivity

---

## ✅ 问题2: Table 7格式覆盖问题

### 已修复 ✅

**问题**: `\end{table*}` 与 `\begin{table}[h]` 不匹配  
**修复**: 改为 `\end{table}`  
**状态**: ✅ 已修复并验证无linter错误

### 验证

两个strata表格现在都是：
```latex
\begin{table}[h]
...
\end{table}
```

环境匹配，格式应该正确。

**请编译PDF确认格式是否正常。**

---

## ❓ 问题3: Table 10 HR_new@10重复

### 检查结果

**Table 10 (Aggressive config, line 1080-1094)**:
```latex
\textbf{Model} & HR@10 & NDCG@10 & HR\_new@10 & HR\_few@10 \\
```

**列**: 5列，每个指标出现1次  
**结论**: ❌ **未发现重复**

### 可能的情况

1. **Table 9 (Scale Law)** 有 HR_new@10 + NDCG_new@10
   - 两个都是"new"相关，但不是重复
   - 一个是HR，一个是NDCG

2. **其他表格**: 未发现重复

**请您明确指出**:
- 具体哪个表格？
- 哪一行或列有重复？
- 或者提供截图/行号

---

## 📊 当前实验状态

### Running (8个)
```
批次2 (GPU 0-3): No-Whiten × 4 (RUNNING)
批次3 (GPU 4-7): SE-net/Cross × 4 (RUNNING)
```

### 待创建的实验

如需补充，我可以创建：

1. **Infer boost扩展** (2个)
   - infer=2.0, 2.5
   - 找到性能峰值

2. **Appendix承诺** (2个)
   - Cold-start reweighting ablation
   - Center-only normalization

3. **(可选) Beauty敏感性分析** (4个)
   - 如果需要双数据集验证

**是否需要我创建这些实验脚本？**

---

## 📝 待办事项

### 立即确认
- [ ] Table 10具体重复位置（需要用户指出）
- [ ] 是否需要Beauty敏感性分析
- [ ] 是否需要创建补充实验脚本

### 等待实验完成
- [ ] 提取no-whiten结果
- [ ] 提取SE-net/Cross结果
- [ ] 更新论文表格

### 可选补充
- [ ] 扩展infer boost
- [ ] Appendix承诺实验
- [ ] (可选) Beauty sensitivity

---

## ✅ 总结

**已修复**:
1. ✅ Table strata format (table* → table)
2. ✅ Method添加inference boost
3. ✅ Seed stability caption

**待确认**:
1. ❓ Table 10重复位置
2. ❓ 是否需要Beauty sensitivity

**建议**:
1. ✅ 敏感性分析保持仅Toys（符合惯例）
2. ✅ 编译PDF验证表格格式
3. ⏳ 等待当前8个实验完成

**需要我做什么**:
- 创建补充实验脚本？
- 修改特定表格？
- 其他？

