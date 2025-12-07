# 架构公平性 - 快速参考

## 🎯 核心问题

**当前 Multi-View Split 和 TF-IDF+LLM 对比不公平！**

## 📊 关键差异（一图看清）

```
┌─────────────────────────────────────────────────────────────────┐
│                        TF-IDF + LLM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Base[256] + LLM[256] = 512                                     │
│         ↓                                                        │
│  Text Processing (512 → 256)                                    │
│         ↓                                                        │
│  ID[256] + Text[256] = [512] ← 融合维度                          │
│         ↓                                                        │
│  Cross(512) + Deep(512→256) + Pred(1024→256)                    │
│         ↓                                                        │
│  Output[256]                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                      Multi-View Split                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Base[256] + 4×View[64] = 512                                   │
│         ↓                                                        │
│  ⭐ Per-View SENet (4× enhancement)                              │
│         ↓                                                        │
│  ⭐ Per-View Gates (learnable weights)                           │
│         ↓                                                        │
│  Concat: Base[256] + Weighted_Views[256] = 512                  │
│         ↓                                                        │
│  ⭐ Projection (512 → 512)  ← 扩展了！                           │
│         ↓                                                        │
│  ID[256] + Text[512] = [768] ← 融合维度 ⚠️ 比 TF-IDF+LLM 大 50%! │
│         ↓                                                        │
│  Cross(768) + Deep(768→256) + Pred(1024→256)  ← 更大的网络       │
│         ↓                                                        │
│  Output[256]                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## ⚠️ 不公平点汇总

| # | 不公平之处 | TF-IDF+LLM | Multi-View | 影响程度 |
|---|-----------|-----------|-----------|---------|
| 1 | **融合维度** | 512 | **768** (+50%) | 🔴 **高** |
| 2 | SENet 增强 | ❌ | ✅ 4个 | 🟡 中 |
| 3 | Per-View Gates | ❌ | ✅ 可学习 | 🟡 中 |
| 4 | Cross Network 大小 | 512维 | 768维 | 🔴 **高** |
| 5 | 特征来源数量 | 2个 | 5个 | 🟡 中 |

## 💡 快速修复方案

### 推荐：统一融合维度到 512

修改文件：`recbole/model/sequential_recommender/sasrecalignmultiview.py`

```python
# 找到 Line 102 左右:
# 将
self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.hidden_size * 2)

# 改为
self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.hidden_size)

# 找到 Line 111 左右:
# 将
fusion_input_dim = self.hidden_size + self.hidden_size * 2  # 256 + 512 = 768

# 改为
fusion_input_dim = self.hidden_size + self.hidden_size  # 256 + 256 = 512
```

**效果**: Multi-View 的融合维度从 768 降为 512，与 TF-IDF+LLM 对齐。

## 📈 预期性能变化

### 当前（不公平）

```
TF-IDF+LLM:      Recall@10 = 0.0285  (假设)
Multi-View:      Recall@10 = 0.0315  (假设)
提升:            +3.0% (绝对值: +0.003)
```

### 公平配置后

```
TF-IDF+LLM:      Recall@10 = 0.0285  (不变)
Multi-View:      Recall@10 = 0.0300  (预期下降)
提升:            +1.5% (绝对值: +0.0015)
```

**解读**:
- 如果公平后还有 +1.5%，说明多视图方法本身有价值
- 如果公平后没有提升，说明之前的提升主要来自更大的模型

## 🔬 验证步骤

```bash
# 1. 当前配置（记录baseline）
bash two_phase_run_tfidf_llm.sh
# 结果: Recall@10 = _____ (A)

bash two_phase_run_multiview_split.sh
# 结果: Recall@10 = _____ (B)
# 不公平优势 = (B - A) / A

# 2. 修改代码（统一融合维度）
vim recbole/model/sequential_recommender/sasrecalignmultiview.py
# 应用上述修改

# 3. 重新运行 Multi-View
bash two_phase_run_multiview_split.sh
# 结果: Recall@10 = _____ (C)
# 公平优势 = (C - A) / A

# 4. 分析
# 架构优势 = (B - C) / A  (来自更大融合维度)
# 方法优势 = (C - A) / A  (来自多视图本身)
```

## 📋 对比检查清单

在发表结果前，请确认：

- [ ] **明确说明**两个模型的融合维度不同
- [ ] **报告**公平配置和不公平配置的结果
- [ ] **解释**性能差异来源（架构 vs 特征）
- [ ] **提供**消融实验（SENet、Gates 的贡献）
- [ ] **讨论**是否应该统一架构

## 🎓 科学性建议

### 论文中应该写

**❌ 错误**:
> "Multi-view方法比TF-IDF+LLM提升3%"

**✅ 正确**:
> "Multi-view方法在当前配置下提升3%。注意到Multi-view模型使用768维融合（vs TF-IDF+LLM的512维），控制融合维度后，提升降至1.5%，表明真实方法优势约为1.5%，另外1.5%来自更大的模型容量。"

### Ablation Study 应包含

| 实验 | 融合维度 | SENet | Gates | Recall@10 | 分析 |
|------|---------|-------|-------|-----------|------|
| TF-IDF+LLM | 512 | ❌ | ❌ | 0.0285 | Baseline |
| Multi-View (Full) | 768 | ✅ | ✅ | 0.0315 | +3.0% (不公平) |
| Multi-View (Fair) | 512 | ✅ | ✅ | 0.0300 | +1.5% (公平) |
| Multi-View (No SENet) | 512 | ❌ | ✅ | 0.0295 | SENet贡献0.5% |
| Multi-View (No Gates) | 512 | ✅ | ❌ | 0.0297 | Gates贡献0.3% |
| Multi-View (Minimal) | 512 | ❌ | ❌ | 0.0290 | 纯多视图贡献0.5% |

## 🚨 重要提醒

**当前配置下的对比结果不应该直接发表！**

原因：
1. 融合维度不对等（768 vs 512）
2. 额外模块（SENet、Gates）未消融
3. 无法区分性能来自"多视图方法"还是"更大模型"

**建议**：
1. 先统一融合维度
2. 做完整消融实验
3. 分析各组件贡献
4. 在论文中明确说明所有差异

---

**结论**: 修复公平性问题后，如果多视图方法仍有提升（预期1-2%），就是真正的方法创新价值。

**下一步**: 应用快速修复方案，重新运行实验！

