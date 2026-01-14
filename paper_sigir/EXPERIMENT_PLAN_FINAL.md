# 论文实验完整规划 - 最终版 (2026-01-14)

## 🎯 论文完成度总览

| 部分 | 进度 | 状态 | 说明 |
|------|------|------|------|
| **Main Results** (Table 3) | 100% | ✅ 完成 | Beauty 14B, Toys 7B, 层级满足 |
| **Ablation** (Table 4, 5) | 100% | ✅ 完成 | SENet, Cross, Whiten 消融 |
| **Sensitivity** (Table 10) | 75% | 🔄 运行中 | λ, τ 完成; cold, infer 运行中 |
| **Uni100 Sanity** (App Table) | 0% | ⏳ 待开始 | 3个实验脚本已创建 |
| **Scale Law** (App Table) | 33% | ⏸️ 可选 | 7B Standard 已有 |

---

## 🔄 当前运行中的实验 (3个)

| GPU | 实验 | 配置 | 用途 | 预计完成 |
|-----|------|------|------|----------|
| **5090-0** | cold_25 | cold=2.5 baseline | 敏感性分析 | ~30分钟 |
| **4090-1** | infer_05 | infer=0.5 | 敏感性分析 | ~2小时 |
| **4090-2** | cold_30 | cold=3.0 | 敏感性分析 | ~2小时 |

---

## ⏳ 待执行的关键实验

### 第一优先级: Uni100 对比实验 (Appendix 必需)

**论文位置**: Appendix §A.1 - Sampled Evaluation Sanity Check

**目的**: 
- 验证 uni100 采样评估与 full-ranking 的一致性
- 分析采样评估对模型排序的影响
- 计算 Pearson/Spearman 相关系数

**实验** (等 GPU 3-5 空闲后):

```bash
# 4090-3: Multi-view 7B (Toys, Uni100)
GPU_ID=3 nohup bash experiments/exp_uni100_toys_7b.sh > uni100_toys_7b.log 2>&1 &

# 4090-4: TF-IDF+LLM (Toys, Uni100)
GPU_ID=4 nohup bash experiments/exp_uni100_tfidf_llm_toys.sh > uni100_tfidf_llm_toys.log 2>&1 &

# 4090-5: TF-IDF (Toys, Uni100)
GPU_ID=5 nohup bash experiments/exp_uni100_tfidf_toys.sh > uni100_tfidf_toys.log 2>&1 &
```

**预计时间**: 各 2-3 小时（完整二阶段训练）

**对比数据** (Full-ranking 基线):

| Model | Full HR@10 | Full MRR@10 | Full HR_new@10 |
|-------|-----------|-------------|----------------|
| Multi-view 7B | 6.92 | 3.76 | 1.99 |
| TF-IDF+LLM | 6.66 | 3.71 | 1.84 |
| TF-IDF | 6.60 | 3.73 | 1.83 |

**分析内容**:
1. Overall correlation (Spearman/Pearson)
2. Per-stratum correlation (new/few/frequent)
3. Model mis-ranking cases (是否改变模型排序)

---

### 第二优先级: 完整 Sensitivity Analysis (可选)

如果需要更完整的敏感性分析，可补充：

```bash
# infer_boost = 2.0 (验证上限)
GPU_ID=6 nohup bash experiments/exp_sensitivity_infer_20.sh > sensitivity_infer_20.log 2>&1 &
```

---

## 🔬 Whiten 方案真实情况 (重要发现)

### 从特征生成脚本分析：

**gen_text_emb_beauty_qwen2.5_7b.sh** 使用统一的 `--center --whiten` 参数：

1. **TF-IDF**: 
   ```bash
   --center --whiten
   → Center + ZCA Whiten + L2
   ```

2. **Single-view LLM**:
   ```bash
   --center --whiten
   → SVD → Center + ZCA Whiten + L2
   ```

3. **Multi-view (per-view)**:
   ```bash
   --center --whiten (每个视图独立)
   → SVD → Center + ZCA Whiten + L2 (4个独立 whiten matrix)
   ```

### 关键差异：

**不是 "有/无 ZCA"，而是 "Shared vs Per-view ZCA"：**

- **Single-view**: 所有特征用同一个 whiten matrix
- **Multi-view**: 
  - TF-IDF base: 与 single-view 共享 whiten matrix
  - 4 Views: 各自独立的 whiten matrix

### 架构图已修正：

- ✅ Single-view: `ZCA (shared)`
- ✅ Multi-view TF-IDF: `ZCA (shared)`
- ✅ Multi-view 4 Views: `ZCA (per-view)`

---

## 📊 实验完成时间线

### Week 1 (当前)
- ✅ 主表数据 (Table 3)
- ✅ 消融实验 (Table 4, 5)
- 🔄 敏感性分析 (Table 10) - 75% 完成

### Week 2 (预计)
- ⏳ 敏感性分析完成 (infer_05, cold_30)
- ⏳ Uni100 对比实验 (3个)
- ⏳ 论文格式最终检查

### 投稿时间
- **预计**: 2周内完成所有实验
- **论文状态**: 7页，主要内容完整，附录待补充

---

## 🚀 一键启动命令 (分批执行)

### 批次1: 当前运行中
```bash
# 已启动，等待完成
GPU_ID=0: cold_25
GPU_ID=1: infer_05
GPU_ID=2: cold_30
```

### 批次2: Uni100 对比 (等批次1完成后)
```bash
GPU_ID=3 nohup bash experiments/exp_uni100_toys_7b.sh > uni100_toys_7b.log 2>&1 &
GPU_ID=4 nohup bash experiments/exp_uni100_tfidf_llm_toys.sh > uni100_tfidf_llm_toys.log 2>&1 &
GPU_ID=5 nohup bash experiments/exp_uni100_tfidf_toys.sh > uni100_tfidf_toys.log 2>&1 &
```

---

## 📁 已创建的脚本

- ✅ `experiments/exp_sensitivity_cold_30.sh`
- ✅ `experiments/exp_uni100_toys_7b.sh`
- ✅ `experiments/exp_uni100_tfidf_llm_toys.sh`
- ✅ `experiments/exp_uni100_tfidf_toys.sh`

**总计**: 4 个新脚本，所有实验准备就绪！

---

## 🎯 最终论文检查清单

- [x] 主表数据（层级满足）
- [x] 消融实验
- [x] 提示词模板
- [x] 架构图（完整数据流）
- [x] Whiten 方案说明
- [ ] 敏感性分析（等 infer_05, cold_30）
- [ ] Uni100 对比（待执行）
- [ ] 格式检查（Overfull hbox）
- [ ] 参考文献补全

**预计投稿**: 完成 Uni100 实验后即可！
