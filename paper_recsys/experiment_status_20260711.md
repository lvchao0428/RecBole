# 实验进展状态报告 (2026-07-11 20:53 更新)

## 一、当前三机运行状态

### 5090 (RTX 5090, 32GB)
- **Beauty V2 Pilot 全部完成** ✅ (20:37)
- **GPU 空闲，等待下一阶段任务**

### log10 (GTX 1080Ti, 11GB)
- **当前任务**: ID-only Toys seed=42 运行中 (~2h left)
- **已完成**: Beauty s2024✅, s42✅; Toys s2024✅, s2025✅
- **剩余**: Toys s42(运行中) → Grocery s2025, s2024, s42
- **⚠️ 注意**: 当前跑的旧脚本没有 `--min_train_interactions 5`

### 本机 Mac (M3 Max, MPS)
- **状态**: 环境已验证(MPS可用)，待命中

---

## 二、Beauty V2 Pilot 最终结果 (seed=2025, TS split, min5 filter, no boost, no SE)

> 全部使用 TS-aware 特征 + min_train_interactions=5 过滤
> 确认: cold_text_boost=0.0, infer_boost=0.0, cold_threshold=0

### Phase-B 最终结果（全模型训练完成后）

| 配置 | valid MRR@10 | valid NDCG@10 | valid R@10 | test MRR@10 | test NDCG@10 | test R@10 | 耗时 |
|------|-------------|--------------|-----------|------------|-------------|----------|------|
| **TF-IDF** | 0.0281 | 0.0342 | 0.0545 | 0.0157 | 0.0197 | 0.0326 | 78min |
| **TF-IDF+LLM** | 0.0284 | 0.0345 | 0.0549 | 0.0157 | 0.0195 | 0.0320 | 101min |
| **MV-Align** | 0.0282 | 0.0345 | 0.0552 | 0.0157 | 0.0198 | 0.0333 | 204min |

### 分层 Recall@10 (test, Phase-B)

| 配置 | R_new@10 | R_few@10 | R_freq@10 | NDCG_new@10 | NDCG_few@10 | NDCG_freq@10 |
|------|---------|---------|----------|-----------|-----------|------------|
| **TF-IDF** | 0.0183 | 0.0345 | 0.0719 | 0.0128 | 0.0214 | 0.0421 |
| **TF-IDF+LLM** | 0.0198 | 0.0308 | 0.0717 | 0.0133 | 0.0203 | 0.0422 |
| **MV-Align** | 0.0224 | 0.0324 | 0.0735 | 0.0138 | 0.0223 | 0.0415 |

### log10 上的 ID-only 结果 (⚠️ 无 min5 过滤，评测用户群不同)

| 数据集 | seed | valid MRR@10 | valid NDCG@10 | test MRR@10 | test NDCG@10 | test R@10 | 耗时 |
|--------|------|-------------|--------------|------------|-------------|----------|------|
| Beauty | 2024 | 0.0224 | 0.0284 | 0.0155 | 0.0198 | 0.0338 | 117min |
| Beauty | 42 | 0.0221 | 0.0281 | 0.0156 | 0.0198 | 0.0335 | 117min |
| Toys | 2024 | 0.0253 | 0.0318 | 0.0187 | 0.0227 | 0.0355 | 147min |
| Toys | 2025 | 0.0253 | 0.0318 | 0.0182 | 0.0221 | 0.0349 | 147min |

---

## 三、关键分析

### 1. 三个 text 模型分数高度接近
Phase-B 之后，TF-IDF / TF-IDF+LLM / MV-Align 在 test MRR@10 上几乎一致 (0.0157):
- **TF-IDF 和 TF-IDF+LLM 层级无区分**：test MRR@10 = 0.0157 vs 0.0157
- 但 **MV-Align 在 new/few 分层上有微弱优势**（R_new=0.0224 vs 0.0198/0.0183）

### 2. ID-only 评测不一致问题
log10 的 ID-only（无 min5）test MRR@10 ≈ 0.0155-0.0156
text 模型（有 min5）test MRR@10 ≈ 0.0157
**数值碰巧相近，但评测用户群不同，不可直接对比**

### 3. TF-IDF+LLM ≈ TF-IDF 的原因分析
Qwen single-view 特征 (`item_text_emb.qwen2.5_7b.base.ts.npy`) 经过 TS-aware center+whiten 后:
- 与原始版本 mean abs diff = 1.065（确实做了变换）
- 但 Phase-B 解冻 backbone 后，ID embedding 主导了预测能力
- LLM 特征的增益被 backbone 的学习能力吸收了

---

## 四、耗时记录（5090 Beauty V2 Pilot, 实测）

| 配置 | 耗时(秒) | 耗时(分钟) |
|------|---------|-----------|
| TF-IDF (SASRecAlignV3) A+B | 4702 | **78 min** |
| TF-IDF+LLM (SASRecAlignV3) A+B | 6048 | **101 min** |
| MV-Align (SASRecAlignMultiViewV3) A+B | 12222 | **204 min** |

---

## 五、Bug 修复记录 (2026-07-11)

1. **`--config_dict` 解析 bug**: `ast.literal_eval` 不识别 JSON `false/true/null`。已修复并三机同步。
2. **MPS 设备支持**: `configurator.py` 新增 MPS fallback。
3. **代码三机同步**: Mac→5090→log10 全量 rsync 完成，MD5 校验通过。

---

## 六、5090 夜间流水线 (21:11 启动, PID=1344387)

`run_5090_overnight_20260711.sh` 按序执行：

| # | 任务 | 预估耗时 | 预计完成 |
|---|------|---------|---------|
| 1 | **ID-only min5** (Beauty, seed=2025) | ~26min | ~21:40 |
| 2 | **TF-IDF + freq-weight boost** (Beauty, cold_boost=3.0) | ~78min | ~23:00 |
| 3 | **TF-IDF+LLM + freq-weight boost** (Beauty, cold_boost=3.0) | ~101min | ~00:40 |
| 4 | **MV-Align + freq-weight boost** (Beauty, cold_boost=3.0) | ~204min | ~04:05 |
| 5 | **Toys/Grocery Qwen embedding 生成** (TS-aware) | ~10min | ~04:15 |
| 6 | **Beauty 3-seed TF-IDF no-boost** (seed=2024,42) | ~156min | ~06:50 |

Boost 参数: `cold_text_boost=3.0, infer_boost=0.0, cold_threshold=10`

### log10 当前状态
- Toys seed=42 运行中 (~21:30 完成)
- 后续: Grocery s2025, s2024, s42 (~4.5h → ~02:00 完成)
- ⚠️ 旧脚本（无 min5），后续可用模型做 min5 重评测

## 七、明天早上待做

### 到手结果 (预计 07:00)
- ID-only min5 baseline → 与 text 模型直接对比
- 3 组 freq-weight boost 实验 → 判断 boost 是否恢复层级
- Toys/Grocery Qwen TS-aware embedding → ready for 3-seed
- Beauty TF-IDF 3-seed 中的另 2 seeds

### 明早 P0
1. 收集夜间结果，分析 boost vs no-boost 层级变化
2. 启动 Beauty 3-seed 剩余配置 (TF-IDF+LLM, MV-Align, seed=2024,42)
3. 启动 Toys/Grocery 第一轮实验

### 明早 P1
4. log10 部署 V2 脚本（min5），排 Grocery ID-only
5. UniSRec portability / ablation 实验准备
