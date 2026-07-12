# 实验计划 — 2026-07-12 (基于老师 0712 指导)

> 核心转变：**WSDM 版改为 leakage-aware controlled study**
> - TF-IDF 是最重要的 strong baseline，不能删
> - 复杂文本模型在严格时间协议下边际收益收缩 → 这本身就是 finding
> - 不追求所有配置形成漂亮层级，真实、可复现、解释得通

---

## 一、老师最新指示摘要 (0712)

1. **V1 no-boost TF-IDF+LLM=0.0042 很像异常，必须排查**
   - 需用同一 seed 复跑四个核心配置，确认打平现象非实现问题
2. **简化模型**: SE 删掉(已完成)、training/inference cold boost 从主方法删掉、两阶段只作 optimization protocol
   - 如 Cross 没带来稳定净收益就不硬塞进最终模型
3. **item-frequency 分桶保留**，改名 low/mid/head，不再叫 cold-start
4. **机制分析**: 先定义假设再验证，不预设方向
   - Cross: per-user target rank transition, score entropy, Top-1~Top-10 margin, Top-K head/tail 比例
   - Align/Multi-view: frequency bucket target-rank improvement, leave-one-view-out, cosine redundancy
5. **UniSRec portability + 第三数据集**必须在新 eval protocol 下重跑

---

## 二、资源与耗时

| 机器 | GPU | 显存 | 单实验耗时参考 (Beauty) |
|------|-----|------|------------------------|
| 5090 | RTX 5090 | 32 GB | ID:24m, TF:78m, LLM:101m, MV:204m |
| log10 | 1080 Ti | 11 GB | ID: ~40m (估) |

> log10 已恢复 (192.168.0.107)，ID-only 全量实验已在 log10 启动。

---

## 三、实验队列 (5090, 串行)

### Phase 1 — 高优排查 + 简化确认 (~7h)

| # | 实验 | 预计耗时 | 目的 |
|---|------|---------|------|
| 1 | **ID-only** (Beauty, seed=2025, min5, no-boost) | 24 min | 已有，作为 baseline anchor |
| 2 | **TF-IDF** no-boost (Beauty, seed=2025, min5) | 78 min | 已有 ✅ V2 结果 |
| 3 | **TF-IDF+LLM** no-boost (Beauty, seed=2025, min5) | 101 min | ✅ V2 已有，确认 |
| 4 | **MV-Align** no-boost (Beauty, seed=2025, min5) | 204 min | ✅ V2 已有，确认 |
| — | *以上 4 个 V2 均已完成，结果一致 → Phase 1 可跳过直接进 Phase 2* | | |

> **V1 异常排查结论**: V1 no-boost TF-IDF+LLM 的 0.0042 是因为 `freeze_backbone` 解析 bug（`false` 未被正确解析为 `False`），Phase-A 冻结 backbone 导致严重欠训练。V2 已修复此 bug。四个 V2 配置 (no-boost, TS-aware, min5) 结果稳定：TF=LLM=MV≈0.0157, ID=0.0112。此打平现象确认非实现问题。

### Phase 2 — 3-seed 稳定性验证 (~12h)

| # | 实验 | 预计耗时 | 目的 |
|---|------|---------|------|
| 5 | TF-IDF+LLM no-boost (seed=2024) | 101 min | 补 LLM 3-seed |
| 6 | TF-IDF+LLM no-boost (seed=42) | 101 min | 补 LLM 3-seed |
| 7 | MV-Align no-boost (seed=2024) | 204 min | 补 MV 3-seed |
| 8 | MV-Align no-boost (seed=42) | 204 min | 补 MV 3-seed |

> TF-IDF 3-seed 已完成: mean MRR@10=0.0156 (std很小)

### Phase 3 — Cross ablation (~4h)

| # | 实验 | 预计耗时 | 目的 |
|---|------|---------|------|
| 9 | MV-Align **no-Cross** (seed=2025) | ~200 min | 评估 Cross 净收益 |

> 对比 #4 (with Cross) vs #9 (no Cross)
> 如 Cross 无稳定净收益 → 从最终模型移除

### Phase 4 — Toys/Grocery 重跑 (~12h)

| # | 实验 | 预计耗时 | 目的 |
|---|------|---------|------|
| 10 | Toys ID-only (seed=2025, min5) | ~24 min | 新 protocol baseline |
| 11 | Toys TF-IDF no-boost (seed=2025, min5) | ~78 min | |
| 12 | Toys TF-IDF+LLM no-boost (seed=2025, min5) | ~101 min | |
| 13 | Toys MV-Align no-boost (seed=2025, min5) | ~204 min | |
| 14 | Grocery ID-only (seed=2025, min5) | ~24 min | |
| 15 | Grocery TF-IDF no-boost (seed=2025, min5) | ~78 min | |
| 16 | Grocery TF-IDF+LLM no-boost (seed=2025, min5) | ~101 min | |
| 17 | Grocery MV-Align no-boost (seed=2025, min5) | ~204 min | |

### Phase 5 — UniSRec portability (~TBD)

| # | 实验 | 预计耗时 | 目的 |
|---|------|---------|------|
| 18 | UniSRec Base (Beauty, seed=2025, min5) | ~TBD | |
| 19 | UniSRec +Align (Beauty, seed=2025, min5) | ~TBD | |
| 20 | UniSRec +MultiView (Beauty, seed=2025, min5) | ~TBD | |

---

## 四、log10 ID-only 实验 (已启动)

> log10 已恢复，2026-07-12 13:42 启动 `run_log10_id_only_all_min5.sh`

| 实验 | 状态 |
|------|------|
| Beauty ID-only 3-seed (min5) | 🔄 运行中 |
| Toys ID-only 3-seed (min5) | ⏳ 排队 |
| Grocery ID-only 3-seed (min5) | ⏳ 排队 |

> 完成后自动 rsync 到 5090 `logs/log10/`

---

## 五、实验原则

1. **所有实验统一**: TS split + min_train_interactions=5 + TS-aware 特征 + no boost
2. **不再使用**: cold_text_boost, infer_boost, cold_threshold (主方法中删掉)
3. **item frequency 分桶**: 保留 low/mid/head 三档，不再叫 cold-start
4. **Cross 作为机制旋钮**: 不默认加入最终模型，看 ablation 决定

---

## 六、总预计时间

| 阶段 | 估时 | 累计 |
|------|------|------|
| Phase 1 (排查) | 已完成 | 0h |
| Phase 2 (3-seed) | ~12h | 12h |
| Phase 3 (Cross ablation) | ~4h | 16h |
| Phase 4 (Toys/Grocery) | ~14h | 30h |
| Phase 5 (UniSRec) | ~TBD | ~35h+ |

> 5090 串行约需 30-35h，即约 1.5 天。
