# 5090 实验状态梳理（2026-06-22）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-22 20:42（5090 实时，已停 bc 队列、恢复 Food）  
> **GPU 状态**: RTX 5090 训练中（Food TF-IDF Phase-A，util ~94%，显存 ~6GB）  
> **当前主线**: **Food 第三域四配置**（book-crossing 已暂停，见 6/21 快照）  
> **Balanced 参数**: `align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10`  
> **RUN_ID**: `20260622_food` · **SEED**: `2024`

**相关文档**:
- 实验设计: [`experiment_food_20260621.md`](experiment_food_20260621.md)
- 执行顺序总表: [`experiment_order_20260618.md`](experiment_order_20260618.md)（6/22 起 Food 升为 P0）
- book-crossing 暂停快照: 5090 `paper_recsys/book_crossing_status_snapshot_20260621.md`

---

## 1. 主线进度总览

```
Food 流水线 (run_5090_food_serial.sh)
[✅] 0_setup      数据集解压
[✅] 1_emb        TF-IDF + LLM 1v/4v（6/22 05:13 完成，~5.5h）
[✅] 2_id         SASRec ID baseline 50ep（6/22 16:02 完成，~45min）
[🔄] 3_tfidf      two-phase TF-IDF V3 — **重启** Phase-A（phaseb50 协议，6/22 20:42）
[⏸] 4_tfidf_llm  two-phase single-view LLM
[⏸] 5_mv         two-phase MV 7B
```

| Step | 配置 | 状态 | 开始 | 结束/当前 | 耗时 | test MRR@10 |
|------|------|------|------|-----------|------|-------------|
| 0 | setup | ✅ | 6/21 23:37 | 6/21 23:38 | ~1 min | — |
| 1 | embeddings | ✅ | 6/21 23:38 | 6/22 05:13 | **~5.5 h** | — |
| 2 | **ID-only** | ✅ | 6/22 15:17 | 6/22 16:02 | **~45 min** | **0.0060** |
| 3 | **TF-IDF** | 🔄 Phase-A 重启（phaseb50） | 6/22 20:42 | — | — | 旧 run Phase-B ep16/40 被 bc 抢占中断；见 §4 |
| 4 | TF-IDF+LLM | ⏸ 排队 | — | — | ~5 h 估 | — |
| 5 | MV-Align 7B | ⏸ 排队 | — | — | ~5 h 估 | — |

**四配置训练进度**: 1/4 完成 · 1/4 进行中 · 2/4 待跑  
**剩余预估**: TF-IDF Phase-B ~1h + LLM ~5h + MV ~5h ≈ **11h**

---

## 2. Step 2 — ID baseline 结果（seed=2024）

| 指标 | valid | test |
|------|-------|------|
| MRR@10 | 0.0052 | **0.0060** |
| Hit@10 | 0.0182 | 0.0200 |
| NDCG@10 | 0.0082 | 0.0093 |
| Hit_new@10 | 0.0010 | 0.0007 |
| Hit_few@10 | 0.0024 | 0.0036 |
| Hit_freq@10 | 0.0322 | 0.0359 |

- 日志: `logs/5090_food_20260622_food/2_id.log`
- checkpoint: `saved/`（RecBole 默认目录）

---

## 3. Step 3 — TF-IDF 进行中

- 脚本: `two_phase_run_tfidf_v3_food_stratified.sh`
- checkpoint: `saved/two_phase_run_tfidf_v3_food_stratified_seed2024/`
- 日志: `logs/5090_food_20260622_food/3_tfidf.log`

**Phase-A**（20ep, aw=0.1, tau=0.05）已完成:
- best valid MRR@10 = **0.0034**（低于 ID baseline 0.0052，属训练早期正常）
- Phase-A test MRR@10 = 0.0034

**Phase-B** 已自动续跑（40ep, eval_step=5, backbone_lr_scale=0.1）:
- 当前: **epoch 16/40**（~130s/epoch）
- 中途 valid MRR@10: ep4 **0.0059** · ep9 0.0052 · ep14 0.0054（已接近/略超 ID baseline 0.0052）
- 最新 checkpoint: `SASRecAlignV3-Jun-22-2026_16-34-13.pth`（970MB，6/22 16:45）

---

## 4. 已知问题 / 待处理

| 项 | 说明 | 建议 |
|----|------|------|
| **Phase-B epoch 协议** | 主表 ID=50ep；text 配置应为 Phase-A 20ep (warmup) + **Phase-B 50ep**。旧脚本/当前 Food TF-IDF 跑为 phaseb40 | 已更新脚本默认 `PHASE_B_EPOCHS=50`、`CHECKPOINT_TAG=phaseb50`；Food 当前 TF-IDF 可跑完或重跑 phaseb50 |
| `METRIC_BASELINE` | 流水线自动解析失败，仍用占位 **0.015**；Food ID valid MRR@10 实际为 **0.0052** | 后续 run 设 `METRIC_BASELINE=0.0052`；或修复 `run_5090_food_serial.sh` 解析 lowercase `mrr@10` |
| Phase-A gate | TF-IDF Phase-A MRR=0.0034 < threshold 0.015 → 仍 auto-continued Phase-B | 正常；gate 仅日志提示 |
| 6/21 v2 流水线 | embedding 完成后父进程退出，训练未续跑 | 已由 6/22 `RUN_ID=20260622_food` 从 `2_id` 重启 |
| book-crossing | 6/21 23:37 暂停；6/22 18:02 **误启** `run_5090_queue_bc_mv_phaseb50.sh` 抢占 GPU → **6/22 20:42 已停** | Food 四配置完成前勿再启 bc 队列 |

---

## 5. 5090 执行备忘

```bash
# 监控总进度
ssh charlie@www.ultrapp.online \
  'tail -f /home/charlie/project/RecBole/logs/5090_food_20260622_food_history.log'

# 当前 step
ssh charlie@www.ultrapp.online \
  'tail -f /home/charlie/project/RecBole/logs/5090_food_20260622_food/3_tfidf.log'

# 断点续跑（若中断）
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && \
  START_FROM=3_tfidf SKIP_SETUP=1 SKIP_EMB=1 RUN_ID=20260622_food SEED=2024 \
  METRIC_BASELINE=0.0052 bash run_5090_food_serial.sh'
```

**本地同步代码**（不含 dataset/saved）:

```bash
./sync_recbole.sh
```

---

## 6. 历史：book-crossing 状态（已暂停）

6/21 暂停前 book-crossing 已有较完整 SASRec 矩阵（baseline 4 seeds、GRU4Rec 四配置 seed2025、aligned 调参 g0–g3 等）。详见 5090 上 `book_crossing_status_snapshot_20260621.md` 与 [`experiment_status_20260618.md`](experiment_status_20260618.md)。

**策略变更（6/21）**: 第三域候选由 book-crossing 切换为 **Food**（非 Amazon、文本覆盖 100%、分层与 B/T 更接近）。见 [`dataset_strata_analysis_20260621.md`](dataset_strata_analysis_20260621.md)（5090）。
