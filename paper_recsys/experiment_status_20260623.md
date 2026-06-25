# 5090 实验状态梳理（2026-06-23）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-23 23:56（5090 实查 · 本地+5090 已同步）  
> **GPU 状态**: RTX 5090 训练中（BC TF-IDF+LLM Phase-B ep41/50，util ~91%，显存 ~12GB）  
> **当前主线**: **book-crossing phaseb50 收尾** → 自动接 **Grocery 第三域**  
> **Balanced 参数**: `align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10`  
> **SEED**: `2024`（BC 文本配置已确认；ID baseline 见 §3 已知问题）

**相关文档**:
- BC 结果汇总: [`book_crossing_stratified_results_20260623.md`](book_crossing_stratified_results_20260623.md)
- Grocery 规划: [`grocery_experiment_plan_20260623.md`](grocery_experiment_plan_20260623.md)
- 分层扫描: [`dataset_strata_all_20260623.md`](dataset_strata_all_20260623.md)
- Food 已完成: [`experiment_status_20260622.md`](experiment_status_20260622.md) · [`experiment_food_20260621.md`](experiment_food_20260621.md)

---

## 1. 5090 资源调度（当前）

```
现在 ────────────── BC phaseb50 step 3/4 LLM (~40min) + step 4 MV (~3h)
BC 完成 (~6/24 03:30) ── Grocery embedding (~8h) + 三文本配置 (~10h)
Grocery 完成 (~6/24 22:00) ── trend 判断 → 可选 multiseed
```

| 进程 | PID | 状态 |
|------|-----|------|
| `run_5090_queue_bc_four_configs_phaseb50.sh` | 1486437† | 🔄 step 3/4 |
| `run_5090_queue_after_bc_grocery.sh` | 1499287† | ⏸ 轮询等待 BC |

† PID 换机后可能变化，以 `pgrep -af run_5090` 为准。
| Food `run_5090_food_serial.sh` | — | ✅ 6/23 09:38 完成 |

---

## 2. book-crossing phaseb50（seed=2024）

```
[✅] 1/4 ID-only 50ep          17:57  test MRR@10=1.95%  HR@10=3.87%
[✅] 2/4 TF-IDF phaseb50       20:45  test MRR@10=2.20%  HR@10=3.65%  (+13% MRR)
[🔄] 3/4 TF-IDF+LLM phaseb50  Phase-B ep41/50  valid MRR@10≈3.37%
[⏸] 4/4 MV phaseb50           排队 ~3h
```

**日志**:
- 队列: `logs/queue_bc_four_phaseb50.log`
- 当前: `logs/exp_bc_llm_phaseb50_20260623_bc_four_phaseb50.log`
- checkpoint: `saved/two_phase_run_tfidf_llm_v3_book_crossing_stratified_phaseb50_seed2024/`

**BC 剩余 ETA**: LLM ~30min + MV ~3h → **约 6/24 03:30 完成**

**趋势（vs seed≈2025 Phase-B 40ep）**: TF-IDF MRR 2.20% ≈ 2.22%；增益仍为 **+13% MRR**，确认 BC 不适合第三域主表（appendix 定性不变）。

---

## 3. Grocery 第三域（等待 BC）

```
[✅] 0_setup       dataset 解压（6/23 18:00）
[✅] log10 ID-only 50ep  19:16  test MRR@10=2.08%  HR@10=6.35%
[⏸] 1_emb         TF-IDF + Qwen 1v/4v（BC 完成后启动）
[⏸] 3_tfidf       two-phase TF-IDF
[⏸] 4_llm         two-phase single-view LLM
[⏸] 5_mv          two-phase MV 7B
```

| Step | 配置 | 状态 | test MRR@10 | test HR@10 |
|------|------|------|-------------|------------|
| log10 ID | SASRecAlign 50ep | ✅ | **2.08%** | **6.35%** |
| TF-IDF | two-phase | ⏸ | — | — |
| TF-IDF+LLM | two-phase | ⏸ | — | — |
| MV | two-phase | ⏸ | — | — |

**Grocery 剩余 ETA**（BC 完成后）: embedding ~8h + 三配置 ~10h ≈ **~18h** → 约 **6/24 22:00**

**日志 / 标记**:
- 等待队列: `logs/queue_after_bc_grocery.log`
- log10 ID: `log10:~/project/RecBole/logs/log10_grocery_id_20260624_grocery.log`
- setup 标记: `logs/.grocery_setup_done`
- pipeline 脚本: `run_5090_grocery_pipeline.sh`（`RUN_ID=20260624_grocery`）

**Go/No-Go 参考**（Beauty 主表）: TF-IDF/ID MRR ≥ **1.30×**；MV MRR > LLM MRR。

---

## 4. Food 第三域（已完成 · 6/23 09:38）

> RUN_ID=`20260622_food` · seed=2024 · 非 Amazon 对照域

| 配置 | test MRR@10 | test HR@10 | vs ID (MRR) |
|------|-------------|------------|-------------|
| **ID-only** | **0.60%** | **2.00%** | — |
| TF-IDF | 0.68% | 1.70% | +13% |
| TF-IDF+LLM | 0.65% | 1.30% | +8% |
| MV | 0.65% | 1.46% | +8% |

**结论**: ID > 文本（overall），与 BC 类似、与 B/T 主表趋势相反；适合 **非 Amazon tail-heavy 对照**，不作第三域主表。

**日志**: `logs/5090_food_20260622_food/` · history: `logs/5090_food_20260622_food_history.log`

---

## 5. 已知问题

| 项 | 说明 | 建议 |
|----|------|------|
| **ID seed 覆盖** | BC / Grocery log10 ID 的 run_metrics 中 `seed=2020`，yaml 默认覆盖 CLI `--seed 2024` | 检查 `sasrec_baseline_50ep_*_stratified.yaml` 是否硬编码 seed；必要时重跑 ID |
| Grocery METRIC_BASELINE | log10 ID valid MRR@10=**0.0204**；`.grocery_metric_baseline_*` 待 pipeline 启动后写入 | BC 完成后 pipeline 自动解析 |
| BC multiseed | 已定性 appendix，当前仅 seed=2024 phaseb50 收尾 | 不必 multiseed |

---

## 6. 5090 监控命令

```bash
# BC 当前 step
ssh charlie@www.ultrapp.online \
  'tail -f /home/charlie/project/RecBole/logs/exp_bc_llm_phaseb50_20260623_bc_four_phaseb50.log'

# Grocery 等待队列
ssh charlie@www.ultrapp.online \
  'tail -f /home/charlie/project/RecBole/logs/queue_after_bc_grocery.log'

# BC 完成后 Grocery pipeline 总日志
ssh charlie@www.ultrapp.online \
  'tail -f /home/charlie/project/RecBole/logs/grocery_pipeline_20260624_grocery.log'

# 进程一览
ssh charlie@www.ultrapp.online \
  'pgrep -af "run_5090|book_crossing|grocery|two_phase_train" | grep -v pgrep'
```

---

## 7. 换机追踪备忘（2026-06-24 起）

**5090 连接**（两台机器通用）:

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole
```

**本地同步代码+文档到 5090**（不含 dataset/saved/logs）:

```bash
./sync_recbole.sh          # 全项目
# 或仅文档:
rsync -avz paper_recsys/*.md charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
```

**明日优先检查顺序**:

| 时间窗 | 预期事件 | 检查命令 |
|--------|----------|----------|
| 6/24 00:00–01:00 | BC step 3 LLM 完成 | `tail logs/queue_bc_four_phaseb50.log` |
| 6/24 03:00–04:00 | BC step 4 MV 完成，Grocery pipeline 自动启动 | `tail logs/queue_after_bc_grocery.log` |
| 6/24 04:00–12:00 | Grocery embedding 进行中 | `ls dataset/Amazon_Grocery_and_Gourmet_Food/*.npy` |
| 6/24 12:00–22:00 | Grocery 三文本配置 | `tail logs/grocery_pipeline_history.log` |

**关键 RUN_ID / 标记**（不因换机变化）:

| 实验 | RUN_ID | 关键日志 |
|------|--------|----------|
| BC phaseb50 | `20260623_bc_four_phaseb50` | `logs/queue_bc_four_phaseb50.log` |
| Grocery | `20260624_grocery` | `logs/grocery_pipeline_20260624_grocery.log` |
| Food（已完成） | `20260622_food` | `logs/5090_food_20260622_food_history.log` |

**更新文档流程**: 5090 实查 → 改本地 `paper_recsys/experiment_status_*.md` → `rsync paper_recsys/*.md` 推回 5090。

详见 [`experiment_handoff_20260624.md`](experiment_handoff_20260624.md)。
