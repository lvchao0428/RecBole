# 实验执行顺序 TodoList

> **日期**: 2026-06-18（初版）· **主线更新 2026-06-22**  
> **最后核对**: 2026-06-22 17:12（5090 实时；详见 [`experiment_status_20260622.md`](experiment_status_20260622.md)）  
> **机器**: 5090 (`charlie@www.ultrapp.online:/home/charlie/project/RecBole`)  
> **Balanced 参数**: `align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10`  
> **当前 P0 主线**: **Food 四配置**（seed=2024，`RUN_ID=20260622_food`）— book-crossing **已暂停**

**图例**: `- [ ]` 待跑 · `- [x]` 已完成 · `🔄` 进行中

> **换机续跑**: 在新电脑上 `git pull` / `./sync_recbole.sh` 同步代码即可；**不要**用旧版 sync（已修复 `--exclude dataset/`）。实验产物只在 5090 的 `saved/`、`logs/`，不在本机。

---

## P0 — Food 第三域四配置（**当前最高优先级**，2026-06-22 起）

> 设计文档: [`experiment_food_20260621.md`](experiment_food_20260621.md) · 状态: [`experiment_status_20260622.md`](experiment_status_20260622.md)  
> 脚本: `run_5090_food_serial.sh` · seed=2024 · 超参与 Beauty/Toys 主表 V3 一致

- [x] 同步 Food 脚本/YAML 到本地 + 5090
- [x] 0_setup — 数据集 `dataset/Food/`
- [x] 1_emb — TF-IDF + Qwen2.5-7B 1v/4v（6/22 05:13）
- [x] 2_id — SASRec ID baseline 50ep（test MRR@10=0.0060）
- [🔄] 3_tfidf — two-phase TF-IDF V3（Phase-B epoch 16/40，valid MRR@10 最高 0.0059）
- [ ] 4_tfidf_llm — two-phase TF-IDF+LLM
- [ ] 5_mv — two-phase MV 7B
- [ ] 对比 Beauty/Toys 同配置趋势，决定是否 multiseed / 入主表
- [ ] 修复 `METRIC_BASELINE` 自动解析（valid MRR@10=0.0052）

~~6/21 `RUN_ID=20260621_food_v2` embedding 完成后中断~~ → 6/22 已从 `2_id` 重启。

---

## 代码 / 脚本状态（2026-06-18 已同步 5090）

| 模块 | 本地 | 5090 | 说明 |
|------|------|------|------|
| GRU4RecAlignV3 + MultiViewV3 | ✅ | ✅ | 模型 + 12 YAML + 15 脚本 |
| FDSAAlignV3 + MultiViewV3 | ✅ | ✅ | 同上 |
| UniSRec | ✅ | ✅ | 模型 + 3 YAML + 4 脚本 |
| SASRec book-crossing V3 脚本 | ✅ | ✅ | `run_0125Batch_v3_book_crossing.sh` 等 |
| `scripts/recbole_env.sh` | ✅ | ✅ | 5090 自动走 anaconda python |
| `sync_recbole.sh` | ✅ | — | **`--exclude dataset/`**，不覆盖数据 |

**本地未同步、下次再推**: 无（本次改动已全部 rsync；若换电脑后本地有新 commit，再 `./sync_recbole.sh`）。

---

## Book-Crossing 完整实验矩阵（5090 状态 — **⏸ 已暂停 6/21**）

> **4 配置**：baseline（ID-only）· TF-IDF · single-view（`tfidf_llm` / Qwen3）· multi-view（4-view）  
> **默认 4 seeds**：42 / 2024 / 2025 / 2026  
> 暂停快照: 5090 `paper_recsys/book_crossing_status_snapshot_20260621.md`

| Backbone | baseline | TF-IDF | single-view | multi-view | 备注 |
|----------|----------|--------|-------------|------------|------|
| **SASRec** | ✅ 4 seeds + aligned | ✅ 多 seed | ✅ 多 seed | ✅ 4 seeds | 6/21 暂停；aligned g0–g3 已跑 |
| **GRU4Rec** | ✅ seed2025 | ✅ seed2025 | ✅ seed2025 | ✅ seed2025 | 同上 |
| **FDSA** | ⏸ | ⏸ | ⏸ | ⏸ | 未跑 |
| **UniSRec** | — | ⏸ | — | — | 未跑 |

~~**当前最高优先级**：book-crossing 趋势验证~~ → **已切换至 Food**（见上方 P0）。

~~Apr 旧 checkpoint 仅作参考，趋势验证以 6/18 新 baseline 为锚重跑 text 三配置。~~

~~`run_all_book_crossing_remaining_serial.sh` 已暂停（原为 baseline 3 seeds + backbone）。~~

---

## P0 — book-crossing 趋势验证（**最高优先级**，先于 multiseed / backbone）

> **gate**：单 seed 四配置主指标走势需与 Beauty/Toys 一致（TF-IDF 常最强或 near-best；single-view / MV 在 cold/new 有增益）；否则先调 `align_weight` / `cold_text_boost` / `infer_boost`  
> 回应 R2-W3 / R3-Q2

- [x] 同步代码/脚本到 5090
- [x] 恢复 book-crossing 数据 + timestamp
- [x] SASRec V3 **baseline seed=2025**（test MRR@10≈0.0199，valid≈0.0278）  
  - `saved/baseline_v3_book_crossing_stratified_seed2025/`
- [◐] **`run_book_crossing_v3_trend_check.sh`** — seed=2025：baseline(skip) ✅ · TF-IDF ✅ · single-view ✅ · **multi-view ❌**  
  - 日志: `logs/exp_bc_v3_trend_check_seed2025.log`（21:04 崩溃：`views.json` 缺失）  
  - 修复: `python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views`  
  - 24h 续跑: `run_5090_24h_book_crossing_serial.sh`
- [ ] 对比 Beauty/Toys 同配置 test 指标，记录是否需调参
- [ ] 趋势 OK 后 → multiseed（`run_0125Batch_v3_book_crossing_multiseed.sh`）→ backbone（P1）

- [x] ~~Apr 旧跑~~ tfidf / single-view / multi-view 4 seeds（**不用于趋势结论**，将被 seed2025 新跑覆盖 checkpoint）
- [ ] V3 baseline 其余 3 seeds — **等趋势验证通过后**
- [ ] （可选）汇总指标入表

~~已暂停~~：`run_all_book_crossing_remaining_serial.sh`（baseline 3 seeds + GRU4Rec/FDSA/UniSRec）

---

## P1 — book-crossing 新 Backbone 矩阵（**趋势验证通过后**）

> 原 `run_all_book_crossing_remaining_serial.sh` 已暂停

- [ ] `run_gru4rec_v3_batch_book_crossing.sh` — baseline · TF-IDF · single-view · multi-view
- [ ] `run_fdsa_v3_batch_book_crossing.sh` — 同上
- [ ] `run_unisrec_book_crossing_stratified.sh` — UniSRec

---

## P2 — Beauty / Toys SASRec 补全 & 多 seed

- [ ] Beauty SASRec 4 配置 multiseed（若主表仍缺）
- [ ] Toys SASRec 4 配置 multiseed（若主表仍缺）

---

## P3 — GRU4Rec Backbone 泛化（R2-W2 / R3-Q2）

- [ ] `run_gru4rec_v3_batch_beauty.sh`
- [ ] `run_gru4rec_v3_batch_toys.sh`
- [ ] 上述 × 4 seeds

---

## P4 — FDSA 文本增强全矩阵（R2-W1）

- [ ] `run_fdsa_v3_batch_beauty.sh`
- [ ] `run_fdsa_v3_batch_toys.sh`
- [ ] 上述 × 4 seeds

---

## P5 — UniSRec 外部 PLM 基线（R2-W1 / R3-Q3）

- [ ] `run_unisrec_beauty_stratified.sh`
- [ ] `run_unisrec_toys_stratified.sh`
- [ ] `run_unisrec_batch_all.sh`
- [ ] 上述 × 4 seeds

---

## P6 — 消融实验（机制论文）

- [ ] 单视角消融 — `run_per_view_ablation.sh`（Toys）
- [ ] MV-Align-Lite — Toys 精简变体

---

## P7 — 全矩阵 multiseed 汇总

- [ ] GRU4Rec / FDSA / UniSRec：3 数据集 × 4 seeds
- [ ] SASRec book-crossing V3 baseline：补全 3 seeds（见 P0）

---

## P8 — 可选 / 低优先级

- [ ] FDSA 原版 RecBole（无 text pipeline）
- [ ] S3-Rec 基线
- [ ] 机制分析：Gini/entropy、gate bucket、case study

---

## 5090 执行备忘

```bash
# 新电脑首次：克隆/拉代码后
./sync_recbole.sh          # 只推代码，不碰 dataset/

# P0 收尾 + 24h 串行:
#   python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views
#   nohup bash run_5090_24h_book_crossing_serial.sh > logs/exp_5090_24h_book_crossing_serial.log 2>&1 &
#   tail -f logs/5090_24h_serial/00_master.log
```

---

## 预估耗时（5090 单卡，粗估）

| 块 | 内容 | 单次 | ×4 seeds |
|----|------|------|----------|
| P0 | 趋势验证 seed2025 四配置 | ~8–12h | — |
| P0 后续 | multiseed + baseline 3 seeds | ~2h/seed | ~32h |
| P1 | GRU4Rec+FDSA+UniSRec（gate 后） | ~15h | ~60h |
| P3–P5 | Beauty/Toys 全矩阵 | ~30h/数据集 | ~120h |

**建议**: P0 趋势验证（seed2025 四配置）→ 调参（如需）→ multiseed → P1 backbone → Beauty/Toys。
