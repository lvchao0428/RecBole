# 5090 实验状态梳理（2026-06-18）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-18 21:04（5090 实时）  
> **GPU 状态**: RTX 5090 空闲（util 0%，无训练进程）  
> **Balanced 参数**: `align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10`  
> **当前 METRIC_BASELINE**: `0.028`（对齐 6/18 V3 baseline valid MRR@10）

---

## 1. 今日已完成训练（含耗时）

| # | 实验 | 脚本 / 日志 | 开始 | 结束 | 耗时 | test MRR@10 | 状态 |
|---|------|------------|------|------|------|------------|------|
| 1 | SASRec V3 baseline seed=2025 | `run50epBase_v3_book_crossing_stratified.sh` · `logs/exp_20260618_p0_sasrec_bc_baseline.log` | 13:48 | 14:39 | **~51 min** | ~0.0199 | ✅ |
| 2 | SASRec TF-IDF V3 seed=2025 | `two_phase_run_tfidf_v3_book_crossing_stratified.sh` · trend log | 15:40 | 18:01 | **~2 h 21 min** | 0.0222 | ✅ |
| 3 | SASRec TF-IDF+LLM (Qwen3) seed=2025 | `two_phase_run_tfidf_llm_v3_book_crossing_stratified.sh` · trend log | 18:01 | 21:03 | **~3 h 02 min** | 0.0223 | ✅ |
| 4 | SASRec Multi-View seed=2025 | `two_phase_run_multiview_v3_book_crossing_stratified_7b.sh` · trend log | 21:04 | — | — | — | ❌ 失败 |

**P0 趋势验证进度**: 4 配置中 3/4 完成；multi-view 在初始化阶段崩溃。

**单配置耗时粗估（5090，book-crossing）**:

| 配置 | Phase | 典型耗时 |
|------|-------|----------|
| baseline (50ep, only Phase-A) | — | ~50 min |
| TF-IDF / single-view (20ep A + 40ep B) | 两阶段 | ~2.0–3.0 h |
| multi-view (同上) | 两阶段 | ~3.0–3.5 h（待补跑验证） |
| GRU4Rec / FDSA 单配置 | 同上结构 | ~1.5–3.0 h（GRU 略快于 SASRec） |
| UniSRec | 单阶段 | ~1.0–1.5 h |

---

## 2. 训练失败 / 中断记录

| 时间 | 实验 | 原因 | 处理 |
|------|------|------|------|
| **6/18 21:04** | SASRec multi-view seed2025（P0 step 4/4） | `ValueError: views.json not found in dataset/book-crossing/qwen3_4views`；目录有 `view_0..3.npy` 但缺元数据 | 运行 `python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views` 后重跑 |
| **6/18 15:27–15:39** | SASRec baseline seed=42（`run_all_book_crossing_remaining_serial.sh`） | 跑到 epoch ~12 时被 P0 趋势验证脚本抢占/中断；checkpoint 不完整 | 纳入 24h 计划 Step 2 重跑 |
| **4 月旧跑** | SASRec text 三配置 4 seeds | 使用 `METRIC_BASELINE=0.015`、旧 60ep baseline 锚点；**不用于 6/18 趋势结论** | 保留 checkpoint 作参考，seed2025 新跑覆盖 |
| **2025-01-12** | 8 个实验 | 5090 磁盘满（见 `archive_0113/CHANGELOG_0112_scale_law.md`） | 已清理；当前磁盘正常 |
| **2026-06-18 前** | sync 误删 dataset | 旧版 `sync_recbole.sh` 无 `--exclude dataset/` | 已修复；数据已从 backup 恢复 |

---

## 3. 5090 当前 checkpoint 快照

```
saved/baseline_v3_book_crossing_stratified_seed2025          ✅ 6/18 13:53
saved/two_phase_run_tfidf_v3_book_crossing_stratified_seed2025     ✅ 6/18 16:31
saved/two_phase_run_tfidf_llm_v3_book_crossing_stratified_seed2025  ✅ 6/18 19:09
saved/baseline_v3_book_crossing_stratified_seed42             ⚠️ 不完整（epoch~12 中断）
saved/two_phase_run_multiview_v3_book_crossing_stratified_7b_seed2025  📁 Apr 旧跑（2 ckpt，非 6/18 配置）
Apr 旧跑（4 seeds 齐）: two_phase_run_tfidf_v3_*, tfidf_llm_*, multiview_*_seed{42,2024,2025,2026}
```

---

## 4. 下一批次实验列表（按优先级）

### Gate：P0 趋势验证收尾

- [ ] 修复 `views.json` → 补跑 **SASRec multi-view seed=2025**（~3.5 h）
- [ ] 对比 Beauty/Toys 同配置 test 指标，判断趋势是否一致
- [ ] 若 OK → 进入 multiseed / backbone；若否 → 调 `align_weight` / `cold_text_boost` / `infer_boost`

### 24h 串行计划（趋势 gate 通过后执行）

| Step | 内容 | 脚本 | 预估 |
|------|------|------|------|
| 0 | 修复 views.json | `tools/repair_qwen3_views_json.py` | 1 min |
| 1 | SASRec multi-view seed2025 | `two_phase_run_multiview_v3_book_crossing_stratified_7b.sh` | 3.5 h |
| 2 | SASRec baseline seed 42 / 2024 / 2026 | `run50epBase_v3_book_crossing_stratified.sh` ×3 | 2.6 h |
| 3 | GRU4Rec ×4 配置 seed2025 | `run_gru4rec_v3_batch_book_crossing.sh` | 8–9 h |
| 4 | FDSA ×4 配置 seed2025 | `run_fdsa_v3_batch_book_crossing.sh` | 8–9 h |
| 5 | UniSRec seed2025 | `run_unisrec_book_crossing_stratified.sh` | 1.5 h |
| | **合计** | `run_5090_24h_book_crossing_serial.sh` | **~22–24 h** |

### Gate 之后、24h 之外的排队

| 优先级 | 内容 | 脚本 | 备注 |
|--------|------|------|------|
| P0+ | SASRec 四配置 multiseed（3 seeds 待补） | `run_0125Batch_v3_book_crossing_multiseed.sh` | 4 seeds × ~10 h ≈ 40 h |
| P2 | Beauty / Toys SASRec multiseed | 各 batch 脚本 | 主表补全 |
| P3–P5 | Beauty/Toys GRU4Rec / FDSA / UniSRec | `run_gru4rec_v3_batch_*.sh` 等 | ~120 h |

---

## 5. 启动命令（5090）

```bash
# 1) 本地同步代码（含 repair 脚本 + 24h 串行脚本）
./sync_recbole.sh

# 2) 5090 上先单独验证 repair（可选）
python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views

# 3) 启动 ~24h 串行流水线
nohup bash run_5090_24h_book_crossing_serial.sh \
  > logs/exp_5090_24h_book_crossing_serial.log 2>&1 &

# 4) 监控
tail -f logs/exp_5090_24h_book_crossing_serial.log
tail -f logs/5090_24h_serial/00_master.log
```

**仅补跑 P0 multi-view**（不跑 24h 全量）:

```bash
python tools/repair_qwen3_views_json.py --split_dir dataset/book-crossing/qwen3_4views
METRIC_BASELINE=0.028 SEED=2025 GPU_ID=0 \
  bash two_phase_run_multiview_v3_book_crossing_stratified_7b.sh
```

---

## 6. 与 `experiment_order_20260618.md` 的对应

| 文档章节 | 当前状态 |
|----------|----------|
| P0 趋势验证 | 3/4 完成；multi-view 待 repair + 重跑 |
| P1 backbone 矩阵 | 纳入 24h Step 3–5 |
| P0 multiseed | Step 2 + 后续 `run_0125Batch_v3_book_crossing_multiseed.sh` |
| Apr 旧跑 | 作废对比，不写入主表 |
