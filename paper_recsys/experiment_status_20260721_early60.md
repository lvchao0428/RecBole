# Early60 overnight — stop-gate 第二窗口 + Cross Top-K seal

**启动时间**: 2026-07-21 下午（5090）  
**目标**: 明早可对老师说 stop-gate 是否正式触发；Cross 默认删除有 Top-K head/tail 证据。

## 时间预算（可行）

| 步骤 | 预估 | 说明 |
|------|------|------|
| 重建 early60 TF-IDF + Qwen whiten | ~15–30 min | 不重跑 Qwen inference |
| P1 Top-K head/tail share（Main Cross ckpt 离线） | ~30–60 min | LLM/TF/MV 三对 |
| Early TF/LLM/MV × seed2025 | ~4 h | 关死 stop-gate 的最低要求 |
| + seed42 / 2024 | ~8 h | 理想 3-seed；时间不够则明早至少有 2025 |
| Early LLM +Cross seed2025 | ~1.2 h | 可选，回应“依赖窗口” |

**结论**: 一天/过夜可验证完。默认跑满 3 seeds + Early Cross；若只想保底，设 `SEEDS=2025`。

## P0 协议

- Split: `TS: [0.6, 0.1, 0.3]`（与 Main 同一 `time_based_split`：全局 test≈70th pct，valid 为更早窗口）
- Features: train-only TF-IDF/SVD/whiten 按 early cutoff 重拟合  
  - `item_text_emb.base.ts_early60.npy`  
  - `item_text_emb.qwen2.5_7b.base.ts_early60.npy`  
  - `qwen2.5_7b_4views_ts_early60/`
- 冻结 Main 超参（valid 选出）: TF `5e-4/0.3`，LLM `1e-4/0.1`，MV `5e-4/0.3`
- no-boost / no-Cross / min_train≥5 / Phase-A+B

## 判定（明早看）

对比 Main（已有 3-seed）与 Early：

1. 两窗口排序是否同为 **TF ≈ LLM ≥ MV**（test MRR；附 valid）
2. MV 相对单视图是否仍 **不能稳定超过 1 std**  
→ 是：stop-gate 正式触发，MV 不作主方法

## P1 Cross 封板

- 不重跑全套 Cross 3-seed
- 离线补 Top-K head/tail share → `paper_recsys/rank_transition_beauty_{llm,tf,mv}.md`
- 可选：Early LLM ±Cross（seed2025）→ CSV 中 `use_cross=true` 行

## 产物路径（5090）

```
logs/early60_overnight.log
logs/early60_overnight_nohup.log
logs/early60_stopgate_beauty.csv
saved/ew_beauty_{tf,llm,mv}_*_seed*/
paper_recsys/rank_transition_beauty_*.md
paper_recsys/rank_transition_beauty_*.json
```

## 启动命令

```bash
nohup bash run_5090_early60_overnight.sh > logs/early60_overnight_nohup.log 2>&1 &
# 或保底：
# SEEDS="2025" nohup bash run_5090_early60_overnight.sh > logs/early60_overnight_nohup.log 2>&1 &
```
