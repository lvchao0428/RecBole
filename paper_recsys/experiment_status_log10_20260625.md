# log10 实验状态（2026-06-25）

> **机器**: `charlie@192.168.0.137`（5090 上 alias `log10`）  
> **项目**: `/home/charlie/project/RecBole`  
> **GPU**: GTX 1080Ti 11GB  
> **核对时间**: 2026-06-25 22:25

**主文档**: [`experiment_status_20260625.md`](experiment_status_20260625.md)（5090 总览）

---

## 1. 任务范围

Phase 4 GRU4Rec **baseline 分流**（5090 跑 LLM+MV，log10 跑轻量配置）:

| 数据集 | 配置 | Seeds | Runs |
|--------|------|-------|------|
| Beauty | ID-only + TF-IDF | 42, 2024, 2025, 2026 | 8 |
| Toys | ID-only + TF-IDF | 42, 2024, 2025, 2026 | 8 |
| **合计** | | | **16** |

脚本: `run_log10_gru4rec_baselines.sh --all-seeds`  
启动: 5090 上 `scripts/remote_start_log10_gru4rec_baselines.sh`（21:00 首次 · 22:03 重复启动已清理）

---

## 2. 当前进度

| 项 | 状态 |
|----|------|
| 总进度 | **1/16** |
| 当前 run | Beauty · GRU4Rec ID-only · **seed=42** |
| epoch | **~32/50**（~64%） |
| GPU | ~97% · ~2.9GB VRAM |
| checkpoint | `saved/gru4rec_baseline_v3_beauty_stratified_seed42/` |
| run_metrics | 0（首个 run 未完成） |

**串行队列**（每个 seed）: ID-only 50ep → TF-IDF two-phase → 下一数据集/seed

---

## 3. 数据预检 ✅

| 资源 | 状态 |
|------|------|
| Beauty/Toys `.inter` / `.item` | ✅ |
| TF-IDF `item_text_emb.base.npy` | ✅（Toys 165MB 已从 5090 rsync） |
| LLM / MV 4views | ⚠️ 仅 5090（本机不跑） |

---

## 4. 已修复问题

| 问题 | 处理 |
|------|------|
| `initializer_range` None | `gru4recalignv3.py` 默认 0.02 · 已 sync |
| 22:03 重复启动 duplicate python | 已 kill 重复进程，保留 21:00 任务 |
| `two_phase_run_gru4rec*.sh` 续行空行 | 6/25 22:20 已从 5090 sync · TF-IDF 下一 run 可正常启动 |

---

## 5. 监控

```bash
ssh charlie@192.168.0.137
cd ~/project/RecBole
tail -f logs/log10_gru4rec_beauty_id_seed42.log
tail -f logs/log10_gru4rec_baselines_20260625_210045.log
pgrep -af two_phase_train; nvidia-smi
```

---

## 6. 完成后

5090 上 `run_5090_phase4_text_only.sh` 末尾自动执行:
```bash
bash scripts/wait_and_pull_log10_gru4rec.sh
```
回拉 `saved/gru4rec_*` · `run_metrics/` · `logs/log10_gru4rec_*.log` 到 5090。

**预估 log10 单独耗时**: ~**2–3 天**（16 runs × ~2h/run）
