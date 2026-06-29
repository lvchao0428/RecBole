# log10 实验状态（2026-06-26）

> **机器**: `charlie@192.168.0.137`（5090 上 alias `log10`）  
> **项目**: `/home/charlie/project/RecBole`  
> **GPU**: GTX 1080Ti 11GB  
> **核对时间**: 2026-06-26 08:20

**主文档**: [`experiment_status_20260626.md`](experiment_status_20260626.md)（5090 总览）

---

## 1. 任务范围

Phase 4 GRU4Rec **baseline 分流**（5090 跑 LLM+MV，log10 跑轻量配置）:

| 数据集 | 配置 | Seeds | Runs |
|--------|------|-------|------|
| Beauty | ID-only + TF-IDF | 42, 2024, 2025, 2026 | 8 |
| Toys | ID-only + TF-IDF | 42, 2024, 2025, 2026 | 8 |
| **合计** | | | **16** |

脚本: `run_log10_gru4rec_baselines.sh --all-seeds`  
启动: 2026-06-25 21:00 · `logs/log10_gru4rec_baselines_20260625_210045.log`

---

## 2. 当前进度

| 项 | 状态 |
|----|------|
| 总进度 | **2/16** |
| 已完成 | Beauty GRU4Rec **ID-only seed=42** |
| 当前 run | Beauty · GRU4Rec **TF-IDF** · **seed=42**（two-phase Phase-A） |
| GPU | ~97% · ~2.9GB VRAM |
| run_metrics | 0（首个完整 two-phase 未完成） |

**串行队列**（每个 seed）: ID-only 50ep → TF-IDF two-phase → 下一 seed / 数据集

**Balanced 参数**（与主表一致）: `align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10`

---

## 3. 与 5090 衔接

5090 `run_5090_phase4_text_resume.sh` 全部 LLM+MV 跑完后执行:
```bash
bash scripts/wait_and_pull_log10_gru4rec.sh
```
回拉 `saved/gru4rec_*` · `run_metrics/` · `logs/log10_gru4rec_*.log` 到 5090。

之后自动接 **Phase 5**: UniSRec +Align/+MV balanced（Beauty seed=2024）。

---

## 4. 监控

```bash
ssh charlie@192.168.0.137
cd ~/project/RecBole
tail -f logs/log10_gru4rec_baselines_20260625_210045.log
pgrep -af two_phase_train; nvidia-smi
```

**预估 log10 单独耗时**: ~**2–3 天**（16 runs × ~2h/run）
