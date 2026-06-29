# GRU4Rec Phase 4 任务分配（自包含 · 2026-06-29 更新）

> **原则**: **5090 跑完 = 全部完成** · 5090 5090 更快 → **baseline 主力在 5090** · log10 当前 job 结束后暂停

---

## 分工（2026-06-29 重分配）

| 机器 | 职责 | batch | 人工干预 |
|------|------|-------|----------|
| **5090** | **全部 baseline 2024/2025/2026**（skip 已有 ckpt）· batch=512 | 512 | ❌ tail 自动 |
| **log10** | 仅跑完 **当前** Beauty ID 2025 → **pause watcher 自动停** | 256 | ✅ 已挂 PID 84705 |

**剩余缺口（5090 独占）**: seed 2025 ×3 + seed 2026 ×4 = **7 块** · ETA ~**20–28h** @5090

---

## 5090 自动收尾

```bash
# 5090 上（watchdog 管理 tail）
nohup bash scripts/5090_auto_tail_watchdog.sh >> logs/5090_auto_tail_watchdog.log 2>&1 &
# 或手动：
nohup bash run_5090_phase4_tail_rebalanced.sh >> logs/5090_phase4_tail_rebalanced.log 2>&1 &
```

**tail 脚本顺序**:
```
pull log10 → Phase5 UniSRec (skip if ckpt) → baseline 2024/2025/2026 @512 → pull + collect → done
```

**修复 (6/29)**: 补全 `run_unisrec_v3_align_balanced_batch_beauty.sh`；Phase 5 skip 已有 balanced ckpt；watchdog 不再空转。

---

## log10 暂停

```bash
# 5090 上触发（log10 当前 two_phase_train 结束后停队列）
ssh charlie@192.168.0.107 'nohup bash ~/project/RecBole/scripts/log10_pipeline_pause_after_current.sh \
  >> ~/project/RecBole/logs/log10_pipeline_pause_nohup.log 2>&1 &'
```

恢复: `bash run_log10_gru4rec_baselines_from_2024.sh`（通常不需要，5090 已补齐）

---

## 完成判定

```bash
test -f logs/5090_pipeline_all_complete.done && echo DONE
bash scripts/collect_gru4rec_phase4_results.sh --pull-log10
# baseline 16/16 + text 16/16 → 32/32
```

---

## 日志

| 文件 | 内容 |
|------|------|
| `logs/5090_auto_tail_watchdog.log` | watchdog |
| `logs/5090_phase4_tail_rebalanced.log` | Phase5 + baseline |
| `logs/5090_gru4rec_baselines.log` | 5090 baseline 明细 |
| `logs/log10_pipeline_pause.log` | log10 暂停 |
| `logs/5090_pipeline_all_complete.done` | 全部完成标记 |
