# GRU4Rec Phase 4 任务分配（自包含 · 2026-06-28）

> **原则**: **5090 跑完 = 全部完成** · log10 可继续多跑（重复无妨）· **无需半夜人工切换**

---

## 分工

| 机器 | 职责 | 人工干预 |
|------|------|----------|
| **5090** | LLM+MV 收尾 → UniSRec → **baseline 2024/2025/2026 全部补齐** | ❌ 无（watchdog 自动 tail） |
| **log10** | baseline 2024→2026 继续跑 | ❌ 无（多跑 OK） |

---

## 5090 自动收尾

```bash
# 5090 上（已部署则无需再跑）
nohup bash scripts/5090_auto_tail_watchdog.sh >> logs/5090_auto_tail_watchdog.log 2>&1 &
```

**watchdog 行为**:
1. 等 Toys MV seed=2026 checkpoint 出现
2. 若旧脚本卡在 `wait_and_pull` → **自动 kill**
3. 自动启动 `run_5090_phase4_tail_rebalanced.sh`

**tail 脚本顺序**:
```
pull log10 → Phase5 UniSRec → baseline 2024/2025/2026 (skip ckpt) → pull + collect → logs/5090_pipeline_all_complete.done
```

---

## 完成判定

```bash
# 5090 上
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
| `logs/5090_pipeline_all_complete.done` | 全部完成标记 |
| `logs/log10_gru4rec_baselines.log` | log10（可选参考） |

---

## ETA

5090 全部完成（含 baseline 2024–2026 + UniSRec）: **~1–1.5 天**  
log10 并行多跑不影响 5090 完成时间。
