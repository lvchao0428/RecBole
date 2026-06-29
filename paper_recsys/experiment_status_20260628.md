# 5090 + log10 状态（2026-06-28 · 自包含）

> **5090 跑完 = 全部完成** · log10 多跑无妨 · watchdog 已部署，无需半夜观测

---

## 当前

| 5090 | log10 |
|------|-------|
| Toys LLM/MV seed=2026 进行中 | Toys ID seed=2024 进行中 |
| watchdog 监控中 | 自由跑到 2026 |

---

## 自动收尾（5090）

- `scripts/5090_auto_tail_watchdog.sh` — Phase4 text 完成后自动 tail
- `run_5090_phase4_tail_rebalanced.sh` — UniSRec + baseline 2024/2025/2026
- 完成标记: `logs/5090_pipeline_all_complete.done`

---

## 一键检查（白天看一次即可）

```bash
ssh charlie@www.ultrapp.online 'cd ~/project/RecBole && \
  test -f logs/5090_pipeline_all_complete.done && echo "✅ ALL DONE" || echo "🔄 running"; \
  tail -3 logs/5090_auto_tail_watchdog.log; \
  bash scripts/collect_gru4rec_phase4_results.sh 2>&1 | grep 计数'
```

---

## ETA

**~6/29–6/30**（5090 全部完成，~1–1.5 天）
