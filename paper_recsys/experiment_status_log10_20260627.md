# log10 实验状态（2026-06-27）

> **机器**: `charlie@192.168.0.107`（5090 内网 · `alias log10="ssh charlie@192.168.0.107"`）  
> **核对时间**: 2026-06-27 ~16:05  
> **GPU**: GTX 1080Ti · **99%** · Beauty TF-IDF seed=2024 Phase-A 早期

**主文档**: [`experiment_status_20260627.md`](experiment_status_20260627.md)

---

## 进度 4/16

| Seed | Beauty ID | Beauty TF-IDF | Toys ID | Toys TF-IDF |
|------|-----------|---------------|---------|---------------|
| 42 | ✅ 6/26 | ✅ | ✅ 6/27 | ✅ 6/27 |
| 2024 | — | 🔄 进行中 | 📋 | 📋 |
| 2025–2026 | 📋 | 📋 | 📋 | 📋 |

**脚本**: `run_log10_gru4rec_baselines.sh --all-seeds`（since 6/25 21:00）

---

## 断电暂停（可选）

若 log10 与 5090 同步断电：

```bash
nohup bash scripts/log10_pipeline_pause_after_current.sh \
  >> logs/log10_pipeline_pause_nohup.log 2>&1 &
tail -f logs/log10_pipeline_pause.log
```

**恢复**:

```bash
nohup bash run_log10_gru4rec_baselines_from_2024.sh \
  >> logs/log10_gru4rec_resume_from_2024_nohup.log 2>&1 &
```

---

## 监控

```bash
ssh charlie@192.168.0.107   # 或 5090 上: ssh log10
tail -f ~/project/RecBole/logs/log10_gru4rec_beauty_tfidf_seed2024.log
pgrep -af two_phase_train; nvidia-smi
```
