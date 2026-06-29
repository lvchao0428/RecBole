# 5090 + log10 实验状态（2026-06-29）

> **5090**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **log10**: `charlie@192.168.0.107`（5090 内网 · `scripts/log10_env.sh`）  
> **核对时间**: 2026-06-29 15:45 CST（收尾检查 · 三端同步）

**相关文档**:
- 任务分配: [`pipeline_task_allocation_20260628.md`](pipeline_task_allocation_20260628.md)
- 导师指导: [`0626zhidao.txt`](0626zhidao.txt)
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- GRU4Rec 矩阵: [`gru4rec_phase4_matrix_20260629.md`](gru4rec_phase4_matrix_20260629.md)

---

## 1. 当前进度

```
[✅] SASRec 主表 · Grocery 4-seed · BC · Phase 3 机制分析
[✅] UniSRec balanced（Base/Align/MV · Beauty seed=2024）
[✅] Phase 4 GRU4Rec text (5090 LLM+MV)                 16/16
[🔄] Phase 4 GRU4Rec baseline                           14/16 · 合计 30/32
[⏸] log10 baseline                                     当前重复 job 后自动停
[🔄] 5090 独占 seed=2026                               仅剩 Toys 2026 ×2
```

| 任务 | 状态 | 备注 |
|------|------|------|
| GRU4Rec LLM+MV (5090) | ✅ **16/16** | 已完成 |
| GRU4Rec baseline | 🔄 **14/16** | seed **2026** 只剩 `Toys ID` + `Toys TF-IDF` |
| 5090 当前 | 🔄 | `Beauty TF-IDF 2026` 已出 ckpt，正在向 `Toys 2026` 推进 |
| log10 当前 | ⏸ | 仍在跑重复的 `Beauty TF-IDF 2025` |
| log10 停机 | ✅ | `log10_pipeline_pause_after_current.sh` 已挂，另补 direct stop watcher |

---

## 2. 收尾判断

- **5090 当前训练健康**：GPU 约 **96%**，checkpoint 正常新增，已从 `Beauty ID 2026` 顺利推进到 `Beauty TF-IDF 2026`。
- **今晚可正常结束**：如果不再出现新异常，剩余仅 `Toys 2026` 两块，ready 目标仍可按 **约 20:00 左右** 预期。
- **log10 不再作为关键路径**：它现在只是在消化一个重复任务，结束后应自动停队列，不影响最终 ready。

---

## 3. log10 停重复任务

| 动作 | 状态 |
|------|------|
| 常规 watcher | `log10_pipeline_pause_after_current.sh` 已启动 |
| 加固 watcher | 额外按当前 `two_phase_train` PID 启动 stop-after-current |
| 目标 | 当前 `Beauty TF-IDF 2025` 结束后，停止 `run_log10_gru4rec_baselines_from_2024.sh` |

**勿再手动启动** `run_log10_resume_after_poweroff.sh` / `run_log10_gru4rec_baselines_from_2024.sh`。

---

## 4. 三端同步

| 机器 | 状态 |
|------|------|
| 本机 | ✅ 最新文档已写入 |
| 5090 | ✅ 已同步最新状态与脚本 |
| log10 | ✅ 已同步文档与暂停脚本 |

---

## 5. 监控

```bash
ssh charlie@www.ultrapp.online 'cd ~/project/RecBole && \
  bash scripts/collect_gru4rec_phase4_results.sh | tail -10; \
  nvidia-smi; \
  pgrep -af "5090_gru4rec|two_phase_train" | grep -v pgrep | head -3'
```

完成标记：`logs/5090_pipeline_all_complete.done`
