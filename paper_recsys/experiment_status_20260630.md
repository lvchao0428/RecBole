# 5090 + log10 实验状态（2026-06-30）

> **5090**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **log10**: `charlie@192.168.0.107:/home/charlie/project/RecBole`（经 5090 内网检查）  
> **核对时间**: 2026-06-30 07:23 CST  

**相关文档**:

- 任务分配: `pipeline_task_allocation_20260628.md`
- 结果矩阵: `gru4rec_phase4_matrix_20260630.md`
- GRU4Rec 四配置结果: `gru4rec_results_snapshot_20260629.md`
- 完成总览: `experiment_completed_snapshot_20260629.md`

---

## 1. 当前结论

```text
[✅] 5090 主线任务全部完成
[✅] GRU4Rec Phase 4 checkpoint = 32/32
[✅] GRU4Rec LLM / MV checkpoint 补评完成并已拉回本地
[✅] UniSRec Toys seed=2024 已完成并同步到本地
[🔄] log10 队列已停，但仍残留 1 个 duplicate 训练任务（Toys ID seed=2025）
```

---

## 2. 5090 当前状态

| 项目 | 状态 | 备注 |
|------|------|------|
| GPU | ✅ 空闲 | `0%`, `508 MiB / 32607 MiB` |
| 完成标记 | ✅ | `logs/5090_pipeline_all_complete.done` 已存在 |
| GRU4Rec text eval | ✅ | `gru4rec_text_eval_summary_20260629.md` 已生成 |
| GRU4Rec baseline | ✅ | `2024/2025/2026` 全部补齐 |
| UniSRec Toys | ✅ | `saved/unisrec_toys_stratified_seed2024/UniSRec-Jun-29-2026_20-07-57.pth` |

`5090_phase4_tail_rebalanced.log` 收尾信号：

- `ALL COMPLETE — 5090 pipeline done`
- `collect_gru4rec_phase4_results.sh` 已生成 `gru4rec_phase4_matrix_20260630.md`

---

## 3. GRU4Rec 总进展

来自 `gru4rec_phase4_matrix_20260630.md`：

- text (`LLM + MV`): `16/16`
- baseline (`ID + TF-IDF`): `16/16`
- 合计 checkpoint 块: `32/32`

当前正式结果表也已完整：

- `gru4rec_results_snapshot_20260629.md`
- `gru4rec_text_eval_summary_20260629.md`

---

## 4. log10 当前状态

### 4.1 队列状态

`log10_pipeline_pause.log` 显示：

- `2026-06-30 04:58:24` 当前 target job 已结束
- `2026-06-30 04:58:24` pause watcher 已执行 `Stopping log10 baseline queue PID=3077`
- `2026-06-30 04:58:27` 已写出 `Safe to power off log10 now`

所以：

- **log10 队列已经停掉**
- 但**不是完全空闲**

### 4.2 当前残留任务

当前仍在跑的进程：

- `python scripts/two_phase_train.py ...`
- 任务口径: `GRU4Rec Toys ID-only seed=2025`
- PID: `112561`
- 运行时长: `02:27:32`（核对时）
- GPU: `97%`, `3578 MiB / 11264 MiB`

这说明 log10 上还残留了一个已经起跑的 duplicate job。它**不在关键路径上**，因为对应 checkpoint 已经由 5090 侧补齐。

---

## 5. 三端同步状态

| 端 | 状态 | 已同步内容 |
|----|------|------------|
| 本机 | ✅ | `gru4rec_text_eval_summary_20260629.md`、`gru4rec_text_eval_20260629/*.txt`、`gru4rec_phase4_matrix_20260630.md`、`UniSRec Toys` checkpoint/log、关键 `log10` 状态日志 |
| 5090 | ✅ | 最新脚本、Phase 4 收尾日志、`DONE` 标记、GRU4Rec 补评结果 |
| log10 | ✅ | 通过 5090 已核对当前进程、pause watcher、最近日志；关键日志已回拉到本机 |

---

## 6. 当前建议

### 建议 A（最稳）

- 保持现状
- 让 log10 上这一个残留 `Toys ID seed=2025` 自己跑完
- 由于主结果已经在 5090 收口，不影响论文表和总进度

### 建议 B（若想彻底收口）

- 手动停掉 log10 当前这个 duplicate job
- 然后 log10 就可以视为真正 idle

当前从实验完整性角度看，**A 已经足够**。
