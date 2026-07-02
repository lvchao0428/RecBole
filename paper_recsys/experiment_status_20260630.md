# 5090 + log10 实验状态（2026-06-30）

> **5090**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **log10**: `charlie@192.168.0.107:/home/charlie/project/RecBole`（经 5090 内网检查）  
> **核对时间**: 2026-06-30 21:26 CST  

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
[✅] log10 当前已无活跃训练进程
```

---

## 2. 5090 当前状态

| 项目 | 状态 | 备注 |
|------|------|------|
| GPU | ✅ 空闲 | `0%`, `508 MiB / 32607 MiB` |
| 完成标记 | ⚠️ | `done` 文件当前未见，但收尾日志已明确写出 `ALL COMPLETE` |
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

### 4.2 当前是否还有残留任务

最新核对结果：

- `pgrep` 未发现 `two_phase_train.py / run_log10_gru4rec_baselines / run_log10_resume_after_poweroff`
- `nvidia-smi` 显示 `0%`, `88 MiB / 11264 MiB`

因此，**log10 现在也已经空闲**。

补充说明：

- 早上检查时曾看到一个 `Toys ID seed=2025` 的 residual duplicate job
- 但按今晚再次核对，它现在已经结束，不再占用 GPU
- 该任务本来就不在关键路径上，因为对应 checkpoint 已经由 5090 补齐

---

## 5. 三端同步状态

| 端 | 状态 | 已同步内容 |
|----|------|------------|
| 本机 | ✅ | `gru4rec_text_eval_summary_20260629.md`、`gru4rec_text_eval_20260629/*.txt`、`gru4rec_phase4_matrix_20260630.md`、`UniSRec Toys` checkpoint/log、关键 `log10` 状态日志 |
| 5090 | ✅ | 最新脚本、Phase 4 收尾日志、GRU4Rec 补评结果；当前 GPU 空闲 |
| log10 | ✅ | 通过 5090 已核对 pause watcher、最近日志；当前无活跃训练进程 |

---

## 6. 当前建议

当前从训练收口角度看，已经可以视为：

- `5090` 空闲
- `log10` 空闲
- 主结果与补评结果都已同步到本机

也就是说，**当前训练已经收口**。
