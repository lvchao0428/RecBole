# 归档清单 2026-11-11（5090）

只移动，不删除。论文用的 yaml、评测脚本、CSV、划分和 checkpoint 都还在原处。

## 已移动

`Users/a58/project/RecBole/`（8 个文件）挪到 `archive/20261111/Users_a58_nested_copy/a58/project/RecBole/`。

这是误嵌进仓库的副本，不是入口。7 个文件与仓库根同名文件字节相同。`scripts/analyze_rank_transition.py` 与正式的 `scripts/analyze_rank_transition.py` 不同，所以整份留在归档里。正式脚本没动。

嵌套副本里的三份 `*ts_early60.yaml` 仍是旧 split `[0.6, 0.1, 0.3]`。不要拷回仓库根。根上的同名文件才是论文 Early 用的 `[0.6, 0.1, 0.1]`。

本地同一份清单在仓库的 `archive/20261111/README.md`。本地还另存了这三份旧 Early yaml 的副本，并补了一份原来只在 5090 上的 `eval_gts_random.py`。5090 的 `eval_gts_random.py` 没有动。

## 留在原处，未归档

主表 yaml、`scripts/run_*0818.sh`、`scripts/audit_ckpt_protocol.py`、`sasrec_baseline_50ep_stratified_ts_early60_eq.yaml`、`eval_gts_random.py`、数据、checkpoint、结果和 log。
