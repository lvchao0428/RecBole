# 归档清单 2026-11-11

只移动，不删除。论文表用到的 yaml、评测脚本、结果 CSV、数据划分和 checkpoint 都不在这里。

## 已移动

| 现路径 | 原路径 | 为什么可以挪 | 日期 |
|---|---|---|---|
| `archive/20261111/Users_a58_nested_copy/a58/project/RecBole/`（8 个文件） | 仓库里的 `Users/a58/project/RecBole/` | 误嵌进仓库的副本，不是入口，也没有被 import。其中 7 个与仓库根上的同名文件字节相同。`scripts/analyze_rank_transition.py` 与 `scripts/analyze_rank_transition.py` 不同（嵌套副本 sha256 前缀 `f2d55343face78a0`，12182 字节；正式脚本 `d0417824f79558cf`，13352 字节），所以整份副本留在归档里，正式脚本留在 `scripts/`。 | 2026-11-11 |
| `archive/20261111/config_superseded_early60_local_20260721/` 下 3 个 `*ts_early60.yaml` | 仓库根上的同名文件 | 本地这三份 mtime 2026-07-21，split 仍是 `[0.6, 0.1, 0.3]`。5090 上同路径 mtime 2026-08-19，只把 split 改成论文 Early 用的 `[0.6, 0.1, 0.1]`，其余行相同。训练 log 里的 `eval_args` 也是 `[0.6, 0.1, 0.1]`。本地旧文件归档后，根上已换成与 5090 相同的字节。5090 原文件未动。 | 2026-11-11 |

5090 上同样把 `Users/a58/` 挪到了 `/home/charlie/project/RecBole/archive/20261111/Users_a58_nested_copy/`。两边这份嵌套副本的 `analyze_rank_transition.py` 哈希一致。5090 上的 Early yaml 没有挪，它们就是论文那次运行时的后继版本。

嵌套副本里的三份 `*ts_early60.yaml` 仍是旧的 `[0.6, 0.1, 0.3]`。不要把它们拷回仓库根。根上现在是 `[0.6, 0.1, 0.1]`。

## 补到本地、但不是归档

`eval_gts_random.py` 原来只在 5090 仓库根上（11866 字节，sha256 前缀 `da4c6e00afc6a8c1`，mtime 2026-08-18）。`scripts/run_beauty_ts_maintable_0818.sh` 调用的是这一份，不是 `scripts/eval_gts_random.py`。本地原来没有这个路径，现已按原路径补上。`scripts/eval_gts_random.py` 两边哈希相同，留在原处。

## 留在原处，未归档

- 主表 / Early / LOO 用到的根目录 yaml，以及 `scripts/eval_gts_last.py`、`scripts/eval_gts_random.py`、`scripts/two_phase_train.py`。
- 5090 上才有的 `scripts/run_beauty_ts_maintable_0818.sh`、`scripts/run_early60eq_fair_0818.sh`、`scripts/run_unisrec_toys_ts_0818.sh`、`scripts/audit_ckpt_protocol.py`、`sasrec_baseline_50ep_stratified_ts_early60_eq.yaml`。只存在于一侧，没有证明是废弃文件，不挪。
- 根目录 `exp*.sh`、`yelp_exp*.sh`、`debug_*.py` 一类早期脚本。看起来像调试残留，但没有逐个证明未被引用，不挪。
- `paper_recsys/main_wsdm2027_v3_0822.tex` 只在本地。`0823.tex` 两边哈希相同。都不归档。
- 数据划分、特征 npy、`saved/` checkpoint、结果 CSV、log。

## 5090 上同日较短清单（原文）

# 归档清单 2026-11-11（5090）

只移动，不删除。论文用的 yaml、评测脚本、CSV、划分和 checkpoint 都还在原处。

## 已移动

`Users/a58/project/RecBole/`（8 个文件）挪到 `archive/20261111/Users_a58_nested_copy/a58/project/RecBole/`。

这是误嵌进仓库的副本，不是入口。7 个文件与仓库根同名文件字节相同。`scripts/analyze_rank_transition.py` 与正式的 `scripts/analyze_rank_transition.py` 不同，所以整份留在归档里。正式脚本没动。

嵌套副本里的三份 `*ts_early60.yaml` 仍是旧 split `[0.6, 0.1, 0.3]`。不要拷回仓库根。根上的同名文件才是论文 Early 用的 `[0.6, 0.1, 0.1]`。

本地同一份清单在仓库的 `archive/20261111/README.md`。本地还另存了这三份旧 Early yaml 的副本，并补了一份原来只在 5090 上的 `eval_gts_random.py`。5090 的 `eval_gts_random.py` 没有动。

## 留在原处，未归档

主表 yaml、`scripts/run_*0818.sh`、`scripts/audit_ckpt_protocol.py`、`sasrec_baseline_50ep_stratified_ts_early60_eq.yaml`、`eval_gts_random.py`、数据、checkpoint、结果和 log。

## 2026-11-11 同步补记

上面「留在原处」里写的、当时只在 5090 上的 `scripts/run_beauty_ts_maintable_0818.sh`、`scripts/run_early60eq_fair_0818.sh`、`scripts/run_unisrec_toys_ts_0818.sh`、`scripts/audit_ckpt_protocol.py`、`sasrec_baseline_50ep_stratified_ts_early60_eq.yaml`，这次按原路径补到本地。没有改这些文件的字节。`paper_recsys/main_wsdm2027_v3_0823.tex` 没有覆盖。
