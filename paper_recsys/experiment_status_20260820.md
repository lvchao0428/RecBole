# 实验进展 2026-08-20

全文截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京 **Aug 25 19:59**）。摘要已交。

**时间锚点：北京 17:46。** `0820zhidao` 十条已全部写入正文；Last∩Random 交集检查已跑完并写入附录。RecBole GPU 空闲（5090 上另有 `qwen3.5-0.8b-sft-v67`，与本论文无关，不要 kill）。

稿本：`paper_recsys/main_wsdm2027_v3.tex` + `refs.bib` + `figures/fig_protocol_contrast.pdf`。快照说明：`VERSION_0820.md`。导师意见原文：`0820zhidao.txt`。

本机没有 `pdflatex`，PDF 需在 Overleaf 编译。

---

## 现在（17:46）

| 块 | 状态 |
|---|---|
| `0820zhidao` 十条（命名 / warm-shift / cohort / 引用 / LOO / UniSRec / 限定句 / Hybrid / cost） | **已写入** `main_wsdm2027_v3.tex` |
| **Last∩Random intersection-user check**（导师第 3 条的更干净选项） | **17:23 齐**，附录 `app:intersection` / `tab:intersection` 已写入 |
| 主表 Last/Random、Early v2、LOOV、Hybrid | 沿用 0819–0820 上午数字，**主表格子未因交集检查改动** |
| RecBole GPU | **空闲**（交集评测约 17:23 结束） |
| 论文 | 本地 906 行；已含交集附录。5090 下午 15:59 稿（854 行、无附录）**本文件写完后同步** |

**不要：** 把训练 log 的 GTS-All `test_mrr@10` 写进主表；不要动 5090 上的 `train_phase2.py`（非 RecBole）。

---

## 一、今天完成的两块

### 1. `0820zhidao` 十条（下午已落实，见 `VERSION_0819.md` 文末表）

要点（正文已是终稿口径，不必再改命名或 shift 表）：

1. 变体改名为 **TF / TF+LLM / TF+MV**（后两者是 TF-IDF base 上的增量，不是独立 LLM 表示）。
2. `tab:shift` / `tab:bucket` 改成 **warm 队列** 内统计（`scripts/analyze_warm_shift.py` → `warm_shift_{beauty,toys,grocery}*.json`）。
3. 3.1 节写明 Last/Random 同一 eligible 集合（Beauty 4,520 / Toys 4,873 / Grocery 4,152）；warm 子队列略有差别。
4–8. Intro 引用改写 + Recformer/FDSA；effective rank 补 `roy2007effective`；Clean LOO 写成 all-but-final-two；UniSRec 写成 adapted backbone；“MV leads” 加 SASRec-variants 限定。
9. Hybrid 实测写入（不再写 paused）：Last Beauty **.0070±.0006** / Toys **.0049±.0011**；Random **.0128±.0003** / **.0114±.0008**。
10. `tab:cost` 标题改为 recommendation-stage（offline encoding excluded）；正文补 TF-IDF+SVD ~1 min CPU vs Qwen 每视图一次全目录推理。

Fig.1 已按 TF+LLM / TF+MV 重画（`figures/fig_protocol_contrast.pdf`，15:21）。

### 2. Last∩Random 交集检查（今晚 GPU 空闲时跑完）

**问题：** warm filter 作用在 *选中的* target 上，Last 与 Random 的 warm 子队列不完全重合。Last→Random 排序翻转有没有可能是队列构成差异，而不只是 target selection？

**做法：** 新脚本 `scripts/eval_gts_intersection.py`，每个 ckpt **一次前向** 同时算 Last 与 Random（抽样种子 `{42,123,456}`），再限制到两边都 warm 的用户上重聚。默认环境即可跑（`/home/charlie/anaconda3/bin/python`）。

**覆盖：** Beauty/Toys/Grocery × ID/TF/TF+LLM/TF+MV × 训练种子 `{42,2024,2026}` = **36 ckpt × 3 抽样种子 = 108 行**，零失败。  
CSV：`paper_recsys/gts_intersection_eval.csv`（5090 上另有 `logs/gts_intersection_eval.csv`）。  
汇总：`paper_recsys/gts_intersection_summary_20260820.txt`。

探针已核对 Beauty TF seed42 与已发表 CSV 逐位一致：Last warm **0.004927**；Random seed 42/123/456 = **0.013666 / 0.013406 / 0.009367**；人数 Last 3137、Random 3054–3083。

**人群规模（与正文 3.1 一致）：**

| | eligible | warm Last | warm Random | **交集**（占 eligible） |
|---|---|---|---|---|
| Beauty | 4,520 | 3,137 | 3,054–3,083 | **2,846–2,858（63%）** |
| Toys | 4,873 | 3,007 | 2,932–2,963 | **2,651–2,686（54%）** |
| Grocery | 4,152 | 2,923 | 2,926–2,936 | **2,671–2,678（64%）** |

**Warm MRR@10（3 训练种子；Random 再对 3 个抽样种子平均）：**

| | | Last∩ | Last 全 warm | Random∩ | Random 全 warm |
|---|---|---|---|---|---|
| Beauty | ID | .0033±.0006 | .0032±.0005 | .0066±.0004 | .0074±.0004 |
| | TF | .0054±.0003 | .0053±.0003 | .0110±.0002 | .0124±.0002 |
| | TF+LLM | .0063±.0004 | .0060±.0003 | .0107±.0003 | .0120±.0001 |
| | TF+MV | **.0065±.0003** | **.0064±.0002** | **.0113±.0002** | **.0124±.0002** |
| Toys | ID | .0032±.0001 | .0031±.0001 | .0073±.0001 | .0076±.0001 |
| | TF | .0039±.0005 | .0040±.0004 | **.0110±.0003** | **.0113±.0003** |
| | TF+LLM | .0035±.0002 | .0035±.0002 | .0106±.0005 | .0108±.0006 |
| | TF+MV | **.0040±.0004** | **.0042±.0005** | .0105±.0008 | .0108±.0008 |
| Grocery | ID | .0031±.0010 | .0030±.0010 | .0047±.0010 | .0050±.0010 |
| | TF | .0049±.0005 | .0046±.0005 | .0083±.0010 | .0087±.0009 |
| | TF+LLM | .0050±.0004 | .0047±.0004 | .0071±.0002 | .0076±.0003 |
| | TF+MV | **.0063±.0005** | **.0058±.0005** | **.0087±.0007** | **.0089±.0006** |

**结论（六个 dataset×mode 块全部成立）：**

1. 限制到交集后绝对值最多差 **.0014**（Beauty Random 上 TF .0124→.0110），ID / TF / TF+LLM / TF+MV **排序不变**。
2. 我们依赖的 Last→Random 变化在同一批用户上仍然成立：Last 下 TF+MV 三个数据集领先；Random 下 TF 在 Beauty（.0110 vs .0113）和 Grocery（.0083 vs .0087）追到 1 std 内，并在 Toys 反超（.0110 vs .0105）。Random 对每个变体仍大约是 Last 的两倍容易。
3. 因此翻转是 **纯 target-selection 效应**，不是队列构成差异。主表数字 **不用改**。

正文改动：3.1 Leakage controls 指向附录；Discussion「Why do Last and Random disagree?」加一句；新增 `app:intersection` + `tab:intersection`。

---

## 二、论文当前口径（0820 终稿，主表未动）

主协议：GTS-Last warm、min5、train-catalog candidates；GTS-Random 为稳健性（3 个抽样种子）；Clean LOO 为对比。

- 主表 Beauty/Toys/Grocery：cutoff-fit、no-cross、`[0.8,0.1,0.1]`（`wrap_*`）。
- `tab:early` Early 行：v2，cutoff-fit、no-cross、`[0.6,0.1,0.1]`。
- 变体名：TF / TF+LLM / TF+MV。
- Hybrid：公平协议已写入，不是 paused。
- UniSRec：adapted backbone；`tab:gts-all` Beauty/Toys 已是 cutoff-fit warm（含 UniSRec 列 .0180 / .0139）。

---

## 三、还剩什么（都不是门槛）

| 项 | 要不要做 | 说明 |
|---|---|---|
| **Overleaf 编译** | **要** | 核对 `tab:intersection` 表宽、总页数、引用。本地/5090 都没有 texlive |
| Discussion Hybrid / RQ2 warm-shift 措辞过一眼 | 建议 | 0820 下午改过，今晚没再动这两段 |
| GTS-Successive（sampled-user） | 导师：有余量再做 | 不阻塞 |
| Grocery CKA float32 | 可选 | raw 3584-d 曾因 float16 失败 |
| Early-Random / Toys·Grocery Early | **不要** | 导师已否 |

不要重跑、不要写进附录当主结论：0817 Hybrid（NON_TS+cross+boost）；`sg_beauty_*` 当 cutoff-fit；Grocery LOO。

---

## 四、路径速查

| 用途 | 路径 |
|---|---|
| 今晚论文 | `paper_recsys/main_wsdm2027_v3.tex` |
| 稿本说明 | `paper_recsys/VERSION_0820.md` |
| 导师意见 | `paper_recsys/0820zhidao.txt` |
| 交集 CSV / 汇总 | `paper_recsys/gts_intersection_eval.csv`、`gts_intersection_summary_20260820.txt` |
| 交集脚本 | `scripts/eval_gts_intersection.py`、`scripts/agg_gts_intersection.py` |
| ckpt 清单（5090） | `logs/gts_intersection_ckpt_list.txt` |
| warm-shift | `paper_recsys/warm_shift_*.json`、`scripts/analyze_warm_shift.py` |
| Fig.1 | `paper_recsys/figures/fig_protocol_contrast.pdf` |
| Overleaf 目录 | `paper_recsys/overleaf_deploy/`（`main.tex` 已与 v3 对齐） |
