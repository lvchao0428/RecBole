# WSDM 2027 稿本 2026-08-20 晚

源文件：`main_wsdm2027_v3.tex` + `refs.bib` + `figures/fig_protocol_contrast.pdf`

**今晚改稿先读** `experiment_status_20260820.md`。

本机没有 `pdflatex`，PDF 需在 Overleaf 编译。

## 相对 0819 / 0820 下午多了什么

- `0820zhidao` 十条下午已写入（命名、warm-shift、cohort 人数、引用、LOO、UniSRec、限定句、Hybrid、cost）。细节见 `VERSION_0819.md` 文末表。
- **今晚新增：** Last∩Random 交集检查跑完，附录 `app:intersection` / `tab:intersection` 写入。3.1 与 Discussion 各加一句指向。主表格子 **未改**。

## 已写入、可当主结论用的数字

- `tab:protocol` / `tab:ndcg_hit` / `tab:perseed` / `fig:protocol`：GTS-Last / GTS-Random warm，cutoff-fit，no-cross。
- `tab:early`：Main 与主表 Last 一致；Early 为 v2（cutoff-fit、no-cross、`[0.6,0.1,0.1]`）。
- `tab:intersection`（新）：同一批 warm 交集用户上 Last vs Random；排序不变，翻转是 target selection。
- CKA / effective rank / LOOV / Hybrid / cost：可用。
- `tab:gts-all` Beauty/Toys 含 UniSRec warm 列（.0180 / .0139）。

## 数据文件

- 交集：`gts_intersection_eval.csv`（36 ckpt × 3 sampling seeds）
- warm-shift：`warm_shift_{beauty,toys_and_games,grocery_and_gourmet_food}.json`
- Hybrid：`hybrid_ts_gts_last.csv` + `hybrid_ts_gts_random_seed{42,123,456}.csv`

## 尚未做（非门槛）

- Overleaf 编译：核对 `tab:intersection` 表宽和总页数。
- GTS-Successive sampled-user；Grocery CKA float32。
- 不要做：Early-Random、Toys/Grocery Early、Grocery LOO。
