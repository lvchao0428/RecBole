# 论文表 → 5090 运行 冻结对照（2026-11-11）

数字只从 CSV / JSON / log 重算后，和 tex 里已经印出来的格子对照。对不上就写未对上，不补一个“应该是”的数。

## 对照的是哪一版 tex

没有选定投稿版。skill 把 `main_wsdm2027_v3_0822.tex` 标成 stable mainline，把 `main_wsdm2027_v3_0823.tex` 标成后来的 de-AI 稿。本笔记不改这个判断。

| 文件 | 在哪 | sha256 前缀 | 本笔记怎么用 |
|---|---|---|---|
| `paper_recsys/main_wsdm2027_v3_0823.tex` | 本地和 5090 字节相同 | `ddd7740a87c21008`（60579 字节） | **下面的格子按这一份对。** |
| `paper_recsys/main_wsdm2027_v3_0822.tex` | 只在本地。5090 没有这个文件 | 未单列 | `tab:protocol` / `tab:perseed` 抽查行的数字与 0823 相同，加粗不同（0822 在 Beauty Random 上把 TF `.01239` 和 TF+MV `.01244` 都加粗）。0823 的变更说明写了主表格子未改。没有逐格证明两份 tex 每一格都相同。 |

0823 相对 0822 还有附录措辞和加粗差异。图文件名两边都是 `figures/fig_protocol_contrast_v2.pdf` 和 `figures/fig_cka_heatmap.pdf`。

## 代码版本（所有行共用）

训练 log 没有记下 git commit。

| 位置 | HEAD | 日期 | 说明 |
|---|---|---|---|
| 5090 `/home/charlie/project/RecBole` | `4c030be46f2e59878d9a04ea76f28ca374edc4d8` | 2026-07-21 | 分支 `exp1110`。工作区是脏的：相对这次 HEAD 有修改，8 月的论文脚本和结果大多是未跟踪文件。这不是一次干净的冻结提交。 |
| 本地 | `8c34cea4b5a1ff2fae47dc3b19b3d9e438fbe672` | 2026-11-11 13:33 | 分支 `exp1110`。共同祖先是 `7cad8c16`（2026-07-20）。本地这次提交是事后快照，**不能写成跑表时的 commit**。 |

两边历史分叉，没有做 merge，也没有互相覆盖工作区。抽查的三份结果 CSV 本地与 5090 哈希相同：`gts_last_eval_beauty_ts.csv` `5317d93417b51fae`，`gts_last_eval.csv` `d0d1e322b88e56a7`，`early60eq_gts_last_eval_ts_v2.csv` `b4ba5d8414aa2f40`。

没有 git 冲突标记（`<<<<<<<`）。文档里的 `=======` 分隔线不是冲突。

种子从这些记录里原样看到的，没有新加、没有删：训练 `42, 2024, 2026`；GTS-Random 的 target-sampling `42, 123, 456`。均值是 3 个训练种子的 mean，样本标准差 `ddof=1`；Random 先对每个训练种子的 3 个 sampling seed 取平均，再对 3 个训练种子做 mean/std。四位小数四舍五入后与 tex 对照。

路径都在 5090：`/home/charlie/project/RecBole/`。下面写相对路径。checkpoint 在 `saved/<tag>/<pth>`。

## 主表和主图

共同 cutoff，除非某行另写：Main 是 log 里的 `eval_args = {split: TS [0.8, 0.1, 0.1], order: TO, group_by: user, mode: full}`。评测是 warm target、`min_train_interactions=5`、候选 `train_catalog`。交互文件没有单独的 cutoff 时间戳文件。

| 表/图 | 协议 | 数据集 | 方法行 | 结果路径 | config（log 里的 `--config_files`，特征路径以 log 解析值为准） | 特征 |
|---|---|---|---|---|---|---|
| `tab:protocol` `tab:ndcg_hit` `tab:perseed` `fig:protocol` 的 GTS-Last / GTS-Random 文本三行 | GTS-Last；GTS-Random | Beauty | TF, TF+LLM, TF+MV | Last：`paper_recsys/gts_last_eval_beauty_ts.csv`。Random：`gts_random_eval_beauty_ts_seed{42,123,456}.csv` | TF `sasrec_align_base_stratified_v3_ts.yaml`；LLM `sasrec_align_qwen3_stratified_v3_ts.yaml`；MV `sasrec_align_multi_view_v3_stratified_ts.yaml`。`use_cross/use_seq_text_cross/use_multiview_text_cross=false`，boost=0。入口 `scripts/run_beauty_ts_maintable_0818.sh`（只在 5090） | TF：`dataset/Amazon_Beauty/item_text_emb.base.ts.npy`。LLM 另加 `item_text_emb.qwen2.5_7b.base.ts.npy`。MV 另加 `qwen2.5_7b_4views_ts/view_0..3.npy`。hash 见文末 |
| 同上，ID | 同上 | Beauty | ID | 同一套 Last 数字在 `paper_recsys/gts_last_eval.csv` 的 `wrap_beauty_id_*` 行（`.0032±.0006`，逐种子 `.0038/.0032/.0026`）。Random 的 ID 在 `gts_random_eval_seed{42,123,456}.csv` 的 Beauty ID 行（sampling 种子均值 `.0081/.0080/.0063`，总均值 `.0074±.0005`） | **未对上。** checkpoint 在 `saved/wrap_beauty_id_lr1e4_do0.1_seed{42,2024,2026}/`，pth 为 `SASRecAlign-Jul-23-2026_23-37-27.pth`、`Jul-24-2026_01-18-26.pth`、`Jul-24-2026_02-59-25.pth`。没有 `logs/wrap_beauty_id_*.log`。旁边的 `logs/ts_amazon_beauty_id_only_seed42.log` 和 `logs/ts_beauty_id_only_min5_seed42.log` 写的是别的 `checkpoint_dir`，不能算这一批 | ID 关闭文本。特征文件不适用 |
| 同上，UniSRec | 同上 | Beauty | UniSRec | Last / NDCG / Hit：`paper_recsys/gts_last_eval.csv` 的 `unisrec_beauty_gts_seed*`（Last `.0052±.0002`，逐种子 `.0050/.0054/.0051`）。Random：`gts_random_eval_seed*.csv` 里同名 tag（`.0096±.0001`） | `unisrec_beauty_stratified_ts.yaml`，命令行只覆盖了 seed。log 解析值 `item_text_emb_path = dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy` | 见上，hash 见文末。**不是** 8 月 23 日的 `unisrec_beauty_gts_ts_*` / `gts_last_eval_unisrec_beauty_ts.csv`（那份 Last 均值 `.0051±.0007`，对不上正文 `.0052±.0002`） |
| 同上 | 同上 | Toys | ID, TF, TF+LLM, TF+MV | `paper_recsys/gts_last_eval.csv` 的 `wrap_toys_*`。Random：`gts_random_eval_seed*.csv` 的 Toys 行（不含下面的 UniSRec） | ID `sasrec_baseline_50ep_stratified_toys_ts.yaml`（`disable_text_feature=true`）。TF `sasrec_align_toys_base_stratified_v3_ts.yaml`。LLM `sasrec_align_toys_qwen3_stratified_v3_ts.yaml`。MV `sasrec_align_multi_view_v3_toys_stratified_7b_ts.yaml`。cross/boost 关 | TF-IDF `dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy`。LLM 加 `item_text_emb.qwen2.5_7b.base.ts.npy`。MV 加 `qwen2.5_7b_4views_ts/view_0..3.npy` |
| 同上，UniSRec | 同上 | Toys | UniSRec | `paper_recsys/gts_last_eval_unisrec_toys_ts.csv` 与 `gts_random_eval_unisrec_toys_ts_seed{42,123,456}.csv`。Last `.0040±.0008`，逐种子 `.0044/.0045/.0031`。`gts_last_eval.csv` 里更早的 `unisrec_toys_gts_seed*` 是 `.0041±.0001`，**对不上** | `unisrec_toys_stratified_ts.yaml`，命令行把路径覆盖成 `.ts.npy`。log：`item_text_emb_path = .../item_text_emb.qwen2.5_7b.base.ts.npy`。旧的 7 月 28 日跑次解析值是 `item_text_emb.qwen2.5_7b.base.npy`，那一列不进表 | `.ts.npy`，hash 见文末 |
| 同上 | 同上 | Grocery | TF, TF+LLM, TF+MV | Last 的这三行在 `gts_last_eval_grocery.csv` 与 `gts_last_eval_grocery_v2.csv` 里数值相同。Random：`gts_random_eval_grocery_v2.csv` 配 `gts_random_eval_grocery_seed{123,456}.csv` | TF `sasrec_align_grocery_base_stratified_v3_ts.yaml`。LLM `sasrec_align_grocery_qwen_stratified_v3_ts.yaml`。MV `sasrec_align_multi_view_v3_grocery_stratified_7b_ts.yaml` | `dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.ts.npy`，LLM 加 `item_text_emb.qwen2.5_7b.base.ts.npy`，MV 加 `qwen2.5_7b_4views_ts/view_0..3.npy` |
| 同上，ID | 同上 | Grocery | ID | **逐种子和 NDCG/Hit 对上 `gts_last_eval_grocery.csv`，对不上 v2。** 同一个 pth `SASRecAlign-Aug-06-2026_07-05-46.pth`：csv 的 seed2026 warm MRR `0.004260`（正文 `.0043`），v2 是 `0.004226`。Hit `.0107±.0041`、NDCG `.0048±.0019` 只对上 csv。均值四位小数两边都还能写成 `.0030±.0012` | `sasrec_baseline_50ep_stratified_grocery_ts.yaml`，文本关 | 不适用 |
| 同上，UniSRec | 同上 | Grocery | UniSRec | `gts_last_eval_grocery_v2.csv` 的 `unisrec_grocery_gts_seed*`（8 月 14 日 pth）。NDCG `.0160±.0003`、Hit `.0338±.0007` 对上这一份。8 月 22 日 `gts_last_eval_unisrec_grocery_ts.csv` 的 Hit 均值是 `.0341`，**对不上** | `unisrec_grocery_stratified_ts.yaml`，命令行只覆盖 seed。log：`item_text_emb_path = dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.qwen2.5_7b.base.ts.npy` | 该 `.ts.npy` |
| `tab:protocol` 的 LOO 行 | LOO，`split: LS valid_and_test`。只作对照 | Beauty | ID, TF, TF+LLM, TF+MV | `paper_recsys/beauty_loo_results.csv` | ID log：`sasrec_baseline_50ep_stratified.yaml`。TF log：`sasrec_align_base_stratified_v3.yaml`，特征 `item_text_emb.base.npy`（不是 `.ts.npy`）。LLM/MV 的 yaml 名这次没有逐个打开 | TF 用 `item_text_emb.base.npy`。hash 见文末 |
| 同上 | 同上 | Toys | ID, TF, TF+LLM, TF+MV | `paper_recsys/toys_loo_results.csv` | ID log：`sasrec_baseline_50ep_toys_stratified.yaml` | 未逐个再打开 LLM/MV 的特征行 |
| 同上 | 同上 | Beauty, Toys | UniSRec | `paper_recsys/unisrec_loo_results.csv`。均值 Beauty `.0239`、Toys `.0196` 对上。样本标准差分别约 `0.00026` 和 `0.0013`，正文写成 `.0002` 和 `.0010` | **标准差这一位未对上。** 候选就是这一份 CSV，没有第二份更贴的文件 | 未从 LOO UniSRec log 再抄特征路径 |
| `tab:early` Main 行 | 与主表 Beauty Last 的 TF/LLM/MV 相同 | Beauty | TF, TF+LLM, TF+MV | 同 `gts_last_eval_beauty_ts.csv` | 同 Beauty 主表文本三行 | 同主表 |
| `tab:early` Early60-EQ | GTS-Last，`TS [0.6, 0.1, 0.1]` | Beauty | TF `.0042±.0005`，TF+LLM `.0037±.0003`，TF+MV `.0039±.0003` | `paper_recsys/early60eq_gts_last_eval_ts_v2.csv` | TF `sasrec_align_base_stratified_v3_ts_early60.yaml`；LLM `sasrec_align_qwen3_stratified_v3_ts_early60.yaml`；MV `sasrec_align_multi_view_v3_stratified_ts_early60.yaml`。5090 上这三份现在就是 `[0.6, 0.1, 0.1]`。本地 7 月 21 日的 `[0.6, 0.1, 0.3]` 已归档 | `item_text_emb.base.ts_early60.npy`，LLM 加 `item_text_emb.qwen2.5_7b.base.ts_early60.npy`，MV 加 `qwen2.5_7b_4views_ts_early60/view_0..3.npy` |
| `tab:shift` `tab:bucket` | Main GTS 的 all-target 目录统计，不是 warm MRR | Beauty, Toys | 无方法行 | `paper_recsys/distribution_shift_beauty_formal.json`、`distribution_shift_toys_formal.json`。流行度和 head/mid/low/unseen 与表内百分数一致（例如 Beauty 112.8→101.8、21→15；Toys test unseen 21.3%） | 不是一次训练。脚本未在 json 里写死 | 不涉及模型特征 |
| `tab:shift` Grocery 行 | 同上 | Grocery | 无方法行 | **未对上。** 没有 `distribution_shift_*grocery*formal.json`。`warm_shift_grocery_and_gourmet_food.json` 是另一套 per-user last 统计，数字不是 87.3→76.9、16→12。CSV 末尾的 `29.6` 是耗时秒数，不是 unseen 比例 | — | — |
| `fig:cka` `tab:cka` `tab:effrank` | 3 次 3000 item 子样本 | Beauty, Toys | 四个 prompt view 与 TF-IDF | 数值对上 `paper_recsys/cka_subsample_beauty.json`、`cka_subsample_toys.json`（Beauty Description–Function 均值 `0.8457` → `.846`） | 生成脚本 `cka_subsample.py`：若 `qwen2.5_7b_4views_ts` 存在就读它，否则读 `qwen2.5_7b_4views`。**json 本身没写当时的路径** | 候选是已哈希的 `qwen2.5_7b_4views_ts`。不能把候选写成已经证实的那一次输入 |
| `tab:lovo` | GTS-Last warm，zero-mask，3 种子均值 | Beauty, Toys | TF+MV 的四个 view | `paper_recsys/loov_gts_last_warm_{beauty,toys}_seed{42,2024,2026}.md`。Beauty Full 均值 `.00636`，Style 的 Δ `-.00032`。同目录不带 `warm` 的 md 是 GTS-All 量级（Beauty Full `0.016`），**不是这张表** | 用主表 MV checkpoint：`saved/wrap_{beauty,toys}_mv_lr5e4_do0.3_seed*` | 同主表 MV |
| `tab:cka-prepost` 与正文 raw 3584-d 句 | pre/post whiten；raw hidden | Beauty, Toys | 四个 view | 64-d 阶段对上 `cka_pre_vs_post_beauty.json`、`cka_pre_vs_post_toys_games.json` 这一档的表内区间需要再对一次字段；本次没有把 json 的每个键重算进表。raw 句 `.961/.963` 对上 `cka_raw_4096d_beauty.json`、`cka_raw_4096d_toys_games.json` 的文件名和已有记录，**json 内路径未再打开** | 候选目录 `dataset/<ds>/qwen2.5_7b_4views_raw_4096d/`（每份约 1.8–2.3 GB） | **未哈希。** 路径没有从 json 里钉死，所以不算冻结特征 |
| `tab:cost` | seed 42 的推荐阶段参数和耗时 | Beauty（Toys 只在表注里写约 87M） | ID / TF / TF+LLM / TF+MV / UniSRec | 数字对上 `paper_recsys/model_cost_table.csv`（67.16 / 67.43 / 67.56 / 67.71 / 67.69 M，0.02 与 0.04 ms） | **未对上主表 checkpoint。** TF/LLM/MV 的 tag 是 `sg_beauty_*`（7 月 20 日，0818 已记为 NON_TS），不是主表的 `wrap_beauty_*`。ID 和 UniSRec 的 tag 与主表 Beauty 相同 | 这次计时没有新的特征文件 |
| `tab:gts-all` | GTS-All warm MRR，`test_n_all` Beauty 12001、Toys 13186 | Beauty | TF `.02411`，TF+LLM `.0237`，TF+MV `.02413` | `paper_recsys/gts_all_eval_beauty_ts.csv`，checkpoint 与主表 `wrap_beauty_*` 相同 | 同 Beauty 主表文本三行 | 同主表 |
| 同上 | 同上 | Beauty, Toys | ID | `paper_recsys/warm_only_eval_min5.csv` 里 `wrap_*_id_*` 行（Beauty `.0156`，Toys `.0118`）。同一文件里的 Beauty TF 仍是 `sg_beauty_*`，那些行**不是**表里的 TF | 同各自 ID。Beauty ID 的 yaml 仍未对上 | ID |
| 同上 | 同上 | Toys | TF `.0191`，TF+LLM `.0180`，TF+MV `.0184` | 同一 `warm_only_eval_min5.csv` 的 `wrap_toys_{tf,llm,mv}_*` | 同 Toys 主表 | 同 Toys 主表 |
| 同上；`tab:unisrec-app` 的 GTS-All 列 | 同上 | Beauty `.0180±.0001`，Toys `.0139±.0003` | UniSRec | `paper_recsys/gts_all_eval_unisrec_ts.csv`。Beauty 是 7 月 28 日 `unisrec_beauty_gts_seed*`；Toys 是 8 月 19 日 `unisrec_toys_gts_ts_seed*`。和各自 Last 列是同一批 checkpoint | 同各自 UniSRec 主表行 | 同各自 UniSRec |
| `tab:grocery-perseed` | GTS-Last 与 GTS-Random | Grocery | ID, TF, TF+LLM, TF+MV | Last 见上：TF/LLM/MV 两份 grocery CSV 相同；ID seed2026 只对上 `gts_last_eval_grocery.csv`。Random 逐种子表的 `.0075/.0095/.0091` 等来自 grocery random CSV，本次没有把 9 个格子逐一重印，主表 Random 均值已经对上 | 同 Grocery 主表 | 同 Grocery 主表 |
| `tab:random-seeds` | 每个 target-sampling 种子上，再对 3 个训练种子取均值 | Beauty | TF/LLM/MV 对上 `gts_random_eval_beauty_ts_seed*.csv`（42 号：`.0138/.0133/.0134`）。ID 与 UniSRec 对上 `gts_random_eval_seed*.csv` 的 Beauty 行 | 见上 | 同 Beauty 主表对应行 | 同主表 |
| 同上 | 同上 | Toys | ID/TF/LLM/MV 对上 `gts_random_eval_seed*.csv` 的 Toys 行。UniSRec 对上 `gts_random_eval_unisrec_toys_ts_seed*.csv`（123 号 `.0078`，456 号 `.0075`；旧文件这两格是 `.0082/.0074`，对不上） | 见上 | 同 Toys 主表 | 同主表 |
| `tab:intersection` | 同一批 checkpoint 上，Last 与 Random 的 warm 用户交集 | Beauty, Toys, Grocery | ID, TF, TF+LLM, TF+MV | `paper_recsys/gts_intersection_eval.csv`，tag 与主表 checkpoint 同名（Beauty 文本是 `wrap_beauty_*`，Toys `wrap_toys_*`，Grocery `wrap_grocery_*`）。本次没有把交集列的 24 个均值全部重算 | 同主表 | 同主表 |

`fig:protocol` 的柱高就是 `tab:protocol` 的点估计，没有另一套结果文件。

## 明确不要拿来换表的后来跑次

0823 变更说明写过：Beauty/Grocery UniSRec 的 cutoff-fit 重训 CSV 出来之后再对表，表先不动。本次重算确认表确实没换成这些文件：

| 后来的文件 | checkpoint | 为什么不是表现在的格子 |
|---|---|---|
| `gts_last_eval_unisrec_beauty_ts.csv` 及对应 random | `UniSRec-Aug-23-2026_*`，`unisrec_beauty_gts_ts_seed*` | Last `.0051±.0007`，逐种子 `.0050/.0058/.0044`。正文是 `.0052±.0002` 和 `.0050/.0054/.0051` |
| `gts_last_eval_unisrec_grocery_ts.csv` | `UniSRec-Aug-22-2026_*` | Hit 均值 `.0341`。正文 Grocery UniSRec Hit 是 `.0338±.0007`，对上 8 月 14 日的 v2 |

## 特征与划分的 hash

sha256，2026-11-11 在 5090 上对当时磁盘文件计算。本地仓库没有这些 npy。

| 路径 | 字节 | sha256 |
|---|---:|---|
| `dataset/Amazon_Beauty/item_text_emb.base.ts.npy` | 132713088 | `29522897954355999b764f379f6105fa8443ae0d774af290889fbb44343ce9ac` |
| `dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy` | 132713088 | `f60328c6ab4726fff4fffa55701e214f168599f59a9f831d19f127a8f9f99b02` |
| `dataset/Amazon_Beauty/item_text_emb.base.ts_early60.npy` | 132713088 | `3706e2fce0332f77a4f5f8bbcd5ab63bb8cff6cc546ad6330f8efdf959cb26ee` |
| `dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts_early60.npy` | 132713088 | `21b6eef04b951516c981d4bd3a9d510e76e1e17e51a65810274c1ada278cd754` |
| `dataset/Amazon_Beauty/item_text_emb.base.npy` | 132713088 | `35accf793581b10aeb8ed307904700bcf9822faf2e39cb7152f4a5e029fb2d36` |
| `dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.npy` | 132713088 | `de2e29fc836d4e569445784232cd43885020c606c32547b3a4abb4ad3bc37b7d` |
| `dataset/Amazon_Beauty/qwen2.5_7b_4views_ts/view_0.npy` | 33178368 | `8fd3ef87e379be03faa1f41d25562f7cf3c96d3a5174dce102b368ee9206cc01` |
| `.../view_1.npy` | 33178368 | `6dae0e454ee050131f07d8477ddbb5a0a0f5a3bce9e0d7d73f1a3bf3e810a23a` |
| `.../view_2.npy` | 33178368 | `4979d31bf2b129d634095f4e8dc54af8f096a06d1a5ffd7bee8a8ca3a94d2a55` |
| `.../view_3.npy` | 33178368 | `b043235b567855a2b47c89d91e70103fdefa86010ee406f7550dca1e651a0b11` |
| `dataset/Amazon_Beauty/qwen2.5_7b_4views_ts_early60/view_0.npy` | 33178368 | `a941488bea1e7095985dc07b7a98ebf596e1a512a6194f9ca64ca18b40c36f9c` |
| `.../view_1.npy` | 33178368 | `694c3eb5f2c750f468ec2237afccb009e7b84ac1c35ffaa97df7769b76bce84d` |
| `.../view_2.npy` | 33178368 | `5f42bcc38bd6336b1cda8dd9e613d33478949522cbde4c65029eff2aa5c627d9` |
| `.../view_3.npy` | 33178368 | `25413f677cc30f554cdf5babfc6fde96d4822e76152335c724ccc4fa4500b3e8` |
| `dataset/Amazon_Toys_and_Games/item_text_emb.base.ts.npy` | 172069504 | `3c51c53bc116b1952ef8f98e398263d7a2f244c71ac6098728a2b9cdff673d89` |
| `dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.ts.npy` | 172069504 | `c3fc355c9bfa35e7075d74ee5d7e5fef31f95088ca846841f41084dfc209b2ca` |
| `dataset/Amazon_Toys_and_Games/item_text_emb.qwen2.5_7b.base.npy` | 172069504 | `e6e496982259edd088137ee229a48aee6ff435acae022537d0331d65f736114a` |
| `dataset/Amazon_Toys_and_Games/item_text_emb.base.npy` | 172069504 | `cf8f91b79011ac521124a74cdfde9440372e4abc48ba400e41e83ce1c10577df` |
| `dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_ts/view_0.npy` | 43017472 | `229fafeffd6d74e078f5c108f2ed020f710319ad04ea1f9583c2867de3df31f3` |
| `.../view_1.npy` | 43017472 | `07630d92339e6db1e92ada71dd895267812c78c3f46354105c80b269efd7e91b` |
| `.../view_2.npy` | 43017472 | `13e99a347c6d38e52918fe17b012eecf2ad6b57c691bb5058a1d1870105cb680` |
| `.../view_3.npy` | 43017472 | `522cd1b076634deb89110f724f746fb956c49da32036ed4d7506290d18b8e0e0` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.base.ts.npy` | 87941760 | `6f821eade6283961570ce9a38b52cac3f9126d8673864f4ecd3d901b75d05b7d` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/item_text_emb.qwen2.5_7b.base.ts.npy` | 87941760 | `47f23fc3a9b60a118c9cdeb8238dbd1fcb482ce28f3ac0519b3825379c2ac669` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/qwen2.5_7b_4views_ts/view_0.npy` | 21985536 | `f8a349a324285c55bdc166d38d3ac2d167de48e13e4a6bd32203fb9d30f32bd3` |
| `.../view_1.npy` | 21985536 | `92ff4a90485bbc367f4f449e9922caf3551b7bdbebbf3658424d0bde2f4bf99b` |
| `.../view_2.npy` | 21985536 | `929d80cc689fab803689be6a9ea2a418d3d9a03f8ee0a65a5b03c8bbf6c434f2` |
| `.../view_3.npy` | 21985536 | `98bddc8dd7b991cc947a400349496de5db32f5b1f7a7c769e2986f6d25d573f2` |
| Beauty 交互（符号链接目标 `/home/charlie/project/recbole/dataset/Amazon_Beauty/Amazon_Beauty.inter`，注意小写 `recbole`） | 82432221 | `2e24b454c4b17675b4bb697ba17d60c1de3de2fd6ec98fbae3e980cbb8e2aa9a` |
| `dataset/Amazon_Toys_and_Games/Amazon_Toys_and_Games.inter` | 91781980 | `219a39119c1007498e7b5e72bf654753f17700c7802a0440832e0bc1ba7e183b` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/Amazon_Grocery_and_Gourmet_Food.inter` | 52847599 | `7da49600683a36592d594c77722db14ed6db3be72f7b0d0f49738cf19c028cab` |

仓库里的 `dataset/Amazon_Beauty/Amazon_Beauty.inter` 是指向上述小写 `recbole` 目录的符号链接。Toys / Grocery 的 `.inter` 是本仓库里的普通文件，mtime 2020-10-31。

## 配置怎么处理的

抽查的主表 yaml（Beauty/Toys/Grocery 的 ts 与 unisrec、LOO 的 baseline）本地和 5090 哈希相同，没有互拷。

三份 Early yaml 是唯一一处“同一路径、一边是另一边的后继”：

- 本地旧内容只有 split `[0.6, 0.1, 0.3]` 和对应注释。
- 5090 在 2026-08-19 12:54 写成 `[0.6, 0.1, 0.1]`，与 Early 训练 log 一致。
- 旧文件在 `archive/20261111/config_superseded_early60_local_20260721/`。本地根上已换成 5090 的字节。5090 未覆盖。

`eval_gts_random.py` 只存在于 5090 根目录，且是 8 月 18 日主表 Random 的入口（比 `scripts/eval_gts_random.py` 多了“文件或目录都能收 checkpoint、按文件大小挑 pth”）。已补到本地同路径。`scripts/` 那一份两边相同，留着。

以下没有自动合并，留给人看：

- git：本地 `8c34cea` 与 5090 `4c030be` 从 `7cad8c16` 分叉。工作区里抽查过的已跟踪文件（`WSDM_experiment_spec.md`、`refs.bib`、`recbole_env.sh`、`scripts/analyze_*.py`、`tools/rebuild_qwen_emb_ts.py` 等）哈希已经相同。没有用 merge，也没有把一边的工作区整棵盖到另一边。
- Beauty ID 的训练 yaml 仍然未对上。
- Grocery `tab:shift` 仍然未对上。
- `tab:cost` 的 TF/LLM/MV 仍指向 `sg_beauty_*`，没有改成 `wrap_beauty_*`，也没有重测参数。
