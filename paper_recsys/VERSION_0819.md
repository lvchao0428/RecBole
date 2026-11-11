# WSDM 2027 稿本 2026-08-19（0820 中午更新：0820zhidao 十条已全部落实，见文末）

**0820 晚：** 交集检查已跑完并写入附录，当前稿本说明改看 `VERSION_0820.md` / `experiment_status_20260820.md`。

源文件：`main_wsdm2027_v3.tex` + `refs.bib` + `figures/fig_protocol_contrast.pdf`

**今晚改稿先读** `NIGHT_HANDOFF_0819.md`（Early60 v2 口径、主表不动、CSV 齐了怎么换 `tab:early`）。

本机没有 `pdflatex`，PDF 需在 Overleaf 或 5090 编译。

## 已写入、可当主结论用的数字

- `tab:protocol` / `tab:ndcg_hit` / `tab:perseed` / `fig:protocol`：GTS-Last / GTS-Random warm，cutoff-fit，no-cross。Beauty TF/LLM/MV 为今早 `wrap_beauty_*`；Toys UniSRec 为 `unisrec_toys_gts_ts_*`。
- `tab:early`：Main 行与主表 Last 一致；**Early 行已换成 v2**（20:46 CSV）TF **.0042±.0005** / LLM .0037±.0003 / MV .0039±.0003。匹配协议：cutoff-fit、no-cross、`[0.6,0.1,0.1]`。Main 上 MV 领先，Early 上 TF 略高（1 std 内）。
- CKA / effective rank / raw 3584-d audit：可用。
- Discussion Pure LLM：GTS-Last Beauty .0046 / Toys .0018。

## 本版故意没有当成 GTS-Last 主协议的

- `tab:loov`：GTS-Last warm 3-seed（`wrap_*_mv_*`，zero-mask）。Beauty Full **.00636** / Toys **.00417**，与主表 MV 对齐。最大 |Δ| **0.00032**（Beauty Style）。已去掉 pending hedge。
- 附录 `tab:gts-all` Beauty TF/LLM/MV：cutoff-fit `wrap_beauty_*` GTS-All warm（23:40）**.0241 / .0237 / .0241**；Toys wrap 未改。UniSRec 列仍是 RecBole all-target，不是 warm。
- Hybrid：公平重训已暂停，无增益句；结论改为 withhold。

## 附录还要不要跑（2026-08-19）

**只 eval、不重训（GPU 空时值得做）：**
1. ~~Beauty GTS-All warm `wrap_beauty_*`~~ **23:40 已写入** TF/LLM/MV **.0241 / .0237 / .0241**。
2. （可选）UniSRec GTS-All **warm**：Beauty `unisrec_beauty_gts_*` + Toys `unisrec_toys_gts_ts_*`，避免附录 UniSRec 与 TF 列口径混用。当前已在 caption/脚注写明，不做也能交。
3. ~~Grocery GTS-Random 补 sampling seeds {123,456}~~ **00:13 已补齐并写入**（见下）。

**要重训才有数（不是门槛，导师：有余量再做）：**
4. 公平 Hybrid（cutoff-fit、no-cross）：Beauty seed42 可能已有 `hybrid_ts_beauty_seed42`；Toys seed42 不完整。要 3-seed Last/Random 才能进附录。
5. GTS-Successive（sampled-user）。
6. Grocery CKA（raw 3584-d 曾因 float16 失败，可改 float32 重算，不重训）。
7. Early-Random / Toys·Grocery Early：导师说不要。

**不要重跑、不要写进附录当主结论：**
- 0817 Hybrid（NON_TS + cross + boost）的 +20% / +83%。
- `sg_beauty_*` 当 cutoff-fit。
- Grocery LOO（导师：Last/Random 复制即可）。

## 引用（对照 ACM / DBLP / 导师给的 Scholar BibTeX）

0819 之前已改：

| key | 问题 | 修正 |
|---|---|---|
| `gusak2025time` | 旧稿写成 liang2024、年份/会议错 | RecSys 2025, pp. 874–883, doi 10.1145/3705328.3748164（与导师 Scholar 条目一致） |
| `bao2023tallrec` | 页码 100–110 | **1007–1014**（ACM RecSys ’23） |
| `zhao2021recbole` | DOI `…3482010` | **`…3482016`**（CIKM 2021 pp. 4653–4664） |
| `qwen2024qwen25` | note 写成 Model Studio 产品说明 | 收成 arXiv:2412.15115, Qwen Team, 2024 |
| `kornblith2019cka` | 缺 PMLR volume | volume **97**, pp. 3519–3529 |

**0819 深夜第二轮核查又改：**

| key | 问题 | 修正 |
|---|---|---|
| `sun2019bert4rec` | 页码 1443–1452（错） | **1441–1450**（CIKM ’19，ACM DL / DBLP 一致） |
| `hou2023vqrec` | 页码 2372–2383（错） | **1162–1171**（WWW ’23） |
| `xu2013multiview` | `note` 是自撰评语，不是文献字段 | 删掉，补 arXiv DOI |
| `liang2024timetosplit` | 重复别名条目，无人引用 | 删除（脚本已核：正文无引用、bib 无孤儿条目） |

bib 里的过程注释（"Supplement 2026-06-18…"、"Alias kept so older drafts…"）已清掉。

自动核过：`\cite` 键全部能解析，bib 无未被引用条目，`\begin/\end` 与花括号配平。

核对过、未改：Kang ICDM 2018 pp. 197–206；Krichene KDD 2020 pp. 1748–1757；Wang CDL KDD 2015 pp. 1235–1244；Hou UniSRec KDD 2022 pp. 585–593；Geng P5 RecSys 2022 pp. 299–315；Salton IPM 24(5) 513–523；Ethayarajh EMNLP-IJCNLP 2019 pp. 55–65；Rendle BPR UAI 2009 pp. 452–461；Hidasi CIKM 2018 pp. 843–852；Meng RecSys 2020 pp. 681–686；Ji TOIS 41(3) 75:1–27 doi 3569930；Sun SIGIR 2023 pp. 2629–2638；Hidasi GRU4Rec ICLR 2016；McAuley SIGIR 2015 pp. 43–52；Ni EMNLP 2019 pp. 188–197。

正文对 Time-to-Split 的陈述已对齐原文 Table 6：Random 与 Successive 相关最高；Last 次之；All 最低；Random 非确定性故平均 3 个 sampling seed。

## 数字前后一致性核查（0819 深夜，逐格对 CSV/JSON 重算）

**改掉的真错：**

| 位置 | 原文 | 实际 |
|---|---|---|
| §RQ1 text ×2 + Practitioner | LOO 比 Last 高 **3–6×** | **4–7×**（实测 3.95–7.37；Toys LLM 7.4×） |
| §RQ1 Implications | 文本比 ID **+29–100%** | **+13–100%**（Toys LLM 只 +12.9%） |
| `tab:cka` caption | TF vs LLM "stays below 0.20" | Beauty 最大 **.201**，改成 "within 0.18–0.20" |
| 摘要/Contribution/Synthesis/Discussion/结论 | CKA **≈0.19**，但图注写 ≈0.20 | 统一成 **0.18–0.20**；正文给实测 .196–.201 / .184–.189 |
| RQ3 whitening | post-whiten CKA 0.87、降 ~0.09 | Beauty **0.88**、降 **0.08**；Toys 0.87、降 0.10 |
| `tab:gts-all` Toys LLM | .0181 | **.0180** |
| `tab:protocol` Grocery UniSRec Last | .0106±.0001 | **.0107±.0002**（截断而非四舍五入） |
| `tab:ndcg_hit` Grocery UniSRec | NDCG .0159±.0002 / Hit .0337±.0005 | **.0160±.0003 / .0338±.0007** |
| `tab:ndcg_hit` Toys UniSRec Hit | .0153 | **.0154** |
| `tab:cka-prepost` vs `tab:effrank` | Toys post-whiten 140 vs 141 | 两个独立子样本，caption 已注明 |
| `app:unisrec` | Toys GTS-All 与 Last/Random 混了 ts / 非 ts ckpt | 加脚注说明 |
| **`tab:protocol` Grocery GTS-Rand 整行** | 只用了 sampling seed 42，但 caption 声明 3 seed | **补跑 123/456**（00:13 完），换成真 3-seed |

**Grocery Random 补齐后（`gts_random_eval_grocery_seed{123,456}.csv`）：**

| | ID | TF | LLM | MV | UniSRec |
|---|---|---|---|---|---|
| 旧（仅 seed42） | .0044 | **.0085** | .0079 | **.0085** | .0118 |
| 新（42/123/456） | .0050±.0012 | .0087±.0011 | .0076±.0004 | **.0089**±.0008 | .0121±.0002 |

Grocery Random 赢家由「TF 与 MV 并列」变成 **MV 微弱领先 .0002**（远小于 1 std）。已连带改：`tab:protocol` caption、RQ1 bullet、Practitioner #2、结论第 1 条、`tab:gts-all` 讨论句、`tab:grocery-perseed`（Random 行改为 3-seed 均值 + caption 注明）。**主结论未变**：Last 三个数据集都是 MV 略高；Random 上 TF 追平（Toys 反超），所有差距在 1 std 内。

**逐格重算无误：** `tab:protocol` Beauty/Toys 全部 mean±std；`tab:perseed` 18 格；`tab:ndcg_hit` Beauty/Toys；`tab:loov` 10 格（Beauty Full .00636、Toys .00417、max |Δ| .00032）；`tab:cka` 12 格；`tab:effrank`；`tab:cka-prepost`；raw 3584-d（.961/.963、1567/1622）；`tab:random-seeds` 30 格；`tab:shift` / `tab:bucket`；Grocery per-seed；Pure LLM .0046±.0001 / .0018±.0007；cosine −0.01 / 0.42；数据集规模。

**注意：** Grocery Last 用 `gts_last_eval_grocery.csv`（非 v2），UniSRec Grocery 只在 v2 里；两份 CSV 的 Grocery ID seed2026 相差 .004260 / .004226，正文取非 v2。改这一行前先确认取哪份。

## 0820zhidao 落实记录（2026-08-20 下午）

| # | 建议 | 落实 |
|---|---|---|
| 1 | TF/LLM/MV 命名歧义 | 全文改为 **TF / TF+LLM / TF+MV**（宏 `\tfllm` `\tfmv`）；3.2 明写 LLM 变体是"增量到 TF base"；LLM-only ablation 段落改为解释命名依据；7 张表表头 + Fig.1 图例已换 |
| 2 | shift 证据与 warm 协议不齐 | 按"更好"选项重算：`scripts/analyze_warm_shift.py` → `warm_shift_{beauty,toys,grocery}.json`。`tab:shift` 改为 warm share + warm 目标流行度（84.8→72.1% 等）；`tab:bucket` 改为 warm 桶内 head/mid/low（无 unseen 行）；unseen 27.9/34.6/30.6%（last-target 口径）降级为 catalog-level context |
| 3 | Last/Random cohort | 3.1 Leakage controls 补 eligible=4520/4873/4152、warm 3137 vs 3054–3083 等；Implications 加 "same eligible users" |
| 4 | Intro 引用句 | 改写为 pretrained language representations…transferability，加 `li2023recformer`、`zhang2019fdsa`（bib 不写不确定字段） |
| 5 | effective rank 出处 | `roy2007effective`（导师给的 bib 原文），Related Work + Intro 两处引用 |
| 6 | Clean LOO 描述 | 精确写成 all-but-final-two / penultimate / final（RecBole LS） |
| 7 | UniSRec 描述 | 改为 "UniSRec backbone adapted to our cutoff-fit features"：MoE adaptor 原架构、从零训练、无跨域预训练、输入换 cutoff-fit Qwen |
| 8 | "MV leads on all three" | RQ1 bullet 和结论第 1 条都加 "among the SASRec-based variants (adapted UniSRec strongest on Grocery)" |
| 9 | 过程句 | Hybrid 已跑完并写入：Last Beauty .0070±.0006 / Toys .0049±.0011；Random .0128±.0003 / .0114±.0008（≥ TF+MV，读作 no-loss + small edge）。结论/未来工作同步改。附录 E "initially evaluated"→filter 敏感性表述，footnote 改中性协议注 |
| 10 | cost 表 | 标题改 recommendation-stage…(offline text encoding excluded)；正文补 TF-IDF+SVD ~1 min CPU（实测 67s）vs Qwen 每视图一次 GPU 全目录推理 |

数据来源：`hybrid_ts_gts_last.csv`、`hybrid_ts_gts_random_seed{42,123,456}.csv`（0820 09:07 完成，log 已核 use_cross=False、boost=0、.ts.npy、[0.8,0.1,0.1]、num_text_views=1）。
tab:gts-all 移除了口径混用的 UniSRec 列（UniSRec GTS-All 留在附录 E，注明 all-target）。
Fig.1 已重画（顶部图例，TF+LLM/TF+MV 标签）。

尚未做（可选，非门槛；**0820 晚已清**）：~~UniSRec GTS-All warm 重评~~ 已写入 `tab:gts-all`（Beauty .0180 / Toys .0139）；~~Last∩Random intersection-user check~~ 已写入 `app:intersection`（见 `VERSION_0820.md`）。
