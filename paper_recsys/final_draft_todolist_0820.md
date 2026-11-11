# 终稿 Todolist：0820zhidao → 当前稿对照

源：`0820zhidao.txt`  
稿本：`main_wsdm2027_v3.tex`（2026-08-21：摘要仅统一变体名为 TF / TF+LLM / TF+MV；含 `app:intersection`）  
状态：`[x]` 已写入当前稿 · `[ ]` 终稿仍建议做 · `[~]` 有意不做 / 可选

提交截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京 Aug 25 19:59）。

---

## 0. 摘要已交，PDF 里能不能微调命名？

**可以微调；不要改 claim。** 这是顶会常见做法，不是违规。

WSDM 2027 CFP（[cffp](https://www.wsdm-conference.org/2027/cffp.html)）把 Abstract deadline（Aug 17）和 Papers Due（Aug 24）分开。Abstract 截止后**明确锁死的是作者名单**（*Authorship changes, including additions or removals, will not be permitted after the abstract deadline*），**没有**写死 PDF 摘要必须与系统里那一版逐字相同。

惯例（WSDM / KDD / WWW / SIGIR 同一套 CMT 流程）：

| 系统里已提交的 abstract | Full-paper PDF 里的 abstract |
|---|---|
| 给 PC **bidding / 分稿** 用 | **审稿人实际读的是 PDF** |
| 锁作者、锁题目大方向 | 允许澄清用词、修语法、统一术语 |
| 不要换成另一个研究问题 | 不要新增贡献、不要改三条发现的方向 |

老师这条意见的核心是：**摘要如果还写 “TF-IDF vs single-view LLM vs multi-view LLM”，正文表头却是 TF+LLM / TF+MV，reviewer 第一眼就会抓歧义。** 这比 “系统摘要和 PDF 摘要差几个词” 更危险。

建议的微调范围（只改命名，三条发现不动）：

- 保留提交稿的研究问题、三个数据集、GTS-Last / Random / LOO、CKA 0.82–0.94、ER 53–55%、CKA ≈ 0.19、near-ties。
- 把 study 句和 First 句里的 *single-view LLM / four-view / multi-view LLM* 改成 **TF+LLM / TF+MV**，并加半句 *added to the TF-IDF base*。
- **不要**把 RQ2 改回 “Early 会翻转排序”——那是 0820 下午误写进摘要的 claim 升级，提交稿没有这句话。

下面 §1 的摘要项已完成（只改命名，不改三条发现）。

---

## 1. TF / LLM / MV 命名歧义（claim 级，最容易被抓）

**老师原文要点：** 3.2 看起来像三种独立 text representation；§7 Pure LLM-only 又低于 Table 1 的 LLM 列。Table 1 的 LLM 实际是 TF-IDF base + 单视图；MV 是 TF base + 四视图。应改名为 TF、TF+LLM、TF+Multi-view LLM，并统一 Table 1、Figure 1、Abstract、RQ1。否则 claim 会变成 “比 representation family”，实际是 incremental augmentation。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| 宏 | `\llm` 当主变体名 | `\tfllm` = TF+LLM，`\tfmv` = TF+MV | [x] |
| §3.2 `sec:models` | **LLM** / **MV** 像独立表征 | TF+LLM = TF 基座 **plus** 单视图 Qwen；TF+MV = TF 基座 **plus** 四提示视图；并写明 RQ1 比的是增量，不是三个家族 | [x] |
| Table 1 `tab:protocol` 表头 | ID TF LLM MV UniSRec | ID TF **TF+LLM TF+MV** UniSRec | [x] |
| Fig.1 `fig:protocol` | 图例 LLM / MV | 已重画 TF+LLM / TF+MV（`figures/fig_protocol_contrast.pdf`） | [x] |
| RQ1 题述 `sec:intro` / `sec:rq1` | TF-IDF vs LLM vs MV | `\tfidf` vs `\tfllm` vs `\tfmv` | [x] |
| 其余主表 | `tab:ndcg_hit` `tab:perseed` `tab:early` `tab:gts-all` `tab:grocery-perseed` `tab:random-seeds` `tab:cost` | 表头同步 TF+LLM / TF+MV | [x] |
| §7 Pure LLM `sec:discussion` | 低于 “TF-anchored LLM column”，暴露歧义 | 改为命名依据：无基座 LLM-only Beauty `.0046` / Toys `.0018`，低于 TF+LLM 列；词法基座承重，所以不叫 LLM / MV | [x] |
| **Abstract** | 提交稿：single-view Qwen / four-view LLM | study 句与 First 句改为 \tfllm / \tfmv，并写明 added to the TF-IDF base；Second/Third 与 CKA ≈ 0.19 未动 | [x] |

**终稿待办（摘要轻改，已写入 2026-08-21）：** study 句与 First 句已用 TF+LLM / TF+MV，并写明 added to the TF-IDF base。

---

## 2. RQ2 shift 证据必须对齐 warm 主协议

**老师原文要点：** 主表只评 warm target。Table 5/6 用 31.9%/40.8% unseen 解释性能下降，逻辑不通。二选一：标成 all-target catalog context，或在 warm 子集内重算 popularity median / item age / head-mid-low。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| `tab:shift` / `tab:bucket` | unseen 用来解释 warm MRR | **采用第一种方案（不重算）：** 恢复全目标数字；caption 写明 *all-target catalog-level context, not a direct explanation of warm-target MRR* | [x] |
| RQ2 正文 | 用 unseen% 讲 warm drop | `tab:early` 才是 warm MRR；shift/bucket 只作目录背景 | [x] |
| `app:alltarget` | unseen 像主解释 | 写明从不进入主表 warm 分数 | [x] |
| item age | 老师点名可加 | **不做** | [~] |

不再使用 warm-subset 重算。`tab:early` 热目标结果未改。

---

## 3. Last vs Random：同一批用户 + 交集检查

**老师原文要点：** warm filter 作用在选中的 target 上，Last/Random 可能连 user set 都不同。至少写 eligible 人数；成本低则做 intersection-user sanity check。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| §3.1 Leakage controls | 未写人数 | 同一 eligible 集：Beauty 4,520 / Toys 4,873 / Grocery 4,152；warm 子队列略有差（Beauty Last 3,137 vs Random 3,054–3,083） | [x] |
| RQ1 Implications | “same checkpoints → 纯 target-selection” 未钉死 user set | 加 *same eligible users* | [x] |
| 交集检查 | 未做 | `app:intersection` + `tab:intersection`：36 ckpt × 3 sampling seed；交集 Beauty 63% / Toys 54% / Grocery 64%；排序不变，翻转仍在，主表不改 | [x] |
| RQ1「Why Last and Random disagree?」 | 无指向 | 一句指向附录：不是队列构成伪影 | [x] |

数据：`gts_intersection_eval.csv`、`gts_intersection_summary_20260820.txt`。

---

## 4. Intro 引用句写错了（TALLRec / UniSRec）

**老师原文要点：** “LLM-generated descriptions beat TF-IDF/Word2Vec [TALLRec, UniSRec]” 不是这两篇的贡献。改成 pretrained language representations 改进 text-enhanced rec / transferability。可补 Recformer、FDSA/S3-Rec。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| Intro 第 2 段 | routinely reports LLM-generated descriptions improve over TF-IDF/Word2Vec `\cite{bao2023tallrec,hou2022unisrec}` | pretrained language representations can improve text-enhanced recommendation and transferability `\cite{hou2022unisrec,li2023recformer,bao2023tallrec}`；feature-level attention `\cite{zhang2019fdsa}` | [x] |
| bib | 无 Recformer / FDSA | `li2023recformer`、`zhang2019fdsa` | [x] |
| S3-Rec | 老师列为可补 | 未加 | [~] 可选 |

---

## 5. effective rank 缺原始引用

**老师原文要点：** 定义了 entropy of normalized singular values，没给出处。补 Roy & Vetterli 2007（EUSIPCO pp. 606–610），bib 原文已给。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| Intro 诊断句 | CKA + ER + LOOV，ER 无 cite | ER 加 `\cite{roy2007effective}` | [x] |
| Related Work | 定义无出处 | 同 cite；bib 按老师条目写入 `roy2007effective` | [x] |

---

## 6. Clean LOO 写成 “last-3”，像只取最后三条

**老师原文要点：** 写成 all but the final two for training, penultimate for validation, final for test（若实际如此）。protocol 是贡献之一，必须写严。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| §3.1 Clean LOO | last-3 interactions as train/valid/test | *all but the final two* 训练，倒数第二次验证，最后一次测试（RecBole `LS`） | [x] |

---

## 7. UniSRec 不是原版，要说清适配了什么

**老师原文要点：** “same TS-whitened LLM embeddings” 已经不是原版（原方法有 item text、parametric whitening、MoE、contrastive pretraining）。应叫 UniSRec backbone adapted to cutoff-fit Qwen features。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| §3.2 UniSRec 条 | external baseline + 同一套 whitened LLM | 原架构 MoE adaptor；各数据集从头训、无跨域预训练；输入换成 cutoff-fit 白化 Qwen；**adapted backbone, not original pretrained UniSRec** | [x] |
| Intro 变体枚举 | UniSRec 外部基线 | *an adapted UniSRec baseline* | [x] |

---

## 8. “MV leads on all three datasets” 漏了 UniSRec 限定

**老师原文要点：** Grocery 上 UniSRec 的 MRR/NDCG/Hit 更高。应写成 Among the controlled SASRec-based variants, MV leads narrowly under GTS-Last。Conclusion 同样加限定。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| RQ1 GTS-Last bullet | MV leads on all three datasets | *among the controlled SASRec-based variants*, TF+MV 略领先；adapted UniSRec 在 Grocery 最强（`.0107`） | [x] |
| `tab:ndcg_hit` caption | 未限定 | Among the SASRec-based variants, NDCG agrees with MRR；UniSRec 在 Hit@10（三数据集）和 Grocery NDCG 最强 | [x] |
| Conclusion 第 1 条 | MV leads on all three | 同样加 SASRec-based 限定 + Grocery UniSRec | [x] |

---

## 9. 过程句：Hybrid paused + 附录踩坑史

**老师原文要点：** “retrain was paused” 不要留在论文里。要么跑完，要么删掉只在 Future Work 写 open。Appendix E 的 initially evaluated / earlier catalog-fit run 全部去掉。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| §7 Hybrid | matched-protocol hybrid retrain was paused | 公平协议结果写入：Last Beauty `.0070±.0006` / Toys `.0049±.0011`；Random `.0128±.0003` / `.0114±.0008`；读作无损失 + 小优势 | [x] |
| Conclusion / Future Work | additive gains remain open（因暂停） | 与 Hybrid 数字对齐；未来工作改成更复杂融合是否显著，而非 “还没跑” | [x] |
| `app:unisrec` | initially evaluated without… / earlier catalog-fit | 中性 filter 敏感性：无 min5 时 Beauty 0.0204–0.0208；三列均为 cutoff-fit warm | [x] |

---

## 10. Cost 表漏了离线 LLM 编码

**老师原文要点：** “同一参数量、同一 0.02 ms” 只覆盖推荐阶段。标题改为 recommendation-stage … (offline text encoding excluded)。有数据再加一行 offline cost，否则和 practitioner “TF-IDF 便宜、LLM 贵” 打架。

| 位置 | 改前 | 改后（当前稿） | 状态 |
|---|---|---|---|
| `tab:cost` 标题 | same parameter budget and inference cost | *Recommendation-stage parameter count and online scoring cost (offline text encoding excluded)* | [x] |
| Cost 段正文 | 只强调模型侧一样 | 补：Beauty 上 TF-IDF+SVD ~1 min CPU；Qwen 每视图一次全目录 GPU 推理，多视图按 prompt 倍增；差别在特征侧 | [x] |
| 表内 GPU 小时行 | 老师说有数据最好加 | 未加测过的 GPU-hour 数字 | [~] 可选 |

---

## 终稿还要做的（按优先级）

- [x] **摘要轻改命名**（§1）：study 句与 First 句已改为 TF+LLM / TF+MV（加到 TF-IDF 基座上）；三条发现未改。中文对照稿与 `overleaf_deploy/` 已同步。
- [ ] Overleaf 编译核对 `tab:intersection` 表宽、总页数 ≤ 9 + ethics + refs。英文稿 pdfLaTeX；中文稿 XeLaTeX。
- [x] `overleaf_deploy/main_wsdm2027_v3.tex` 已换成仓库根下含交集附录的 907 行稿；`refs.bib` 已对齐。
- [~] 可选：Intro 补 S3-Rec；`tab:cost` 加 Qwen GPU-hour。都不是门槛。

**不要做：** Early-Random、Toys/Grocery Early、Grocery LOO、把训练 log 的 GTS-All `test_mrr@10` 写进主表、把 0817 Hybrid（NON_TS+cross+boost）写进主结论。
