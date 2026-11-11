# 今晚改稿交接（5090）— 2026-08-19 18:12 写，分析同步自本机

全文截止：WSDM 2027 **Aug 24 23:59 AoE**（北京 **Aug 25 19:59**）。摘要已交。

**先读这份。** 论文源：`paper_recsys/main_wsdm2027_v3.tex` + `refs.bib` + `figures/fig_protocol_contrast.pdf`。快照：`main_wsdm2027_v3_0819.tex` / `refs_0819.bib` / `VERSION_0819.md`。

本机没有 pdflatex，PDF 在 Overleaf 或本机（5090 若有 texlive）编。

---

## 不要动

- **不要 kill** parent **877810**（`scripts/run_early60eq_fair_0818.sh`）及其 `two_phase_train`。
- 不要再开 `run_maintext_first_0818.sh` / `run_paper_rerun_queue_0818.sh` / Hybrid。
- 不要手跑 yaml（主表 V3 磁盘默认仍可能是 NON_TS + `use_cross:true` + boost=3）。信 ckpt/log，不信 yaml 默认。
- 训练 log 里的 `test_mrr@10`（如 TF 0.0111）是 **GTS-All**，正文不用。

---

## 18:12 GPU 实况 → 今晚数字几点齐

当前：`early60eq_ts_v2_beauty_llm_*_seed2026`（18:02:57 起；18:12 约 Phase-A epoch 15）。

已完（GTS-All 仅作进度，**勿写入 tab:early**）：

| tag | GTS-All test MRR |
|---|---|
| TF 42 / 2024 / 2026 | 0.0111 / 0.0099 / 0.0102 |
| LLM 42 / 2024 | 0.0096 / 0.0097 |
| MV 42 / 2024 | 0.0107 / 0.0108 |

还剩：LLM 2026 收尾 → MV 2026（约 73 min）→ `eval_gts_last.py` 9 个 ckpt（约 30–40 min）。

**ETA：北京 20:40–21:10** 出

`paper_recsys/early60eq_gts_last_eval_ts_v2.csv`

齐了才换 `tab:early` 的 **Early 行**（用 CSV 的 **warm** MRR@10，3-seed mean±样本 std）。Main 行已经是新 Beauty Last，不要改。

---

## `[0.6,0.1,0.1]` 口径（今天下午已核，没有白跑）

**只约束 RQ2 Early。** 主表 Last/Random 必须是 `[0.8,0.1,0.1]`。

| 谁 | split | 怎么核的 |
|---|---|---|
| 主表 Beauty/Toys/Grocery wrap、UniSRec、Pure LLM、Hybrid 残留 | `[0.8,0.1,0.1]` | 训练 log `eval_args` |
| Beauty ID | 同 Main（GTS-Last n_users=4520，与 wrap 一致） | 评测人数 |
| **正在跑的 v2** TF/LLM/MV seed42 **Phase-A 和 Phase-B** | **`[0.6,0.1,0.1]`** | dump + test **5269 users / 14707** |
| 8/16 `tab:early` 现用 Early 行 | 已经是 `[0.6,0.1,0.1]` | 同 5269 |
| 7/21 `ew_beauty_*` | `[0.6,0.1,0.3]`（n_users=7470） | **作废，不要用** |
| Clean LOO / 旧 GRU4Rec | `LS` | RQ1 对比列，不是 Early |

12:54 改 yaml 时 TF/LLM seed42 已在跑。split **不是靠那次改文件**：11:24 脚本 `config_dict` 写死了完整 `eval_args`。Phase-A/B 两边 dump 都是 `[0.6,0.1,0.1]`，人数与 MV（yaml 改完后才开）相同。**不要因「改到一半」重跑。**

磁盘 `sasrec_baseline_50ep_stratified_ts_early60.yaml` 仍是 `[0.6,0.1,0.3]`，当前 job 没用它。

---

## 影响范围：主表零格子；只动 RQ2

**主表不动**（已写入，可当结论）：

- `tab:protocol` / `tab:ndcg_hit` / `tab:perseed` / `fig:protocol`
- Beauty Last：TF **.0053±.0004** / LLM **.0060±.0004** / MV **.0064±.0003**
- Beauty Random：TF **.0124** / LLM **.0120** / MV **.0124**
- Toys UniSRec Last **.0040±.0008**，Random **.0085**
- `tab:early` **Main 行**已与上面对齐

**v2 只换：**

| 位置 | 现在 | 换成 |
|---|---|---|
| `tab:early` Early 行 | 8/16：.0036 / .0038 / **.0043** | v2 Last warm 3-seed |
| caption L372、正文 L433–434 | “stress test / retrain in progress / fusion 不一致” | 齐了可改成 matched-protocol cutoff ablation |
| 摘要第二条、Contribution 2（L119）、结论第 2 条（L682） | 等 horizon + stress test + 三者差 < 0.001 | 按新排序改一句 |

`tab:shift` / `tab:bucket` 是 **Main 窗** valid→test，不算 Early 重训。跌幅数字会随 Early 行变，因果仍写 association。

Toys/Grocery Early、Early-Random、Early-ID/UniSRec：**不在这轮**（导师：Beauty TF/LLM/MV × 3 seed 即可）。

---

## 今晚改稿清单（CSV 出现之后）

1. 打开 `early60eq_gts_last_eval_ts_v2.csv`，用 **warm** 列，不要 all、不要训练 log。
2. 只改 `tab:early` Early 行 + 上述三处措辞。
3. 若仍近并列、绝对值仍掉：删 hedge，Contribution 2 升成「只动 cutoff」。
4. 若排序翻了（Early 上 TF 明显赢）：改 Contribution 2 / “两窗都是 MV 略高”；**RQ1 主表不动**。
5. 不要把 LOOV 今早 md 的 ~.016 写进 `tab:loov`（那是 GTS-All）。
6. Hybrid 继续 withhold。有余量再 resume `hybrid_ts_*`，不是今晚门槛。

---

## 论文里已经锁住、不要回退的结论

- Last vs Random = **同 checkpoint 的 target-selection**；LOO vs GTS = **split/训练协议**（不是 same ckpt）。
- Primary = GTS-Last warm、min5、train catalog；Random = 同 ckpt、sampling seeds {42,123,456}。
- cutoff-fit：`*.ts.npy` / `4views_ts`；Early 特征 `*.ts_early60.npy` / `4views_ts_early60`。
- Pure LLM Last：Beauty **.0046±.0001**，Toys **.0018±.0007**（Toys 低于 ID）。不要写 GTS-All .0256/.0190。
- 引用：`gusak2025time`；TALLRec 1007–1014；RecBole DOI 3482016；CKA PMLR vol 97。见 `VERSION_0819.md`。

导师 0813 原话在 `0813zhidao.txt`。RQ2 最大风险是旧 `[0.6,0.1,0.3]` 把训练量/预测跨度/target 组成混在一起——**现用 Early 行 split 已是等 horizon**；v2 是再把融合协议拉齐到主表。
