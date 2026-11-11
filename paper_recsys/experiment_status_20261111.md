# 实验进展 2026-11-11（阶段 1 第 1 天，下午）

> 日记沿用 `paper_recsys/experiment_status_YYYYMMDD.md`。这一串最近写到 [`experiment_status_20261104.md`](experiment_status_20261104.md)，再往前连续写到 20260822。  
> 另一套日计划是 `paper_recsys/experiment_plan_YYYYMMDD.md`，停在 [`experiment_plan_20260730.md`](experiment_plan_20260730.md)，这次不另起。根目录还有 `experiment_status_20260726.md` 和 `experiment_plan_20260727.md` 的旧副本。  
> 本周任务来自 [`notes/wsdm_next_round_plan_20261111.md`](notes/wsdm_next_round_plan_20261111.md) 第 H 节。不改 tex，不跑阶段 2，不安排 PAKDD。

**时间锚点：2026-11-11 下午。** 本日无新训练、未改论文、未改数字。5090 只读对照，并归档误嵌的副本。

阶段 1 的窗口是 **2026-11-11 到 2026-11-18**。11 月 11 日只剩下午，下面这一天写剩余时间。

---

## 用的是哪本日记

主日记就是本文件所在的 `experiment_status_YYYYMMDD.md`。周计划不新开一套目录。

---

## 一、这 8 天

每天都有一个可以交出来的文件，和一个停。停了就不把下一阶段的训练提前拉进来。

### 2026-11-11（周三，下午剩余）— 本日

**产出。** 本文件；[`notes/table_run_freeze_map_20261111.md`](notes/table_run_freeze_map_20261111.md)；`archive/20261111/README.md`。

**已经知道的。** 5090 连上了（`charlie@www.ultrapp.online`，主机名 `charlie-ultra`）。主表大部分格子能对上 `saved/` 里的 checkpoint 和 `paper_recsys/` 里的 CSV。训练当时没有单独的干净 commit。5090 HEAD 仍是 2026-07-21 的 `4c030be`，工作区是脏的。本地 `8c34cea`（今天下午）是事后快照，不是跑表时的 commit。

**仍未对上，不在今天下午再猜。** Beauty ID 的训练 log / yaml；Grocery 的 `tab:shift`；`tab:cost` 的 TF/LLM/MV 仍记在 `sg_beauty_*` 上；LOO UniSRec 的标准差与 CSV 差一位显示；CKA 的 json 没有写下特征路径。

**停。** 不写 protocol，不写检查脚本，不跑抽取，不改 tex。

### 2026-11-12（周四）— 把未对上的格子收口

**产出。** 更新冻结对照表：每个未对上的格子要么钉到一个 log，要么维持“未对上”并写出看过的候选路径。不新跑。

优先看这四件：

1. Beauty ID：`saved/wrap_beauty_id_lr1e4_do0.1_seed{42,2024,2026}/` 已有 pth，缺训练 log。只许从 checkpoint 里读出它当时记下的 config，不许换一个“像”的 yaml。
2. Grocery `tab:shift` 的 87.3→76.9、16→12、29.6%。`warm_shift_grocery_and_gourmet_food.json` 不是这一行。
3. `tab:cost`：确认 `model_cost_table.csv` 的 `sg_beauty_*` 要不要继续当作这张表的来源。不重测。
4. 8 月 22–23 日的 UniSRec Beauty/Grocery 重训 CSV 继续排除在表外，除非某格被证明就是正文数字。

**停。** 五者（论文版本、commit、划分、特征、主表结果）仍对不齐的格子，不拿去选模型，也不进入抽取训练。投稿版 tex 继续不猜：0822 与 0823 都留在表上。

### 2026-11-13（周五）— 一页 protocol

**产出。** 一页 experiment protocol，单独一个 md，不写进 tex。

必须写明：

- 主评测是 leakage-free GTS。旧的、带全局时间重叠的 LOO 只作协议对照，不参与选型。
- 确认集的名字：没有用来反复调参的时间窗口，或一份外部数据。Beauty 上已经反复看过的 test 不是这套确认集。
- 探索可以回看已经看过的 test。确认集在探索时不打开。谁不能打开，写成人名或角色，不写“请遵守”。

**停。** protocol 文本冻结。不因为某张旧 LOO 表更好看就改选型规则。

### 2026-11-14（周六）— 检查脚本的规格，先不写脚本

**产出。** 一份检查脚本规格。五项都要写成可以判定过/不过的句子：

1. Early 的特征统计只在 Early cutoff 之前拟合。
2. Last、Random、Successive 共用一套历史构造和候选规则。Successive 现稿没跑，规格里写“尚未有运行”而不是假装已对齐。
3. 各模型的 alignment、融合容量、训练开关一致。对照对象用冻结表里的那几份 yaml 和 log，不用磁盘上别的草稿。
4. 特征文件 hash 与冻结表文末的 sha256 一致。
5. config diff 和时间边界：Main `TS [0.8, 0.1, 0.1]`，Early `TS [0.6, 0.1, 0.1]`，LOO `LS valid_and_test`。

**停。** 规格冻结之后才允许写脚本。规格没写完，周日不写代码。

### 2026-11-15（周日）— 只有规格已冻结才写检查脚本

**产出。** 二选一，都要留在文件里：

- 规格已在周六冻结，并且脚本只读冻结的 CSV / yaml / hash、不改数据划分、不改正式评测脚本、不改 checkpoint：写下脚本路径。
- 否则写一句“本周不实现”，原因是规格未冻结，或实现会碰到冻结数据或正式评测脚本。

**停。** 不改 `dataset/`、不改 `saved/`、不改 `scripts/eval_gts_last.py` 和 5090 上的 `eval_gts_random.py`。检查不通过时，不改协议去迎合某一组指标。

### 2026-11-16（周一）— 种子名单

**产出。** 一份种子名单。从冻结表和主表记录原样抄：

- 训练种子：`42`，`2024`，`2026`
- GTS-Random target-sampling：`42`，`123`，`456`

排除规则只保留事先写明的程序故障，并保留 log。不新增种子，不因为结果差或方差大删种子。

**停。** 名单冻结。不另开一组“更稳”的种子。

### 2026-11-17（周二）— 抽取样本设计，不训练

**产出。** 一份抽取样本设计。字段只包括属性、功能、适用对象，或原文里同类的信息。每条要能指回原文位置。原文没有的格子允许空。审计列：原文归因、coverage、正确率、幻觉比例、按 item 频次的缺失。通过标准写成“哪些 view 值得进入训练”，不写成 coverage 阈值。

**停。** 不跑抽取训练，不调用模型填空，不把缺失补成 90% coverage。

### 2026-11-18（周三）— 贡献–证据–边界空表，并做阶段 1 的停机检查

**产出。** 一张三列表：贡献、证据、边界。现稿已经写进 0823、并且被 skill 锁住的观察，证据列只指向现稿哪一节，并标明这是现稿事实，不是下一轮的新机制。至少这三行：

| 贡献 | 证据 | 边界 |
|---|---|---|
| 同一批 checkpoint 上改变目标选择会改变排序 | 待填：0823 的哪一节 / 哪张表。这是现稿事实 | 待填现稿已经写明的范围。机制列空着，标待核查 |
| Early 上 Main 的排序在 Beauty 不能保持；Early 表只有 Beauty | 待填：`tab:early` 与正文那一句 | 待填。不把 Toys/Grocery Early 写成已经跑过 |
| 这套问法设计下，冗余在 SVD / whitening 之前的 Qwen hidden states 里已经出现；现稿写明不能把 encoder、共享正文和有限 prompt 差异分开 | 待填：RQ3 那一节 | 待填。不写成 encoder 造成的 |

下一轮的机制行先空着，标「待核查」。摘要和贡献点这一周不锁定。

**停。** 要能说出：协议和检查是否已经冻结，以及哪些抽取出的 view 值得训练。说不出后者，就停在“还不能点名进入训练的 view”，不开始阶段 2。停的标准不是 coverage，也不是模型排序。

---

## 二、本周明确不做

不跑阶段 2 的正式训练。不选最终会议。不改论文句子。不把 PAKDD 的 2026-11-20 放进提交安排。不改原始数据、划分清单和正式评测脚本。

---

## 三、今日实际做完的对照（给 11/12 用）

细节在冻结对照表。这里只留明天还要处理的缺口：

- 主表 Beauty/Toys/Grocery 的 TF、TF+LLM、TF+MV，以及 Toys UniSRec、Grocery UniSRec（8 月 14 日那批）、Beauty UniSRec（7 月 28 日那批），数字对上了指定 CSV。8 月 22–23 日的 UniSRec 重训没有进表。
- Early 对上 `early60eq_gts_last_eval_ts_v2.csv`，cutoff 是 `TS [0.6, 0.1, 0.1]`。
- Beauty ID 的结果 CSV 和 pth 对上了，训练配置没有对上。
- Grocery `tab:shift` 没有对上。
- `tab:cost` 的数字对上 `model_cost_table.csv`，但 TF/LLM/MV 的 tag 是 `sg_beauty_*`，不是主表的 `wrap_beauty_*`。
