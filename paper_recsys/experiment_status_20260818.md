# 实验进展 2026-08-18（补跑优先级重排版）

全文截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京时间 Aug 25 19:59）。摘要已交，不依赖本轮重跑。

---

## 一、最近几次老师指导要点（0717 → 0806 → 0813）

大方向已锁：**从「MV-Align 新模型」彻底改成 leakage-aware controlled study**；三个 RQ = 协议是否改结论 / temporal robustness / 为何 multi-view 没稳定增益。以下是对补跑仍有约束的条目：

**协议（贯穿）**
- 主协议 = **GTS-Last warm**；GTS-Random = robustness（同 checkpoint，只换 target，3 个 sampling seed 平均）；GTS-All 只进附录（task mismatch）。
- 主表**全部对齐到这一协议**；TF 词表/IDF/SVD、LLM center/whiten 只用 train-cutoff 前拟合（cutoff-fit）。
- 结论定调：**text > ID 稳**；**TF / LLM / MV 之间对 protocol 和 window 敏感**。多 seed 完成前不得写「MV 稳定更强」。

**0813 明确要 fix 的点（与数据/实验相关的）**
1. RQ1 措辞拆两种 effect：Last vs Random = target-selection（同 ckpt）；LOO vs GTS = split/训练协议。→ 纯文字。
2. **RQ2 最大方法学风险**：Early 必须做成**等 horizon rolling window**（valid/test 跨度与 Main 一致，只移 cutoff），否则训练量/预测跨度/target 组成三者混淆。→ 已在正文改成 `[0.6,0.1,0.1]`（`tab:early` 用等 horizon）；**但数字仍需用等 horizon + cutoff-fit + 公平协议重训**。
3. **Table 9 LOOV** 现在是 seed42 + GTS-All，与主协议不符，reviewer 一眼可见。→ 用 GTS-Last warm ckpt 重算，3 seeds，zero-mask，不重训。
4. raw 4096-d（实为 3584-d）Qwen 上做 pre-SVD CKA/effective-rank，闭环 RQ3。→ 数据侧 Beauty/Toys 已有；Grocery 在 0817 队列尾。
5. TF + single-view LLM hybrid（公平 GTS-Last/Random，Beauty/Toys 3 seeds）：价值高于继续跑 Cross/SE/boost，但**是 Discussion，不是主表门槛**。
6. 降温措辞、修 refs（`gusak2025time` 等）。→ 纯文字。
7. Grocery 只作 same-platform replication，有 Last/Random 即可，不必补齐 LOO/UniSRec。

---

## 二、逐块核对：正文数据的「正确性 + 训练有效性」

判定依据 = **checkpoint 内记录的实际路径/开关**（不是磁盘上现在的 yaml；yaml 默认仍是 `.npy`）。今日已在 5090 用 `audit_ckpt_protocol.py` 逐个读取确认。

| 正文表/图 | 具体格子 | ckpt 实际协议 | 判定 |
|---|---|---|---|
| `tab:protocol` Beauty **Last/Rand** — TF/LLM/MV | `sg_beauty_*` | `.npy` / `4views`（**NON_TS**），no-cross，boost=0 | **无效，必须重训（P1）** |
| `tab:protocol` Beauty — ID | `wrap_beauty_id_*` | 无文本 | 有效，留 |
| `tab:protocol` Beauty — UniSRec | `unisrec_beauty_gts_*` | `qwen2.5_7b.base.ts.npy` | 有效，留 |
| `tab:protocol` Beauty **LOO** | `loo_beauty_*` | LOO 对照协议 | 有效，留（勿与 GTS 混） |
| `tab:protocol` Toys — TF/LLM/MV | `wrap_toys_*` | `*.ts.npy` + `4views_ts`，no-cross，boost=0 | **有效，留** |
| `tab:protocol` Toys — UniSRec | `unisrec_toys_gts_*` | `qwen2.5_7b.base.npy`（**NON_TS**） | 主表这列不干净，**建议重训（P2）** |
| `tab:protocol` Grocery — TF/LLM/MV | `wrap_grocery_*` | `*.ts.npy` + `4views_ts` | **有效，留** |
| `tab:protocol` Grocery — UniSRec | `unisrec_grocery_gts_*` | `*.ts.npy` | 有效，留 |
| `tab:ndcg_hit` / `tab:perseed` Beauty | 同 `sg_beauty_*` | NON_TS | **随 P1 一起换** |
| `tab:ndcg_hit` / `tab:perseed` Toys/Grocery | wrap_* | TS | 有效，留 |
| `tab:early`（RQ2） | Aug-16 early60 文件 | 非 TS + cross + boost=3 + yaml lr/do | **无效，必须重训（P3）**；split 已是等 horizon |
| `tab:loov`（RQ3） | Beauty .01571 / Toys .01089 | 绝对值≈GTS-Last 的 2.5×，实为 **GTS-All/其它 target** | **无效**：Beauty 随 P1 重算；**Toys 也要重算**（用已有 `wrap_toys_mv_*`，只 eval 不重训） |
| `tab:cka` / `tab:effrank` / `tab:cka-prepost` / raw 3584-d | 与 rec 训练无关 | 脚本读 `4views_ts` / raw | 有效，留 |
| `tab:shift` / `tab:bucket` | 数据统计 | — | 有效，留 |
| `tab:cost` | 参数/时延 | — | 有效，留 |
| Discussion: Pure LLM | `purellm_*`（0817） | `.ts.npy`，no-cross，min5，已 `--save` | 训练有效，**只缺 Last/Random eval（P0）** |
| Discussion: TF+LLM hybrid | 0817 hybrid | 非 TS + cross + boost + yaml lr | **不当对比**；公平版 = P4（可砍） |

**两个纯文字问题（不需重跑，但要改）**
- Setup §3.2 写 `d=64`，但 yaml `hidden_size:256`、参数 67M 与 256 对得上 → 正文改成 256（或核对后统一）。
- 「cutoff-fit throughout」在 Beauty 主表重训完成前不能作为既成事实。

---

## 三、正在跑 / 排队现状（13:26 核对）

- **GPU 上**：0817 队列最后一条 hybrid（`hybrid_toys_seed2026`，Phase-A 已过、Phase-B epoch 0）。协议：`.npy` + `4views`、PA8/PB40 → **对正文无用**，只进 Discussion 且是错误对比。
- **不要 kill**。0817 之后：对这批 hybrid 的旧 Last/Random eval（不进表）+ **Grocery raw 3584-d CKA（要留，闭环 RQ3）**。
- **Watcher** PID 642904 `watch_and_run_p3_fixed.sh` 等 GPU 空 → 跑 `run_paper_rerun_queue_0818.sh`（顺序 P0→P1→P2→P3→P4）。**只有一个 watcher，勿再开第二个。**
- GPU 预计空闲：**今天 16:00–16:30**。

排队脚本已核对，协议正确：
- P1 `run_beauty_ts_maintable_0818.sh`：`item_text_emb.base.ts.npy` / `qwen2.5_7b.base.ts.npy` / `qwen2.5_7b_4views_ts`，no-cross，boost=0，PA20/PB50，新目录 `wrap_beauty_*`，**不覆盖 `sg_beauty_*`**；含 Beauty LOOV 重算。
- P3 `run_early60eq_fair_0818.sh`：`*.ts_early60.npy` / `4views_ts_early60`，split `[0.6,0.1,0.1]`（等 horizon），同一公平协议。特征文件已在盘（7/21 生成）。
- P2 `run_unisrec_toys_ts_0818.sh`：`item_text_emb.qwen2.5_7b.base.ts.npy`，Phase-A 50。
- **缺口**：P1 只重算 **Beauty** LOOV，**Toys LOOV 未重算**（正文 `tab:loov` Toys 绝对值同样错）。需补一条 Toys LOOV（用 `wrap_toys_mv_*` 现成 ckpt，纯 eval）。

---

## 3.5、实测单机时长（用于 ETA，非拍脑袋）

今日从历史日志实测（同协议 PA20/PB50，仅特征路径不同，代表性强）：

| 单元 | 实测壁钟 | 来源 |
|---|---|---|
| Beauty TF（1 seed） | ~74 min | `sg_beauty_tf_*` 16:17→17:31 |
| Beauty LLM（1 seed） | ~73 min | `sg_beauty_llm_*` 15:04→16:17 |
| Beauty MV（1 seed） | ~88 min | `sg_beauty_mv_*` 17:31→18:59 |
| **Beauty 1 seed 三路** | **~3.9h** | 合计 |
| Early60 TF/LLM/MV（各 1 seed） | ~35 min | Aug-14 early60 日志（60% cutoff，item 少，更快） |
| GTS-Last eval / ckpt | ~28s | `gts_last_eval_grocery_reloc` |
| GTS-Random ×3 seed | ~15 min（9 ckpt） | 同上 ×3 |
| LOOV / seed | ~2 min | `loov_gts_last_*` |
| UniSRec Toys（1 seed，PA50） | ~1.5h（5090） | `unisrec_toys_gts_*`（历史含 1080Ti 偏慢） |

**关键校正**：
- **Toys LOOV 已有**（Aug-14，GTS-Last，跑在 `wrap_toys_mv_*` 有效 ckpt，Full MRR=0.011340）→ 正文 `tab:loov` Toys 行只需**文本更新**，**无需重跑**。
- **Beauty LOOV** 现有 Aug-14 版跑在 `sg_beauty_mv_*`（NON_TS），要被 P1 的 `wrap_beauty_mv` 取代 → 已含在 P1，~6min。
- P3 实测比原估 10h 快：**~6h**。

---

## 四、补跑优先级（重排，已落地到队列脚本）

**已把 `run_paper_rerun_queue_0818.sh` 顺序改成正文优先并同步 5090**（watcher PID 642904 按路径调用同一脚本，无需改 watcher）：

| 队列新序 | 任务 | 正文必须 | 进哪张表 | 实测估时 |
|---|---|---|---|---|
| **[1]** | Beauty TF/LLM/MV × 3 seed 重训 + Last/Random + Beauty LOOV | **必须** | `tab:protocol`/`tab:ndcg_hit`/`tab:perseed` Beauty + `fig:protocol` + `tab:loov` Beauty | **~12h** |
| **[2]** | Early60-EQ TF/LLM/MV × 9 重训 + Last | **必须** | `tab:early` | **~6h** |
| **[3]** | UniSRec Toys `.ts.npy` 重训 + Last/Random | 建议 | `tab:protocol`/`tab:ndcg_hit` Toys UniSRec 列 | ~5h |
| [4] | Pure LLM Last/Random eval | 否（Discussion） | Pure LLM 段 | ~2h |
| [5] | 公平 Hybrid TF+1view × 6 + Last/Random | 否（Discussion，可砍） | Hybrid 段 | ~15h |

**Toys LOOV**：已有（Aug-14 GTS-Last on `wrap_toys_mv_*`），**只改正文文本**，不进队列。

### 三个 ETA（GPU 空闲锚点 ≈ 今天 16:15–16:30）

hybrid_toys PhaseB 13:49 到 epoch 7/40，~190s/epoch → 约 15:35 训完 → 旧 hybrid eval（几分钟）+ Grocery raw CKA ~35min → **GPU 空 ≈ 16:15–16:30**。

1. **正文主表数据（[1] Beauty 主表）跑完 = 8/19（周三）清晨 4:00–5:00**
   - Beauty 训练 ~12h（含 Last/Random/LOOV）。16:30 起 → 次日 ~4:30。
   - 其中 **GTS-Last 数字**（`tab:protocol` Beauty 三行的关键值）在 Beauty 训练一结束即产出，同点。

2. **全部正文数据（[1]+[2]+[3]）跑完 = 8/19（周三）夜间 ~23:00–24:00**
   - [1] ~4:30 → [2] Early60 ~6h → ~10:30（`tab:early` 可改）→ [3] UniSRec Toys ~5h + eval → ~16:00 训完。
   - 保守到 **8/19 夜里 24 点前**全部正文表数据齐（含 UniSRec Toys 这个建议项）。
   - 若只算「三大 finding 门槛」（不含 UniSRec Toys 列），**8/19 上午 10:30 前**（[1]+[2]）就够。

3. **含附录/Discussion 全部数据（[1]→[5]）跑完 = 8/20（周四）下午 ~15:00–17:00**
   - 续 [3] ~16:00 → [4] Pure LLM eval ~2h → ~18:00（8/19）→ [5] 公平 Hybrid ~15h → **8/20 ~9:00–11:00**；留 buffer 到 **8/20 傍晚**。
   - [5] 可砍：砍掉则含附录也在 **8/19 夜 ~20:00** 完成。

距 8/24 23:59 AoE：正文必须项有 **~5 天** buffer，含附录也有 **~4 天**。

**明确不跑**：Beauty/Grocery UniSRec 重训、0817 hybrid 的 Last 进表、覆盖 `sg_beauty_*`、4×256 扩维、Cross/SE/boost 救 MV。

---

## 五、正文改动（8/18 13:56 落地，已同步 5090）

改了 `main_wsdm2027_v3.tex` 两处，均**有实测依据、不臆造**：

**改动 1 — Setup 骨干维度 `d=64` → `d=256`（第 190 行）**
- 依据：`audit_ckpt_protocol.py` + 每个主表 yaml/ckpt 的 Run Summary JSON 都是 `hidden_size:256, inner_size:256`；参数量 67M（Beauty）/87M（Toys）与 256 对得上，与 64 对不上。全部 5 个主表 config（TF/LLM/MV Beauty & Toys）一致。
- 注意区分：文本 pipeline 仍是「每视图 64-d → 拼 256-d」（第 194/196/536/547 行的 64-d 指的是**投影后单视图维度**，正确，不动）。改的只是 **SASRec 骨干 hidden_size**。

**改动 2 — `tab:loov`（第 500–520 行）加 pending 注释；数值本身未臆改**
- **核实结论：Toys 行是对的**。Aug-14 已有的 GTS-Last 3-seed（跑在 cutoff-fit `wrap_toys_mv_*`）算出的 3-seed 均值正好 = 正文现值：Full .01089 / −Desc −.00035 / −Func −.00011 / −Aud −.00015 / −Style −.00033。**Toys 无需改，也无需重跑**（推翻了昨天「Toys LOOV 要重算」的判断）。
- **Beauty 行有问题**：现值 Full .01571 来自被淘汰的 NON_TS `sg_beauty_mv`（且连 NON_TS 3-seed 均值 .01604 都对不上，更像更旧的快照）。已在表上方加 `% TODO(P1)` 注释：将由 `run_beauty_ts_maintable_0818.sh` 在 cutoff-fit `wrap_beauty_mv` 上按 GTS-Last 3-seed 重算；并注明「NON_TS 版 3-seed 均值为 Full .01604，不作最终值」。等 P1 出数后替换整行。

**未改（留给 P1/P3 出数后统一处理，避免臆造）**：`tab:protocol`/`tab:ndcg_hit`/`tab:perseed` Beauty 三行、`tab:early` 数字、以及 refs（`gusak2025time` 等，属另一轮纯文字校对）。

---

## 六、在跑 + 排队脚本依赖审查（8/18 13:56，全部通过）

逐脚本核对入口、特征文件、YAML、eval 参数、目录冲突。结论：**队列可安全自动启动，无缺依赖、无覆盖风险。**

**入口脚本**（全部存在）：`scripts/two_phase_train.py`、`scripts/eval_gts_last.py`、`eval_gts_random.py`、`scripts/analyze_leave_one_view_out.py`、`scripts/recbole_env.sh`。

**[1] Beauty 主表** `run_beauty_ts_maintable_0818.sh`
- 特征齐：`item_text_emb.base.ts.npy` / `qwen2.5_7b.base.ts.npy` / `qwen2.5_7b_4views_ts/{view_0..3}`。
- 3 个 yaml 齐；`run_two_phase` 7 参签名与 7 参调用匹配（TAG/MODEL/YAML/LR/DO/SEED/EXTRA）。
- MV EXTRA（`num_text_views` 默认 4、`text_view_indices:null`、`mv_include_base:true`、`use_per_view_align:true`）与历史可跑的 `sg_beauty_mv` 配置一致。
- LOOV 调用参数（`--ckpt/--dataset/--output/--gpu_id/--min_train_interactions`）与脚本 arg 面匹配。
- 新目录 `wrap_beauty_*`，**当前 saved/ 下不存在**，不覆盖 `sg_beauty_*`。

**[2] Early60-EQ** `run_early60eq_fair_0818.sh`
- **关键点已核**：`sasrec_baseline_50ep_stratified_ts_early60_eq.yaml` 在 **5090 与本地都是单一 split `[0.6,0.1,0.1]`（等 horizon）**，无 `[0.6,0.1,0.3]` 残留（昨天的「双 split」担忧是本地 grep 跨文件误读，已排除）。
- 特征齐：`*.ts_early60.npy` / `qwen2.5_7b_4views_ts_early60/{view_0..3}`。
- yaml 默认 `item_text_emb_path_base:""`，靠 config_dict 的 EXTRA 覆盖 → 正确。
- `run_two_phase` 6 参签名与调用匹配，YAML 固定用 `$EARLY_YAML`。

**[3] UniSRec Toys** `run_unisrec_toys_ts_0818.sh`
- `unisrec_toys_stratified_ts.yaml` 存在，split = `TS[0.8,0.1,0.1]`；config_dict 用 `item_text_emb.qwen2.5_7b.base.ts.npy` 覆盖 yaml 里旧的 `.npy` → ckpt 会记 TS。
- 新目录 `unisrec_toys_gts_ts_*`（区别于旧 `unisrec_toys_gts_*`），不覆盖。

**[4] Pure LLM eval** `run_p3_fixed_0818.sh`：纯 eval，动态收集 `purellm_*` ckpt；hybrid 部分只 audit 不进表。

**[5] 公平 Hybrid** `run_hybrid_fair_ts_0818.sh`
- 特征齐；写 `hybrid_ts_*`（区别于正在跑的 0817 `hybrid_toys_seed2026`），**不覆盖运行中任务**。
- ckpt-list 命名（`hybrid_ts_beauty/toys_seed*`）与 run_one 的 TAG 一致。

**运行中 & watcher**：GPU 上仍是 0817 `hybrid_toys_seed2026`（PID 644066）；watcher PID 642904 存活、按路径调用已重排的队列脚本。**不要 kill、不要开第二个 watcher。**

**唯一需人工注意**：本地/5090 的 Early60 yaml 都已正确，**不要再从别处覆盖**；正文 refs 校对未做。

---

## 七、18:30 事故与修复（队列没按重排跑 → 已用正文优先启动器修好）

**现象**：0817 队列 ~16:42 全部结束（含 Grocery raw CKA），GPU 空。但 **12:56 就在运行的 watcher（PID 642904）用的是旧版 `watch_and_run_p3_fixed.sh`**，它 18:02 直接调 `run_p3_fixed_0818.sh`（只做 Pure LLM Last/Random eval），**跑完即退，不会启动 13:50 重排的 `run_paper_rerun_queue_0818.sh`**。即：重排队列成了孤儿，Beauty 主表不会自动开跑。

**根因**：13:50 我改了磁盘上的 watcher + 队列脚本，但**已在运行的 watcher 进程仍执行旧逻辑**（旧版直接 `bash run_p3_fixed_0818.sh`，无后续链）。磁盘上的 watcher 现在虽写着调用队列，但没有新进程去执行它。

**未跑错、未污染正文**：
- 正在跑的 Pure LLM eval 协议正确（`.ts.npy`/no-cross/min5）。**GTS-Last 已出**（`purellm_hybrid_gts_last.csv`，7 行=6 ckpt+表头）：Beauty warm≈0.0045–0.0047、Toys≈0.0013–0.0026 → 进 Discussion「Pure LLM-only」。GTS-Random 正在补（seed42 完成、123 进行中、456 待），此前 0817 日志的 `had errors` 未复现。
- 0817 hybrid（`.npy`+cross+boost）本就标「不进正文」，未变。

**修复动作**：
- 新建 `scripts/run_maintext_first_0818.sh`（PID 702435，nohup 后台），**先等 Pure LLM eval 让出 GPU**，再按正文优先顺序跑：**[1] Beauty 主表 → [2] Early60 → [3] UniSRec Toys → [5] 公平 Hybrid**；**跳过 [4] Pure LLM**（旧流程已完成，不重复）。
- 旧 watcher（642904）跑完 Pure LLM 后自行退出，之后只剩这个启动器管事，不会双开抢 GPU。

**当前（18:35）**：GPU 空闲；Pure LLM Random 收尾中；启动器在等待；Beauty **尚未开跑（正确，等评测结束）**。

**ETA 顺延**：Beauty 主表因 Pure LLM eval（~到 18:45–19:00）小幅推迟，主表数据仍 **8/19 清晨**；全正文 **8/19 夜**；含附录 **8/20 下午**（[5] 可砍）。

**给自己/接手人的提醒**：
- 不要 kill 702435；不要再起第二个 watcher/队列。
- 确认点：Pure LLM eval 结束后 `saved/wrap_beauty_tf_*` 应出现、GPU 占用上升 → Beauty 已接上。
- 若要临时插播，记住旧 watcher 的教训：**改脚本后必须重启对应进程**才生效。
