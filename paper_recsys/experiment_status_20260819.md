# 实验进展 2026-08-19

全文截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京 **Aug 25 19:59**）。摘要已交。

**时间锚点：北京 23:35。** 正文 `tab:loov` 已换 GTS-Last。附录 `tab:gts-all` Beauty 文本列仍是 `sg_*`，caption 已标明。

5090 工作区同步前的 20:56 开头（原文，未改写）：

**时间锚点：北京 20:56。** Early60 v2 Last 已齐，论文 `tab:early` Early 行已换成匹配协议数字。

---

## 现在（20:56）

| 块 | 状态 |
|---|---|
| 主表 Beauty/Toys/Grocery Last/Rand、UniSRec | 单位 18:13 稿已写入；Toys UniSRec Last **.0040±.0008** / Random **.0085** |
| **RQ2 Early60-EQ v2 GTS-Last warm** | **20:46 齐** `early60eq_gts_last_eval_ts_v2.csv` |
| GPU | **空闲**。Hybrid toys seed42 上午已停，未 resume |
| 论文 | `main_wsdm2027_v3.tex` 已换 Early 行 + 去掉 stress-test hedge |


---

## 现在（23:35）

| 块 | 状态 |
|---|---|
| 正文表 | **齐**（含 LOOV Last） |
| 附录 `tab:gts-all` Beauty TF/LLM/MV | catalog-fit `sg_*`；cutoff-fit 只 eval 即可 |
| GPU | **空闲**（LOOV 23:19 结束） |
| Hybrid / Successive | **不重训**；不是门槛 |

详见 `VERSION_0819.md` 附录清单。

---

## 23:19 记录（LOOV 写入）

| 块 | 状态 |
|---|---|
| 主表 / Early / CKA | 已写入 |
| **RQ3 LOOV GTS-Last warm** | **23:19 齐**。Beauty Full **.00636** / Toys **.00417**。最大 \|Δ\| **0.00032** |
| GPU | 当时刚空闲 |

---

## 21:31 记录（历史：LOOV 刚启动）

| 块 | 状态 |
|---|---|
| 主表 Beauty/Toys/Grocery Last/Rand、UniSRec | 已写入；Toys UniSRec Last **.0040±.0008** / Random **.0085** |
| **RQ2 Early60-EQ v2 GTS-Last warm** | **20:46 齐**；`tab:early` 已换 |
| **RQ3 LOOV GTS-Last warm** | **21:26 启动** Beauty+Toys × 3 seed。PID **985406**。Beauty seed42 Full = **0.006291**（n_warm=3137） |
| GPU | **当时占用**（eval） |

---

## 20:56 记录（历史：Early 已齐、GPU 当时空闲）

| 块 | 状态 |
|---|---|
| 主表 Beauty/Toys/Grocery Last/Rand、UniSRec | 单位 18:13 稿已写入；Toys UniSRec Last **.0040±.0008** / Random **.0085** |
| **RQ2 Early60-EQ v2 GTS-Last warm** | **20:46 齐** `early60eq_gts_last_eval_ts_v2.csv` |
| GPU | **当时空闲**。Hybrid toys seed42 上午已停，未 resume |
| 论文 | `main_wsdm2027_v3.tex` 已换 Early 行 + 去掉 stress-test hedge |

**Early v2 Last warm（3-seed，用 CSV warm 列）：**

| | TF | LLM | MV |
|---|---|---|---|
| Main（已在表） | .0053±.0004 | .0060±.0004 | **.0064**±.0003 |
| Early v2 | **.0042**±.0005 | .0037±.0003 | .0039±.0003 |

相对 8/16 旧 Early（.0036 / .0038 / **.0043**）：排序翻了。Main 仍是 MV；Early 是 TF 略高（相对 MV 在 1 std 内）。三者仍差 < 0.001。TF 从 Main→Early 掉得比 LLM/MV 少。

n_users=5269 / n_warm=3564（等 horizon `[0.6,0.1,0.1]`，与 8/16 同一 test 人数；主表 Beauty 是 4520）。

**不要：** 把训练 log GTS-All（0.0111 等）写进表；不要把今早 LOOV .016 写进 `tab:loov`；不要重启 Hybrid 当门槛。

已拉回本地：UnisRec Toys Last/Random CSV、Early v2 Last CSV、单位 18:13 的 tex/refs/`VERSION_0819.md`/`NIGHT_HANDOFF_0819.md`。

---

## 18:12 记录（历史：当时 v2 还在跑）

**今晚改稿先读** `paper_recsys/NIGHT_HANDOFF_0819.md`（口径审计、影响范围；CSV 当时未齐）。

**当时不要 kill**：`run_early60eq_fair_0818.sh`（PID **877810**）。该 job **已结束**。

---

## 现在这一刻（18:12）

| 块 | 状态 |
|---|---|
| 主表 Beauty/Toys/Grocery Last/Rand、UniSRec、措辞、图 | **已写入** `main_wsdm2027_v3.tex`（12:51 已在 5090；本机同步同一版） |
| **RQ2 Early60-EQ v2** | 8/9 训完。GPU：`llm_seed2026`（18:02 起）。还剩 MV seed2026 + GTS-Last 评 9 ckpt |
| 产出 | `paper_recsys/early60eq_gts_last_eval_ts_v2.csv`（尚无）→ 只换 `tab:early` Early 行 |
| ETA | **20:40–21:10** |
| LOOV / Hybrid | 仍不对齐 / 已停；不是今晚门槛 |

训练 log 的 mrr（TF 0.0111 等）是 **GTS-All**，不要写进表。

**split 口径（已核，v2 没有混 `[0.6,0.1,0.3]`）**：主表 `[0.8,0.1,0.1]`；Early v2 三路 Phase-A/B 都是 `[0.6,0.1,0.1]`，test 5269 users。12:54 改 yaml 时 TF/LLM seed42 已在跑，但 `config_dict` 从 11:24 就写死了等 horizon，不必重跑。

---

## 11:30 记录（历史：v2 刚启动）

## 当时这一刻

| 块 | 状态 |
|---|---|
| 主表 Beauty Last/Rand TF/LLM/MV | **已写入** `main_wsdm2027_v3.tex`（`wrap_beauty_*` cutoff-fit） |
| 主表 Toys UniSRec Last/Rand | **已写入**（`.ts.npy` 3-seed） |
| 措辞 | Last vs Random = 同 ckpt；LOO vs GTS = split；Early 标明未完全匹配；Pure LLM 用 Last；Time-to-Split 引用改为 `gusak2025time`；图 Random 改为 3-seed 均值 |
| **RQ2 公平 Early60-EQ v2** | **11:24 已启动**，GPU 在训 TF seed42。日志已确认 `disable_text_feature = False`、Phase-A epoch 0/1 在跑（上一版 45 秒崩溃的坑已修好） |
| Beauty LOOV GTS-Last | 仍未评对协议；**不要**把今早 md 的 ~.016 写进 `tab:loov` |
| 公平 Hybrid | **已停**（Toys seed42 训到一半被停，给正文 Early60 让路）。ckpt：`saved/hybrid_ts_beauty_seed42` 可能已完；`saved/hybrid_ts_toys_seed42` 不完整。有余量再 resume |

**ETA（北京，11:24 起算）**：TF/LLM ~35 min、MV ~47 min ×3 seeds + Last eval ~30 min → 约 **17:30–19:00** 出 `paper_recsys/early60eq_gts_last_eval_ts_v2.csv`。齐了再换 `tab:early` 的 Early 行。

---

## 论文已改的格子（GTS-Last warm MRR@10，3-seed）

Beauty Last：TF **.0053±.0004** / LLM **.0060±.0004** / MV **.0064±.0003**（ID/UniSRec 未动）。  
Beauty Random：TF **.0124** / LLM **.0120** / MV **.0124**（两者并列加粗）。  
Toys UniSRec Last **.0040±.0008**（seed2026=.0031 拉低了均值），Random **.0085±.0003**。  
`tab:early` Main 行已换成新 Last；Early 行仍是 8/16，caption/正文写成 temporal stress test，**不是**「与 Main 只差 cutoff」。

Discussion Pure LLM：Beauty **.0046±.0001**、Toys **.0018±.0007**（Toys 低于 ID）。不要写 GTS-All .0256/.0190。Hybrid 增益句已扣住，等公平重训。

图：`figures/fig_protocol_contrast.pdf` 已按表里的 3-seed Random 重画（旧图误用 sample-seed 42，Beauty TF 会画成 .0139）。

---

## Early60 修了什么、为什么这次能跑

上一版 `run_early60eq_fair_0818.sh` 套了 ID-only yaml，dict 没写 `disable_text_feature:false` → `text_mode=none` → Phase-A 无梯度。

现在：V3 yaml + 显式 `disable_text_feature:false` + 完整 `eval_args` 覆盖成 `TS[0.6,0.1,0.1]`（V3 early60 文件里仍是 `[0.6,0.1,0.3]`，只改 split 会把 `order:TO` 冲掉，所以必须写全）。新 tag：`early60eq_ts_v2_beauty_*`，不覆盖 8/16 ckpt。

进程：parent **877810**，日志 `logs/early60eq_fair_0819.log` / `logs/early60eq_ts_v2_beauty_tf_lr5e4_do0.3_seed42.log`。

11:25 已看到 `disable_text_feature = False` 且 epoch 0/1 有 train loss，不是 45 秒挂。

---

## 附录：08:42 记录（历史，队列状态已过时）

# 实验进展 2026-08-19（Beauty TS 主表已齐 / Early60 公平重训失败 / 协议核对）

全文截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京 **Aug 25 19:59**）。摘要已交，不依赖本轮。

本文档时间锚点：**北京 08:42**。前一天长审计见 `experiment_status_20260818.md`。本文件只记 8/18 夜–8/19 晨的实况和几次分析结论，方便单位拉下来改论文。

**（历史，PID 已过时。）** 当时还在跑 UniSRec / after_beauty；那些进程已结束或已停。当前以文首 11:30 节为准。

---

## 一、一句话状态

| 块 | 状态 | 何时齐 |
|---|---|---|
| **主表 Beauty Last/Rand TF/LLM/MV**（cutoff-fit、no-cross、boost=0） | **04:56 已齐**，CSV 已在本地 | 估的 05:00，实际 04:56 |
| 主表 Toys/Grocery 整块、Beauty ID / UniSRec / LOO | 本来就齐 | — |
| 主表 Toys UniSRec 列（`.ts.npy`） | seed42 / 2024 完；**seed2026 进行中**（08:05 起，已跑 ~37 min，前两颗各 ~94 min） | 训练约 **09:40**，Last/Random 评完约 **10:00–10:20** |
| 正文 RQ2 `tab:early` 公平 Early60-EQ | **失败后被跳过**，队列不会自动重跑 | 须手动修 yaml 再插；约 6–8h |
| Beauty LOOV GTS-Last | 脚本跑了，但评的是 **GTS-All**（md 有数、csv 空） | 须另评，未排 |
| 附录公平 Hybrid | UniSRec 之后自动开（可砍） | 不插 Early60：**周四 00:00–02:00**；插了则周四上午 |
| Pure LLM Last/Random | 0817 已齐 | 只需改 Discussion 文字（勿再用 GTS-All） |

**论文可改主表数字：现在就可以**（Beauty 用新 CSV；Toys UniSRec 列等到 ~10:20）。正文 RQ2 公平数字要等 Early60 重跑。

---

## 二、今天已完成：Beauty cutoff-fit 主表重训

脚本：`scripts/run_beauty_ts_maintable_0818.sh`  
日志：`logs/beauty_ts_maintable_0818.log`  
ckpt：`saved/wrap_beauty_{tf,llm,mv}_*_seed{42,2024,2026}`（**不覆盖** `sg_beauty_*`）

协议（日志已核）：V3 yaml + `config_dict` 覆盖 → `use_cross=False`，boost=0，`item_text_emb.*.ts.npy` / `qwen2.5_7b_4views_ts`，PA20/PB50，min5，`--save`。

实测：TF ~52 min，LLM ~55 min，MV ~78 min。18:44 开训 → 03:58 九跑完 → Last/Random/LOOV 脚本 → **04:56 ALL DONE**。

### 本地已拉回（单位可直接用）

- `paper_recsys/gts_last_eval_beauty_ts.csv`
- `paper_recsys/gts_random_eval_beauty_ts_seed{42,123,456}.csv`
- `paper_recsys/loov_gts_last_beauty-ts_seed{42,2024,2026}.md`（见下：协议不对）
- `paper_recsys/loov_gts_last_beauty_ts.csv`（只有表头）

### 新数字 vs 论文旧表（GTS warm MRR@10，3 train-seed）

Random = 每 ckpt 先对 sampling seed {42,123,456} 平均，再对 3 个训练 seed 均值±样本 std。

| | Last 旧 `sg_beauty_*` NON_TS | Last 新 `wrap_beauty_*` TS | Random 旧 | Random 新 |
|---|---|---|---|---|
| TF | .0051±.0008 | **.0053±.0004** | .0122±.0010 | **.0124±.0002** |
| LLM | .0059±.0005 | **.0060±.0004** | .0118±.0007 | **.0120±.0002** |
| MV | **.0061**±.0005 | **.0064±.0003** | **.0123**±.0008 | **.0124±.0002** |

Last NDCG@10：TF .0074 / LLM .0080 / MV .0087。Hit@10：.0146 / .0146 / .0167。

结论未翻：Last 仍 MV 略高、三者近并列；Random 仍 TF≈MV > LLM。方差比 NON_TS 更小。换表时 `tab:protocol` / `tab:ndcg_hit` / `tab:perseed` / `fig:protocol` 的 Beauty GTS 行以及 `tab:early` 的 **Main 行**都应改成新 Last。

---

## 三、队列实况（08:42）

```
00:51  run_after_beauty_0819.sh 挂起等 Beauty PID 705779
04:56  Beauty 完；Early60 起了两次（04:56:04 与 04:56:49）
04:57  Early60 非零退出 → 直接 [3] UniSRec Toys
04:57–06:31  UniSRec Toys seed42
06:31–08:05  seed2024
08:05–     seed2026（进行中）
之后脚本会跑 Last/Random eval，然后自动 [5] 公平 Hybrid
Early60 不会自动重来
```

GPU 上：`unisrec_toys_gts_ts_seed2026`，特征 `item_text_emb.qwen2.5_7b.base.ts.npy`。

---

## 四、分析结论（8/19 晨几次核对，改论文时按这个理解）

### 4.1 Early60 在论文里对应哪一块

只服务 **RQ2**，不是主表。

- 问题：Main 80% cutoff 的结论，训练窗缩到 60% 后还在不在。
- 协议：等 horizon `TS [0.6,0.1,0.1]`（valid/test 仍各 10%，只移 cutoff）；超参从 Main 冻结。
- 吃数字的表：**`tab:early`**（Main vs Early60-EQ，GTS-Last warm MRR@10）。
- 同一主张还在：摘要第二条、Contribution 2、结论第 2 条。
- **不依赖** Early60 重训：`tab:shift` / `tab:bucket`（Main 分布偏移）、`tab:protocol`。

表上现用数字：

| Window | TF | LLM | MV | 实际来源 |
|---|---|---|---|---|
| Main | .0051 / .0059 / **.0061** | 7/20–21 `sg_beauty_*` Last warm（NON_TS） |
| Early60-EQ | .0036 / .0038 / **.0043** | 8/16–17 ckpt + `early60eq_gts_last_eval.csv` Last warm |

### 4.2 表里 Early 行 **不是 ID-only**（读表判断对）

三列差得开、TF < LLM < MV，MV 明显高一截。若关文本，三列会贴在一起，并靠近 Beauty ID-only Last（~.0032）。

8/16 日志：`disable_text_feature = False`，`use_cross = True`，路径 `item_text_emb.base.npy`。文本是开的。

**今早 45 秒崩溃没有写进这张表。** 崩溃的是另一次公平重训。

### 4.3 今早 Early60 为什么挂（公平脚本独有）

`run_early60eq_fair_0818.sh` 为了 `[0.6,0.1,0.1]`，套了 **ID-only yaml**  
`sasrec_baseline_50ep_stratified_ts_early60_eq.yaml`（`disable_text_feature: true`）。

`config_dict` 只改了 embedding 路径和 `text_mode:'base'`，**没写** `disable_text_feature: false`。  
日志：`text_mode = base`（dict 成功）但 `disable_text_feature = True` → 模型 `text_mode=none` → Phase-A 冻 backbone 后只剩 `text_gate_param` → `does not require grad`。

起了两次：Beauty 04:56 结束，watcher/队列立刻拉一次；`run_after_beauty_0819.sh` 45 秒后再拉一次。两次都是同一 yaml 错误。`||` 吃掉失败后跳去 UniSRec。

**同一坑有没有出现在别处：** 没有出现在主表 / UniSRec / Hybrid。

| 脚本 | yaml | 覆盖 `disable_text_feature:false` | |
|---|---|---|---|
| `run_5090_0813_experiments.sh` / `resume2.sh` | 同一个 baseline eq yaml | **有** | 能训 |
| **`run_early60eq_fair_0818.sh`** | 同上 | **没有**（0813 抄漏） | **本次崩溃** |
| Beauty 主表 0818 | **V3** yaml（默认 false） | 不需要 | 已成功 |
| Hybrid 0818 | MV V3 + dict 显式 false | 有 | 尚未跑，不会这样挂 |
| Pure LLM 0817 | baseline_ts yaml（默认 true） | **有** false | 已完成，Last .0046 证明文本开着 |
| 8/16 `tab:early` 现用 ckpt | V3 yaml，只改 split | yaml 自带 false | **不会** 45 秒挂 |

危险组合只有：**baseline yaml + 当文本模型训 + dict 不关 disable_text**。修法：EXTRA 补上 0813 那两行，或改用 `sasrec_align_*_v3_ts_early60.yaml` 再覆盖 split 为 `[0.6,0.1,0.1]`。

### 4.4 协议锁定：no-cross 是对的；yaml 默认仍是 true

0806 / spec：主方法 **不要 Cross / 不要再扩 boost**。主表脚本用 `config_dict` 覆盖为 false。

但 `sasrec_align_*_v3_ts.yaml` **文件里仍写着** `use_cross: true`、`cold_text_boost: 3.0`、路径 `item_text_emb.base.npy`。手跑 yaml 或 dict 只改 split，会把 Cross+boost+非 TS 带回来。

| 实验 | `use_cross` | 文本路径 |
|---|---|---|
| `wrap_beauty_*`（今晨主表） | **False** | `*.ts.npy` |
| `tab:early` Early 行（8/16–17） | **True** | `*.npy` |
| 今早挂掉的公平 Early60 | False（dict 写了） | `*.ts_early60.npy`，但文本开关没开 |

Early 行「有问题」= **和 Main 不是只差训练窗**（多了 Cross、boost=3、yaml lr/do、非 cutoff-fit），不是看起来像 ID。split 已经是等 horizon。

### 4.5 「TS 重跑」≠「报 test 不报 valid」。泄露在特征拟合

三件事不要混：

1. **交互 GTS**  
   RecBole `TS [0.8,0.1,0.1]` / Early `[0.6,0.1,0.1]`。训练交互不含未来窗口。表里 Main/Early **都已经是这一层**。

2. **文本 cutoff-fit（文件名 `.ts.npy`）**  
   TF-IDF 词表/IDF/SVD、LLM center/whitening **只在 train-cutoff 前出现过的 item 上 fit**。  
   `*.npy` / `4views`（无 `_ts`）= 全 catalog 拟合 → **catalog 泄露**（未来商品的文本统计进了 SVD/白化）。  
   Qwen 对 title 的前向编码是静态的；有泄露的是 **拟合统计量**。  
   Toys/Grocery 主表、今晨 `wrap_beauty_*` 用 `.ts.npy`。`sg_beauty_*` 和 8/16 Early 用 `.npy`。

3. **指标：valid vs test vs GTS-Last**  
   选模可以用 valid。正文数字要用测试窗 **GTS-Last warm**（`eval_gts_last.py`），不是 RecBole `best_valid`，也不是 GTS-All。  
   当前 `tab:early` Early 行已经是 Last warm，**不是因为报了 test 才泄露**。泄露是第 2 条。

### 4.6 Beauty LOOV 仍不能换 `tab:loov`

`analyze_leave_one_view_out.py` 走 RecBole 默认 test loader：eval targets=12001，Full MRR≈.016，与训练日志 GTS-All 一致，**不是** GTS-Last warm（n_warm=3137，MV Last ~.0064）。csv 没有写入逻辑所以是空的。Toys LOOV 正文绝对值同样可疑（0818 已记，用 `wrap_toys_mv_*` 只 eval 即可）。

### 4.7 Discussion 文字坑（不需等 GPU）

- Pure LLM：已有 GTS-Last Beauty **.0046±.0001**、Toys **.0018±.0007**。正文若仍写 GTS-All .0256/.0190「超过 ID」是错的。
- 0817 Hybrid：NON_TS + cross + boost，**不能**写 +20% / +83%。公平 Hybrid 可砍；砍则现在改掉这些句子。

---

## 五、ETA（北京，08:42 起算）

| 完成度 | 时刻 |
|---|---|
| 主表 Beauty Last/Rand | **04:56 已齐** |
| 主表含 Toys UniSRec 列 | **今天 10:00–10:20** |
| 正文全部新数据（含公平 Early） | 须手动插 Early60 → **今天 16:30–18:30**；不插则 RQ2 继续用 8/16 不公平数字 |
| 附录含 Hybrid | 不插 Early：**周四 00:00–02:00**；插了：周四上午 |
| 砍 Hybrid | 附录在 **今天 10:20** 没有未完成实验 |

---

## 六、单位接下来改论文（建议顺序）

1. **现在就能改**：`tab:protocol` / `tab:ndcg_hit` / `tab:perseed` / `fig:protocol` 的 Beauty GTS Last/Rand（新 CSV）；`tab:early` **Main 行**改成 .0053 / .0060 / .0064。Early 行先留 8/16，caption 不要写成「与 Main 只差 cutoff」。
2. Setup：已是 cutoff-fit 的 Toys/Grocery/新 Beauty 可以写死；Beauty 旧 `sg_*` 不要再当 cutoff-fit。`d=256`（不要写 64）。
3. Discussion：Pure LLM 换成 Last 数字；删或改 Hybrid 增益句。
4. **先不要等 Early60** 再改主表。公平 Early 修脚本后再跑。
5. ~10:20 拉 `gts_last_eval_unisrec_toys_ts.csv` 和 Random 三个 seed，换 Toys UniSRec 列。
6. 不要把今早 LOOV md 的 .016 写进 `tab:loov`。

5090 拉数：

```bash
scp charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/gts_*unisrec_toys_ts*.csv \
    charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/experiment_status_20260819.md \
    paper_recsys/
```

（Beauty TS CSV 本地已有，不必再拉。）

---

## 七、PID / 路径速查

| 用途 | 路径或 PID |
|---|---|
| 跟随队列 | 775096 `run_after_beauty_0819.sh` |
| UniSRec 脚本 | 814388 `run_unisrec_toys_ts_0818.sh` |
| GPU | UniSRec Toys seed2026，`saved/unisrec_toys_gts_ts_seed2026` |
| Beauty Last | `paper_recsys/gts_last_eval_beauty_ts.csv` |
| Beauty Random | `paper_recsys/gts_random_eval_beauty_ts_seed{42,123,456}.csv` |
| 8/16 Early Last（现表） | `paper_recsys/early60eq_gts_last_eval.csv` |
| 公平 Early 脚本（坏） | `scripts/run_early60eq_fair_0818.sh` |
| 正确覆盖范例 | `run_5090_0813_experiments.sh` 的 `TF_EXTRA` 含 `disable_text_feature:false` |
| 正确 split yaml | `sasrec_baseline_50ep_stratified_ts_early60_eq.yaml` 是 ID-only；V3 用 `sasrec_align_*_v3_ts_early60.yaml`（注意 split 可能是 `[0.6,0.1,0.3]`，覆盖成 `[0.6,0.1,0.1]`） |
