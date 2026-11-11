# 0803 指导：整体梳理 + 结论对照

> 依据：[`0803zhidao.txt`](0803zhidao.txt)  
> 更新：2026-08-05  
> 原则：**GTS-Last / GTS-Random 优先于 GTS-All**；暂不拆 `sections_*.tex`。

---

## 一、老师定调（一句话）

不要再维护 MV-Align 新模型叙事。  
真正故事是：**严格时间评测会改变 TF-IDF / LLM / multi-view 的结论**，以及 **更多 prompt view 未必带来更多有效信息**。

切入点必须是：protocol 改变的是 *text representation 排序*，再用表示冗余解释 MV 无增益——而不是“又一篇 LOO vs GTS”。

---

## 二、论文三问 → 当前结论（方向已立，主表协议未锁死）

| RQ | 老师期望 | 现有证据（协议标注） | 结论状态 |
|---|---|---|---|
| **RQ1** Protocol effect | LOO vs GTS 是否改变 TF/LLM/MV 排序 | Last 主表已写入；LOO vs Last vs All† 对照表已有；**Random 跑中** | **方向✅**；写作已切 Last；差 Random 列合并 |
| **RQ2** Temporal robustness | Main/Early 谁更稳；与 shift 同现 | Beauty Early60 现为 All；`ew_beauty_*` ckpt 在 | **方向✅**；**Early Last/Random 已排入队列（P1）** |
| **RQ3** Why no MV gain | CKA + ER + LOOV 三角；勿单靠 CKA | CKA/ER/LOOV/cost ✅（LOOV 仍 All） | **机制✅**；LOOV Last 可选不阻塞 |

**跨 RQ 稳定结论（可写进摘要）**
1. 文本始终显著优于 ID-only。  
2. TF / LLM / MV **相对排序对协议与时间窗口不 robust**（乱序本身是发现）。  
3. Prompt 四 view 高冗余；TF–LLM 低 CKA → 互补存在，但简单堆 view 吃不到。

**尚不能写成“主表最终数字”的部分**：凡依赖 GTS 绝对值/细排序的句子，需以 **Last（已有）或 Random（未做）** 重述后定稿。

---

## 三、协议身份：已确认 + 缺口

| 问题 | 答案 |
|---|---|
| 训练日志里的 GTS 是什么？ | **Successive-All（GTS-All）**：每 user test 期内全部 next-item step |
| 实证规模 | Beauty 4520 users → 12001 samples；Toys 4873 → 13186 |
| 与标准 next-item 的关系 | All 存在 task mismatch；老师明确 **Last / Random 更重要** |
| **GTS-Last** | ✅ 已对 30 ckpt 重评（`gts_last_eval.csv`），含 warm 列 |
| **GTS-Random** | ❌ **未做**（无脚本结果、无 CSV） |
| GTS-All | ✅ 有（训练 log / wrapup）；应降为 **Appendix / contrast** |

### GTS-Last 已得数字（test MRR@10，3-seed mean，min5）

| | Beauty All-user | Beauty Warm | Toys All-user | Toys Warm |
|---|---:|---:|---:|---:|
| ID | 0.0022 | 0.0032 | 0.0019 | 0.0031 |
| TF | 0.0036 | 0.0051 | 0.0025 | 0.0040 |
| LLM | 0.0041 | 0.0059 | 0.0022 | 0.0035 |
| MV | **0.0042** | **0.0061** | **0.0026** | **0.0042** |
| UniSRec | 0.0036 | 0.0052 | 0.0026 | 0.0041 |

读法（对 RQ1）：
- Last 下绝对值远低于 All；Beauty 细排序变为 **MV≈LLM>TF≈UniSRec≫ID**（与 All 上 TF 略强不同）。  
- 文本变体差距多在 ~1 std 内 → **“排序不 robust / 无稳定 MV 增益”** 与 All 故事兼容。  
- **不能**再把 All 的 “TF 最强” 写成唯一主结论；主文需以 Last（+ 理想情况再加 Random）为准。

---

## 四、老师“只做三件事”逐条勾选

### 4.1 锁定 protocol

| 要求 | 状态 | 说明 |
|---|---|---|
| 主表用 **GTS-Last 或 GTS-Random** | ⚠️ 进行中 | Last ✅ 已写入 `tab:main`；**Random 5090 重评中**（`run_5090_gts_random.sh`） |
| warm targets + train-cutoff catalog | ✅ | 主表同时报 All-user / Warm；主张以 Warm 为准 |
| TF/SVD/whiten 仅训练期拟合 | ✅ 实验设定如此 | 正文已写清 |
| All-target / unseen → Appendix | ✅ | `app:gts-all`；主对比表仅作 $^\dagger$ 参考行 |
| Clean LOO 只作 contrast | ✅ | `tab:protocol` 对比行，不承担主结论 |

### 4.2 锁定主文数据

| 主文要素 | 状态 | 备注 |
|---|---|---|
| LOO vs GTS 聚合表 | ✅/⏳ | `tab:protocol` = LOO vs **Last** vs All†；Random 列待 CSV 合并 |
| GTS Main / Early 3-seed | ⚠️ | Early 仅 Beauty、且为 **All**；未 Last/Random |
| UniSRec external | ⚠️ | All+min5 修正 ✅；**Last 已在 gts_last 中有 6 行**，主表可直接用 |
| temporal-shift 表 | ✅ | 关联表述，非因果 |
| CKA heatmap + ER + LOOV 组合图 | ✅ | 图已有；LOOV 为 All |
| Warm / per-seed / Cross / unseen → 附录 | ⚠️ | 材料齐，正文布局未统一 |

### 4.3 轻量补实验

| 项 | 状态 |
|---|---|
| CKA 3 次子样本 mean±std | ✅ Beauty + Toys |
| LOOV 四 view 全量 | ✅ Beauty + Toys（zero-mask） |
| 参数/推理成本表 | ✅ `model_cost_table.csv` |
| **不要**再扩 4×256 / boost / 救 Cross | ✅ 未再扩 |

---

## 五、三个科学问题：是否“已经得出结论”？

### ✅ 已可下结论（不依赖 Random）

1. **叙事转向成立**：乱序 + 文本>ID + MV 无稳定增益，足够支撑新题目。  
2. **RQ1 方向**：LOO vs 时间协议会改变/拉近 TF–LLM–MV 排序，并抬升绝对值。  
3. **RQ2 方向**（Beauty）：Early 窗口 TF 仍最稳；性能跌与 shift 统计同现。  
4. **RQ3**：三角证据完整，可写“prompt view 冗余；TF 与 LLM 更互补”。  
5. **协议身份已澄清**：原 GTS = All；Last 重评已完成。  
6. **Cross**：只能 negative appendix。  
7. **投 Main、Findings 兜底**：老师已背书；重点是重写叙事与统一协议，不是继续炼模型。

### ⚠️ 结论需改口 / 降级表述（有 Last，缺 Random）

| 原 All 主文说法 | Last 下应如何说 |
|---|---|
| “GTS 上 TF 最强” | 改为：“在 Successive-All 上 TF 略强；在 **GTS-Last** 上 Beauty 为 MV≈LLM≳TF，差距近方差；**相对优劣不 robust**” |
| 主表数字 0.0165 / 0.0113 | 主表应换 Last（或并列 All 于附录）；摘要同步 |
| Early “与 Main 同序” | 仅对 All 成立；若主协议改 Last，Early 需重评或标明协议 |

### ❌ 尚未得出结论 / 明确缺口

| 缺口 | 重要性 | 建议 |
|---|---|---|
| **GTS-Random 全表** | **高**（老师点名 Last *或* Random；两者都更重要） | 对同一批 ckpt 做 Random-target 重评（不重训），与 Last 并列主文或主文其一、另一附录 |
| Early60 的 Last/Random | 中 | 有算力则补；否则正文写明 Early 证据来自 All，作 robustness 辅助 |
| LOOV / 机制分析在 Last 上 | 低 | RQ3 是表征性质，All 上 LOOV 可保留；可选 Last 抽查 1 seed |
| Toys Early | 低 | 不阻塞 |
| 正文主表切换 + 8 页扩写 | 高（写作） | 不拆细 tex；在 `main_wsdm2027.tex` 整篇推进 |

---

## 六、推荐主文协议（按 0803 更新）

```
主结论协议（锁定）:
  GTS-Last  + warm metrics 为主报告
  （理想）并报 GTS-Random 同表或紧邻列  ← 目前缺，优先补

对比协议（非主结论）:
  Clean LOO  — protocol contrast only
  GTS-All    — Appendix / “RecBole native successive”

特征泄漏控制（全程）:
  train-cutoff fit TF-IDF / SVD / whiten
  min_train_interactions = 5
```

**Random 与 Last 的关系（写作口径）**  
- 两者都比 All 更接近标准 next-item。  
- Last = 每用户最后一个 test target（确定性）。  
- Random = 每用户从 test 期 successive targets 中随机抽 1 个（应固定 seed 可复现）。  
- 若 Random 与 Last 排序一致 → 主文更硬；若不一致 → 恰成 “protocol sensitivity” 的一部分，不可藏。

---

## 七、整体论文骨架（不拆文件，单 tex 推进）

仍对齐 `main.tex` 六段，但 **Method → Study Design**，主表数字跟 Last（+Random）：

1. **Introduction** — 协议改变文本表征结论；RQ1–3  
2. **Related** — eval leakage + multi-view complementarity（少写 Cross）  
3. **Study Design** — LOO / All / **Last / Random** 定义 + leakage + 四表征 + 共享 SASRec  
4. **Setup** — 数据表、超参、复现  
5. **Results**  
   - RQ1: LOO vs **Last**（Random 若有则三列）；UniSRec  
   - RQ2: Main/Early（标明协议）；shift 表  
   - RQ3: CKA+ER 图 + LOOV + cost  
6. **Conclusion + Limitations** — 缺 Random/Successive 全量可写进 limitation（若 Random 补上则删）  
7. **薄 Appendix** — All、unseen、Cross、per-seed  

页数：现 ~4 → 目标 ~8（见 `outline_wsdm2027_8page.md`，主表协议改为 Last/Random 优先）。

---

## 八、总判

| 维度 | 判读 |
|---|---|
| 科学故事 / 三 RQ 方向 | **已够写**，不必再炼模型 |
| 老师点名的 Last | **实验已完成**，**写作未切换为主表** |
| 老师点名的 Random | **未完成** → 当前最大实验缺口 |
| 轻量分析（CKA/LOOV/cost） | **已完成** |
| 主文数据清单 | **材料齐、组装未齐** |
| 能否说“0803 问题都已得出结论” | **不能**：方向结论大多已有，但 **Random 缺失 + 主表仍偏 All** 使协议锁定未闭环 |

**今晚优先级队列（5090，`run_5090_after_random_then_grocery.sh`）**
1. ⏳ **P0** Beauty/Toys GTS-Random（进行中）→ 自动 `build_main_tables_gts.py`  
2. ⏳ **P1** Beauty Early60 Last+Random（`run_5090_early60_last_random.sh`，9 ckpt，不重训）  
3. ⏳ **P2** Grocery 对齐协议重训 + Last/Random（`run_5090_grocery_gts_wrapup.sh`）  
4. 写作：主表/摘要已切 Last；Random/Early/Grocery CSV 齐后合并进 `main_wsdm2027.tex` + 凑 8 页。
