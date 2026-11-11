# 0823 去 AI 味道之后：全文复核与全部改动

日期：2026-08-22（北京）至 08-23 凌晨。  
工作稿：`paper_recsys/main_wsdm2027_v3_0823.tex` + `_zh_0823.tex`（Overleaf 同名拷贝已同步）。  
回退点：`main_wsdm2027_v3_0822.tex`。  
规则：`.cursor/skills/wsdm-paper-de-ai/`。逐条流水也在 `experiment_status_20260822.md`。

原则：只改句子和加粗规则，**不改正文主表格子**（Table 1–6、CKA/ER/LOVO、Hybrid 均值）。UniSRec Grocery Last `.0107` 等 cutoff-fit 重训 CSV，未动表。

---

## 1. 本轮全文复核结论

### 1.1 数据

主表数字与已锁定 CSV 一致，本轮没有改格子。复核中修的是**口径和加粗**，不是均值：

| 核查 | 结果 |
|------|------|
| Table 1 Last / Random / LOO 均值 | 与既有 GTS-Last / Random / LOO 源表一致 |
| Beauty Random TF/MV 五位小数 `.01239` / `.01244` | 保留；四位都是 `.0124` |
| Table 4 Early 仅 Beauty；`.0042 / .0037 / .0039` | 对；彼此差 ≤ 0.001 |
| CKA 区间 0.82–0.94；Toys Function–Style `.818` | 对（该格均值 0.8176） |
| TF–LLM CKA 0.18–0.20；拼接 ER 53–55%；`64→136` | 对 |
| Hybrid Last `.0070/.0049`，Random `.0128/.0114` | 对 |
| 热末项目标排除 30–38%（Beauty 30.6 / Grocery 29.6 / Toys 38.3） | 对；表 5 全目标 unseen 仍是 30–41% |
| 附录 GTS-All Beauty TF `.02411` vs MV `.02413` | 源 `.024107` / `.024126`；只加粗最大值 |
| 附录交集**均值** | 与 `gts_intersection_summary_20260820.txt` 一致 |
| 附录 UniSRec 表 All/Last/Random | `.0180/.0052/.0096`（Beauty），`.0139/.0040/.0085`（Toys）对 |
| 旧句「无 min-5 则 Last ≈ 0.020 / 0.016」 | **错**。`0.0204` 是 valid，不是去掉过滤后的 GTS-Last。已改为 All vs Last |

**未改、仍待实验：** Grocery/Beauty UniSRec cutoff-fit 重训进行中。表 1 Grocery UniSRec `.0107` / 「Grocery 上最强」在新 CSV 对完之前不动。

### 1.2 去 AI 味道

老师点名的模板是「结论 → 限定 → 再降调」。0823 已按段压成**一层证据匹配的限定**。摘要 `Three findings emerge`、RQ2 顾问原句、RQ3 locus、结论定位句按锁定保留。

本轮复核后仍可留的轻痕迹（故意不自动再削）：

- 引言 `This hypothesis rests on fragile evaluation foundations`
- 引言 `specifically decisive`
- 结论 `not a free upgrade`
- Related Work `not a niche issue but a systemic concern`（综述收束，未再动）
- 附录 `Second, and more importantly`；`Five questions remain open`

### 1.3 引用

已按原文主张收过一轮（TALLRec / UniSRec / Salton / Xu / Kornblith / Ethayarajh）。本轮再核：

| 引用 | 正文用法 | 判断 |
|------|----------|------|
| UniSRec, RecFormer | 预训练语言表征可改进文本增强 SR / 迁移 | 对（0820 口径） |
| TALLRec, P5 | LM 本身当推荐器；TALLRec = instruction tuning | 对；不再挂「LLM 描述胜过 TF-IDF」 |
| FDSA | 物品属性上的 feature-level attention | 对 |
| CDL | 早期内容 bag-of-words + CF | 对；已去掉 Salton |
| Xu 2013 | 多视图**需要**互补 | 对；more-views 无效是本文结论，不挂 Xu |
| Kornblith | CKA = Gram 矩阵相似性 | 对；不再挂「不能证明任务无关」 |
| Ethayarajh | 语境向量各向异性 / 窄锥 | 仍引用；已删越界的「谱结构遗传 / embedding quality」 |
| Roy | 有效秩 = 归一化奇异值熵的指数 | 对 |
| Gusak 2025 | Last/Random/Successive/All 定义与相关 | 对。0.001–0.003 **是我们测的**，已不再挂 Gusak |
| Ji / Sun / Meng | LOO vs 时间分割会改分数与选型 | 对 |
| Krichene | 采样指标扭曲排序 | 对 |
| Salton 1988 | `refs.bib` 仍有，**正文已无引用** | 可删条目，或挂方法节当 TF-IDF 来源 |

**未改、仅记录：** `hidasi2018recurrent`（Top-k Gains / 排序损失）被用来支撑「细微协议选择会大幅改序列模型排序」以及「会话推荐里的时间协议」。更贴的是评测/损失论文，不是 GTS vs LOO。截稿前未重写 Related Work 这段，避免再开一轮润色。

---

## 2. 今日全部改动清单（0822 → 0823，含本轮 E26）

编号与 `experiment_status_20260822.md` 一致。中英文同步。

### A. 叠层 hedge

| # | 位置 | 原来 | 现在 |
|---|------|------|------|
| A1 | RQ1 LOO 正文 | 标题已是 plausible，正文再写 *A plausible explanation is* | *This overlap may…* |
| A2 | RQ1 Last vs Random | hypothesis + *only as an association* + *the solid claim is* | *Last may…; our claim is only that changing target selection changes the winner* |
| A3 | Hybrid | *we read this as no loss plus a small edge* | *margins within one std---no loss, not a large additive gain* |
| A4 | Intro | *widely held*；*seldom examined* | 删 |
| A5 | RQ1 LOO 首句 | *not merely an additive shift; they also change* | *also change* |
| A6 | RQ1 Implications | *narrow claim … not a broad one* | 删 |
| A7 | Implications 原第 4 条末 | *Such claims should be verified rather than assumed* | 删 |
| A8 | Conclusion | *Three conclusions follow*；*community should first lock…* | 删（不克隆摘要骨架） |
| A9 | Ethical | *mitigate the former… rather than* | 收成一句 |

### B. 口径补丁

| # | 位置 | 原来 | 现在 |
|---|------|------|------|
| B1 | UniSRec 过滤 | *inflates* … 0.02 | *raises*（后被 E25 换成 All vs Last） |
| B2 | 中文 Intro LOO | 抬高绝对指标 | 给出更高的绝对指标 |
| B3 | Future (2) | *confirms the redundancy locus in Qwen* | *already in Qwen outputs under our prompt design* |
| B4 | RW 收尾 | *stronger text encoders and more views* | *adding LLM views—and adding more of them* |

### C. 引用张冠李戴

| # | 位置 | 原来 | 现在 |
|---|------|------|------|
| C1 | Intro 第 2 句 | LLM-generated semantics；enrich ID；cite UniSRec+TALLRec | 标题与描述；enrich sequential recommenders；UniSRec+RecFormer |
| C2 | 工作假设 | LLM 应胜过 bag-of-words | pretrained embeddings should beat TF-IDF；同一物品的多视图 |
| C3 | Recent work | 把 TALLRec/FDSA/侧面名写成文献结论 | UniSRec+RecFormer 迁移；FDSA=属性；侧面名是本文设定 |
| C4 | 从业者段 | Practitioners face… bag-of-words | cutoff-fit TF-IDF 是否已够，取决于协议 |
| C5 | RW 早期 | bag-of-words or review text + CDL+Salton | item-content bag-of-words + CDL only |
| C6 | RW Modern | 四篇都挂「编码标题做迁移」 | UniSRec/VQ-Rec 迁移；P5+TALLRec = LM as recommender |
| C7 | RW LLM 段 | growing practice… complementary facets | 可选项；*This paper studies the second choice* |
| C8 | RW 多视图 | *more views do not guarantee benefit* [Xu] | *requires complementarity* [Xu] |
| C9 | Discussion | *Why can a bag-of-words…* | *Why can TF-IDF…* |

### D. 腔调 / Successive

| # | 位置 | 原来 | 现在 |
|---|------|------|------|
| D1 | §7 小标题 | Practitioner implications | Implications |
| D2 | Implications 3 | *before investing in multi-view pipelines* | *before adding more prompt views* |
| D3 | §3.1 Random | *Random has the highest rank-correlation with Successive* | 删（RW 已引 Gusak，写在设置里像自己跑过） |
| D4 | Implications 5 | *Last and Random (or Successive)* | *Last and Random* |

### E. 分节复核（RQ1 → 附录）

| # | 位置 | 原来 | 现在 |
|---|------|------|------|
| E1 | RW 多视图 | *we test whether prompt views provide it* | 只留 Xu 的互补条件 |
| E2 | RW Information geometry 整段 | Ethayarajh failure mode + 谱结构 + ER/dim 0.53 vs 0.98 | **整段删除**（越界引用 + 与表冲突） |
| E3 | RW 多视图 4 处 | cosine≈0 仍能测冗余；geometric redundancy；erank=熵；embedding quality [ethayarajh] | 删越界句；Roy 改为 exp(entropy)；Kornblith 只比层/架构 |
| E4 | §3.4 | *mean±sample-std over 3 seeds* | 写清训练种子 vs 目标采样种子 |
| E5 | Table 3 Beauty seed 42 | `.00627` 与 `.00629` 双加粗 | 只加粗 `.00629` |
| E6 | RQ1 Implications / UniSRec | *wider under Random*；*into the 0.02 range* | *absolute scores are higher*；随后被 E25 替换 |
| E7 | RQ1 图注 / Last / LOO / Implications | 图注复述 Toys 排序；*often within one std*；*advantage vanishes*；Implications 再讲一遍 checkpoints；*min-5* + *dominates* | 图注只留 point estimates+≈；seed-dependent；LOO 排序不再出现；Implications 只留 vs-ID +13–100%；five training interactions + leads |
| E8 | RQ2 | *isolate training-window size*；占优独句；*Despite catalog shift* + 第三层 | 顾问 Early 口径；表 4 点明 Beauty；锁住 *empirical robustness observation* |
| E9 | RQ3 表 3 | *all pairs ≥ 0.82* | 0.82–0.94，点名 Toys Function–Style `.818` |
| E10 | RQ3 开篇 | *cannot prove task-irrelevance [kornblith]* | CKA *measures Gram-matrix similarity* |
| E11 | RQ3 3584-d | *$4×256$ not justified (tab:cost)* | 更宽投影也难消重叠，指向 pre/post 表 / 3584-d |
| E12 | §7 concat | *adds ~50% new capacity* | *only about doubles ER ($64→136$)* |
| E13 | Limitations locus | *256-d SVD-projected* | *64-d pre-whitening* |
| E14 | §7 TF-IDF | 标题后再问 Why；*each dimension distinct*；geometric diversity 两遍 | Random 事实 → First/Second；收束一句 |
| E15 | §7 hybrid | *pair carries the useful signal*；*3 sampling seeds* | 删升级句；*target-sampling seeds* |
| E16 | §7 Implications | Early 未点 Beauty；第 5 条 *do not overclaim* | Early *on Beauty*；删第 5 条 |
| E17 | §7 热目标 / cost | 排除 28–35%；*multi-view failure* | 30–38%；*limited multi-view gains are not a recommendation-stage capacity issue* |
| E18 | §8 第 2 条 | Early 未点 Beauty；*same span as Main*；*which text helps most* | *on Beauty*；60/10/10 + 末 20% 不用；删收束 |
| E19 | §8 第 3 条 | *prompt-view budget is better spent* | 只留 hybrid matches or slightly exceeds |
| E20 | §8 收束 | *LLM upgrade / monotonically improve* | 只留实践默认 + *not a free upgrade* |
| E21 | Future (1) / Ethical | Future 重报 hybrid 分；伦理 *object of study rather than default* | Future 只问融合能否显著；伦理只留热目标诊断 |
| E22 | 附录 GTS-All | Beauty TF/MV 双加粗；*leads or ties* | 只加粗 MV `.02413`；Toys TF 领先，Beauty 低 `.00002` |
| E23 | 附录交集表 | 只给完整热列加粗；std 偏小 | 四列都标列内最高；std 改回 3 训练种子 sample std |
| E24 | 附录交集正文 | *ordering untouched*；Toys 54%；*roughly twice as easy* | 只保证 TF vs MV 谁领先；54–55%；*easier* |
| E25 | 附录 UniSRec / RQ1 | *无 min-5 则 Last≈0.020/0.016* | All `.0180/.0139` vs Last `.0052/.0040` |
| E26 | 贡献 2 / RW / §3.1 / Limitations | Early 未点 Beauty；*sampling seeds*；0.001–0.003 挂 Gusak；§3.1 *ordering unchanged* | 与 E8/E18/E24 对齐；std 标明是我们测的 |
| E27 | 附录全目标 / 交集 / 中文 caption | 交叉注意力负结果；*preserves ranking*；*more importantly*；ZH 两处 caption 未闭合 | 删交叉注意力；节名 All-Target；去掉无表排序句；补 ZH `}`；本机 CJK 字体回退 | 无消融表；未闭合 caption + 非 XeLaTeX / 缺 Noto 会让中文稿编不全 |
| E28 | §3.2 / 附录 Grocery | *same-platform* 与 Limitations 再降调 | 只留 Limitations | 跨段叠层 |
| E29 | 跨段叠层 | 设置/RQ/附录/Limitations 各写同一限定 | 删复述；附录全目标整节去掉 | 定义与锁定句留原处 |

---

## 3. 锁定未改（不要为文风再动）

- 摘要 `Three findings emerge` 及三条 finding 骨架  
- RQ2：*we treat this as an empirical robustness observation rather than evidence for a causal mechanism*  
- RQ3：*already present in the raw Qwen outputs … under our prompt design*；*consistent with insufficient view complementarity*  
- 结论：*empirical investigation rather than a model-proposal paper*；*not a free upgrade*  
- 表 5 caption：*associations with catalog shift, not causal claims*  
- 变体名 TF / TF+LLM / TF+MV；UniSRec = 改编骨干  
- Last vs Random：同一 checkpoint，换目标选择，换赢家  
- Early：60/10/10，末 20% 不用；表 4 只有 Beauty  
- 主表与 CKA/ER/LOVO/Hybrid **格子数字**

---

## 4. 仍开放（非本轮正文）

1. UniSRec Beauty/Grocery `.ts.npy` 重训与 Last/Random 重评（5090）。新 CSV 出来再对表 1 / 表 2。  
2. `refs.bib` 中 Salton 1988 无引用。  
3. Hidasi 2018 是否值得换成更贴「时间协议」的文献：记录在案，未改。  
4. Successive：不重训；batch 评测约 1.5–2.5 h，但截稿前不建议换主表。  
5. 中文稿必须 XeLaTeX（不能 pdfLaTeX）。本机无 Noto CJK 时已回退 Songti/Heiti SC。

---

## 5. 涉及文件

| 文件 | 角色 |
|------|------|
| `paper_recsys/main_wsdm2027_v3_0823.tex` | 英文工作稿 |
| `paper_recsys/main_wsdm2027_v3_zh_0823.tex` | 中文对照 |
| `paper_recsys/overleaf_deploy/main_wsdm2027_v3_0823.tex` | Overleaf 英 |
| `paper_recsys/overleaf_deploy/main_wsdm2027_v3_zh_0823.tex` | Overleaf 中 |
| `paper_recsys/experiment_status_20260822.md` | 逐条流水 A1–E27 |
| `.cursor/skills/wsdm-paper-de-ai/` | 去 AI / 不叠句规则 |

0822 主线文件未改，可整文件回退。
