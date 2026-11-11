# 实验进展 2026-08-22（去 AI 味道 / 引用口径）

全文截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京 **Aug 25 19:59**）。

**时间锚点：北京下午。** 本日未重跑主实验、未改主表数字。工作全是措辞、引用与去 AI 套话。

**记日志：** 自 0823 去 AI 稿起，**每一次**正文改动都追加到文末「变更流水」。改 0823 的同一轮必须写一行，包括小改。

## 稿本怎么分

| 角色 | 文件 | 说明 |
|------|------|------|
| 去 AI 前主线（可回退） | `main_wsdm2027_v3_0822.tex` + `_zh_0822.tex` | 去 AI 前冻结 |
| **当前工作稿（汇报用这个）** | `main_wsdm2027_v3_0823.tex` + `_zh_0823.tex` | 本日全部去 AI / 引用修正 |
| Overleaf | `overleaf_deploy/` 同名 0823 | 与 0823 同步；`0822` 仍留在目录 |

老师意见：`0821zhidao.txt`。去 AI 规则：`.cursor/skills/wsdm-paper-de-ai/`。

**没有改：** 主表、Early、CKA/ER/LOVO、Hybrid 数字；Abstract *Three findings emerge*；RQ2 老师原句；RQ3 *already present … under our prompt design*。Related Work 里 Gusak 对 Successive 的文献句仍在（只删了 §3.1 / Implications 的重复）。

---

## 给老师：今天改了什么、为什么

核心是老师 0822「结论 → 限定 → 再降调」，以及 0821「不要通篇润色、核准过的数字/协议/措辞不要偷偷改」。只动句子，不动格子。细节见文末流水。

口头 1 分钟：本日无新实验。0823 相对 0822 只做文字：压叠层 hedge；引言/Related Work 按原文主张收引用（TALLRec 不当文本特征迁移；Salton 不当早期 CF；Xu 只管互补条件，more-views 结论留给我们；多视图标成本文问题）；*stronger text encoders* 收成「加 LLM 视图、再加更多视图」。Successive 只在文献/局限/未来工作里出现，不写成我们跑过。建议以 **0823** 为当前稿，0822 回退。

---

## 尚未做（非门槛）

- 老师过目后是否把 0823 提成主线。
- Overleaf 用 XeLaTeX 编译中文对照稿。
- `refs.bib` 里 Salton 1988 现无正文引用；可挂方法节当 TF-IDF 来源，或删条目。

---

## 变更流水（0822 → 0823，须逐次追加）

对照：`diff main_wsdm2027_v3_0822.tex main_wsdm2027_v3_0823.tex`。中文稿同步。未改数字。

### A. 叠层 hedge（Pass 1–2）

| # | 位置 | 原来 | 现在 | 理由 |
|---|------|------|------|------|
| A1 | RQ1 LOO 正文 | *A plausible explanation is that this overlap disproportionately benefits…* | *This overlap may disproportionately benefit…* | 标题已是 plausible account |
| A2 | RQ1 Last vs Random | hypothesis + *only as an association* + *the solid claim is* | *Last may…; our claim is only that changing target selection changes the winner* | 三层压成一层 |
| A3 | Hybrid | *we read this as no loss plus a small edge, not a large gain* | *margins within one std---no loss, not a large additive gain* | 去掉解说腔 |
| A4 | Intro | *This hypothesis is widely held*；*seldom examined* | 删 | 无出处人群判断 |
| A5 | RQ1 LOO 首句 | *not merely an additive shift; they also change* | *also change* | 套话节奏 |
| A6 | RQ1 Implications | *narrow claim … not a broad one* | 删 | 前两句已说过 |
| A7 | Implications 原第 4 条末 | *Such claims should be verified rather than assumed…* | 删 | 叠层说教 |
| A8 | Conclusion | *Three conclusions follow*；*community should first lock…* | 删 | 不克隆摘要骨架；去掉对社区说教 |
| A9 | Ethical | *mitigate the former… latter by treating… rather than* | 收成一句 | 避免再叠默认/非推荐 |

### B. 口径补丁（老师更早意见）

| # | 位置 | 原来 | 现在 | 理由 |
|---|------|------|------|------|
| B1 | UniSRec 过滤 | *inflates* … 0.02 | *raises* … 0.02 | 0813 少用 inflates。数字未变 |
| B2 | 中文 Intro LOO | 抬高绝对指标 | 给出更高的绝对指标 | 对齐英文 *yielding higher* |
| B3 | Future work (2) | *confirms the redundancy locus in Qwen* | *finds this redundancy already in Qwen outputs under our prompt design* | 0821：不能写成源自 encoder |
| B4 | Related Work 收尾 | *whether stronger text encoders and more views remain advantageous* | *whether adding LLM views—and whether adding more of them—remains advantageous* | 没有比多种编码器 |

### C. 引言 / Related Work 张冠李戴

| # | 位置 | 原来 | 现在 | 理由 |
|---|------|------|------|------|
| C1 | Intro 第 2 句 | *LLM-generated semantics*；*enrich ID embeddings*；cite UniSRec+TALLRec | 标题与描述；*enrich sequential recommenders*；cite UniSRec+Recformer | TALLRec 不是 ID 增强；UniSRec 不是 LLM 生成语义。0820 |
| C2 | 工作假设 | *LLM embeddings should beat bag-of-words*；*views should beat a single view* | *pretrained language embeddings should beat TF-IDF*；*views of the same item* | 对主表；避免稻草人词袋；多视图=同一物品 |
| C3 | Recent work | 把 TALLRec、FDSA「与文本」、*increasingly adopted* function/audience/style 写成文献结论 | 只留 UniSRec+Recformer 迁移；FDSA=物品属性；多视图另起 *We ask whether… in this study* | 侧面名是本文设定 |
| C4 | Intro 从业者段 | *Practitioners face a concrete decision… bag-of-words… depends critically* | *That comparison, and whether cutoff-fit TF-IDF already suffices, depends on how evaluation is set up* | 假二选一；词袋对上 TF-IDF |
| C5 | RW 早期系统 | *bag-of-words or review text* cite CDL+Salton | *item-content bag-of-words* cite CDL only | Salton 1988 是 IR 词加权；CDL 不是评论 |
| C6 | RW Modern | UniSRec/VQ-Rec/P5/TALLRec 都挂「编码标题做迁移」 | 前两篇迁移；P5+TALLRec=LM as recommender | 0820 同类 |
| C7 | RW LLM 段 | *growing practice… supposedly capture complementary facets* | 写成可选项；*This paper studies the second choice* | 不把本文 facet 写成学界惯例 |
| C8 | RW 多视图理论 | *more views do not guarantee benefit* [Xu] | *requires complementarity* [Xu]; *we test whether prompt views… provide it* | Xu 只管条件；more-views 结论是本文的 |
| C9 | Discussion | *Why can a bag-of-words representation match* | *Why can TF-IDF match* | 与主表同名 |

### D. 腔调 / Successive 重复

| # | 位置 | 原来 | 现在 | 理由 |
|---|------|------|------|------|
| D1 | Discussion 小标题 | Practitioner implications | Implications | 去掉咨询报告腔 |
| D2 | Implications 第 3 条 | *before investing in multi-view pipelines* | *before adding more prompt views* | 去掉 invest |
| D3 | §3.1 Random | *Random has the highest rank-correlation with Successive* | 删 | RW 已引 Gusak；此处像自己跑过 |
| D4 | Implications 第 5 条 | *Last and Random (or Successive)* | *Last and Random* | 我们没跑 Successive。RW / Limitations / Future work 仍保留 |

---

### E. 本段之后追加

| # | 位置 | 原来 | 现在 | 理由 |
|---|------|------|------|------|
| E1 | RW 多视图理论 | *requires complementarity; we test whether prompt views… provide it* | 只留 *requires complementarity* [Xu] | Related Work 惯例先写背景；*we test* 与段末 *We combine CKA…* 重复，且不像综述句 |
| E2 | RW `\paragraph{Information geometry of embeddings.}` 整段 | 4 行：failure modes [ethayarajh] + 谱结构遗传 + ER/dim 0.53 vs 0.98 | 删除 | ①与上一段重复（ethayarajh 各向异性、roy 有效秩、段末 We combine 已覆盖）；②引用越界：singular-value concentration / failure mode 不是 Ethayarajh 结论（他测词级 self-sim/MEV，且指出 BERT 低层 PC 静态向量优于 GloVe）；③"每视图继承相似谱结构"与表 tab:effrank 单视图 63.7/64=99.5% 冲突且无测量；④0.53/0.98 挑边，全文口径为 53–55% / 97–98%；⑤唯一新信息已在 RQ3 "expands the subspace far less than the dimension count suggests" |
| E3 | RW 多视图段 4 处剪枝 | Kornblith 分号后 cosine≈0 仍能测冗余；Ethayarajh 后 *geometric redundancy*；Roy 括注=熵；*These tools … embedding quality [ethayarajh]* | 分号后整句删；*geometric redundancy* 删；Roy 改为 *exponential of the entropy*；改为 *CKA has been used to compare layers and architectures [kornblith]*，缺口句只提 CKA+effective rank | 最小改动去引用越界：cosine–CKA 反差是本文 RQ3；Ethayarajh 未用 CKA/ER、不做 embedding quality；erank=exp(H) 不是 H |
| E4 | §3.4 Evaluation Details | *mean±sample-std over 3 seeds (42, 2024, 2026)* | *mean±std over 3 training seeds (42, 2024, 2026). GTS-Random further averages 3 target-sampling seeds (42, 123, 456) on those same checkpoints* | 与 §3.1 Random 的 (42,123,456) 扫读易混；sample-std 又撞 target-sampling；两组 seed 分工写清，不改数字 |
| E5 | Table~\ref{tab:perseed} Beauty seed 42 | TF+LLM 与 TF+MV 均加粗 `.00627` / `.00629` | 只加粗 `.00629`；`.00627` 去粗 | 与表1规则一致：行内最高；非精确平手不双加粗。五位小数是为了分开四位都显示 .0063 的两格 |
| E6 | RQ1 Implications / UniSRec | *wider under Random*；*into the 0.02 range* | 改为 *absolute scores are higher under Random*；Beauty ≈0.020、Toys ≈0.016 | Random 相对 ID 是 +42–78%，并不比 Last 的 +13–100% 更宽；附录 F Toys 无 min-5 是 0.016 不是 0.02 |
| E7 | RQ1 图注 / Last 条 / LOO 收束 / Implications / UniSRec | 图注复述 Toys 排序+非确定性冠军；*all text margins often within one std*；*advantage vanishes*；Implications 前两句再讲一遍 checkpoints / superiority unreliable；*min-5* + *dominates* | 图注只留 point estimates+≈+见表1；改为 seed-dependent winners；GTS 下 LOO 排序不再出现；Implications 只留 vs-ID +13–100%；five training interactions + leads | 去重复与机制越界；原则写入 wsdm-paper-de-ai skill §RQ1 results prose |
| E8 | RQ2 开篇 / 独句 / TF-IDF 段 | *To isolate training-window size…*；*No single text variant dominates…*；*Despite catalog shift* + *co-occur… do not explain* | 顾问 Early 口径（60/10/10，horizon 固定）；表4正文点明 Beauty；删占优独句；锁住后半 *empirical robustness observation* 保留，去掉 Despite 与第三层 | 表5–6 是 Main valid→test，不是 Main→Early；原则写入 skill §RQ2 results prose |
| E9 | RQ3 表3 caption + (1) | *all pairs exceed / $\ge$ 0.82* | $0.82$--$0.94$，点名 Toys Function--Style $.818$ | 该格均值 0.8176，表内 $.818\pm.002$；Abstract 四舍五入区间可留 |
| E10 | RQ3 开篇 | *cannot prove task-irrelevance [kornblith]* | CKA *measures Gram-matrix similarity [kornblith]* | Kornblith 只定义 CKA/Gram；任务无关是本文加 LOVO 的理由，不挂 [11] |
| E11 | RQ3 3584-d 收束 | *$4\times256$ not justified (tab:cost)* | *wider projection unlikely… already visible at 3584-d (tab:cka-prepost)* | 未做 4×256；cost 表是推荐阶段参数，不是投影维 |
| E12 | Discussion TF-IDF 段 | *adds only ~50% new capacity* | *only about doubles the effective rank ($64\to136$)* | 136/256 是占名义维一半；相对单 view 是约 2.1× 不是「多 50%」 |
| E13 | Limitations Redundancy locus | *256-d SVD-projected* | *64-d SVD-projected pre-whitening* | 正文三阶段是 3584 / 64 白化前 / 64 白化后；256-d 文件未进主表 |
| E14 | §7 TF-IDF 段 | 标题后再问一句 Why；*each dimension distinct*；*geometric diversity* 连说两遍 | 先写 Random 事实，再 First/Second；删 distinct；收束只留一句 | 标题已问 why；ER/dim 0.98 已够；去叠句 |
| E15 | §7 hybrid | *pair carries the useful signal*；*3 sampling seeds* | 删该句；改为 3 target-sampling seeds | CKA 只许可几何不相似已够；勿升级成有用信号 |
| E16 | §7 Implications 2/5 | Early 未点 Beauty；第 5 条 *do not overclaim / rankings reverse* | Early *on Beauty*；删第 5 条 | 表4 只有 Beauty；第 1 条已要求报告 GTS 目标模式 |
| E17 | §7 热目标 / cost | 排除 28–35%；*multi-view failure is not a capacity issue* | 30–38%；*limited multi-view gains are not a recommendation-stage capacity issue* | Beauty 30.6 / Grocery 29.6 / Toys 38.3；30–41% 仍是表5 全目标 unseen |
| E18 | §8 item 2 | Early 未点 Beauty；*same span as Main*；收束 *which text helps most* | *on Beauty*；60/10/10 + 末 20% 不用；删收束句 | 表4 只有 Beauty；horizon 长度固定、切片前移 |
| E19 | §8 item 3 | *prompt-view budget is better spent* | 只留 hybrid matches or slightly exceeds | CKA 不许可“预算该花在哪” |
| E20 | §8 收束 | *LLM upgrade / monotonically improve* + *not a free upgrade* | 只留实践默认 + *not a free upgrade* | 无来源 crowd-claim；Implications 已列默认 |
| E21 | §8 Future(1) / Ethical | Future 重报 hybrid 分数；伦理 *object of study rather than default* | Future 只留融合是否显著；伦理只留热目标诊断 | 第 3 条与收束已说过 |
| E22 | tab:gts-all | Beauty TF/MV 双加粗；正文 *leads or ties* | 只加粗 MV `.02413`；Toys TF 领先、Beauty 低 `.00002` | 非精确平手不双加粗（同 E5）；源 `.024107` / `.024126` |
| E23 | tab:intersection | 只给完整热列加粗；std 比表1 偏小 | 四列都标列内最高；std 改回 3 训练种子 sample std | caption 原先把交集列排除在加粗外 |
| E24 | 交集正文 | *ordering untouched*；Toys 54%；*roughly twice as easy* | 只保证 TF vs MV 谁领先；54--55%；*easier* | Toys Random 上 LLM/MV 会互换；2651/4873=54.4% |
| E25 | app:unisrec / RQ1 | *无 min-5 则 Last≈0.020/0.016* | 改为 All `.0180/.0139` vs Last `.0052/.0040` | `0.0204` 是 valid，不是去掉 min-5 的 GTS-Last |
| E26 | 贡献 2 / RW Gusak / §3.1 / Limitations Random | Early 未点 Beauty；*sampling seeds*；0.001–0.003 挂 [gusak]；§3.1 *ordering unchanged* | Early *on Beauty*+60/10/10；*target-sampling*；std 是我们测的；只保证 TF vs MV 领先 | 与 E8/E18/E24 对齐；Gusak 只说 Random 非确定，没给这个 std |
| E27 | app:alltarget / 交集 / 中文 caption | 交叉注意力负结果段；*preserves ranking*；*more importantly*；ZH Grocery/Random caption 未闭合 | 删交叉注意力；节名 All-Target；去掉无表的排序句；交集去掉 *more importantly*；补 ZH caption `}` | 无消融表；未闭合 caption 会导致中文稿编不全 |
| E28 | §3.2 / 附录 Grocery | *same-platform … not domain diversity* 与 Limitations 再降调 | 设置与附录各删一句；只留 Limitations 的 Amazon-only | 同一限定说三遍 |
| E29 | 跨段叠层 | 同一限定在设置/RQ/附录/Limitations 各写一遍 | §3.1 交集只留指针；RQ2 删 Early 60/10/10 复述；表6 caption 不重复「不进热分」；Limitations Random 不重报种子列表；RQ3 加宽投影句不重报 locus；讨论删命名复述；Implications 1 不重报 4–7×；删附录全目标整节；交集/GTS-All/UniSRec 附录去掉预告结论与再降调 | 定义与锁定句留原处 |
| E30 | 附录/参考文献排版 | `[h]` 表漂走导致节号像 C→E；末页未 `\balance`，[4] 孤悬 | 附录表改 `[ht]`；节前 `\FloatBarrier`；Grocery 正文提前；参考文献前 `\balance` | 只动版式 |
| E31 | 附录/参考文献排版 | `\balance` 放在 E 节前截短第 8 页左栏；Overleaf 残留 `\FloatBarrier`+`\enlargethispage` 把 [26] 拆到几乎空白的末页 | 附录表保持 `[H]`；去掉 C 节前 `\Needspace`；`\balance` 只放在 `\bibliography` 前；附录 `\arraystretch` 0.88 并收紧 caption 间距；同步 overleaf 0823 | 只动版式；表数字未改 |
| E32 | 摘要 / 表1 caption / RW / LOO / RQ2 / RQ3 综合 / §7 | 摘要 Early 未点 Beauty；表1 caption 重报排序；RW *little empirical attention* + *precisely because*；LOO *memorize*；RQ2 正文与顾问原句叠层；RQ3 *not harvested / 复制四次*；TF-IDF 收束 *geometric diversity per dimension*；Hybrid *useful signal*；Implications 3–4 重复 *more useful axis* | 摘要补 *on Beauty*；caption 只留协议/加粗/Grocery UniSRec；RW 改为同一协议下可换序；LOO 只留共现统计仍可用；删 RQ2 叠句；综合停在 CKA $0.18$--$0.20$；Hybrid 改为 *usable at ranking time*；第4条只留 hybrid 事实 | 上次复核 9 句；中英+overleaf 同步；未改格子 |
| E33 | 表1 / 表2 caption / 附录 Random 表 | Beauty Random TF/MV 双加粗；表2 caption 漏 Toys NDCG；列名 *Sample seed* | 只加粗 `.01244`（数字未改）；caption 补 Toys 与 Grocery NDCG；列名 *Target seed* / 目标种子；交集 *target-sampling seed* | 与 E5 同一加粗规则；避免和训练种子撞名 |
| E34 | 引言 / RQ1 / Implications 3 / 附录交集 | 引言 Early 未点 Beauty；Beauty Random 写 *tie*；第3条像 TF+LLM≈TF+MV；交集误引表2 | Early *on Beauty*；*4-decimal tie* / 四位显示打平；第3条只留先诊断冗余；交集只引表1 | 口径；未改格子 |
| E35 | overleaf_deploy | 同名 0823 已对齐，但默认 `main.tex` 仍是 8/12 旧稿（748 行） | `main.tex` 换成当前 0823 英文；重拷中英 0823、`refs.bib`、图 1/CKA | Overleaf 默认编 `main.tex`；0822 未动 |
| E36 | 摘要第二发现 / 图1(a) caption | 摘要 *on Beauty, no text representation is robustly dominant across temporal windows*；图1(a) *LLM views … (0.84--0.94)* | 摘要改 *in a temporal-window experiment, no text representation is robustly dominant: shifting the evaluation window earlier…*；图1(a) 区间改 `0.82--0.94` | ①摘要点名单一数据集不专业，改用「一项时间窗口实验」隐含限定范围（仍满足 skill：不写成三数据集都跑 Early）；②图1(a) 区间与正文/表4 口径不符（Toys Function--Style $.818$ 应含入），统一为 $0.82$--$0.94$；中英同步，未改任何格子数字 |
| E37 | 摘要 TF--LLM CKA / 图1(a)(b) caption | 摘要 `CKA $\approx 0.19$`；图1(a) `TF--LLM CKA is low ($\approx$0.20)`；图1(b) `4-view concat 136/256 (53\%)` | 摘要与图1(a) 统一为 `0.18--0.20`；图1(b) 改 `136--141/256 (53--55\%)` | ①TF--LLM CKA 全文在 0.19/0.20/0.18--0.20 间横跳，统一到表4支持的 $0.18$--$0.20$（正文 144/611/631/697 已是该写法；精确值句 553 保留）；②图1(b) 单写 53% 与正文/表6 的 53--55%（Beauty 136 / Toys 141）不符，改为区间；中英同步，未改表格格子数字 |
| E38 | Introduction L113 | `yielding higher absolute metrics and potentially distorting model comparisons` | `making the impact on absolute metrics unpredictable and potentially distorting model comparisons` | Ji 2023 原文明确说 "It is not true that more future data leads to higher accuracy"、"impact is unpredictable"；"higher" 不准确，改为 unpredictable；中英同步 |
| E39 | LLM embedding pipeline (L235–257) | 黑体标注 checkpoint/layer/pooling；prompt 与 post-processing 连在一段 | 去掉黑体；prompt 和 post-processing 各自独立分段（enumerate list）；ZCA 公式改用 `\bigl/\bigr` 排版 | 老师+用户反馈：去黑体、prompt 单独列出、改善排版密度；中英同步 |
|| E40 | Related Work §评估协议 L169 | `session-based~\cite{ludewig2018evaluation} and general collaborative filtering~\cite{meng2020temporal}` → `general collaborative filtering~\cite{meng2020temporal}` | Ludewig 已在 L167 专门引用，L169 再引一次重复；删除 session-based 分支，只保留 Meng | 中英+overleaf 同步 |
|| E41 | 参考文献排版 L908–914 | `\balance` 在 `\begingroup` 内部 `\bibliography` 前 | `\balance` 移至 `\begingroup` 之前 | `\balance` 在 group 内可能失效或截断引用页平衡；移出确保作用于全文末；中英+overleaf 同步 |


全文复核汇总（数据 / 去 AI / 引用 + A1–E27）：`paper_recsys/0823_deai_full_changelog.md`。

## 之后怎么追加

每改 0823 一句，在上表对应小节末加一行（E1, E2, …），三列：位置 / 原来→现在 / 理由。不要另开日期文件，除非老师把 0823 定成主线后再开 0823 当日档。
