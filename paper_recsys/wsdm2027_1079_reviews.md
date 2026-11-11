# WSDM 2027 审稿意见整理（Paper 1079）

> 来源：CMT，WSDM 2027 Full Paper (Main Track)，Author 视图原文。  
> 整理日期：2026-11-04。本文件只归档和分条，不写 rebuttal，不改论文。  
> 老师据此给出的下一轮安排见 [`experiment_status_20261104.md`](experiment_status_20261104.md)。

| 项 | 内容 |
|----|------|
| Paper ID | 1079 |
| 标题 | Do More Text Views Help? A Leakage-Aware Evaluation of Text Representations for Sequential Recommendation |
| Track | Full Paper (Main Track) |
| 结果 | 拒稿 |
| 分数 | Weak Reject × 3（R1、R2、R4），Weak Accept × 1（R3） |

CMT 这一档没有给出 1–10 的数字分，只有 Overall Recommendation 的文字档。

---

## 一、分数和一句话

| 审稿人 | 推荐 | 他理解的论文 | 他认为没过的那一层 |
|--------|------|----------------|----------------------|
| R1 | Weak Reject | 三个 Amazon 数据集上，四种文本表示 × 不同协议的受控实证；协议改变排序、四 prompt 高冗余、复杂文本管线不稳 | 有现象，没有可操作的机制或可复用的改进；冗余来自 prompt 还是 Qwen 没分开；只有 Amazon、只有一个 LLM |
| R2 | Weak Reject | GTS 下 ID / cutoff-fit TF-IDF / TF+LLM / TF+MV 的 leakage-aware 对照；文本稳定优于 ID，但 TF、TF+LLM、TF+MV 的相对排序随 Last、Random、LOO 改变，且 Last 与 Random 用的是同一批 checkpoint | 主要是设定比较，贡献有限；只在 Amazon；方法之间太接近，难说显著；机制不够；写作还可改进 |
| R3 | Weak Accept | 时间泄漏可控的文本表示实证；协议会明显改变相对表现，多 view 增益有限是因为 view 高度冗余 | 范围窄：只有 Amazon、只有一个 LLM 家族；没跑 Successive；主结果是 warm item；CKA 本身不能证明没有互补的推荐信息 |
| R4 | Weak Reject | leakage-aware 的文本表示实证；文本总体优于 ID，TF-IDF 与 LLM 变体的相对排序对协议敏感，多个 prompt view 高度冗余 | 多 prompt 只测了自己的四 prompt + Qwen2.5-7B + 拼接融合，没有和 LLM-Rec、SC-Rec 比；表示方法窄，文本优于 ID 已有大量工作 |

R3 是唯一倾向接收的人，而且他的限制写得最具体，可以直接变成实验。R1、R2、R4 都是 Weak Reject，但理由不完全相同：R1 要机制和可操作方案，R2 要统计上站得住的差异和更好的机制，R4 要超出这一套 prompt/encoder/融合的比较。

---

## 二、共同认可（四份里反复出现）

下面是整理时归纳的交集，不是某一位的原句。

1. 问题本身值得做：文本增强的序列推荐、以及「更多 LLM view 是否真有互补信息」。
2. 评测是这篇的长处：时间划分、目标选择（Last / Random / LOO）、cutoff 前拟合文本统计，被单独点名为比单协议比较更有信息量。
3. 实验认真、可复现、分析不算只报一个分数。R3 明确写下同一 backbone、只改变输入表示，所以比较的是文本而不是架构。R1 把多维诊断（协议、CKA、复杂管线不稳）当成可靠。
4. 主结果他们读成了两句：文本优于 ID-only；TF-IDF、单 view、多 view 的相对排序随协议变，多 view 增益有限且 view 之间很像。

---

## 三、共同没过的地方

同样是归纳，后面各节保留原意。

| 缺口 | 谁写了 | 他们要看到的东西 |
|------|--------|------------------|
| 发现绑在 Amazon | R1、R2、R3、R4 | 非 Amazon，或至少说明不能外推 |
| 发现绑在一个 LLM | R1、R3、R4 | 另一个 encoder，或不同的 prompt family |
| 多视角冗余不能说成普遍现象 | R1、R3、R4 | 分开「四个 prompt 语义重叠」和「这个 encoder 把它们编成一样」；并和已有多 prompt 方法比 |
| Last / Random 只描述了现象 | R1、R2、R3 | 为什么同一批 checkpoint 上排序会变；Successive 至少小规模补一次；warm 排除了多少、结论变不变 |
| 均值接近，结论偏满 | R2 | 统计显著性和实际差别；R2 直接说 performance similar，hard to say statistical significance |
| 没有可复用的下一步 | R1、R4 | R1 点名 weighted fusion、contrastive constraint、view selection/gating；R4 点名和 LLM-Rec、SC-Rec 比。两位都认为「发现风险」还不是「别人能用的改进」 |
| 文本优于 ID 本身不算新 | R4 | 这点先前工作已经反复做过，不能当主新颖性 |

---

## 四、可以变成实验或改写的条目

只列审稿人明确提出、且能检查做完没有的事项。不在这里决定哪些该做；取舍在 [`experiment_status_20261104.md`](experiment_status_20261104.md) 的四阶段里。

| # | 谁 | 类型 | 原要求 |
|---|----|------|--------|
| 1 | R1、R3 | 实验 | 另一个 LLM encoder，或一套明显不同的 prompt family，用来区分冗余是 Qwen2.5-7B 特有，还是多 view LLM 表示的一般性质 |
| 2 | R1 | 实验 / 分析 | 分开冗余来自四个 prompt（Description, Function, Audience, Style）的语义重叠，还是 Qwen2.5 对它们的编码收敛 |
| 3 | R1 | 分析 | Last 与 Random：抽样差异如何改变估计偏差、训练/测试分布错位、排序稳定性。要有可测的中间变量，不能只写「目标选择的确定性不同」 |
| 4 | R1 | 分析 | Early 窗口里 TF-IDF 更稳：补 TF-IDF 与 LLM 表示在不同时间分布下的机制比较 |
| 5 | R3 | 实验 | Successive 至少做小规模。正文把它写成最接近在线 next-item 的离线近似，主实验却只用 Last 和 Random |
| 6 | R3 | 写作 + 可核数字 | 主文始终写明结果是 warm target；给出排除了多少 post-cutoff target；看定性结论变不变 |
| 7 | R3 | 写作 | 高 CKA 只说明表示相似，不能单独推出「不同 view 没有互补的推荐信息」。下游消融已有证据，两套证据要接上 |
| 8 | R3、R1、R2、R4 | 实验 | Amazon 之外再验证。R3 写：主贡献是关于协议和表示选择的实证结论，而不是某个数据集上的模型提升，因此域外验证会实质加强 |
| 9 | R4 | 实验 | 与明确使用多 prompt / 多 view 的已有方法比较，点名 LLM-Rec、SC-Rec。否则看不出冗余是多 prompt 推荐的普遍限制，还是这套 prompt + 拼接融合的结果 |
| 10 | R4 | 范围 | 相关工作里提到的 vector-quantized、learned language representation 没有进受控比较；主比较是 TF-IDF 与 Qwen2.5-7B，backbone 是一个 SASRec |
| 11 | R1 | 方法 | 缺少系统性的缓解：weighted fusion、contrastive learning constraints、view selection/gating。他认为「TF-IDF + 一个 LLM view」更像经验上的退回，不是方法上的改进 |
| 12 | R2 | 统计 + 写作 | 方法成绩接近，不能靠排序说显著；机制要更好；写作还可改进 |

---

## 五、Reviewer 1 — Weak Reject

### 5.1 Summary（他对贡献的复述）

实证：文本表示对序列推荐表现的影响。三个 Amazon 数据集，四种表示（ID-only、TF-IDF、TF+LLM 单 view、TF+MV 多 view），跨评测协议的受控实验。他把贡献写成三条：

1. 评测协议显著影响模型排序，同一模型的表现会随协议明显变化。
2. 文本表示的多 view 冗余：四个 prompt view 高度重叠（CKA 0.82–0.94）；TF-IDF 与 LLM view 呈互补（CKA 0.18–0.20）。
3. 复杂文本管线并不稳定地带来收益；评估 LLM 增强的好处时要谨慎。

归档时记下他的用词：第 2 条里他把 0.18–0.20 写成 complementarity。这是他的阅读，不是本文件要改的结论。

### 5.2 Strengths

1. 动机强，指出一个重要但被忽略的问题。
2. 多维诊断全面、可靠。
3. 实验设计彻底、可复现。

### 5.3 Weaknesses（他标成最要紧的问题都在这里）

Detailed comments 写的是 Please refer to the Weakness，没有额外技术意见。

**W1. 动机强，但缺少可执行的根本办法。**

- GTS-Last 和 GTS-Random 会得到不同结论，正文主要把现象说成 “differences in the certainty of target selection”。
- 没有底层的偏差机制，也没有可测的中间变量。
- 没有系统性的缓解方法。他举例：weighted fusion、contrastive learning constraints、view selection/gating。
- 最终建议 “TF-IDF + a single LLM view”，在他看来更像经验上的降级，而不是方法上的改进。

**W2. 学术贡献有限，缺少创新。**

- 主要是用更严的评测去观察和诊断，没有提出会改变结论的算法或理论机制。
- 表示和模型大多是现成的：SASRec、UniSRec、TF-IDF、Qwen2.5。
- 亮点是 “discovery of risks”，不是 “formation of reusable improvement plans”。

**W3. 实验范围受限，外推证据不足。**

- 只在三个 Amazon 电商数据集（Beauty, Toys, Grocery）。领域相近，外推弱。
- 多 view 冗余的结论只基于四个固定 prompt view、一个 LLM（Qwen2.5）。
- 缺少跨 LLM、跨 view 设计范式、跨域的交叉验证。

**W4. 根因分析不够。** 分三块：

| 现象 | 他认为缺什么 |
|------|----------------|
| GTS-Last 与 GTS-Random 结果不同 | 只讨论了 random versus deterministic target selection，没有解释抽样差异如何改变估计偏差、造成训练/测试分布不匹配、影响排序稳定性 |
| Early 窗口里 TF-IDF 更稳 | 缺少 TF-IDF 与 LLM 表示在不同时间分布下的机制比较 |
| 多 view 冗余 | 没有区分冗余来自四个 prompt 的语义重叠（Description, Function, Audience, Style），还是来自 Qwen2.5 对这些 prompt 的编码收敛 |

### 5.4 Recommendation

Weak Reject: Borderline paper, tending to reject.

---

## 六、Reviewer 2 — Weak Reject

### 6.1 Summary

序列推荐上的 leakage-aware 受控研究。比较四种：

- ID-only
- cutoff-fit TF-IDF
- TF+LLM：在 TF-IDF 上加一个 Qwen2.5-7B view
- TF+MV：在 TF-IDF 上加四个 prompt view

协议：三个 Amazon 数据集上的 full-ranking global temporal splitting (GTS)。GTS-Last 是主协议，GTS-Random 是稳健性检查，per-user leave-one-out (LOO) 是对照。

他复述的结果：warm target 上文本稳定优于 ID-only；TF-IDF、TF+LLM、TF+MV 的相对排序在 Last、Random、LOO 之间会变，即使 Last 和 Random 评的是同一批训练好的 checkpoint。

### 6.2 Strengths

1. 带文本 embedding 的序列推荐是重要题目。
2. 实验多，分析充分。
3. 关于文本 embedding 的发现对实际推荐系统有帮助。

### 6.3 Weaknesses

1. 贡献可能有限，因为主要是对不同文本 embedding 设定做实证研究。
2. 只在 Amazon 上。不同方法的表现相近，很难说有统计显著性。
3. 这些方法为什么有效、为什么无效，机制需要分析得更好。

### 6.4 Detailed comments

没有展开新的技术点。原句：As in weaknesses. The writing of the paper can also be improved.

### 6.5 Recommendation

Weak Reject: Borderline paper, tending to reject.

---

## 七、Reviewer 3 — Weak Accept

### 7.1 Summary

leakage-aware 时间评测下，序列推荐文本表示的实证研究。比较 ID-only、TF-IDF、单 view LLM、多 view LLM，协议包括 GTS-Last、GTS-Random、LOO。

主发现：评测协议可以大幅改变不同表示的相对表现；再加多个 LLM view 的额外收益有限，因为这些 view 高度冗余。

他的总评：设计仔细、有用，尤其是对时间泄漏的处理和受控比较。主要限制是实证范围相对窄，特别是只用了 Amazon 数据集和单个 LLM 家族。

### 7.2 Strengths

1. 直接问「再加更多 LLM 生成的文本 view，是否真的提供了互补信息」，而不是假设更多语义 view 总是有益。
2. 区分 GTS-Last、GTS-Random 和 LOO，比标准的单协议比较更有信息。文本表示只用 cutoff 之前可用的信息来拟合，使比较明显更可信。
3. 主变体用同一个推荐 backbone，差别主要在输入表示。因此研究的是文本表示的效果，而不是把表示差异和架构改动混在一起。
4. CKA、effective rank、leave-one-view-out 和 single-view 分析，是在解释「为什么额外 LLM view 收益有限」，而不只是报告现象。

### 7.3 Weaknesses

1. 三个数据集都来自 Amazon 生态，协议结论和多 view 冗余结论能否泛化到其他推荐领域，不清楚。
2. 表示分析基于 Qwen2.5-7B。prompt view 之间高相似有信息量，但不能据此建立「多 view 冗余是 LLM 表示的一般性质」。
3. 正文把 Successive 标成更接近在线 next-item 推荐的离线近似，主实验却用 Last 和 Random 作为实际替代。因此不确定与协议相关的主发现在 Successive 下是否还在。
4. 相当一部分 post-cutoff target 被排除，因为评测聚焦训练集目录里已经出现过的物品。主结果主要刻画的是 warm-item 推荐。

### 7.4 Detailed comments（五条，都是可执行的）

1. 若可行，表示分析里再加一个 LLM encoder，或一套差别明显的 prompt family。用来判断观察到的冗余是 Qwen2.5-7B 特有，还是多 view LLM 表示更广的性质。
2. 既然 Successive 被写成最接近在线推荐的离线近似，哪怕小规模的 Successive 评测也会加强「协议敏感」这个中心主张。
3. 主讨论里应更清楚地写明报告的结果基于 warm target。若可能，再分析排除了多少 target、定性结论是否改变。
4. 高 CKA 是表示相似性的证据，但不能单独说明不同 view 没有互补的推荐信息。下游消融已经提供了有用证据；文章应把这两类证据连接得更仔细。
5. 主贡献是关于评测协议和表示选择的实证结论，不是某个数据集上的模型提升，因此 Amazon 生态之外的验证会实质加强这项研究。

### 7.5 Recommendation

Weak Accept: Borderline paper, tending to accept.

---

## 八、Reviewer 4 — Weak Reject

### 8.1 Summary

leakage-aware 的序列推荐文本表示实证研究。在不同时间评测协议下比较 ID-only、TF-IDF、单 view LLM 增强、多 prompt LLM 增强的物品表示，并分析为什么额外的 prompt view 收益有限。三个 Amazon 数据集上：文本总体上优于 ID-only；TF-IDF 与基于 LLM 的变体之间的相对排序对评测协议敏感；多个 prompt view 高度冗余。

### 8.2 Strengths

1. 研究文本增强序列推荐里一个重要的评测问题：不同文本表示的相对表现，会在不同的时间划分和目标选择协议下大幅改变。
2. 组织清楚、好跟。动机、方法和实验设置写得明白，整体框架在概念上直观。
3. 实验评测算比较全面：有代表性 baseline、消融，以及支持所提方法有效性的额外分析。

归档时记下第 3 条的用词：他写了 “the effectiveness of the proposed method”。正文定位是实证研究，不是提出一个新方法。这是他的概括。

### 8.3 Weaknesses

表单要求至少三条，他写了两条。Detailed comments 是 See Strengths and Weaknesses，没有追加。

**W1. 多 prompt 表示的评测基本限制在作者自己的构造上。**

- 四个 prompt，一个 Qwen2.5-7B encoder，简单的 concatenation 融合。
- 没有直接比较明确利用多 prompt 或多 view 的已有推荐方法，例如 LLM-Rec 和 SC-Rec。
- 因此看不出观察到的冗余是多 prompt 推荐的普遍限制，还是主要来自这项研究里特定的 prompt 和融合设计。

**W2. 文本表示方法的实证范围相对窄。**

- 主张是研究序列推荐的文本表示，受控比较主要覆盖的是 TF-IDF 和 Qwen2.5-7B embedding，backbone 是一个 SASRec。
- 相关工作里讨论过的若干已有表示范式没有评，例如 vector-quantized 或 learned language representations。
- 「文本特征优于 ID-only」这一点先前工作已经大量做过。
- 这限制了实证结论的新颖性和一般性。

### 8.4 Recommendation

Weak Reject: Borderline paper, tending to reject.

---

## 九、归档时不要误读的两处用词

只记录差异，不在本文件里改结论。

| 位置 | 审稿人写的 | 和正文锁定口径的关系 |
|------|------------|----------------------|
| R1 Summary 第 2 条 | TF-IDF 与 LLM views exhibit complementarity（CKA 0.18–0.20） | 低 CKA 在正文里是几何上不相似。互补是更强的说法，R3 的详细意见第 4 条也提醒高/低 CKA 不能单独推出有没有推荐互补信息 |
| R4 Strength 第 3 条 | additional analyses that support the effectiveness of the proposed method | 正文没有把主贡献写成一个新方法。他认可实验全，但用了 “proposed method” |
