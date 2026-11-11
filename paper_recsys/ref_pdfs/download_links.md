# 论文引用原文下载链接

共 18 篇引用，按 refs.bib 中首次出现顺序排列。

| # | BibKey | Title | PDF Link |
|---|--------|-------|----------|
| 1 | kang2018sasrec | Self-Attentive Sequential Recommendation | https://arxiv.org/pdf/1808.09781 |
| 2 | krichene2020sampledmetrics | On Sampled Metrics for Item Recommendation | https://dl.acm.org/doi/pdf/10.1145/3394486.3403226 |
| 3 | sun2019bert4rec | BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations | https://arxiv.org/pdf/1904.06690 |
| 4 | hidasi2016gru4rec | Session-based Recommendations with Recurrent Neural Networks | https://arxiv.org/pdf/1511.06939 |
| 5 | wang2015cdl | Collaborative Deep Learning for Recommender Systems | https://dl.acm.org/doi/pdf/10.1145/2783258.2783273 |
| 6 | hou2022unisrec | Towards Universal Sequence Representation Learning for Recommender Systems | https://arxiv.org/pdf/2206.05941 |
| 7 | hou2023vqrec | Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders | https://arxiv.org/pdf/2210.12316 |
| 8 | geng2022p5 | Recommendation as Language Processing (RLP): P5 | https://arxiv.org/pdf/2203.13366 |
| 9 | li2023recformer | Text Is All You Need: Learning Language Representations for Sequential Recommendation | https://arxiv.org/pdf/2305.13731 |
| 10 | zhang2019fdsa | Feature-level Deeper Self-Attention Network for Sequential Recommendation | https://www.ijcai.org/proceedings/2019/0600.pdf |
| 11 | bao2023tallrec | TALLRec: An Effective and Efficient Tuning Framework to Align LLM with Recommendation | https://arxiv.org/pdf/2305.00447 |
| 12 | oord2018cpc | Representation Learning with Contrastive Predictive Coding | https://arxiv.org/pdf/1807.03748 |
| 13 | mcauley2015amazon | Image-based Recommendations on Styles and Substitutes | https://arxiv.org/pdf/1506.04757 |
| 14 | ni2019amazon | Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects | https://arxiv.org/pdf/1811.10247 |
| 15 | zhao2021recbole | RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation | https://arxiv.org/pdf/2011.01731 |
| 16 | qwen2024qwen25 | Qwen2.5 Technical Report | https://arxiv.org/pdf/2412.15115 |
| 17 | ethayarajh2019contextual | How Contextual are Contextualized Word Representations? | https://aclanthology.org/D19-1006.pdf |
| 18 | rendle2009bpr | BPR: Bayesian Personalized Ranking from Implicit Feedback | https://arxiv.org/pdf/1205.2618 |
| 19 | hidasi2018recurrent | Recurrent Neural Networks with Top-k Gains for Session-based Recommendations | https://arxiv.org/pdf/1706.03847 |
| 20 | meng2020temporal | Exploring Data Splitting Strategies for the Evaluation of Recommendation Models | https://arxiv.org/pdf/2007.13237 |
| 21 | ji2023evaluation | A Critical Study on Data Leakage in Recommender System Offline Evaluation | https://arxiv.org/pdf/2207.01371 |
| 22 | sun2023reevaluation | Take a Fresh Look at Recommender Systems from an Evaluation Standpoint | https://dl.acm.org/doi/pdf/10.1145/3539618.3591931 |
| 23 | roy2007effective | The effective rank: A measure of effective dimensionality | https://ieeexplore.ieee.org/document/7098875 |
| 24 | kornblith2019cka | Similarity of Neural Network Representations Revisited | https://arxiv.org/pdf/1905.00414 |
| 25 | xu2013multiview | A Survey on Multi-view Learning | https://arxiv.org/pdf/1304.5634 |
| 26 | gusak2025time | Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders | https://arxiv.org/pdf/2507.16289 |
| 27 | salton1988termweighting | Term-Weighting Approaches in Automatic Text Retrieval | (Elsevier paywall, DOI: 10.1016/0306-4573(88)90021-0) |

## 核查重点

**确认有幻觉的引用：**
- **[19] hidasi2018recurrent** — 论文实际是改进GRU4Rec的loss function（Top-k gains），**不是**研究 protocol sensitivity。我们论文中第168行声称它 "showed that seemingly minor protocol choices can cause large shifts in relative rankings of sequential models"，这是**幻觉**。第170行用它来支持 "session-based setting 中也记录了temporal protocol的影响" 同样不准确——该文使用temporal split做实验但并未研究split effect。

**需要替换的引用建议：**
- Ludewig & Jannach (2018) "Evaluation of Session-based Recommendation Algorithms" (User Modeling & User-Adapted Interaction, Vol 28, pp 331-390) — 这篇才是真正在 session-based 领域展示 protocol 敏感性的文章。
- 或者直接删除 line 168 的整句、line 170 改为其他引用（如 Ludewig & Jannach 2018）。

**其余引用经网搜验证 claim 均匹配原文（但建议人工通读确认）。**
