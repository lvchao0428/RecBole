# WSDM 2027 8-Page Outline

> 目标：把当前 `main_wsdm2027.tex`（约 4 页）扩到 **~8 页正文**（WSDM 上限 9 页；留 0.5–1 页给附录缓冲）。  
> 结构参考：`main.tex`（Intro → Related → Method → Experiments → Results/Analysis → Conclusion）。  
> 叙事锁定（0803）：**不是新模型论文**，而是 leakage-aware study of text representations。  
> 主表协议（0803）：**GTS-Last 与 GTS-Random 优先**（warm + train-cutoff features）；LOO / GTS-All 仅 contrast 或 Appendix。  
> 结论对照详见 [`0803_advisor_checklist.md`](0803_advisor_checklist.md)。不拆细粒度 `sections_*.tex`，单文件推进。

---

## 0. 与 `main.tex` 的结构对照

| `main.tex`（旧 MV-Align） | 8 页新稿（本框架） | 处理原则 |
|---|---|---|
| 1 Introduction (+ RQs) | 1 Introduction (+ RQs) | 保留骨架；痛点从“冷启动增益”改为“协议改变文本表征排序” |
| 2 Related Work | 2 Related Work | 砍 Cross/SE；加长 Evaluation + Multi-view redundancy |
| 3 Method（架构大段） | 3 Study Design（协议 + 表征 pipeline） | **不写新模型卖点**；Method = 可控比较设置 + 共享 backbone |
| 4 Experiments | 4 Experimental Setup | 数据集表、协议定义、超参、复现 |
| 5 Main Results + 6 Analysis | 5 Results and Analysis（按 RQ1–3） | 合并为一节；主结论跟 GTS-Last |
| 7 Conclusion | 6 Conclusion + Limitations | 加 Limitations / Findings 兜底一句 |
| Appendix → 外部 repo | Appendix（计入页限） | 薄附录：GTS-All、Warm、Cross negative、per-seed |

**页数预算（ACM sigconf 双栏，约 8.0 页）**

| 区块 | 预算 | 现状≈ | 需补≈ |
|---|---:|---:|---:|
| Abstract + CCS | 0.35 | 0.35 | 0 |
| §1 Introduction | 1.10 | 0.55 | +0.55 |
| §2 Related Work | 0.90 | 0.35 | +0.55 |
| §3 Study Design | 1.70 | 0.55 | +1.15 |
| §4 Experimental Setup | 1.00 | 0.30 | +0.70 |
| §5 Results & Analysis | 2.60 | 1.60 | +1.00 |
| §6 Conclusion | 0.35 | 0.20 | +0.15 |
| Appendix（薄） | 0.50–1.00 | 0.40 | +0.2 |
| **合计** | **~8.0–8.5** | **~4.0** | **~4.0** |

Ethics + References **不计入** 9 页上限。

---

## 1. 建议目录（可直接变成 `\input{sections_wsdm/...}`）

```
main_wsdm2027.tex                 % driver（对齐 main.tex）
sections_wsdm/
  sec_01_introduction.tex
  sec_02_related_work.tex
  sec_03_study_design.tex         % 对应旧 Method，但改叙事
  sec_04_experimental_setup.tex
  sec_05_results_and_analysis.tex
  sec_06_conclusion.tex
  sec_A_appendix.tex
```

---

## 2. 分节框架（写什么 / 用什么图 / 数据从哪来）

### Abstract（~0.35p）— 已有，微调
- 五句式：背景 → 协议脆弱 → 控制研究 → 三点发现 → 启示  
- 数字改为与**最终主表协议**一致（若主表改 GTS-Last，摘要数字同步）

---

### §1 Introduction（~1.1p）← 扩写重点 #1

对齐 `sec_01_introduction.tex` 的写法：痛点 → 本研究定位 → 贡献 → RQ。

**建议段落**
1. **背景（0.2p）**  
   文本增强序列推荐普及；直觉 = LLM > TF-IDF，多 view > 单 view。
2. **问题（0.35p）** ← 新增篇幅  
   LOO 泄漏 + 绝对指标膨胀；GTS 暴露 shift；  
   已有工作研究过 split，但**很少回答：协议是否改变 TF / LLM / MV 的相对排序**。  
   点一句 All vs Last 的 task mismatch（引用既有 GTS 讨论）。
3. **本研究（0.25p）**  
   leakage-aware controlled study；cutoff-fit features；共享 SASRec backbone；Beauty/Toys。
4. **RQs（0.15p）** — 已有三条，保持。
5. **Contributions（0.15p）** — 三条 bullet，对应 RQ。

**浮动**
- *可选* `fig:teaser`（半栏或通栏矮图）：Beauty 上 LOO vs GTS 排序 flip 迷你条形图  
  （可从 `fig_protocol_contrast` 裁 Beauty 半边，省正文大图页）

**不写**
- MV-Align / Cross / “SOTA gains” 话术

---

### §2 Related Work（~0.9p）← 扩写重点 #2

对齐 `sec_02_related_work.tex` 的 paragraph 风格，但换主题权重：

| 段落 | 行数目标 | 内容 | 可复用旧文？ |
|---|---|---|---|
| Sequential recommendation | 6–8 行 | SASRec/BERT4Rec 一句带过 | 少量 |
| Text-enhanced sequential rec | 12–15 行 | UniSRec/VQ-Rec/TALLRec/P5；强调“编码器更强”假设 | 部分 |
| Evaluation & temporal leakage | **15–20 行** | LOO vs time-split；sampled metrics；GTS-All/Last/Random | **重点新写** |
| Multi-view & representation similarity | **12–15 行** | complementary views；CKA；“more views ≠ better” | 部分 |
| Positioning | 4–6 行 | 我们不做新架构，做协议敏感的表征比较 | 新写 |

**删除/降级**：DCN Cross、SE gating、cold-start boost 长段（旧 Method 叙事）。

---

### §3 Study Design（~1.7p）← 最大扩写块（对应旧 Method）

> 旧 `sec_04_method.tex` 有公式+框架图；新稿保留**可复现 pipeline 细节**，去掉 Cross 卖点。

#### 3.1 Problem Setup（0.25p）
- next-item；full ranking；符号：序列 $s_u$、候选集、MRR/NDCG/HR@10

#### 3.2 Temporal Protocols（0.45p）+ **新图**
- LOO / GTS-All / GTS-Last 定义（各 3–4 句）  
- Main `[0.8,0.1,0.1]` vs Early `[0.6,0.1,0.3]`  
- **主结论协议锁定**：GTS-Last + min_train=5 + warm（正文声明；All/unseen 见附录）

**新图 `fig:protocol_overview`（~0.45p，通栏）**  
研究设计总览（替代旧 `framework` / `mv-cover`）：
```
[Amazon interactions]
        ↓ time-ordered
   �`）：
```
[Amazon interactions]
        ↓ time-ordered
   ┌─ Clean LOO (contrast only)
   └─ GTS cutoff ──► train-fit TF-IDF / SVD / whiten
              ↓
        SASRec + text head (ID / TF / LLM / MV / UniSRec)
              ↓
   eval: Last (main) | All (appendix) | Early window | LOOV/CKA
```
数据：概念图，无需新实验。

#### 3.3 Leakage Controls（0.25p）
- vocabulary/IDF/SVD/center/whiten **仅 train-cutoff items**  
- warm = target ∈ train catalog  
- min_train_interactions=5

#### 3.4 Text Representations（0.40p）← 从旧 Method 精简搬迁
- ID / TF（SVD→256）/ LLM single-view prompts / MV 四 prompt  
- **列出 4 个 prompt 模板**（旧 `sec_07` Prompt Templates 段可搬）  
- whitening / ℓ2（说明是公平预处理，不是贡献点）  
- MV = concat→project（**无 Cross**）

#### 3.5 Shared Backbone & Training（0.25p）
- SASRec d=64, 2 blocks（与真实实验一致）  
- two-phase：text-head warmup → joint（旧 `sec_05` 压缩为一段 + 可选 Algorithm 半栏）  
- InfoNCE align（一句话 + 公式可选）

#### 3.6 Compared Systems（0.15p）
- ID, TF, LLM, MV, UniSRec（外部）  
- Cross → Appendix negative only

---

### §4 Experimental Setup（~1.0p）← 对齐旧 `sec_06_experiments.tex`

#### 4.1 Datasets（表）
**新表 `tab:dataset`**
| Dataset | #Users | #Items | #Inter | GTS users | All targets | Last targets | Unseen% |
（数字已有：Beauty 4520 users / 12001 All；Toys 4873 / 13186）

#### 4.2 Metrics & Splits
- full-ranking HR/NDCG/MRR@10；3 seeds {42,2024,2026}  
- 主报 MRR；附录给 NDCG/HR  
- Early 超参冻结自 Main

#### 4.3 Implementation / Hyperparameters
**新表 `tab:hyper`**（半栏或通栏矮表）  
lr / dropout / batch / epochs / early-stop / align weight — 从 yaml 汇总

#### 4.4 Reproducibility
- RecBole；匿名 repo 链接占位；配置名列表一行

---

### §5 Results and Analysis（~2.6p）← 对齐旧 Results+Analysis 合并

> 老师要求主文：LOO vs GTS 对照、GTS Main/Early、UniSRec、shift 表、CKA+ER+LOOV 组合图。

#### 5.1 RQ1: Protocol Changes Rankings（~0.9p）
**主表策略（二选一，写稿前锁定）**

| 方案 | 主表 | 对比表 | 利弊 |
|---|---|---|---|
| **A（跟 0803）** | GTS-Last ± warm | LOO vs GTS-Last；All→附录 | 协议干净；绝对值低，需解释 |
| B（现状） | GTS-All | LOO vs All；Last→附录 | 与训练日志一致；reviewer 可能挑 All |

**推荐 A**：主表 `tab:main_gts_last`（ID/TF/LLM/MV/UniSRec × Beauty/Toys，MRR±std）  
旁路 `tab:protocol`：LOO vs GTS-Last（或 LOO vs All 保留为“协议敏感性”）  
图：`fig_protocol_contrast`（已有，改标注对齐主协议）

文字要点：
- text ≫ ID 始终成立  
- LOO 抬升绝对值并翻转/拉近 TF vs LLM/MV  
- UniSRec 在同 filter 下不占优  
- 一句指向 Appendix：All-target 数字更高但排序故事同类

#### 5.2 RQ2: Temporal Robustness（~0.7p）
- `tab:early` Main vs Early（Beauty；有则补 Toys）  
- `tab:shift` distribution shift（已有）  
- 段落：associated with，不写因果  
- *可选半栏图*：valid→test MRR drop 条形（TF/LLM/MV）

#### 5.3 RQ3: Why MV Does Not Help（~0.8p）
- `fig_cka_heatmap`（已有；可加 Toys 小面板或附录）  
- `tab:loov` 四 view 全量（已有）  
- `tab:cost` 参数/延迟（已有）  
- 三角证据句式：CKA + ER + LOOV；TF–LLM 低 CKA = 互补但未被四 prompt 利用

#### 5.4 Takeaways（0.2p）
- 4 条 practitioner bullets（已有，略扩）

---

### §6 Conclusion（~0.35p）
- 重述三发现  
- **Limitations**（新增 4–6 行）：两域 Amazon；Qwen 单编码器；未做 GTS-Random/全量 Successive；无 content-only zero-shot  
- Future：hybrid TF+LLM targeting complementarity；Random/Successive  
- 一句：适合 Main；Findings 自动兜底（可不写进正文，写进 cover letter）

---

### Appendix（~0.5–1.0p，计入总页）
| 编号 | 内容 | 数据 |
|---|---|---|
| A | GTS-All 全表 / All vs Last 对照 | `gts_last_eval.csv` + 训练 log |
| B | Warm-only / unseen audit | warm CSV + audit md |
| C | Cross negative（improved≈degraded） | rank transition md |
| D | Per-seed 全量 / NDCG&HR | wrapup CSV |
| E | CKA Toys subsample | `cka_subsample_toys.json` |

---

## 3. 图表清单（凑页的“硬资产”）

| ID | 类型 | 状态 | 放哪 | 预估占位 |
|---|---|---|---|---|
| `fig:protocol_overview` | 研究设计总览 | **待画**（替代旧 framework） | §3 | 0.45p |
| `fig:protocol_contrast` | LOO vs GTS | **已有** | §5.1 | 0.40p |
| `fig:cka` | CKA+ER | **已有** | §5.3 | 0.45p |
| `fig:teaser` | 可选迷你 flip | 可从 contrast 裁 | §1 | 0.25p |
| `fig:shift_drop` | 可选 Early/Main drop | 待画，有数 | §5.2 | 0.25p |
| `tab:dataset` | 数据统计 | 待填 | §4 | 0.20p |
| `tab:hyper` | 超参 | 待从 yaml 抽 | §4 | 0.20p |
| `tab:main` | GTS-Last 主表 | 数已有 | §5.1 | 0.30p |
| `tab:protocol` | LOO↔GTS | 已有 | §5.1 | 0.25p |
| `tab:early` | Main/Early | 已有 | §5.2 | 0.15p |
| `tab:shift` | shift 统计 | 已有 | §5.2 | 0.15p |
| `tab:loov` | LOOV | 已有 | §5.3 | 0.15p |
| `tab:cost` | 成本 | 已有 | §5.3 | 0.15p |
| `tab:gts_all` | All 附录 | 已有 | App | 0.25p |

**旧图处理**
- `framework.pdf` / `mv-cover.png` / `sensitivity*` / `tradeoff*` → **不进主文**（叙事冲突）  
- 若要复用 TikZ，只借灰度风格，重画 `protocol_overview`

---

## 4. 凑满 8 页的优先施工顺序

1. **锁定主表协议 A（GTS-Last）**，改 Abstract / §5.1 数字叙事  
2. **画 `fig:protocol_overview`**（最大视觉占页 + 审稿友好）  
3. **拆 `sections_wsdm/`**，按本大纲 `\input`（对齐 `main.tex`）  
4. 扩 §3（prompt 模板 + leakage + two-phase）与 §4（dataset/hyper 表）  
5. 扩 §2 Related（evaluation + multi-view）  
6. 扩 §1 问题动机 + 可选 teaser  
7. 薄附录 A–C；编译测页数，用 `\balance` 收尾  
8. 若仍 <7.5 页：加 Toys CKA 小图或 Main/Early drop 图；若 >8.5：压缩 Related / 把 All 表整表踢附录

---

## 5. 明确不靠“注水”凑页的内容

- 不再扩 4×256 / boost / 救 Cross  
- 不把 Cross 拉回主方法  
- 不写“我们提出 MV-Align”  
- 不为凑页加无数据的第三数据集

---

## 6. 一页纸检查清单（写满后）

- [ ] `documentclass[sigconf,anonymous,review]{acmart}`  
- [ ] 正文 ≤8.5p，总 PDF ≤9p（含附录）  
- [ ] Ethical Considerations 在参考文献前  
- [ ] 主结论只引用 GTS-Last（或文中明确声明的主协议）  
- [ ] 每张图有 `\Description{}`  
- [ ] LOOV 四个 view 全展示；CKA 报 mean±std（3 subsample）  
- [ ] UniSRec 同 filter 出现在主表或紧邻段  
- [ ] Limitations 段存在  

---

## 7. 下一步（你确认后我执行）

请拍板两件事即可开写：
1. **主表用方案 A（GTS-Last）还是 B（GTS-All）？**（框架默认 A）  
2. 是否按 `main.tex` 拆成 `sections_wsdm/*.tex` 骨架文件并开始填 §3+§4？
