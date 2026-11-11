# 0821zhidao 段落对照（2026-08-21 晚）

源：`0821zhidao.txt`  
稿：`main_wsdm2027_v3.tex`（英文投稿稿）+ `main_wsdm2027_v3_zh.tex`（中文对照，同步改了同一处）  
原则：只改老师点名的 protocol / claim / 图语气；数字、实验、摘要三条发现未动；没有全文润色。

老师总评：可以提交；不要重跑、不要加新故事；最后一轮人工逐句 diff。

---

## 1. Early 协议写死（分数加起来 0.8 ≠ 配置错误）

**老师：** 不要只写 `TS[0.6,0.1,0.1]`；写成 first 60% train / next 10% valid / next 10% test，final 20% unused。把 *only the training cutoff moves* 改成 *the evaluation window is shifted earlier while validation/test horizon lengths remain fixed*。

### §3.1 GTS-Last / 新增 Early 段

**改前**

> Interactions are globally ordered by timestamp; RecBole's `TS` split with `[0.8,0.1,0.1]` (Main) or `[0.6,0.1,0.1]` (Early, equal horizon).

**改后**

> **Main:** first 80% train, next 10% valid, last 10% test (`[0.8,0.1,0.1]`).
>
> **Early window (RQ2):** first 60% train, next 10% valid, next 10% test; **the final 20% is unused** (`[0.6,0.1,0.1]`; fractions are not renormalized). The evaluation window is shifted earlier while validation/test horizon lengths remain fixed at 10% each.

### §3.2 末句

**改前：** only the training cutoff moves.  
**改后：** the evaluation window is shifted earlier while validation/test horizon lengths remain fixed.

### Contribution 2 / `tab:early` caption / RQ2 开篇 / Conclusion 2

四处同样去掉「只移 cutoff / 60% cutoff」的缩写，改成上述四段划分 + unused 20%。`tab:early` 标题改为 point estimates（Main 上 TF+MV 最高，Early 上 TF 最高、1 std 内）。

---

## 2. RQ1/RQ2 机制句降到 hypothesis（证据不够的不写死）

### RQ1「A plausible account of the LOO gap」

**改前**

> Under GTS, where training and test items are temporally disjoint, this advantage vanishes …

**改后**

> Under GTS, where training and evaluation *interactions* are separated by a global temporal cutoff, this advantage is no longer available …
> (Warm evaluation still requires the target item to belong to the train catalog, so the item sets are not temporally disjoint.)

### RQ1「Why do Last and Random disagree?」

**改前**（超证据）

> Last … is biased toward recently popular or trending items.  
> … the target is often a recently introduced or niche item.  
> TF-IDF … may instead benefit from Random's more uniform target distribution …

**改后**（只保留已证明的 + 一句 hypothesis）

> Last and Random score the same trained checkpoints on the same eligible users and differ only in which test interaction is selected; the ranking flip survives on intersecting warm users (Appendix `app:intersection`). That target-selection effect is the claim.  
> One plausible hypothesis is that Last and Random draw different mixes of item popularity within the test window …  
> We do not have a direct statistic showing that Last targets are recently popular, trending, or niche, and we do not claim a lexical-versus-semantic causal mechanism.

### RQ2「Connecting shift to representation type」

**改前**（超证据）

> TF-IDF's discriminative power degrades---but this degradation is *uniform* across items …  
> when targets are dominated by popular items (as in Last …), TF+LLM/TF+MV's semantic precision may help …

**改后**

> We do not have a per-item statistic that TF-IDF degrades uniformly across items, nor that Last targets are dominated by popular items.  
> One plausible hypothesis, consistent with Table `tab:early` where the Main→Early drop is smaller for TF-IDF than for TF+LLM/TF+MV, is that lexical features and LLM-augmented features respond differently when the evaluation window is shifted earlier.  
> 保留原文已有的 association-not-causation 句。

RQ2 开篇删掉「TF-IDF also degrades less …」作为机制句；排序变化改为 point-estimate 陈述（数字未改）。

---

## 3. RQ3 因果收口

### 「Where does redundancy originate?」段末

**改前**

> Redundancy is therefore inherent in the Qwen encoder's representations … the LLM produces near-isomorphic relational structures regardless of the instruction prompt …

**改后**

> Redundancy is therefore already present in the raw Qwen outputs before SVD or whitening *under our prompt design* …  
> This audit cannot separate encoder-intrinsic collapse from using the same item text or from prompts that are insufficiently divergent.

### Synthesis

**改前：** views … fail the complementarity condition  
**改后：** The observed geometry is consistent with insufficient view complementarity …

### Discussion hybrid

**改前：** exactly the behavior predicted by the redundancy analysis  
**改后：** consistent with the redundancy analysis

### Limitations / Future (2) / Contribution 3

- Limitations：originates in the encoder itself → already present in the raw Qwen outputs under our prompt design；并写明无法区分 encoder / 同一 item text / prompt 不够散。
- Future (2)：confirms the redundancy locus in Qwen → observed redundancy already present in Qwen outputs under our prompts。
- Contribution 3：jointly explaining → jointly characterizing。

---

## 4. Figure 1 / 表头 winner 语气（1 std 内用 ≈）

柱状图数字未改；注释改为点估计排序：

| | Last | Random | LOO |
|---|---|---|---|
| Beauty | TF+MV ≈ TF+LLM > TF > ID | TF ≈ TF+MV ≈ TF+LLM > ID | TF+LLM ≈ TF+MV ≈ TF > ID |
| Toys | TF+MV ≈ TF > TF+LLM > ID | TF ≈ TF+MV ≈ TF+LLM > ID | TF+LLM ≈ TF+MV ≈ TF > ID |

`tab:protocol` caption、RQ1 三条 bullet、Conclusion 1 同步改成 point-estimate leader / near-ties，不再把 1 std 内的差画成确定性冠军。

图文件：`figures/fig_protocol_contrast.pdf`（脚本 `figures/plot_protocol_contrast.py`）。

---

## 未改（按老师：不要动）

- 主表 / Early / 交集 / Hybrid / CKA 数字
- 摘要三条发现（命名已在 0820/0821 凌晨统一）
- 没有新实验、没有新数据集、没有全文润色
