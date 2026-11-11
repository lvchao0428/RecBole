# 0803 指导：已完成事项梳理（供老师过目）

> 日期：2026-08-06  
> 范围：**仅列已完成**；进行中的 Random / Early-Last / Grocery 不写入正文清单。  
> 依据：`0803zhidao.txt`

---

## 〇、总览（给老师的一页结论）

按老师建议的三件事，当前完成度为：

| 老师要求 | 完成情况 |
|----------|----------|
| ① 锁定 protocol | **已完成主体**：主结论协议定为 GTS-Last + warm + train-cutoff 特征；All/LOO 降级 |
| ② 锁定主文数据 | **已完成**：主文所需表/图材料齐，初稿已按此组装 |
| ③ 轻量关键分析 | **已完成**：CKA 3-subsample、LOOV 四 view、成本表 |

叙事已从「MV-Align 新模型」转为「leakage-aware 文本表征比较」；三 RQ 方向结论均有数据支撑。

---

## 一、按时间/逻辑顺序：已完成工作清单

### Step 1 — 叙事转向与问题定义（对齐 0803 定调）

1. 停止维护 MV-Align / Cross 作为主贡献。  
2. 确定论文三问：  
   - RQ1 Protocol effect（LOO vs GTS 是否改变 TF/LLM/MV 结论）  
   - RQ2 Temporal robustness（Main/Early + 分布漂移）  
   - RQ3 Why no MV gain（冗余机制）  
3. 工作标题定为：  
   *When More Text Is Not Better: A Leakage-Aware Study of Text Representations for Sequential Recommendation*

---

### Step 2 — 确认原 GTS 协议身份（老师硬要求）

4. 查明原训练/评测 GTS 为 **Successive-All**（非 Last/Random）。  
5. 实证规模：Beauty 4,520 users → 12,001 test samples；Toys 4,873 → 13,186。  
6. 明确 All 与标准 next-item 存在 task mismatch → 主表必须改 Last（或 Random）。

---

### Step 3 — 锁定 protocol（老师「第一件事」）

7. **GTS-Last 重评完成**（不重训）：Beauty/Toys × ID/TF/LLM/MV/UniSRec × 3 seeds = 30 ckpt。  
8. 主表口径锁定：  
   - 主报告：**GTS-Last + warm**（target ∈ train catalog）  
   - `min_train_interactions=5`  
   - TF-IDF / SVD / center-whiten **仅训练期拟合**  
9. All-target、unseen → 附录；Clean LOO → 仅 protocol contrast。  
10. 初稿 `main_wsdm2027.tex` 主表已切换为 GTS-Last（不再以 All 为主结论）。

**已交付主表数字（GTS-Last, warm, MRR@10）**

| Dataset | ID | TF | LLM | MV | UniSRec |
|---------|---:|---:|----:|---:|--------:|
| Beauty | 0.0032 | 0.0051 | 0.0059 | **0.0061** | 0.0052 |
| Toys | 0.0031 | 0.0040 | 0.0035 | **0.0042** | 0.0041 |

---

### Step 4 — 锁定主文数据（老师「第二件事」）

11. **LOO vs GTS 聚合表**  
    - LOO：Beauty/Toys 抬高绝对值，排序偏 LLM≈MV  
    - GTS-All（对照）：TF 略强  
    - GTS-Last（主）：细排序变化，但文本变体差距近方差  

12. **GTS Main / Early 3-seed 表（Beauty）**  
    - Main TF 0.0165 ≥ LLM 0.0161 ≥ MV 0.0160  
    - Early60 TF 0.0053 ≥ MV 0.0052 ≥ LLM 0.0049  
    - 两窗均 TF 最稳，LLM/MV 未稳定超过约 1 std  

13. **UniSRec external baseline**（同 filter）  
    - Last-warm：Beauty 0.0052、Toys 0.0041，不支持「复杂文本方法稳超 TF」  

14. **Temporal-shift 表**  
    - unseen↑、head↓、pop↓、catalog 新品率↑（Beauty/Toys valid→test）  
    - valid→test MRR drop：Beauty ~43%、Toys ~55%，三路同步  
    - 表述为 associated with，不作因果  

15. **CKA heatmap + effective rank + LOOV 组合图/表**（已入初稿）  

16. **附录材料就位**  
    - Warm/unseen audit、per-seed 原始结果、Cross rank transition（improved≈degraded → negative only）

---

### Step 5 — 轻量关键分析（老师「第三件事」）

17. **CKA 3 次独立 item 子样本**（Beauty + Toys）  
    - Inter-view CKA ≈ 0.82–0.94（mean±std）  
    - TF–LLM view CKA ≈ 0.19–0.20  

18. **LOOV 四 view 全量**（zero-mask，Beauty + Toys）  
    - 相对 MRR 变化约 1.9%–2.1%，无「只报有利 view」  

19. **参数 / 推理成本表**  
    - Beauty ID/TF/LLM/MV ≈ 67.16 / 67.43 / 67.56 / 67.71 M，延迟同量级  

20. **明确不做**：不扩 4×256、不调 boost、不救 Cross。

---

### Step 6 — 写作与投递准备（已启动）

21. WSDM 2027 初稿框架与主表口径对齐（ACM `sigconf,anonymous,review`）。  
22. 核心图表已具备：`fig_protocol_contrast`、`fig_cka_heatmap`。  
23. 给老师的把关简报：`0803_advisor_gate_briefing.md`（论点 + 数据 + 章节规划）。

---

## 二、三 RQ：已完成证据对应关系

| RQ | 老师关心的结论 | 已完成证据 |
|----|----------------|------------|
| RQ1 | 协议改变 TF/LLM/MV 结论 | LOO vs All vs **Last** 表；绝对值与排序变化 |
| RQ2 | TF 时间上更稳；漂移与下跌同现 | Main/Early 表；drop 43–55%；shift 四项统计 |
| RQ3 | prompt 冗余，非单纯容量 | CKA 子样本 + ER + LOOV 全量 + 成本表 |

---

## 三、已完成工作的读法（汇报口径）

1. **文本始终显著优于 ID-only**（各协议、两数据集）。  
2. **TF / LLM / MV 相对优劣对协议与时间窗不 robust**——乱序本身是发现。  
3. **GTS-Last 下**文本变体差距接近 seed 方差；不宜再写「某一文本表征全面碾压」。  
4. **MV 无稳定增益**可由「高 CKA + 低 concat 有效秩 + 小 LOOV」共同解释；TF 与 LLM 更互补但未被当前 MV 利用。  
5. 按老师意见：**正常投 Main，Findings 兜底**；下一阶段重点是扩写定稿，而非继续炼模型。

---

## 四、文档索引（便于老师查阅）

| 材料 | 路径 |
|------|------|
| 把关简报（论点+规划） | `paper_recsys/0803_advisor_gate_briefing.md` |
| 完成度对照 | `paper_recsys/0803_advisor_checklist.md` |
| GTS-Last 原始结果 | `paper_recsys/gts_last_eval.csv` |
| Last 主表汇总 | `paper_recsys/main_table_gts_last_random.md` |
| CKA 子样本 | `paper_recsys/cka_subsample_{beauty,toys}.json` |
| LOOV | `paper_recsys/leave_one_view_out_{beauty,toys}.md` |
| 成本表 | `paper_recsys/model_cost_table.csv` |
| 初稿 | `paper_recsys/main_wsdm2027.tex` |

---

*以上均为已完成项，按「定调 → 协议确认 → 锁定 protocol → 主文数据 → 轻量分析 → 写作」顺序整理，供老师审阅。*
