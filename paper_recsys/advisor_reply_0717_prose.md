# 对 0717 指导意见的逐条回复（口头稿）

> 协议：Beauty · Global TS · full ranking · min-history≥5 · train-only 特征 · 无 SE/boost · per-source align · two-phase 仅作优化协议。  
> 超参（valid 网格锁定）：TF 5e-4/0.3，LLM 1e-4/0.1，MV 5e-4/0.3。

---

**1. 公平口径下，是否接受「层级不漂亮、MV/Cross 不稳」，还要不要用 boost/扩维追层级？**

同意，已按此收束。公平对比后 no-Cross test MRR@10：TF 0.0162、LLM 0.0164、MV 0.0158；加 Cross 无稳定净收益。相对 ID-only（≈0.0112）文本约 +45%，说明 time split 下文本有效，但 MV/Cross 额外收益不稳。未再靠 boost、扩维追层级。

---

**2. valid→test 掉约 44% 是不是 temporal shift？五类分布查了吗？**

更像整体时间漂移，不是单模型过拟合。目标 popularity 中位 21→15（−29%）；unseen 目标 11.3%→17.8%；head 桶 63.1%→56.6%；catalog 新 item 约 23.5%→27.3%。目标 item age 中位略降（约 6.37e7→6.03e7 秒）。用户历史中位均为 8，几乎不变。文本分布尚未对比，这一项还不能下结论。

---

**3. 超参是否只用共享 valid 网格？有没有更早的 rolling window？**

已做。lr×dropout 同预算网格，按 valid 锁定上述超参，不用 test 主选参。Early 窗口 TS[0.6,0.1,0.3]、冻结超参、重拟 early 特征后，3-seed test MRR：TF 0.0053±0.0002、MV 0.0052±0.0001、LLM 0.0049±0.0002。与 Main（TF 0.0165±0.0008、LLM 0.0161±0.0005、MV 0.0160±0.0008）合读：两窗口 MV 均未超单视图 1 std，排序大体 TF≈LLM≥MV。

---

**4. item-frequency 分桶是否保留？和 min-history 分开吗？**

保留，且与 min-history≥5 分开：后者管用户序列是否可评，前者按 train cutoff 前频次切 low/mid/head/unseen。公平跑里 MV 在 new 桶更弱（MRR_new≈0.0104 vs LLM≈0.0121）。Item age 有 valid/test 统计，正式 age 分桶表尚未做。

---

**5. 要不要扩 MV 到 4×256？先冗余还是容量不足？**

未扩维。View 对 CKA 均 >0.83；4-view concat effective rank≈136.9/256（0.535）；leave-one-view Δ≈0。判为 prompt multi-view 语义重述，不做 4×256 vs 1024 pilot。

---

**6. Stop-gate 是否触发？MV/Cross/SE·boost 怎么处理？**

可触发。Main 3-seed：TF 0.0165±0.0008、LLM 0.0161±0.0005、MV 0.0160±0.0008；Early 同上，MV 未过 1-std。准备停 MV 作主方法。Cross：Main ΔMRR≈±0.0002；Early LLM±Cross test MRR 同为 0.0051；rank 改善/变差约各半；Top-10 head share LLM 几乎不变（0.665→0.667）。Cross 删默认，留机制分析。SE/boost 已删；two-phase 只作优化协议。
