# 0723 老师指导 — 实验结果汇总

> 生成: 2026-08-02  
> 指导原文: [`0723zhidao.txt`](0723zhidao.txt)  
> 数据源: `logs/{stopgate_3seed_beauty_metrics,early60_stopgate_beauty,wsdm_wrapup_toys_gts,beauty_loo_results,toys_loo_results,warm_only_eval,unisrec_gts_results}.csv`
>
> **已被最新审计替代**：LOVO no-op 后续已用 zero-mask 修复；同时又发现 warm-only、
> UniSRec test 和 LOVO 的 checkpoint 重载未复现训练时的
> `min_train_interactions=5`。请以
> [`0723_full_advisor_report.md`](0723_full_advisor_report.md) 为最终口径。

---

## 〇、先说两个数据有效性问题

在汇总前必须标注，这两项**不能按现状写进论文**：

| # | 项目 | 问题 | 影响 |
|:-:|------|------|------|
| **B1** | **Leave-one-view-out** | 脚本设置 `model.text_view_indices`，但模型前向读的是 `self.active_buffer_positions`（`__init__` 时由 config 固定）。改这个属性**对前向无任何影响** | 4 个 view 的 Δ 全部恰好 `0.000000`，两个数据集都一样 —— 这是 no-op 的特征，不是"完全冗余"的证据。**LOVO 目前无有效结论** |
| **B2** | **UniSRec** | 日志中**只有 `valid result` 区块，没有 test 评测**。提取正则取"最后一个 `mrr@10`"，抓到的是第 50 epoch 的 **valid** 值 | CSV 里 `test_mrr=0.0204/0.0206/0.0208`（Beauty）实为 valid。若误当 test 对比，会得出"UniSRec 大幅超过 TF/LLM/MV"的错误结论。实际同口径 valid 下 UniSRec 0.0206 **低于** TF/LLM/MV 的 ~0.029 |

其余所有结果（G1 warm-target、G2 LOO vs GTS、CKA/effective rank、temporal shift、Cross）经核对**有效**。

---

## 一、G1 — Unseen target audit + warm-only 主表口径

> 指导："先 audit unseen target… 主表建议限制为 train catalog 内的 warm targets。"

### 1.1 Audit 结果

老师引用的是 17.8%，我们实测更高：

| 数据集 | valid unseen | **test unseen** | catalog turnover | train cutoff |
|--------|:------------:|:---------------:|:----------------:|:------------:|
| Beauty | 15.2% | **27.8%** | 23.3% | 2014-03-09 |
| Toys | 21.1% | **34.5%** | 31.1% | 2014-01-17 |

口径说明：audit 脚本按「每用户最后一个 target」统计（Beauty 6332 条 / Toys 6711 条）；warm-only 重评估按全部 test 实例统计（Beauty 79724 条，unseen 27.25%；Toys 90069 条，unseen 32.21%）。两套口径数值接近，结论一致。

`target_bucket_dist` 也印证漂移：Beauty unseen 占比 valid 11.3% → test 17.8%（**这正是老师说的 17.8%，是 valid/test 全实例口径**），head 占比 63.1% → 56.6%。

### 1.2 Warm-only 重评估（3-seed mean，test MRR@10）

复用已有 GTS ckpt 离线重评，未重训。

| 数据集 | 模型 | all-target | **warm-only** | warm/all |
|--------|------|-----------:|--------------:|:--------:|
| Beauty | TF | 0.0205 | **0.0282** | 1.37× |
| Beauty | LLM | 0.0201 | 0.0277 | 1.37× |
| Beauty | MV | 0.0204 | 0.0280 | 1.37× |
| Toys | ID | 0.0175 | 0.0258 | 1.47× |
| Toys | TF | 0.0251 | **0.0370** | 1.48× |
| Toys | LLM | 0.0245 | 0.0361 | 1.48× |
| Toys | MV | 0.0250 | 0.0369 | 1.48× |

> Beauty n_warm/n_all = 58001/79724；Toys 61056/90069。

**结论**：warm-only 把绝对值整体抬高约 1.4–1.5×，但**模型排序完全不变**（两数据集都是 TF ≥ MV ≥ LLM）。所以采用 warm-only 主表是**口径规范性**的改进，不改变任何科学结论 —— 这一点对论文有利，可以直接写"结论对 target 口径 robust"。

---

## 二、G2 — Clean LOO vs GTS protocol 对照

> 指导："用修正后的统一代码，在 Beauty 和另一个数据集上跑 ID/TF/LLM/MV，比较 clean LOO 和 GTS，各 3 个 seed。GTS 再保留 Main/Early 两个窗口。"

统一代码：per-source align、no-boost、no-Cross、min5、超参按 valid 锁定后不再调。

### 2.1 主对照表（test MRR@10, 3-seed mean ± std）

| 数据集 | Protocol | ID-only | TF-IDF | LLM | MV | 排序 |
|--------|----------|--------:|-------:|----:|---:|------|
| Beauty | **GTS Main** | 0.0106±.0005 | **0.0165**±.0006 | 0.0161±.0004 | 0.0160±.0007 | TF ≥ LLM ≈ MV |
| Beauty | **Clean LOO** | 0.0175±.0003 | 0.0251±.0002 | **0.0254**±.0003 | 0.0253±.0004 | LLM ≈ MV ≥ TF |
| Toys | **GTS Main** | 0.0070±.0000 | **0.0113**±.0002 | 0.0107±.0003 | 0.0109±.0005 | TF > MV ≈ LLM |
| Toys | **Clean LOO** | 0.0152±.0004 | 0.0247±.0005 | **0.0258**±.0010 | **0.0258**±.0023 | LLM ≈ MV > TF |

**这是论文最强的一张表**，两个数据集一致地给出：

1. **LOO 把绝对值抬高 1.5–2.3×**（Beauty TF 0.0165→0.0251；Toys ID 0.0070→0.0152）
2. **LOO 翻转排序**：GTS 下 TF 最优；LOO 下 LLM/MV 反超 TF。即"复杂文本方法更好"这个旧结论**是 LOO protocol 的产物**
3. 但 LOO 下的差距同样落在方差内（Toys MV std 0.0023 > 差距 0.0011），所以准确表述是：**LOO 抹掉了 TF 的优势，而非证明 LLM/MV 更好**

### 2.2 第二时间窗口（Beauty Early60, test MRR@10）

| TF | LLM | MV |
|---:|----:|---:|
| **0.0053**±.0002 | 0.0049±.0002 | 0.0052±.0001 |

与 Main 窗口同向：TF ≥ MV ≥ LLM，MV 未超单视图 1 std。**两窗口一致 → stop-gate 成立**。

### 2.3 UniSRec portability

**见 B2 — 当前数据无效**，需补一次 test 评测（约 2h/数据集×2 = 4h，或直接复用已存 ckpt 做 test-only eval，约 20 min）。
现有可用信息：UniSRec Beauty valid MRR@10 = 0.0206±0.0002，Toys = 0.0163±0.0001，**均低于我们三路模型的 valid（Beauty ~0.029, Toys ~0.024–0.025）**。

---

## 三、G3 — Mechanism evidence

> 指导："main paper 放 view redundancy（pairwise CKA、effective rank、leave-one-view-out）+ temporal shift（popularity、unseen share、catalog turnover、valid→test drop）。Cross 放 Appendix。"

### 3.1 View redundancy ✅

| 指标 | Beauty | Toys |
|------|-------:|-----:|
| Inter-view CKA（6 对，范围） | 0.839 – 0.938 | 0.814 – 0.919 |
| TF-IDF vs view CKA | 0.246 – 0.258 | 0.306 – 0.334 |
| 单 view effective rank | 63.7 / 64 (99.6%) | 63.8 / 64 (99.6%) |
| **4-view concat eff. rank** | **136.9 / 256 (53.5%)** | **141.4 / 256 (55.2%)** |
| TF-IDF eff. rank | 253.1 / 256 (98.9%) | 253.1 / 256 (98.9%) |
| TF-IDF + 4views eff. rank | 374.3 / 512 (73.1%) | 382.0 / 512 (74.6%) |
| 诊断 | **REDUNDANT** | **REDUNDANT** |

**这组数据本身就足以支撑「view redundancy 而非容量不足」的论点**：
- 每个 view 单独看满秩（99.6%），说明不是编码器塌缩
- 拼起来只剩 53–55%，说明 4 个 prompt 视角**互相高度线性相关**
- TF-IDF 与 views 的 CKA 只有 0.25–0.33，说明**TF-IDF 提供的是正交信息** —— 这解释了为什么 `mv_include_base=true`（拼上 TF-IDF）后 eff. rank 回到 73–75%
- 最直接的对照：drop 任一 view 后 eff. rank 从 136.9 只降到 115–118（/192），**信息几乎没损失**

> ⚠️ Leave-one-view-out 的 MRR 数据当前无效（B1）。但上面 `drop_view_*` 的 effective rank 是从原始 embedding 算的，**不受该 bug 影响，可用**。

### 3.2 Temporal shift ✅

| 指标 | Beauty valid → test | Toys valid → test |
|------|:-------------------:|:-----------------:|
| Unseen target 占比 | 11.3% → **17.8%** | 13.0% → **21.3%** |
| Head bucket 占比 | 63.1% → 56.6% | 60.1% → 52.3% |
| Target popularity 中位数 | 21 → 15 | 18 → 11 |
| Catalog 新品率 | 23.5% → 27.3% | 27.1% → 32.8% |

### 3.3 valid→test drop：protocol 差异的量化 ⭐

| 数据集 | Protocol | TF | LLM | MV |
|--------|----------|---:|----:|---:|
| Beauty | GTS | 43.1% | 44.3% | 43.9% |
| Beauty | **LOO** | **18.5%** | **18.1%** | **17.7%** |
| Toys | GTS | 54.9% | 55.3% | 54.3% |
| Toys | **LOO** | **20.7%** | **18.2%** | **15.4%** |

**这是我认为最值得放进 main paper 的一张新表**（老师只要求报 GTS 的 drop，但 LOO 对照让论证完整）：

- GTS 下三路模型 drop **43–55% 且高度一致**（组内差 <1.2pp）→ 印证老师判断"更像系统性时间漂移，不是某一路过拟合"
- LOO 下 drop 只有 **15–21%**，不到 GTS 的一半 → **直接证明那 25–35pp 的额外落差来自 temporal shift 本身，而非模型**
- 这把 §2.1 的"排序翻转"和 §3.2 的"分布漂移"用一个量化指标接上了

### 3.4 Cross（→ Appendix）✅

Beauty Main 窗口，rank transition（no-Cross → +Cross）：

| 模型 | Improved | Degraded | Unchanged | Δ score entropy | Δ Top1-Top10 margin |
|------|---------:|---------:|----------:|----------------:|--------------------:|
| TF | 47.6% | 50.9% | 1.4% | +0.0082 | −0.0873 |
| LLM | 49.6% | 49.2% | 1.3% | +0.0014 | −0.0403 |
| MV | 51.5% | 47.3% | 1.2% | −0.0098 | +0.0653 |

improved/degraded 各半、entropy 与 margin 方向在三路间不一致 → **无稳定趋势，按老师意见放 Appendix，不做强解释**。
Early60 窗口 LLM ±Cross：valid 0.0324→0.0320，test 0.0051→0.0051，同样无增益。

---

## 四、超参口径

> 指导："很小的 shared valid grid，只为排除 underfitting…不建议为 MV 单独设计更宽的网格。"

- 共享网格 lr ∈ {1e-4, 5e-4, 1e-3} × dropout ∈ {0.1, 0.3, 0.5}，同预算 9 组/模型，**仅按 valid 选参**
- 锁定：TF/MV `lr=5e-4, do=0.3`；LLM/ID `lr=1e-4, do=0.1`
- 未为 MV 加宽网格 ✅
- valid→test drop 三路一致（§3.3）→ 确认非单路过拟合，符合老师判断

---

## 五、三组收尾完成度

| 组 | 内容 | 状态 |
|:--:|------|:----:|
| G1 | unseen target audit | ✅ |
| G1 | warm-only 主表重评估（21 ckpt） | ✅ |
| G2 | Beauty GTS Main 3-seed | ✅ |
| G2 | Beauty GTS Early60 3-seed | ✅ |
| G2 | Toys GTS Main 3-seed | ✅ |
| G2 | Beauty Clean LOO 3-seed | ✅ |
| G2 | Toys Clean LOO 3-seed | ✅ |
| G2 | UniSRec portability | ❌ **需补 test 评测（B2）** |
| G3 | pairwise CKA + effective rank | ✅ |
| G3 | leave-one-view-out | ❌ **脚本 no-op，需修（B1）** |
| G3 | temporal shift（含 valid→test drop） | ✅ |
| G3 | Cross rank transition → Appendix | ✅ |

---

## 六、建议的下一步（已入队 5090）

> 2026-08-02：`nohup bash run_5090_remaining_p0.sh`  
> 队列：① UniSRec 6 ckpt 真 test(+valid) → `logs/unisrec_gts_test_eval.csv`  
> 　　　② LOVO zero-mask Beauty+Toys → `paper_recsys/leave_one_view_out_{beauty,toys}.md`

完成后 G1/G2/G3 硬缺口收口；Toys Early60 仍为可选加强项。
