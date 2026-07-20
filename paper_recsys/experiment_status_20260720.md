# 实验进展 — 2026-07-20

> 更新时间: 2026-07-20 15:10  
> 上一日进展: [`experiment_status_20260719.md`](experiment_status_20260719.md)  
> 实验规范: [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)  
> 主线追踪: [`WSDM_convergence_tracker.md`](WSDM_convergence_tracker.md)  
> 要点: **Grid Fix 完成 → 下午启动 Stop-Gate 3-seed；三端同步最新结论**

---

## 〇、今日结论（一句话）

**Grid Fix 已于 05:27 跑完。按 valid：MV 0.0284 > TF 0.0280 > LLM 0.0279，但 test 上 LLM 仍最高 (0.0162)。下午 5090 空闲，已启动 Stop-Gate 3-seed（LLM/TF/MV + MV do=0.5 诊断），预计明天白天出均值±std，用于判定 MV 是否进主方法。**

---

## 一、机器快照（7/20 15:10）

| 机器 | 任务 | 状态 | ETA |
|------|------|:----:|-----|
| **5090** | Stop-Gate 3-seed (`run_5090_stopgate_3seed.sh`) | 🟡 已启动 | ~7/21 06:00–10:00 |
| **log10** | 空闲 | ✅ | — |
| **logMac / 本机** | 文档与代码同步 | ✅ | — |

检查命令：
```bash
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && tail -20 logs/stopgate_3seed_nohup.log && cat logs/stopgate_3seed_beauty.csv'
```

---

## 二、Grid Fix 完整结果（seed=2025）

### 2.1 三模型最优（按 valid 选参）

| 模型 | 最优配置 | valid MRR | test MRR | test NDCG | test HR | MRR_new | Gap |
|------|----------|:---------:|:--------:|:---------:|:-------:|:-------:|:---:|
| **MV (正确PA)** | lr=5e-4/do=0.3 | **0.0284** | 0.0158 | 0.0199 | **0.0333** | 0.0119 | -44% |
| **TF-IDF** | lr=5e-4/do=0.3 | 0.0280 | 0.0155 | 0.0191 | 0.0310 | 0.0098 | -45% |
| **LLM** | lr=1e-4/do=0.1 | 0.0279 | **0.0162** | **0.0199** | 0.0319 | **0.0133** | -42% |

### 2.2 Valid vs Test 选参诊断

| 模型 | 翻转？ | 结论 |
|------|:------:|------|
| LLM | **否** | valid-best = test-best |
| TF-IDF | **否** | valid-best = test-best |
| MV | **是** | valid-best do=0.3 (test=0.0158)；test-best do=0.5 (test=0.0162) |

→ 规范仍用 valid；3-seed 主报 MV do=0.3，附带 do=0.5 看稳定性。

### 2.3 关键发现（定稿级草稿）

1. **TF ≈ LLM**：test 差距 0.0007，需 3-seed 确认是否在 1 std 内  
2. **LLM 在 MRR_new 占优**：0.0133 > MV 0.0119 > TF 0.0098  
3. **MV valid 略优、test 未超 LLM**：正确 Phase-A 后仍未稳定赢 single-view  
4. **平均 gap MV 并不最大**（43.6% vs 45.7%）：主因 temporal shift，不是 MV 单独过拟合  
5. 先前 P0/P1：view CKA>0.83、leave-one-view Δ=0、Cross 净负向 — 仍成立

---

## 三、今日下午操作

| # | 操作 | 状态 |
|:-:|------|:----:|
| 1 | 拉取 5090 Grid Fix 结论 + CSV | ✅ |
| 2 | 同步 0719/0720 进展文档到本机 | ✅ |
| 3 | 写 `run_5090_stopgate_3seed.sh`（网格最优超参） | ✅ |
| 4 | 5090 启动 3-seed（GPU 空闲） | 🟡 |
| 5 | 更新收敛文档 + 三端 git 同步 | 🔄 |

### 3-seed 队列（12 runs）

| 顺序 | 模型 | lr / dropout | seeds |
|:----:|------|:------------:|:-----:|
| 1–3 | LLM | 1e-4 / 0.1 | 42, 2024, 2026 |
| 4–6 | TF | 5e-4 / 0.3 | 42, 2024, 2026 |
| 7–9 | MV | 5e-4 / 0.3 | 42, 2024, 2026 |
| 10–12 | MV* (诊断) | 5e-4 / 0.5 | 42, 2024, 2026 |

Stop-gate 判定（0717）：若 MV 相对 LLM 在主指标上不能稳定超过 1 std → MV 降为分析对象。

---

## 四、后续优先级（3-seed 之后）

| 优先级 | 任务 | 说明 |
|:------:|------|------|
| P1 | 汇总 3-seed 均值±std | 写进主表 + stop-gate 结论 |
| P1 | Rolling early-valid window | 检验排序稳定性 |
| P2 | Toys per-source 公平对比 | 跨域；用同一最优超参或小网格 |
| P2 | Figure 整理 | SVD 谱图 + rank-transition 已有 |
| P3 | UniSRec portability | 非阻塞 |

---

## 五、累计关键结论

| # | 结论 | 状态 |
|:-:|------|:----:|
| 1 | 文本特征整体有效 (+40%~50% vs ID-only) | ✅ |
| 2 | TF-IDF ≈ LLM（test 差距 <0.001） | ✅ Grid；待 3-seed |
| 3 | LLM 在 MRR_new 占优 | ✅ |
| 4 | View CKA>0.83 / leave-one-view Δ=0 → 冗余 | ✅ |
| 5 | Cross 净负向 | ✅ |
| 6 | Valid→test ~44% = temporal shift | ✅ |
| 7 | 正确 PA 后 MV valid 略优，test 仍未超 LLM | 🟡 3-seed 中 |
| 8 | MV 唯一 valid/test 选参翻转 | ✅ |

---

## 六、一句话目标

**Grid 完成；Stop-Gate 3-seed 已拉起；今晚/明天收均值±std，决定 MV 是否退出主方法。**
