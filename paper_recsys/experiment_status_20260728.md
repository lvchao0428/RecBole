# Experiment Status — 2026-07-28

> 更新时间: 2026-07-29 21:00  
> 上一日: [`experiment_status_20260727.md`](experiment_status_20260727.md)  
> **5090 已恢复 UniSRec Toys seed2024+2026**

---

## 〇、一句话

主实验与机制分析均已齐；**UniSRec GTS 恢复跑 Toys seed2024+2026**（约 3–3.5h）。暂停记录见 `logs/unisrec_gts_PAUSE.md`；恢复脚本 `run_5090_unisrec_gts_resume.sh`。

---

## 〇-bis、停训节点（23:22）

| 项 | 内容 |
|----|------|
| 停训原因 | 今晚跑其他实验 |
| 队列脚本 | `run_5090_unisrec_gts.sh` |
| 暂停记录 | `logs/unisrec_gts_PAUSE.md` |
| 结果 CSV | `logs/unisrec_gts_results.csv` |

**已完成（可报）**

| tag | test MRR |
|-----|---------:|
| Beauty seed42 / 2024 / 2026 | 0.0204 / 0.0206 / 0.0208（mean≈0.0206） |
| Toys seed42 | 0.0162 |

**勿报 / 需重跑**

| tag | 说明 |
|-----|------|
| Toys seed2024 | 约 epoch 29/50 被 kill；CSV `0.0164` 为中途提取，**无效** |
| Toys seed2026 | 刚启动即被 kill |

**恢复**：从头重跑 Toys seed2024 + seed2026，约 **3–3.5 h**。

---

## 一、今日完成

### 1.1 Toys Clean LOO seed2026 LLM + MV（✅ 收官）

7/28 10:22 从头重跑（昨晚被外部推理挤死），14:28 完成。

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|---------------:|
| ID-only | 0.0150 | 0.0158 | 0.0148 | **0.0152 ± 0.0005** |
| TF | 0.0248 | 0.0252 | 0.0240 | **0.0247 ± 0.0006** |
| LLM | 0.0251 | 0.0250 | 0.0272 | **0.0258 ± 0.0012** |
| MV | 0.0243 | 0.0240 | 0.0290 | **0.0258 ± 0.0028** |

### 1.2 View Redundancy 分析（✅）

| 指标 | Beauty | Toys |
|------|--------|------|
| Inter-view CKA | 0.84–0.94 | 0.81–0.92 |
| 4-view concat effective rank | 136.9/256 (53.5%) | 141.4/256 (55.2%) |
| TF-IDF effective rank | 253.1/256 (98.9%) | 253.1/256 (98.9%) |
| TF-IDF vs views CKA | ~0.25 | ~0.31 |
| Diagnosis | **REDUNDANT** | **REDUNDANT** |

结论：4 个 Qwen prompt view 互相 CKA > 0.8，拼接后维度坍缩至~54%；TF-IDF 与 views 正交（CKA ~0.25–0.31）。

### 1.3 Leave-One-View-Out（✅）

| Dataset | Full MRR@10 | Drop any view | Δ |
|---------|------------:|:-------------:|:-:|
| Beauty | 0.033374 | 0.033374 | **0.000000** |
| Toys | 0.000901 | 0.000901 | **0.000000** |

结论：去掉任何一个 view 对 MRR 完全无影响。**4 views 完全冗余，边际贡献为零。**

### 1.4 Distribution Shift 正式输出（✅）

Beauty + Toys 的 temporal shift 分析 JSON 已正式输出（popularity / unseen share / catalog overlap / item age / user history length）。

---

## 二、UniSRec GTS Baseline（⏸ 暂停于 23:22）

- 配置：TS[0.8,0.1,0.1]，min5，Qwen `.ts.npy`，only_phase_a 50ep
- Beauty 3-seed ✅；Toys seed42 ✅；Toys seed2024/2026 待恢复
- 详见 §〇-bis / `logs/unisrec_gts_PAUSE.md`

---

## 三、全局实验完成度

| 实验 | 状态 | 要求来源 |
|------|------|----------|
| Beauty GTS Main 3-seed | ✅ | 0723 第二组 |
| Toys GTS Main 3-seed | ✅ | 0723 第二组 |
| Beauty Clean LOO 3-seed | ✅ | 0723 第二组 |
| Toys Clean LOO 3-seed | ✅ | 0723 第二组 |
| Beauty Early60 stop-gate 3-seed | ✅ | 0717 stop-gate |
| Beauty Early60 +Cross 1-seed | ✅ | 0717 Cross |
| View redundancy (CKA + eff rank) | ✅ | 0723 第三组 |
| Leave-one-view-out | ✅ | 0723 第三组 |
| Temporal shift analysis | ✅ | 0723 第三组 |
| Rank transition / Top-K share | ✅ | 0717 mechanism |
| Warm-target audit | ✅ | 0723 第一组 |
| **UniSRec GTS baseline** | ⏸ Beauty✅；Toys 差 2 seeds | 0723 第二组 |

### 恢复后即可启动论文撰写

---

## 四、核心数据汇总（论文 ready）

### GTS vs LOO 对照（MRR@10 test, 3-seed mean）

| Dataset | Protocol | ID | TF | LLM | MV |
|---------|----------|---:|---:|----:|---:|
| Beauty | GTS | 0.0106 | **0.0165** | 0.0161 | 0.0160 |
| Beauty | LOO | 0.0175 | 0.0251 | **0.0254** | 0.0253 |
| Toys | GTS | 0.0070 | **0.0113** | 0.0107 | 0.0109 |
| Toys | LOO | 0.0152 | 0.0247 | **0.0258** | **0.0258** |

**GTS 下 TF 最优；LOO 下 LLM/MV 反超 TF。协议决定排序。**

### Beauty Early60 Stop-gate（MRR@10 valid, 3-seed mean）

| TF | LLM | MV |
|---:|----:|---:|
| 0.0336 | 0.0328 | 0.0331 |

MV 未超 single-view 一个 std → stop-gate 触发。

### View Redundancy

- Inter-view CKA: 0.84–0.94（高度冗余）
- Leave-one-view-out Δ = 0（完全冗余）
- Conclusion: prompt-based multi-view is largely redundant

---

## 五、机器快照

| 机器 | 状态 |
|------|------|
| **5090** | WSDM 训练已停；GPU 空闲（0%，≈2 MiB），供其他实验 |
| **本机** | 进展文档 + PAUSE 已写好，已同步 5090 |
