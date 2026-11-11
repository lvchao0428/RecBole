# 实验进展 — 2026-07-21

> 更新时间: 2026-07-21 11:25  
> 上一日进展: [`experiment_status_20260720.md`](experiment_status_20260720.md)  
> 实验规范: [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)  
> 主线追踪: [`WSDM_convergence_tracker.md`](WSDM_convergence_tracker.md)  
> 要点: **Stop-Gate 3-seed 12/12 全部完成；真实 test 指标已重提取；三端同步**

---

## 〇、今日结论（一句话）

**3-seed 于 04:49 跑完。按真实 test MRR（从日志重提取）：TF 0.0165±0.0008 ≥ LLM 0.0161±0.0005 ≈ MV(do=0.3) 0.0160±0.0008；MV 未超过 LLM 一个 std → 按 0717 stop-gate，MV 降为分析对象，主方法倾向 TF-IDF / LLM single-view。**

> 注意：脚本 CSV 里 valid≈test 是提取 bug；以 `logs/stopgate_3seed_beauty_metrics.csv` 为准。

---

## 一、机器快照（7/21 11:25）

| 机器 | 任务 | 状态 |
|------|------|:----:|
| **5090** | Stop-Gate 已结束；GPU 空闲 | ✅ |
| **log10** | 空闲 | ✅ |
| **本机** | 已同步代码 + CSV + 12 个 sg_*.log | ✅ |

---

## 二、Stop-Gate 3-seed 结果（真实 test）

协议：no-boost / no-Cross / proper Phase-A / seeds={42,2024,2026}

| 配置 | seed42 | seed2024 | seed2026 | **mean±std** |
|------|:------:|:--------:|:--------:|:------------:|
| **TF** lr=5e-4/do=0.3 | 0.0160 | 0.0161 | **0.0174** | **0.0165±0.0008** |
| **LLM** lr=1e-4/do=0.1 | 0.0156 | 0.0161 | 0.0165 | **0.0161±0.0005** |
| **MV** lr=5e-4/do=0.3 | 0.0159 | 0.0153 | 0.0169 | **0.0160±0.0008** |
| MV* lr=5e-4/do=0.5 | 0.0154 | 0.0156 | 0.0154 | 0.0155±0.0001 |

### NDCG@10 / Hit@10（test）

| 配置 | NDCG mean | Hit mean |
|------|:---------:|:--------:|
| TF | 0.0207 | 0.0344 |
| LLM | 0.0200 | 0.0329 |
| MV do=0.3 | 0.0202 | 0.0337 |
| MV do=0.5 | 0.0195 | 0.0328 |

### Stop-gate 判定

| 规则（0717） | 结果 |
|--------------|------|
| MV 相对 single-view 是否超过 1 std | **否**（MV 0.0160 vs LLM 0.0161，Δ=-0.0001 ≪ 1 std） |
| MV 是否进主方法 | **否 → 分析对象 / negative result** |
| 主方法候选 | **TF-IDF 或 LLM**（TF 均值略高，LLM 方差更小） |

---

## 三、与 Grid Fix（seed=2025）对照

| 来源 | TF test | LLM test | MV test |
|------|:-------:|:--------:|:-------:|
| Grid Fix seed2025 (valid 选参) | 0.0155 | **0.0162** | 0.0158 |
| 3-seed mean | **0.0165** | 0.0161 | 0.0160 |

→ 单 seed 排序不稳定；3-seed 后三者几乎打平，TF 略优。

---

## 四、同步清单（本机已落地）

| 路径 | 内容 |
|------|------|
| `logs/stopgate_3seed_beauty.csv` | 脚本原始 CSV（valid/test 提取有误，仅作进度记录） |
| `logs/stopgate_3seed_beauty_metrics.csv` | **权威指标** |
| `logs/stopgate_3seed_nohup.log` | 队列日志 |
| `logs/sg_beauty_*.log` | 12 个训练完整日志 |
| `logs/grid_*_beauty_seed2025.csv` | Grid / Fix CSV |
| `run_5090_stopgate_3seed.sh` | 3-seed 脚本 |
| `scripts/extract_stopgate_metrics.py` | 指标重提取 |
| `paper_recsys/experiment_status_2026072*.md` | 进展文档 |

---

## 五、下一步建议

| 优先级 | 任务 |
|:------:|------|
| P0 | 更新 `WSDM_convergence_tracker.md`：写入 3-seed 表 + stop-gate 结论 |
| P1 | Rolling early-valid window（时间稳定性） |
| P1 | Toys 域：TF/LLM/MV 用同一最优超参或小网格 |
| P2 | 论文主表定稿：ID-only / TF / LLM / MV(分析) + 机制图 |

---

## 六、一句话目标

**3-seed 完成且已同步本地；MV 未过 stop-gate；下一步写定稿表 + Toys/rolling window。**
