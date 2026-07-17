# 今日优先级 — 2026-07-16

> 更新时间: 2026-07-16 08:45  
> 详况: [`experiment_status_20260716.md`](experiment_status_20260716.md)  
> 全日规划: [`experiment_plan_20260716.md`](experiment_plan_20260716.md)  
> 状态: **5090 空闲**；MV per-source align 三组已完成

---

## 〇、快照（08:45）

| 机器 | 任务 | 状态 | 备注 |
|------|------|:----:|------|
| **5090** | — | 🟢 **空闲** | 三组 MV 05:17 全部完成 |
| **log10** | — | 空闲 | 可排轻量任务 |

---

## 一、昨夜结果（已完成 ✅）

| # | 实验 | MRR@10 | 判定 |
|:-:|------|:------:|------|
| 1 | MV no-Cross, no-boost | **0.0158** | ≈ TF 0.0157，略低于 LLM 0.0168 |
| 2 | MV no-Cross, **cb3+infer** | **0.0160** | **追平 TF**，差 LLM 0.0011 |
| 3 | MV +Cross, no-boost | **0.0156** | Cross 无增益 |

**决策**：主方法维持 LLM no-Cross；MV 作公平多视图对照。

---

## 二、今日优先级

| 优先级 | 任务 | 机器 | 估时 | 说明 |
|:------:|------|:----:|:----:|------|
| **P0** | TF + LLM per-source align 重跑（Beauty TS） | 5090 | ~4–6h | 与 MV 同 align 协议，公平主表 |
| **P1** | Toys TF/LLM/MV 公平套件 | 5090 | 6–8h | 跨域验证 |
| P1 | UniSRec MV-V3（batch↓） | 5090 | ~2–3h | log10 OOM 备选 |
| P2 | 单 view 消融 `text_view_indices=[i]` | 5090 | ~8h | 机制分析 |
| P2 | Beauty LLM 3-seed | 5090/log10 | ~6h | 稳健性 |

### 建议排程

```
上午  → 写脚本 / 启 TF+LLM per-source align 队列（5090）
下午  → 若 TF/LLM 跑完：启 Toys 公平套件过夜
晚间  → 收数、更新 status
```

---

## 三、对照锚点

| 配置 | MRR@10 | 备注 |
|------|:------:|------|
| ID-only | 0.0112 | — |
| TF no-Cross no-boost | 0.0157 | ★ 旧 concat align |
| TF no-Cross boost | 0.0160 | ★ 旧 concat align |
| LLM no-Cross no-boost | 0.0168 | ★ 旧 concat align |
| **LLM no-Cross boost** | **0.0171** | ★ 当前最优（待重跑） |
| MV no-Cross no-boost | **0.0158** | ✅ per-source align |
| MV no-Cross boost | **0.0160** | ✅ per-source align |
| MV +Cross no-boost | **0.0156** | ✅ per-source align |

---

## 四、一句话

**MV 三组出齐、5090 空闲；白天先重跑 TF/LLM per-source align 公平锚点，再开 Toys。**
