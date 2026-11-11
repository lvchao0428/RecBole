# 实验进展 — 2026-07-23

> 更新时间: 2026-07-23 23:40  
> 上一日进展: [`experiment_status_20260721.md`](experiment_status_20260721.md)  
> 明日计划: [`experiment_plan_20260724.md`](experiment_plan_20260724.md)  
> 老师指导: [`0723zhidao.txt`](0723zhidao.txt)  
> 要点: **止损叙事锁定；5090/log10 串行收尾队列已启动；G1 warm-audit 完成；G2 Toys GTS 与 Beauty ID-only 后台运行中**

---

## 〇、今日结论（一句话）

**0723 确认 stop-gate：不再救 MV。已启动两机串行收尾——5090 跑 Toys GTS Main（ID/TF/LLM/MV×3seed）；log10 跑 warm-audit + Beauty ID-only×3seed + UniSRec。G1 warm-audit：Beauty test unseen≈27.8%（last-target 口径），Toys≈34.5%，主表必须改为 warm-target。**

---

## 一、机器快照（7/23 23:40）

| 机器 | 任务 | 状态 | ETA |
|------|------|:----:|-----|
| **5090** | `run_5090_wsdm_wrapup_serial.sh`：Toys ID seed42 训练中 | 🟡 | Toys 全队列 ~24–36h |
| **log10** | `run_log10_wsdm_wrapup_serial.sh`：Beauty ID seed42 训练中 | 🟡 | ID 3-seed ~6–9h → UniSRec |
| **本机** | 文档 + 脚本已同步 | ✅ | — |

### 启动命令（已执行）

```bash
# 5090
nohup bash run_5090_wsdm_wrapup_serial.sh > logs/wsdm_wrapup_5090_nohup.log 2>&1 &

# log10
nohup bash run_log10_wsdm_wrapup_serial.sh > logs/wsdm_wrapup_log10_nohup.log 2>&1 &
```

---

## 二、0723 止损结论（已锁定）

| 证据 | 结果 |
|------|------|
| Main 3-seed | TF 0.0165±0.0008 ≥ LLM 0.0161±0.0005 ≈ MV 0.0160±0.0008 |
| Early60 3-seed | TF 0.0053±0.0002 ≥ MV 0.0052±0.0001 ≥ LLM 0.0049±0.0002 |
| View redundancy | CKA>0.83, eff-rank 136.9/256, LOVO Δ≈0 |
| Cross | improved≈degraded → Appendix only |

**论文方向**：When More Text Is Not Better（leakage-aware controlled study）  
**停止**：扩 MV / boost / Cross / 漂亮层级

---

## 三、G1 Warm-target Audit（已完成）

产物：[`warm_target_audit_amazon_beauty.md`](warm_target_audit_amazon_beauty.md)、[`warm_target_audit_amazon_toys_and_games.md`](warm_target_audit_amazon_toys_and_games.md)

| 数据集 | Valid unseen | Test unseen | Test catalog turnover |
|--------|:------------:|:-----------:|:---------------------:|
| Beauty | 15.2% | **27.8%** | 23.3% |
| Toys | — | **34.5%** | — |

> 注：与 7/17 distribution_shift 的 17.8%（bucket 口径）不同——本 audit 用 eligible user 的 last-target。两口径都说明 unseen 不可忽视；主表统一 **warm-target only**。

---

## 四、串行队列内容

### 5090（`run_5090_wsdm_wrapup_serial.sh`）

1. G1 audit + distribution_shift（已完成）
2. Toys GTS Main：ID / TF / LLM / MV × {42,2024,2026}
   - 特征：`*.ts.npy` / `qwen2.5_7b_4views_ts`
   - 超参冻结：TF 5e-4/0.3，LLM 1e-4/0.1，MV 5e-4/0.3，ID 1e-4/0.1
   - no-boost / no-Cross
3. CSV：`logs/wsdm_wrapup_toys_gts.csv`

### log10（`run_log10_wsdm_wrapup_serial.sh`）

1. warm-audit +（补齐后）distribution/view analysis
2. Beauty ID-only × {42,2024,2026}
3. UniSRec Beauty seed2025（若脚本可用）
4. CSV：`logs/wsdm_wrapup_log10.csv`

---

## 五、Early60 终态（补记，7/22 完成）

| 模型 | test MRR mean±std |
|------|:-----------------:|
| TF | **0.0053±0.0002** |
| MV | 0.0052±0.0001 |
| LLM | 0.0049±0.0002 |

两窗口同为 **TF ≈ LLM ≥ MV**（Early 上 TF 略优）→ stop-gate 正式触发。

---

## 六、一句话

**止损已执行：两机串行收尾在跑；明天查收 Toys/ID-only 进度，并规划 clean LOO 协议对照。**
