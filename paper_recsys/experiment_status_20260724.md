# Experiment Status — 2026-07-24

> 更新时间: 12:30（本地）  
> **5090 GPU 已空闲**（训练已停，供其他任务使用）

---

## 〇、停训节点（必须先看）

| 项 | 内容 |
|----|------|
| 停训原因 | 临时腾 GPU |
| 队列脚本 | `run_5090_wsdm_wrapup_serial.sh`（WSDM 0723 wrap-up） |
| 暂停记录 | `logs/wsdm_wrapup_5090_PAUSE.md` |
| 结果 CSV | `logs/wsdm_wrapup_toys_gts.csv` → 已同步 `paper_recsys/wsdm_wrapup_toys_gts.csv` |

**中断位置**

- 任务：`wrap_toys_llm_lr1e4_do0.1_seed2026`（Toys · LLM · seed=2026）
- Phase-A：已完成（valid MRR@10≈0.0025）
- Phase-B：约 **epoch 39 / 50** 时被 kill（非最终结果，勿报）
- Checkpoint：`saved/wrap_toys_llm_lr1e4_do0.1_seed2026/`
  - Phase-A：`SASRecAlignV3-Jul-24-2026_11-26-35.pth`
  - Phase-B 中途：`SASRecAlignV3-Jul-24-2026_11-42-32.pth`（不可作正式数）

**恢复建议**

1. 重跑 LLM Toys seed=2026（整段 two-phase，勿用中途 ckpt 报数）
2. 再跑 MV Toys seed=2026（`wrap_toys_mv_lr5e4_do0.3_seed2026`）
3. 当前 wrap-up 队列在 G2 Toys 结束后即 COMPLETE，无 Grocery 后续段

---

## 一、Toys GTS Main 已完成结果（Beauty 锁定超参迁移）

协议：Global TS · min5 · no-boost · no-Cross · train-only `.ts` 特征  
超参：ID/LLM `lr=1e-4, do=0.1`；TF/MV `lr=5e-4, do=0.3`

### Test MRR@10

| 模型 | seed42 | seed2024 | seed2026 | 备注 |
|------|-------:|---------:|---------:|------|
| ID-only | 0.0070 | 0.0070 | 0.0069 | valid CSV 提取坏（显示 0），test 可用 |
| TF | **0.0116** | **0.0112** | **0.0110** | 3-seed 完成 |
| LLM | 0.0111 | 0.0105 | — | **seed2026 未完成** |
| MV | 0.0102 | 0.0112 | — | **seed2026 未跑** |

粗算（仅已完成 seed）：

| 模型 | n | test MRR 粗均值 |
|------|:-:|----------------:|
| ID | 3 | ≈0.0070 |
| TF | 3 | ≈0.0113 |
| LLM | 2 | ≈0.0108 |
| MV | 2 | ≈0.0107 |

初步读数（待 seed2026 补齐后定稿）：Toys 上文本相对 ID-only 仍有约 +50% 量级；TF 略强于 LLM/MV；与 Beauty「TF≈LLM≥MV」方向一致，MV 无显著优势。

### Valid MRR（文本模型，CSV）

| 模型 | seed42 | seed2024 | seed2026 |
|------|-------:|---------:|---------:|
| TF | 0.0250 | 0.0254 | 0.0245 |
| LLM | 0.0237 | 0.0237 | （中断） |
| MV | 0.0238 | 0.0240 | （未跑） |

---

## 二、G1 已完成（同 wrap-up 开头）

- Beauty / Toys distribution shift 已重跑；Toys：`distribution_shift_toys_and_games.json`
- Toys valid→test：unseen 目标 13.0%→21.3%，head 60.1%→52.3%，popularity 中位 18→11（同样呈时间漂移）

---

## 三、机器状态

| 机器 | 状态 |
|------|------|
| **5090** | GPU 空闲（0%，≈2 MiB）；wrap-up **PAUSED** |
| 本机 | 已拉 CSV / queue log / Toys dist-shift / PAUSE 说明 |

---

## 四、今日文件清单

| 路径 | 说明 |
|------|------|
| `logs/wsdm_wrapup_5090_PAUSE.md` | 停训节点（权威） |
| `paper_recsys/wsdm_wrapup_toys_gts.csv` | Toys 已完成分数 |
| `logs/wsdm_wrapup_5090_queue.log` | 队列时间线 |
| `distribution_shift_toys_and_games.json` | Toys 漂移 |
| `paper_recsys/experiment_status_20260724.md` | 本文件 |
