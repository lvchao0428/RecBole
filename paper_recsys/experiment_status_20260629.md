# 5090 + log10 实验状态（2026-06-29）

> **5090**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **log10**: `charlie@192.168.0.137`（5090 内网 · `scripts/log10_env.sh`）  
> **核对时间**: 2026-06-29 11:02 CST（5090 实查）

**相关文档**:
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- GRU4Rec 矩阵: [`gru4rec_phase4_matrix_20260629.md`](gru4rec_phase4_matrix_20260629.md)
- 分析与规划: [`experiment_plan_20260625.md`](experiment_plan_20260625.md)

---

## 1. 进度总览

```
[✅] SASRec 主表 · Grocery 4-seed · BC · Phase 3 机制分析
[✅] 0625 pipeline（UniSRec 旧版 + Grocery multiseed）  6/25 21:54
[✅] Phase 4 GRU4Rec text (5090 LLM+MV)                 16/16
[🔄] Phase 4 GRU4Rec baseline (ID+TF-IDF)               9/16
[✅] Phase 5 UniSRec balanced（AlignV3 + AlignMV）       2/2
[🔄] Tail: GRU4Rec baseline seeds 2024/2025/2026  on 5090
```

| 任务 | 状态 | 备注 |
|------|------|------|
| GRU4Rec LLM+MV (5090) | ✅ **16/16** | 6/27 18:59 末项 Toys 2025 完成 |
| GRU4Rec ID+TF-IDF | 🔄 **9/16** | log10: 42 全套 + 2024 部分；5090 补 2024/2025 |
| UniSRec Base | ✅ | seed=2024 · 外部 baseline |
| UniSRec +Align balanced | ✅ | 6/29 02:38 · infer_boost=0.6 |
| UniSRec +MV balanced | ✅ | 6/29 10:52 · infer_boost=0.6 |
| GRU4Rec baseline 2024–2026 | 🔄 | `run_5090_gru4rec_baselines_seeds.sh` |

**5090 GPU**: ~89% · GRU4Rec Beauty ID-only seed=2025（baseline 补跑）

---

## 2. UniSRec Beauty balanced（seed=2024 · test @10）

| 模型 | MRR@10 | HR@10 | NDCG@10 | infer_boost | 完成 |
|------|--------|-------|---------|-------------|------|
| SASRec MV-Align（主方法） | **3.18%** | 5.79% | 3.79% | 0.6 | 已有 |
| UniSRec Base | 2.47% | **6.60%** | 3.44% | — | 6/25 |
| UniSRec +Align **balanced** | **3.27%** | 6.57% | 4.04% | **0.6** | 6/29 |
| UniSRec +MV **balanced** | **3.25%** | 6.53% | 4.02% | **0.6** | 6/29 |
| UniSRec +Align（旧 infer=0） | 3.25% | 6.52% | 4.02% | 0.0 | 6/25 |
| UniSRec +MV（旧 infer=0） | 3.33% | 6.67% | 4.11% | 0.0 | 6/25 |

**写作**: 主文小表用 **Base + balanced Align/MV**；旧版 infer=0 放 footnote / portability。

---

## 3. GRU4Rec Phase 4 矩阵

详见 [`gru4rec_phase4_matrix_20260629.md`](gru4rec_phase4_matrix_20260629.md)。

| 块 | 进度 |
|----|------|
| 5090 text (LLM+MV) | **16/16** ✅ |
| baseline (ID+TF-IDF) | **9/16** 🔄 |
| 合计 checkpoint | **25/32** |

**待补 baseline**: Beauty/Toys × TF-IDF/ID · seeds 2025/2026（5090 + log10 分工见 matrix）

---

## 4. 5090 当前队列

| 脚本 | 内容 | 状态 |
|------|------|------|
| `run_5090_phase4_tail_rebalanced.sh` | UniSRec balanced + baseline 补跑 | 🔄 watchdog 管理 |
| `run_5090_gru4rec_baselines_seeds.sh` | ID-only 2024/2025/2026 | 🔄 Beauty 2025 |
| log10 `run_log10_resume_after_poweroff.sh` | baseline 续跑 | 🔄 |

完成标记: `logs/5090_pipeline_all_complete.done`（尚未生成）

---

## 5. 监控

```bash
ssh charlie@www.ultrapp.online 'cd ~/project/RecBole && \
  bash scripts/collect_gru4rec_phase4_results.sh | tail -15; \
  pgrep -af "gru4rec|phase4|unisrec" | grep -v pgrep; nvidia-smi'
```

---

## 6. 文档 / 代码同步

```bash
# 本机 ← 5090（代码+文档，不含 npy）
./pull_from_5090.sh

# 本机 → 5090
./sync_recbole.sh

# 5090 → log10（在 5090 上）
bash scripts/sync_log10_code_from_5090.sh
```
