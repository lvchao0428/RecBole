# 5090 实验状态梳理（2026-06-26）

> **⚠️ 已 supersede**: 最新见 [`experiment_status_20260627.md`](experiment_status_20260627.md)

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **核对时间**: 2026-06-26 08:20 CST（5090 + log10 实查）  
> **GPU 状态**: RTX 5090 **训练中**（Phase 4 · GRU4Rec Toys TF-IDF+LLM seed=42）  
> **当前主线**: Phase 4 resume → Phase 5 UniSRec balanced（已挂队列末尾）

**相关文档**:
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- log10 状态: [`experiment_status_log10_20260626.md`](experiment_status_log10_20260626.md)
- 分析与规划: [`experiment_plan_20260625.md`](experiment_plan_20260625.md)
- 导师意见: [`0626zhidao.txt`](0626zhidao.txt) · [`0625zhidao.txt`](0625zhidao.txt)

---

## 1. 进度总览

```
[✅] Beauty/Toys 主表 (SASRec, 4 seeds)
[✅] BC / Grocery phaseb50 + Grocery 4-seed multiseed
[✅] UniSRec Beauty 三配置 seed=2024（infer_boost=0 旧版 · 见 §4）
[✅] Phase 3A–3C 机制分析
[🔄] Phase 4 GRU4Rec（5090 LLM+MV · log10 ID+TF-IDF）
[📋] Phase 5 UniSRec +Align/+MV balanced（Beauty seed=2024 · 2 runs · 队列末尾）
```

| 实验 | 状态 | 备注 |
|------|------|------|
| Phase 4 5090 text | 🔄 **2/16** | Beauty LLM seed=42 ✅ 6/26 04:37；Toys LLM seed=42 训练中 |
| Phase 4 log10 baseline | 🔄 **2/16** | Beauty ID seed=42 ✅；Beauty TF-IDF seed=42 训练中 |
| Phase 5 UniSRec balanced | 📋 **0/2** | 接 Phase 4 自动启动 · `run_unisrec_v3_align_balanced_batch_beauty.sh` |
| UniSRec 旧版 Align/MV | ✅ 已有 | `infer_boost=0` · checkpoint 无 `_balanced` 后缀 |

---

## 2. 5090 当前队列（`run_5090_phase4_text_resume.sh`）

**启动**: 2026-06-26 08:19 · PID `run_5090_phase4_text_resume.sh` · nohup → `logs/phase4_text_resume_nohup.log`

| 顺序 | 阶段 | 内容 | 状态 |
|------|------|------|------|
| 1 | Phase 4 | GRU4Rec LLM+MV · 4 seeds × Beauty/Toys（跳过 Beauty 42） | 🔄 Toys 42 LLM |
| 2 | Phase 4 | `wait_and_pull_log10_gru4rec.sh` | 📋 |
| 3 | **Phase 5** | UniSRecAlignV3 **balanced** | 📋 |
| 4 | **Phase 5** | UniSRecAlignMultiViewV3 **balanced** | 📋 |

**Balanced 参数**（与 SASRec 主表 LLM/MV 一致）:
`align_weight=0.1, cold_text_boost=3.0, infer_boost=0.6, cold_threshold=10`

**脚本 / checkpoint**:
| Run | 脚本 | Checkpoint |
|-----|------|------------|
| +Align balanced | `run_unisrec_align_v3_beauty_stratified_balanced.sh` | `saved/unisrec_align_v3_beauty_stratified_balanced_seed2024/` |
| +MV balanced | `run_unisrec_align_multiview_v3_beauty_stratified_balanced.sh` | `saved/unisrec_align_multiview_v3_beauty_stratified_balanced_seed2024/` |

**日志**: `logs/unisrec_align_v3_beauty_balanced_seed2024.log` · `logs/unisrec_align_multiview_v3_beauty_balanced_seed2024.log`

---

## 3. UniSRec 参数审计（0626）

| 行 | 配置 | infer_boost | 用途 |
|----|------|-------------|------|
| Row 1 | UniSRec Base | N/A（无 V3 插件） | 外部 text baseline ✅ |
| Row 2/3 旧 | +Align / +MV（6/25） | **0.0** | 已有结果 · portability 参考 |
| Row 2/3 新 | +Align / +MV **balanced** | **0.6** | Phase 5 · 与主表对齐 |

> 易混：`cold_threshold=10` ≠ `infer_boost`；主表 `β_infer=0.6`。

**0626 导师口径**（[`0626zhidao.txt`](0626zhidao.txt)）:
- UniSRec = external baseline + portability sanity check，三行够，不加 cross/cold sweep
- 主文小表：Base → +Align → +MV；不与 SASRec+MV 直接比强弱
- 优先：Grocery 四层级 ✅ · UniSRec 表进主文/appendix · BC stress test · GRU4Rec 可缓

---

## 4. Phase 4 GRU4Rec 详情

**分工**:
| 机器 | 任务 | 脚本 | 进度 |
|------|------|------|------|
| **5090** | TF-IDF+LLM + MV | `run_5090_phase4_text_resume.sh` | **2/16** |
| **log10** | ID-only + TF-IDF | `run_log10_gru4rec_baselines.sh` | **2/16** |

**5090 已完成**:
- Beauty GRU4Rec TF-IDF+LLM seed=42（~6.5h · 6/26 04:37 完成）

**5090 当前**:
- Toys GRU4Rec TF-IDF+LLM seed=42 · Phase-A grid · GPU ~97%

**已修复（6/26）**:
- Toys LLM 路径 `qwen3.base.npy` → `qwen2.5_7b.base.npy`（此前 Phase 4 04:38 崩溃）
- Phase 4 resume 重启 + Phase 5 挂队列末尾

**预估**: Phase 4 ~**6/28–6/29** · Phase 5 UniSRec balanced ~**+6h**（Align ~2h + MV ~4h）

---

## 5. 核心结果快照（不变 · seed=2024）

### Beauty: SASRec MV vs UniSRec（旧版 infer_boost=0）

| 模型 | MRR@10 | HR@10 | NDCG@10 |
|------|--------|-------|---------|
| SASRec MV-Align | **3.18%** | 5.79% | 3.79% |
| UniSRec Base | 2.47% | **6.60%** | 3.44% |
| UniSRec +Align（旧） | 3.25% | 6.52% | 4.02% |
| UniSRec +MV（旧） | 3.33% | 6.67% | 4.11% |

→ Phase 5 balanced 跑完后更新上表 Row 2/3；Base 行不变。

---

## 6. 监控

```bash
ssh charlie@www.ultrapp.online
cd ~/project/RecBole
tail -f logs/phase4_text_resume_nohup.log          # 总队列
tail -f logs/gru4rec_toys_text_seed42.log          # 当前 GRU4Rec
tail -f logs/unisrec_align_balanced_batch_beauty_nohup.log  # Phase 5（开始后）
pgrep -af "phase4_text_resume|two_phase_train"; nvidia-smi
```

---

## 7. 下一步

1. **自动**: Phase 4 完成 → pull log10 → Phase 5 UniSRec balanced ×2
2. **写作**: 用 balanced 版更新 UniSRec 小表；旧版 footnote `infer_boost=0`
3. **GRU4Rec**: Phase 4 全完成后汇总 appendix 第二 backbone 小表

**文档同步**:
```bash
rsync -avz paper_recsys/experiment_status_20260626.md \
  paper_recsys/experiment_status_log10_20260626.md \
  paper_recsys/experiment_results_all_20260625.md \
  charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
```
