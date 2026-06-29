# 5090 实验状态梳理（2026-06-27）

> **机器**: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **状态**: 🔄 **已恢复**（2026-06-27 断电后重启）  
> **5090**: Toys LLM seed=2025 训练中  
> **log10**: `charlie@192.168.0.107`（5090 上 `alias log10="ssh charlie@192.168.0.107"`）· 🔄 已恢复

**相关文档**:
- 断电 checkpoint: [`pipeline_checkpoint_20260627.md`](pipeline_checkpoint_20260627.md)（暂停后生成）
- 结果汇总: [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)
- log10: [`experiment_status_log10_20260627.md`](experiment_status_log10_20260627.md)
- 导师意见: [`0626zhidao.txt`](0626zhidao.txt)

---

## 1. 进度总览

```
[✅] SASRec 主表 · Grocery multiseed · Phase 3 机制分析
[✅] UniSRec Beauty 三配置 seed=2024（旧 infer=0）
[🔄] Phase 4 GRU4Rec — 5090 9/16 · log10 4/16
[📋] Phase 5 UniSRec balanced — 接 Phase 4 末尾
[⏸️] Phase 4 已手动停止 — 恢复见 checkpoint
```

| 实验 | 状态 | 备注 |
|------|------|------|
| Phase 4 5090 text | 🔄 **10/16→** | Toys LLM seed=2025 重跑中 · `run_5090_resume_after_poweroff.sh` |
| Phase 4 log10 baseline | 🔄 **4/16→** | Beauty ID seed=2024 重跑中 · 从 5090 `ssh charlie@192.168.0.107` 启动 |

---

## 2. Phase 4 5090 明细（LLM+MV · Beauty/Toys × 4 seeds）

| Seed | Beauty LLM | Beauty MV | Toys LLM | Toys MV |
|------|------------|-----------|----------|---------|
| 42 | ✅ | ✅ | ✅ | ✅ |
| 2024 | ✅ 6/27 03:06 | ✅ | ✅ | ✅ 6/27 12:30 |
| 2025 | ✅ | ✅ | ❌ 丢弃 | ❌ |
| 2026 | 📋 | 📋 | 📋 | 📋 |

**当前 job**: `GRU4RecAlignV3` · Toys LLM seed=2025 · Phase-A ~15/20

**5090 已完成 runs**: 9/16（含 Beauty LLM 2025 test MRR@10 **3.25%**）

**暂停边界**: Beauty MV 2025 完成后 watcher 会 kill `run_5090_phase4_text_resume.sh`，**不会**自动开 Toys 2025。

---

## 3. Phase 4 log10 明细（ID+TF-IDF）

| Seed | Beauty ID | Beauty TF-IDF | Toys ID | Toys TF-IDF |
|------|-----------|---------------|---------|---------------|
| 42 | ✅ | ✅ | ✅ | ✅ |
| 2024 | 📋 | 🔄 Phase-A ep~0 | 📋 | 📋 |
| 2025 | 📋 | 📋 | 📋 | 📋 |
| 2026 | 📋 | 📋 | 📋 | 📋 |

**log10 进度**: 4/16 · 当前 Beauty TF-IDF seed=2024

---

## 4. 恢复（重启后）

**5090**:
```bash
nohup bash run_5090_resume_after_poweroff.sh >> logs/resume_after_poweroff_nohup.log 2>&1 &
```

**log10**:
```bash
nohup bash run_log10_resume_after_poweroff.sh >> logs/log10_resume_after_poweroff_nohup.log 2>&1 &
```

详见 [`pipeline_checkpoint_20260627.md`](pipeline_checkpoint_20260627.md)

---

## 5. 5090 监控

```bash
ssh charlie@www.ultrapp.online
cd ~/project/RecBole
tail -f logs/gru4rec_beauty_text_seed2025.log    # 当前 MV
tail -f logs/pipeline_pause.log                  # 断电 watcher
tail -f logs/phase4_text_resume_nohup.log        # 总队列
pgrep -af "two_phase_train|phase4|pipeline_pause"; nvidia-smi
```

---

## 6. 下一步（断电后）

1. 5090: `run_5090_phase4_text_resume_from_toys2025.sh`
2. log10: `run_log10_gru4rec_baselines_from_2024.sh`（若需要）
3. Phase 4 全部完成 → Phase 5 UniSRec balanced ×2 → 更新结果文档
