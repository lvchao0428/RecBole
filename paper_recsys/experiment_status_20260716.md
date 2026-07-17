# 实验进展 — 2026-07-16

> 更新时间: 2026-07-16 09:15  
> 全日规划: [`experiment_plan_20260716.md`](experiment_plan_20260716.md)  
> 昨夜进展: [`experiment_status_20260715.md`](experiment_status_20260715.md)  
> 关键点: **TF/LLM per-source align 公平重跑已启动**；MV 泛化差(valid→test 漂移)待查

---

## 〇、今日结论（一句话）

**MV 公平融合（TF∥views=512 + per-source align）已跑通：no-Cross boost MRR=0.0160 追平 TF，仍低于 LLM 0.0171（旧 concat align 锚点）；主方法暂维持 LLM no-Cross，MV 作公平多视图对照；下一步优先重跑 TF/LLM per-source align + Toys 域验证。**

---

## 一、机器快照（09:15）

| 机器 | 任务 | 状态 | 备注 |
|------|------|:----:|------|
| **5090** | TF/LLM/MV per-source align 公平重跑 | 🟡 **训练中** | TF no-Cross no-boost Phase-A epoch 0 进行中 |
| **log10** | — | 空闲 | 有 ID-only waiter |

脚本：`run_5090_persource_fair.sh` (PID 2038047)
队列：TF nc → LLM nc → MV nc → TF cross → LLM cross → MV cross
日志：`logs/persource_fair_nohup.log`、`logs/ps_beauty_tf_nc_noboost_seed2025.log` 等

---

## 二、MV per-source align 结果（Beauty TS, seed=2025, min5）✅

配置：`mv_include_base=true`，`raw_concat_dim=512`，SENet 已删，align=5× InfoNCE（base + 4 views）。

| # | 配置 | MRR@10 | HR@10 | NDCG@10 | R_new@10 | R_few@10 | 完成时间 |
|:-:|------|:------:|:-----:|:-------:|:--------:|:--------:|:--------:|
| 1 | no-Cross, no-boost | **0.0158** | 0.0330 | 0.0198 | 0.0168 | 0.0328 | 01:37 |
| 2 | no-Cross, **cb3+infer** | **0.0160** | 0.0342 | 0.0203 | **0.0244** | **0.0378** | 02:55 |
| 3 | +Cross, no-boost | **0.0156** | 0.0322 | 0.0195 | 0.0183 | 0.0328 | 05:17 |

日志 / ckpt：
- #1 `logs/ts_beauty_mv_nc_tfviews_seed2025.log` → `saved/ts_beauty_mv_nc_tfviews_seed2025/`
- #2 `logs/ts_beauty_mv_nc_boost_tfviews_seed2025.log` → `saved/ts_beauty_mv_nc_boost_tfviews_seed2025/`
- #3 `logs/ts_beauty_mv_cross_tfviews_seed2025.log` → `saved/ts_beauty_mv_cross_tfviews_seed2025/`

### 与锚点对比（LLM/TF 为旧 concat align ★）

| 配置 | MV (per-source) | TF★ | LLM★ | 判定 |
|------|:---------------:|:---:|:----:|------|
| no-Cross, no-boost | **0.0158** | 0.0157 | 0.0168 | ≈ TF，略低于 LLM |
| no-Cross, boost | **0.0160** | 0.0160 | 0.0171 | **追平 TF**，差 LLM 0.0011 |
| +Cross, no-boost | **0.0156** | ~0.0157 | ~0.0157 | 与三模型 Cross 格相当 |

★ = 旧 concat align 主表，**需 per-source align 重跑**后才完全公平。

### 决策树结果（#2 boost）

```
#2 MRR = 0.0160
→ 落入「0.0160–0.0170」区间
→ MV ≈ TF；主方法仍可写 LLM no-Cross；MV 作公平多视图对照
→ 白天：重跑 TF/LLM per-source align + Toys 域验证
```

### 亮点

- **冷启动**：#2 boost 的 `R_new@10=0.0244`、`R_few@10=0.0378`，优于整体 MRR 层级表现
- **融合公平性**：#1 per-source align MRR=0.0158，与昨夜 concat align 版相同，说明新 align 未伤害融合路径
- **Cross**：+Cross 未带来增益（0.0156 < 0.0158），主方法仍选 no-Cross

---

## 三、四象限（MV 已更新；LLM/TF 待重跑）

| | **no-Cross** | **+Cross** |
|--|:------------:|:----------:|
| **no-boost** | LLM **0.0168**★ > MV **0.0158** ≈ TF 0.0157★ | MV **0.0156** ≈ ~0.0157 |
| **cb3+infer** | LLM **0.0171**★ > MV **0.0160** = TF 0.0160★ | LLM 0.0164★ / TF 0.0160★ / MV — |

★ = 旧 concat align，待 per-source align 重跑

---

## 四、Align 设计（已落地）

| 模型 | Fusion | Align（pre-concat） | InfoNCE |
|------|--------|---------------------|:-------:|
| TF | `item_text_proj(256→H)` | `align_proj_base` | 1 |
| LLM (`both`) | `item_text_proj(512→H)` | `align_proj_base` + `align_proj_llm` | 2 |
| MV | `mv_text_proj(512→H)` | `align_proj_base` + `shared_view_align_proj(64→H)` ×4 | 5 |

Align 投影与 Fusion 投影**完全独立**；代码已同步三机。

---

## 五、已知问题（Open Issues）

### ⚠️ Issue-1: MV valid→test 泛化差

- **现象**: MV no-Cross no-boost valid MRR ≈ 0.029，test MRR = 0.0158，漂移近 2 倍
- **对比**: TF/LLM 通常 valid 和 test 差距在 10-20%，不应有如此大的倍数漂移
- **可能原因**:
  - view embedding 质量不够强，训练集上过拟合信号到 valid，但无法迁移到 test
  - multi-view 融合在 temporal shift 下比单视图更敏感
- **后续**: 观察 per-source align 重跑后 TF/LLM 的 valid→test gap 做对比

### ⚠️ Issue-2: per-source align 梯度分散

- **现象**: MV 模型需要 5× InfoNCE（1 base + 4 views），LLM 为 2×，TF 为 1×
- **假说**: 多个 align loss 对 Phase-A 的 backbone 训练形成梯度冲突/分散，稀释了有效梯度信号
- **可能影响**: MV MRR (0.0158) 未能超越 LLM (0.0168)，align 过多可能是原因之一
- **后续方案**:
  1. 可视化: 训练时 log 每个 align source 的 grad norm 和 loss curve（per-source grad norm watcher）
  2. 消融: 尝试 `align_weight` 衰减策略（如 warmup-then-decay）或减少 align 频率
  3. 对比: 跑 MV 只用 1× concat-align（关掉 per-view）做消融

---

## 六、今日待办（按优先级）

| 优先级 | 任务 | 机器 | 状态 |
|:------:|------|:----:|:----:|
| **P0** | TF + LLM + MV per-source align 重跑（no-Cross no-boost） | 5090 | ▶️ 训练中 |
| **P0** | TF + LLM + MV per-source align 重跑（+Cross no-boost） | 5090 | ⏳ 队列中 |
| **P1** | Toys TF/LLM/MV 公平套件 | 5090 | 排队 |
| P1 | UniSRec MV-V3（batch↓） | 5090 | log10 OOM 备选 |
| P2 | 单 view 消融 `text_view_indices=[i]` | 5090 | 查哪个 view 贡献最大 |
| P2 | Beauty LLM 3-seed | 5090/log10 | 稳健性 |
| P3 | Issue-2 梯度分散可视化 | — | 需开发 grad norm logger |

---

## 七、已完成（累计）

| 项 | 结果 |
|----|------|
| MV 公平融合重构 | SENet 删；TF∥views=512；`mv_include_base=true` ✅ |
| Per-source align 统一设计 | V3 + MV 代码改完 ✅ |
| MV 三组 per-source align | #1/#2/#3 全部完成 ✅ |
| MV concat align #1 参考 | MRR=0.0158（与 per-source 持平） |
| UniSRec Beauty (log10) | Base✅ Align✅ MV❌ OOM |
| TF/LLM/MV 公平重跑脚本 | `run_5090_persource_fair.sh` 已启动 ✅ |

---

## 八、一句话

**TF/LLM per-source align 公平重跑已启动（5090 PID 2038047），6 组串行队列（no-Cross + Cross × 3 模型）；MV 泛化差(valid→test 2x 漂移)和 per-source 梯度分散问题已记录待查。**
