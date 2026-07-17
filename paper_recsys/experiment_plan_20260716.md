# 7.16 全日实验规划

> 写于: 2026-07-15 22:55（7/16 00:20 更新 per-source align）  
> 机器: **5090** 主跑；**log10** 备用  
> 前置: MV 已改成 TF∥views=512（≡ LLM `both`），SENet 删除，**per-source align 统一设计**  
> 脚本: `run_5090_mv_llm_parity.sh`  
> 关联: [`experiment_status_20260716.md`](experiment_status_20260716.md)

---

## 零、设计变更历史

### 00:20 — per-source align（三模型统一）

所有模型统一改为 **pre-concat per-source align**：每个文本源独立投影后和 ID 做 InfoNCE。

| 模型 | Align 方式 | 投影层 | InfoNCE 次数 |
|------|-----------|--------|:------------:|
| TF | InfoNCE(id, proj_base(TF)) | `align_proj_base(256→H)` | 1 |
| LLM (`both`) | InfoNCE(id, proj_base(TF)) + InfoNCE(id, proj_llm(LLM)) | `align_proj_base` + `align_proj_llm(256→H)` | 2 |
| MV | InfoNCE(id, proj_base(TF)) + 4× InfoNCE(id, shared_view_proj(v_i)) | `align_proj_base` + `shared_view_align_proj(64→H)` | 5 |

- Align 投影层和 fusion 投影层（`item_text_proj`/`mv_text_proj`）**完全独立**
- TF 原先在 LLM 里是"搭便车"，现在三个模型统一 pre-concat 对齐
- **新增参数**：`align_proj_base(256→H)`=65K + `shared_view_align_proj(64→H)`=16K + `align_proj_llm(256→H)`=65K（各模型按需）
- **LLM 主表需重跑**（align 策略改变）

### 00:05 — per-view align（已被 00:20 覆盖）

4 views per-view align，但 TF 未参与。已被上述统一设计替代。

---

## 一、今晚→明早时间线（5090，per-source align 已启动）

耗时参照实测，per-source align（5 次 InfoNCE）预计稍慢（+5–10%）：
- no-Cross 一组 ≈ **2.0–2.4h**（A≈15min + B≈100min）
- +Cross 一组 ≈ **3.3–3.8h**

| 时刻（估） | 事件 | 产出 |
|:----------:|------|------|
| **00:20** ✅ | 新队列启动（per-source align） | 已确认 `[base=7.03, v0=7.05, v1=6.92, v2=6.96, v3=7.06]` |
| **01:37** ✅ | **#1 出 test** — MV no-Cross no-boost | **MRR=0.0158**（= concat align 参考） |
| **02:55** ✅ | **#2 出 test** — MV no-Cross **cb3+infer** | **MRR=0.0160**（追平 TF，未达 LLM 0.0171） |
| **05:17** ✅ | **#3 出 test** — MV +Cross no-boost | **MRR=0.0156** |
| **05:17** ✅ | 三组队列结束 | GPU 空闲 → 接白天计划 |

> 波动 ±20min 正常；以 `logs/mv_tf_views_queue.log` 的「完成:」时间为准。

### 三组对照表（跑完填「新」列）

| # | 配置 | 日志 / ckpt | 对照锚点 | 旧 align 参考 | **per-source align** | 判定 |
|:-:|------|-------------|----------|:-------------:|:--------------------:|------|
| 1 | no-Cross, no-boost | `*_mv_nc_tfviews_*` | TF **0.0157** / LLM **0.0168** | concat 0.0158 | **0.0158** ✅ | ≈ TF，略低于 LLM |
| 2 | no-Cross, **boost** | `*_mv_nc_boost_tfviews_*` | TF 0.0160 / LLM **0.0171** | — | **0.0160** ✅ | **追平 TF**，差 LLM 0.0011 |
| 3 | +Cross, no-boost | `*_mv_cross_tfviews_*` | 三模型曾打平 ~0.0157 | — | **0.0156** ✅ | Cross 无增益 |

> **注意**：LLM/TF 主表也改了 per-source align，需要重跑才能公平对比。白天排入队列。

---

## 二、旧数据速查（Beauty TS, seed=2025, min5）

### 2.1 主表锚点（可信，保留）

| 配置 | HR@10 | MRR@10 | NDCG@10 | R_new@10 |
|------|:-----:|:------:|:-------:|:--------:|
| ID-only | 0.0277 | 0.0112 | 0.0151 | 0.0137 |
| TF no-Cross, no-boost | — | **0.0157** | — | — |
| TF no-Cross, boost | 0.0340 | **0.0160** | 0.0202 | 0.0178 |
| LLM no-Cross, no-boost (`both`) | 0.0349 | **0.0168** | — | — |
| **LLM no-Cross, boost** | **0.0350** | **0.0171** | **0.0213** | **0.0270** |
| LLM +Cross, boost | — | 0.0164 | — | — |

### 2.2 旧 MV（融合不公平 / 已作废，仅对照）

| 来源 | 配置 | MRR@10 | 问题 |
|------|------|:------:|------|
| 早期 no-Cross | no-boost | 0.0141 | 加法融合 / 稀释 |
| aligned | no-Cross boost | 0.0151 | SENet off 但仍旧塔 |
| concat-fix | no-Cross no-boost | 0.0153 | 仍含复杂 view 路径 |
| concat-fix | +Cross boost | 0.0158 | 同上 |
| llmparity 误跑 | 仅 views=256 | — | **已停**，缺 TF |

**结论**：旧 MV 全程 **≤ TF**，远低于 LLM；不能代表公平 TF∥views。

### 2.3 四象限（更新前）

| | no-Cross | +Cross |
|--|:--------:|:------:|
| no-boost | LLM **0.0168** > TF 0.0157 > 旧MV≤0.0153 | ~0.0157 打平 |
| boost | LLM **0.0171** > TF 0.0160 > 旧MV≤0.0158 | LLM 0.0164 ≥ TF |

---

## 三、预期与决策（明早看数用）

### 成功标准（相对旧 MV）

| 级别 | #1 (no-boost) | #2 (boost) | 含义 |
|------|:-------------:|:----------:|------|
| **底线** | ≥ 0.0153（旧 concat） | ≥ 0.0158 | 新融合至少不退步 |
| **公平及格** | ≥ TF 0.0157 | ≥ TF 0.0160 | 不弱于同协议 TF |
| **追平 LLM** | ≥ 0.0165 | ≥ 0.0168 | 接近主表 LLM |
| **翻转** | ≥ 0.0168 | **≥ 0.0171** | 恢复 MV≥LLM 叙事 |

### 决策树（#2 已执行 ✅）

```
#2 MRR = 0.0160 → 落入「0.0160–0.0170」
→ 主方法维持 LLM no-Cross；MV 作公平多视图对照
→ 白天：① TF/LLM per-source align 重跑  ② Toys 域验证
```

---

## 四、7.16 白天时刻表

| 时段 | 优先级 | 任务 | 机器 | 备注 |
|:----:|:------:|------|:----:|------|
| **07:00–08:00** | P0 | 收三组 test；填本节「新 MV」列 + 更新 status | 本机 | 对照 §二锚点 |
| **08:00–09:00** | P0 | 按决策树定主方法一句话；同步 docs 三机 | 本机 | — |
| **09:00–12:00** | P1 | **方案 A**（若 #2 强）：启动 Toys TF/LLM/MV 公平队列 | 5090 | 估 6–8h，过夜 |
| | P1 | **方案 B**（若 #2 平/弱）：UniSRec MV 移 5090（batch↓） | 5090 | ~2–3h |
| **12:00–14:00** | — | 午间检查 Toys/UniSRec 首组是否正常 | — | — |
| **14:00–18:00** | P2 | Beauty LLM no-Cross 3-seed（2024/42）**或** 单 view 消融 `[0]` | 5090/log10 | 视 GPU |
| **14:00–18:00** | P2 | 机制分析（rank-transition / entropy） | 本机 | 不占 GPU |
| **18:00–19:00** | P0 | 写 `experiment_status_20260716.md` 收工小结 | 本机 | — |
| **夜** | — | Toys/Grocery 继续跑；勿手动抢 GPU | 5090 | — |

### log10 建议

| 时段 | 任务 |
|------|------|
| 上午 | 若 5090 满：可跑 Beauty ID-only V2 补点 / 或 LLM 3-seed 之一 |
| 下午 | 小 batch UniSRec 试跑（仅当 5090 不跑 UniSRec 时） |
| 注意 | 现有 `run_log10_ts_beauty_id_v2.sh` waiter 在等 `two_phase`；5090 有任务时它不会误启 |

---

## 五、验收 checklist

- [x] 三组日志均含 `raw_concat_dim=512 (base=256 + views=256)`
- [x] `mv_include_base=True`，无 SENet
- [x] 日志确认 per-source align: `[base=7.03, v0=7.05, v1=6.92, v2=6.96, v3=7.06]`
- [x] #1/#2/#3 test MRR 写入上表「per-source align」列
- [ ] LLM/TF 主表用 per-source align 重跑并对比
- [x] 四象限 MV 格用 per-source align 数据
- [x] 主方法结论写进 7.16 status（一句话）
- [ ] 白天队列已按决策树启动并 `nohup` 落盘

---

## 六、一句话

**MV per-source align 三组完成：boost MRR=0.0160 追平 TF、未达 LLM 0.0171；白天重跑 TF/LLM 公平锚点 + Toys 域。详见 [`experiment_status_20260716.md`](experiment_status_20260716.md)。**
