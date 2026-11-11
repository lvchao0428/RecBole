# Experiment Status — 2026-07-27

> 更新时间: 7/28 16:30（Toys LOO 全部完成 ✅）  
> 上一日: [`experiment_status_20260726.md`](experiment_status_20260726.md) · 计划: [`experiment_plan_20260727.md`](experiment_plan_20260727.md)

---

## 〇、一句话

**全部主实验完成。** Toys GTS 3-seed ✅；Beauty LOO 12/12 ✅；Toys LOO 12/12 ✅。GTS 下 **TF ≥ LLM ≈ MV**；LOO 下 **LLM ≥ MV ≈ TF**（Toys LOO 出现 MV 反超 TF）——**协议决定排序**叙事成立。5090 GPU 空闲。

---

## 一、本机同步清单（从 5090）

| 类别 | 已拉到本机 |
|------|------------|
| 状态/计划 | `experiment_status_20260723/24/26.md`，`experiment_plan_20260724/27.md`，`WSDM_convergence_tracker.md` |
| 结果 CSV | `logs/wsdm_wrapup_toys_gts.csv`，`logs/beauty_loo_results.csv`，`logs/toys_loo_results.csv` |
| 审计 | `warm_target_audit_amazon_{beauty,toys_and_games}.md`，`distribution_shift_toys_and_games.json` |
| 脚本 | `run_5090_wsdm_*`，`run_5090_*_loo_*`，`scripts/audit_warm_targets.py` 等 |

---

## 二、Toys GTS Main 3-seed（✅ 定稿）

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|---------------:|
| ID-only | 0.0070 | 0.0070 | 0.0069 | **0.0070 ± 0.0001** |
| **TF** | 0.0116 | 0.0112 | 0.0110 | **0.0113 ± 0.0003** |
| LLM | 0.0111 | 0.0105 | 0.0104 | 0.0107 ± 0.0004 |
| MV | 0.0102 | 0.0112 | 0.0113 | 0.0109 ± 0.0006 |

相对 ID：文本约 **+50–60%**；跨域复现 Beauty 的 **TF ≥ LLM ≈ MV**，MV 无显著增益。

---

## 三、Beauty Clean LOO 3-seed（✅ 12/12 完成）

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|---------------:|
| ID-only | 0.0179 | 0.0171 | 0.0174 | **0.0175 ± 0.0004** |
| TF | 0.0250 | 0.0253 | 0.0249 | **0.0251 ± 0.0002** |
| LLM | 0.0251 | 0.0252 | 0.0258 | **0.0254 ± 0.0004** |
| MV | 0.0258 | 0.0251 | 0.0250 | **0.0253 ± 0.0004** |

### LOO vs GTS（Beauty）

| 模型 | LOO test | GTS test | LOO/GTS |
|------|---------:|---------:|--------:|
| ID | 0.0175 | 0.0106 | ~1.65× |
| TF | 0.0251 | 0.0165 | ~1.52× |
| LLM | 0.0254 | 0.0161 | ~1.58× |
| MV | 0.0253 | 0.0160 | ~1.58× |

读数：LOO 全面抬高；GTS 上 TF 略领先，LOO 上 **LLM ≈ MV ≈ TF**（seed42 甚至 MV 最高）→ **协议改变模型排序/叙事**。

---

## 四、Toys Clean LOO 3-seed（✅ 12/12 完成）

| 模型 | seed42 | seed2024 | seed2026 | **mean ± std** |
|------|-------:|---------:|---------:|---------------:|
| ID-only | 0.0150 | 0.0158 | 0.0148 | **0.0152 ± 0.0005** |
| TF | 0.0248 | 0.0252 | 0.0240 | **0.0247 ± 0.0006** |
| LLM | 0.0251 | 0.0250 | 0.0272 | **0.0258 ± 0.0012** |
| **MV** | 0.0243 | 0.0240 | 0.0290 | **0.0258 ± 0.0028** |

### LOO vs GTS（Toys）

| 模型 | LOO test | GTS test | LOO/GTS |
|------|---------:|---------:|--------:|
| ID | 0.0152 | 0.0070 | ~2.17× |
| TF | 0.0247 | 0.0113 | ~2.19× |
| LLM | 0.0258 | 0.0107 | ~2.41× |
| MV | 0.0258 | 0.0109 | ~2.37× |

**关键发现**：  
- LOO 抬高幅度比 Beauty 更大（~2.2× vs ~1.6×）  
- GTS 下 TF 最优（0.0113），LOO 下 LLM ≈ MV（0.0258）反超 TF（0.0247）  
- MV seed2026 突变到 0.029（方差大），但均值仍与 LLM 持平  
- 论文叙事：**评估协议决定模型排序，GTS 是更严格的公平对照**

---

## 五、时间线回顾（7/21–7/27）

1. **7/21–22**：Beauty Early60 stop-gate + Cross Top-K → 两窗口 MV 未过 gate  
2. **7/23**：0723 止损；启动 Toys GTS wrap-up + warm-target audit  
3. **7/24**：Toys 跑到 seed2026 LLM 时暂停腾 GPU  
4. **7/26**：恢复补齐 Toys LLM/MV seed2026；Beauty LOO 全量完成；启动 Toys LOO  
5. **7/27 AM**：Toys LOO seed42 + seed2024(ID/TF/LLM/MV) 完成，seed2026 ID/TF 完成  
6. **7/27 13:49**：LLM seed2026 训练中断（用户腾 GPU），已记录 PAUSE.md  
7. **7/27 18:23**：GPU 空闲，恢复 LLM seed2026（从头重跑）  
8. **7/27 ~21:03**：LLM Phase-B 跑到 epoch 17 被外部推理进程挤死；MV 仅加载数据即中断  
9. **7/28 10:22**：GPU 完全空闲，清除错误结果，最终重跑 LLM + MV seed2026  
10. **7/28 14:28**：Toys LOO 全部完成（LLM=0.0272, MV=0.0290），所有主实验到齐  

---

## 六、机器快照（7/28 16:30）

| 机器 | 状态 |
|------|------|
| **5090** | GPU 空闲（0% / 2 MiB），所有训练已完成 |
| **本机** | CSV + 文档已同步 |

---

## 七、全局实验完成度一览

| 实验 | 状态 | 备注 |
|------|------|------|
| Beauty GTS Main 3-seed | ✅ | TF > LLM ≈ MV |
| Toys GTS Main 3-seed | ✅ | TF > LLM ≈ MV |
| Beauty Clean LOO 3-seed | ✅ | LLM ≈ MV ≈ TF |
| Toys Clean LOO 3-seed | ✅ | LLM ≈ MV > TF |
| Beauty Early60 stop-gate 3-seed | ✅ | TF ≈ LLM ≥ MV，stop-gate 触发 |
| Beauty Early60 +Cross 1-seed | ✅ | Cross 无稳定增益 |
| Rank Transition / Top-K share | ✅ | TF/LLM/MV 离线分析完成 |
| Warm-target audit (Beauty + Toys) | ✅ | 分布偏移确认 |

### 尚未完成 / 可选后续

| 实验 | 优先级 | 说明 |
|------|--------|------|
| Toys Early60 stop-gate | P2 | 如审稿要求跨域 early window 复现 |
| Toys +Cross LOO / GTS | P2 | 如需进一步论证 Cross 无稳定增益 |
| UniSRec portability test | P3 | 老师 0717 提到的 transferability 验证 |
| View redundancy (CKA / effective rank) | P3 | MV 分析补充 |
