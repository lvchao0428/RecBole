# Beauty TS 实验：no-boost vs boost 跨版本对比 (2026-07-12)

> 收集时间: 2026-07-12
> 日志已同步至本地 `logs/ts_beauty*.log`

## 实验版本说明

| 版本 | 特征 | 评测过滤 | Boost 设置 | 说明 |
|------|------|---------|-----------|------|
| **V1 验证** | 旧 TF-IDF/Qwen (有泄露风险) | 全量用户 | cold_boost=**3.0**, infer_boost=**0.6** | 2026-07-10 首次 TS 验证 |
| **V1 no-boost** | 旧 TF-IDF/Qwen | 全量用户 | cold_boost=0, infer_boost=0 | 插队实验，旧特征 |
| **V2 no-boost** | TS-aware 特征 (leakage-free) | min_train≥5 | cold_boost=0, infer_boost=0 | Pilot V2，当前主设定 |
| **V2 boost** | TS-aware 特征 | min_train≥5 | cold_boost=**3.0**, infer_boost=0 | 夜间流水线，仅训练期 frequency weight |

⚠️ V1「有 boost」和 V2「boost」**不完全等价**：V1 额外开了 infer_boost=0.6。

---

## 一、总表对比 (test @10, seed=2025)

| 配置 | V1 boost<br>(old, full, cb3+inf0.6) | V1 no-boost<br>(old, full) | V2 no-boost<br>(TS, min5) | V2 boost<br>(TS, min5, cb3) |
|------|:---:|:---:|:---:|:---:|
| | MRR / NDCG / R@10 | MRR / NDCG / R@10 | MRR / NDCG / R@10 | MRR / NDCG / R@10 |
| **ID-only** | 0.0150 / 0.0188 / 0.0310 | 0.0150 / 0.0188 / 0.0310 | **0.0112** / 0.0151 / 0.0277 | 0.0112 / 0.0151 / 0.0277 |
| **TF-IDF** | **0.0204** / **0.0245** / **0.0383** | 0.0203 / 0.0244 / 0.0377 | 0.0157 / 0.0197 / 0.0326 | 0.0162 / 0.0201 / 0.0328 |
| **TF-IDF+LLM** | **0.0206** / **0.0248** / **0.0384** | 0.0042† / 0.0050 / 0.0075 | 0.0157 / 0.0195 / 0.0320 | 0.0158 / 0.0196 / 0.0319 |
| **MV-Align** | **0.0208** / **0.0250** / **0.0386** | — | 0.0157 / 0.0198 / 0.0333 | 0.0159 / 0.0201 / 0.0338 |

† V1 no-boost TF-IDF+LLM 仅完成 Phase-A，Phase-B 未完成，数值不可比。

### MRR@10 单独对比

| 配置 | V1 boost | V1 no-boost | V2 no-boost | V2 boost | V2 Δ(boost−noboost) |
|------|:---:|:---:|:---:|:---:|:---:|
| ID-only | 0.0150 | 0.0150 | 0.0112 | 0.0112 | 0 |
| TF-IDF | 0.0204 | 0.0203 | 0.0157 | 0.0162 | **+0.0005** |
| TF-IDF+LLM | 0.0206 | — | 0.0157 | 0.0158 | +0.0001 |
| MV-Align | 0.0208 | — | 0.0157 | 0.0159 | +0.0002 |

---

## 二、分层 Recall@10 对比 (test)

### New items

| 配置 | V1 boost | V1 no-boost | V2 no-boost | V2 boost |
|------|:---:|:---:|:---:|:---:|
| ID-only | 0.0155 | 0.0155 | 0.0137 | 0.0137 |
| TF-IDF | 0.0166 | 0.0158 | 0.0183 | 0.0224 |
| TF-IDF+LLM | 0.0165 | 0.0000† | 0.0198 | **0.0244** |
| MV-Align | 0.0164 | — | 0.0224 | 0.0234 |

### Few items

| 配置 | V1 boost | V1 no-boost | V2 no-boost | V2 boost |
|------|:---:|:---:|:---:|:---:|
| ID-only | 0.0395 | 0.0395 | 0.0386 | 0.0386 |
| TF-IDF | 0.0413 | 0.0403 | 0.0345 | 0.0320 |
| TF-IDF+LLM | 0.0410 | 0.0000† | 0.0308 | 0.0320 |
| MV-Align | 0.0408 | — | 0.0324 | 0.0332 |

### Frequent items

| 配置 | V1 boost | V1 no-boost | V2 no-boost | V2 boost |
|------|:---:|:---:|:---:|:---:|
| ID-only | 0.0578 | 0.0578 | 0.0563 | 0.0563 |
| TF-IDF | 0.0771 | 0.0766 | 0.0719 | 0.0722 |
| TF-IDF+LLM | 0.0779 | 0.0217† | 0.0717 | 0.0682 |
| MV-Align | **0.0786** | — | 0.0735 | **0.0740** |

---

## 三、排序分析 (MRR@10)

| 版本 | 排序 | 层级清晰? |
|------|------|----------|
| V1 boost (old, full) | MV(0.0208) > LLM(0.0206) > TF(0.0204) > ID(0.0150) | ✅ 清晰 |
| V1 no-boost (old, full) | TF(0.0203) > ID(0.0150); LLM 未完成 | ⚠️ 不完整 |
| V2 no-boost (TS, min5) | TF/LLM/MV 均 0.0157, ID(0.0112) | ❌ text 内部无层级 |
| V2 boost (TS, min5, cb3) | MV(0.0159) ≈ LLM(0.0158) ≈ TF(0.0162) > ID(0.0112) | ⚠️ 微弱 |

---

## 四、关键结论

### 1. 评测体系变化导致绝对值不可直接横比
- V2 min5 过滤后 ID-only MRR@10 从 0.0150 降到 0.0112（-25%）
- 所有 V2 绝对值低于 V1，但这是因为评测用户群更严格（train≥5 条历史）

### 2. V1 下 boost 几乎不改变排序
- TF-IDF: 0.0204 (boost) vs 0.0203 (no-boost)，差异 <0.1%
- V1 的层级主要来自 infer_boost=0.6 + 旧特征，而非 cold_boost 单独作用

### 3. V2 下 no-boost 完全拉平 text 模型
- TF-IDF = TF-IDF+LLM = MV-Align = 0.0157（MRR@10）
- 但三者均显著优于 ID-only (0.0112, +40%)

### 4. V2 cold_boost=3.0 效果微弱
- MRR@10 提升仅 +0.0001~0.0005
- **唯一明显改善**: new item Recall@10（TF-IDF+LLM: 0.0198→0.0244, +23%）
- 未能恢复 V1 时代的 MV > LLM > TF 层级

### 5. 下一步建议
- 考虑 V2 也加 infer_boost（对齐 V1 完整 boost 配置）做一组对比
- 或增大 cold_boost / 调整 cold_threshold
- Beauty TF-IDF 3-seed (V2 no-boost): mean MRR@10=0.0156, 稳定

---

## 五、V2 TF-IDF 3-seed (no-boost, TS-aware, min5)

| seed | test MRR@10 | test NDCG@10 | test R@10 |
|------|------------|-------------|----------|
| 2024 | 0.0149 | 0.0187 | 0.0310 |
| 2025 | 0.0157 | 0.0197 | 0.0326 |
| 42 | 0.0163 | 0.0202 | 0.0332 |
| **mean** | **0.0156** | **0.0195** | **0.0323** |

---

## 六、本地日志索引

| 版本 | 日志文件 |
|------|---------|
| V1 boost | `logs/ts_beauty_{id_only,tfidf,tfidf_llm,mv}_seed2025.log` |
| V1 no-boost | `logs/ts_beauty_tfidf_noboost_seed2025.log`, `ts_beauty_tfidf_llm_noboost_seed2025.log` |
| V2 no-boost | `logs/ts_beauty_{id_only_min5,tfidf_v2,tfidf_llm_v2,mv_v2}_seed2025.log` |
| V2 boost | `logs/ts_beauty_{tfidf_boost,llm_boost,mv_boost}_seed2025.log` |
| V2 3-seed | `logs/ts_beauty_tfidf_v2_seed{2024,2025,42}.log` |
| 调度 | `logs/overnight_20260711.log`, `logs/ts_beauty_pilot_v2.log` |
