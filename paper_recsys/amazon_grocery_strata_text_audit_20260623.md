# Beer / Yelp 分层可评估性 + 文本多样性审计

> 分层: new=1–2, few=3–9, freq≥10 次交互（全量 .inter item degree）
> eval 可行性: 按「每用户 1 条 test、test item  strata 占比≈R-strata%」粗估

> 参照 **Amazon_Beauty**: R-new=9.3% R-few=15.8% R-fr=74.9% coldR=25.1%

## 1. 分层与 eval 可行性（能否看出 new/few 效果）

| Dataset | R-new% | R-few% | R-fr% | est.test new | est.test few | new 评级 | few 评级 | cold 总体 |
|---------|--------|--------|-------|--------------|--------------|----------|----------|-----------|
| Amazon_Grocery | 9.5 | 17.3 | 73.2 | 72,913 | 133,247 | moderate | good | good |

**评级**: `good`≥15% R-strata & ≥10K est.test; `moderate`≥8%; `marginal`≥3%; else `unlikely`

## 2. 文本多样性（MV vs single-view 潜力）

| Dataset | 主字段 | 副字段 | 词数均值 | TTR | TF-IDF均相似度 | 跨字段多样性* | MV潜力 |
|---------|--------|--------|----------|-----|----------------|---------------|--------|
| Amazon_Grocery | title | categories | 13.6 | 0.017 | 0.062 | 0.933 | high |

*跨字段多样性 = 1 − mean_cosine(TF-IDF(主字段), TF-IDF(副字段))，越高表示两路文本越互补

## 3. 与 Beauty 对比

| 指标 | Beauty | Amazon_Grocery |
|---|---|---|
| R-new% | 9.287963342840337 | 9.5 |
| R-few% | 15.844780457423619 | 17.3 |
| coldR% | 25.132743800263956 | 26.8 |
| AvgSeq | 1.6715842980621696 | 1.7 |
| 词数均值 | - | 13.6 |
| TTR | - | 0.0 |