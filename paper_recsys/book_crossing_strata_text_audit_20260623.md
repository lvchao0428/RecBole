# Beer / Yelp 分层可评估性 + 文本多样性审计

> 分层: new=1–2, few=3–9, freq≥10 次交互（全量 .inter item degree）
> eval 可行性: 按「每用户 1 条 test、test item  strata 占比≈R-strata%」粗估

> 参照 **Amazon_Beauty**: R-new=9.3% R-few=15.8% R-fr=74.9% coldR=25.1%

## 1. 分层与 eval 可行性（能否看出 new/few 效果）

| Dataset | R-new% | R-few% | R-fr% | est.test new | est.test few | new 评级 | few 评级 | cold 总体 |
|---------|--------|--------|-------|--------------|--------------|----------|----------|-----------|
| book-crossing | 27.5 | 25.8 | 46.7 | 28,970 | 27,175 | good | good | good |

**评级**: `good`≥15% R-strata & ≥10K est.test; `moderate`≥8%; `marginal`≥3%; else `unlikely`

## 2. 文本多样性（MV vs single-view 潜力）

| Dataset | 主字段 | 副字段 | 词数均值 | TTR | TF-IDF均相似度 | 跨字段多样性* | MV潜力 |
|---------|--------|--------|----------|-----|----------------|---------------|--------|
| book-crossing | book_title | book_author | 8.6 | 0.049 | 0.031 | 0.940 | high |

*跨字段多样性 = 1 − mean_cosine(TF-IDF(主字段), TF-IDF(副字段))，越高表示两路文本越互补

## 3. 与 Beauty 对比

| 指标 | Beauty | book-crossing |
|---|---|---|
| R-new% | 9.287963342840337 | 27.5 |
| R-few% | 15.844780457423619 | 25.8 |
| coldR% | 25.132743800263956 | 53.3 |
| AvgSeq | 1.6715842980621696 | 10.9 |
| 词数均值 | - | 8.6 |
| TTR | - | 0.0 |