# Beer / Yelp 分层可评估性 + 文本多样性审计

> 分层: new=1–2, few=3–9, freq≥10 次交互（全量 .inter item degree）
> eval 可行性: 按「每用户 1 条 test、test item  strata 占比≈R-strata%」粗估

> 参照 **Amazon_Beauty**: R-new=9.3% R-few=15.8% R-fr=74.9% coldR=25.1%

## 1. 分层与 eval 可行性（能否看出 new/few 效果）

| Dataset | R-new% | R-few% | R-fr% | est.test new | est.test few | new 评级 | few 评级 | cold 总体 |
|---------|--------|--------|-------|--------------|--------------|----------|----------|-----------|
| Amazon_Beauty | 9.3 | 15.8 | 74.9 | 112,410 | 191,765 | moderate | good | good |
| BeerAdvocate | 2.8 | 5.4 | 91.8 | 938 | 1,798 | unlikely | marginal | marginal |
| yelp2018 | 0.0 | 8.7 | 91.3 | 20 | 115,223 | unlikely | moderate | marginal |
| yelp2021 | 0.0 | 3.7 | 96.3 | 0 | 80,147 | unlikely | marginal | unlikely |
| yelp2022 | 0.0 | 4.7 | 95.3 | 0 | 93,271 | unlikely | marginal | unlikely |
| Food | 16.1 | 30.6 | 53.4 | 36,454 | 69,222 | good | good | good |

**评级**: `good`≥15% R-strata & ≥10K est.test; `moderate`≥8%; `marginal`≥3%; else `unlikely`

## 2. 文本多样性（MV vs single-view 潜力）

| Dataset | 主字段 | 副字段 | 词数均值 | TTR | TF-IDF均相似度 | 跨字段多样性* | MV潜力 |
|---------|--------|--------|----------|-----|----------------|---------------|--------|
| Amazon_Beauty | title | categories | 16.9 | 0.014 | 0.080 | 0.848 | high |
| BeerAdvocate | name | style | 5.8 | 0.069 | 0.053 | 0.752 | high |
| yelp2018 | item_name | categories | 9.2 | 0.031 | 0.062 | 0.826 | high |
| yelp2021 | item_name | categories | 10.2 | 0.027 | 0.068 | 0.822 | high |
| yelp2022 | item_name | categories | 10.3 | 0.026 | 0.069 | 0.826 | high |
| Food | name | tags | 36.1 | 0.003 | 0.181 | 0.804 | high |

*跨字段多样性 = 1 − mean_cosine(TF-IDF(主字段), TF-IDF(副字段))，越高表示两路文本越互补

## 3. 与 Beauty 对比

| 指标 | Beauty | Amazon_Beauty | BeerAdvocate | yelp2018 | yelp2021 | yelp2022 | Food |
|---|---|---|---|---|---|---|---|
| R-new% | 9.287963342840337 | 9.3 | 2.8 | 0.0 | 0.0 | 0.0 | 16.1 |
| R-few% | 15.844780457423619 | 15.8 | 5.4 | 8.7 | 3.7 | 4.7 | 30.6 |
| coldR% | 25.132743800263956 | 25.1 | 8.2 | 8.7 | 3.7 | 4.7 | 46.6 |
| AvgSeq | 1.6715842980621696 | 1.7 | 47.5 | 4.0 | 3.9 | 3.5 | 5.0 |
| 词数均值 | - | 16.9 | 5.8 | 9.2 | 10.2 | 10.3 | 36.1 |
| TTR | - | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |