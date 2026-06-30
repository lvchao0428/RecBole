# Beauty 机制分析：Score Concentration（2026-06-29）

> 数据源: `ablation_study_doc/scores/beauty_{id_only,tfidf_v3,mv_v3}_seed2024_mechanism_phase_b_topk_scores.npy`  
> 图: `ablation_study_doc/figures/beauty_mechanism_concentration_rank_curve_20260629.png`  
> 口径: 对每个 user 的 top-K score 做 softmax 归一化后，统计 `Top-1 share`、`Top-5 share`、`Entropy`、`Normalized Entropy`、`Gini`

---

## 1. Top-20 / Top-100 concentration

### Top-20

| Model | Top-1 share | Top-5 share | Entropy | Normalized entropy | Gini |
|------|-------------:|------------:|--------:|-------------------:|-----:|
| ID-only | 0.0542 | 0.2660 | 2.9943 | 0.9995 | 0.0274 |
| TF-IDF | 0.0532 | 0.2624 | 2.9948 | 0.9997 | 0.0213 |
| MV | 0.0532 | 0.2623 | 2.9949 | 0.9997 | 0.0212 |

### Top-100

| Model | Top-1 share | Top-5 share | Entropy | Normalized entropy | Gini |
|------|-------------:|------------:|--------:|-------------------:|-----:|
| ID-only | 0.5425 | 0.7067 | 2.1948 | 0.4766 | 0.7656 |
| TF-IDF | 0.1041 | 0.2473 | 4.1998 | 0.9120 | 0.3613 |
| MV | 0.0969 | 0.2354 | 4.2367 | 0.9200 | 0.3495 |

---

## 2. 直接结论

### 2.1 变化不是随机波动

- 在 `Top-20` 上，三种模型的 softmax share 几乎都接近均匀，说明差异主要不体现在最前 20 个位置内部的极端尖锐化。
- 在 `Top-100` 上，差异非常明显：
  - `ID-only` 的 `Top-1 share = 0.5425`，远高于 `TF-IDF (0.1041)` 和 `MV (0.0969)`
  - `ID-only` 的 `Normalized entropy = 0.4766`，显著低于 `TF-IDF (0.9120)` 和 `MV (0.9200)`
  - `ID-only` 的 `Gini = 0.7656`，远高于 `TF-IDF (0.3613)` 与 `MV (0.3495)`

这说明文本增强和 multi-view 确实改变了 score distribution，而且幅度很大。

### 2.2 这次不是“cross 让分数更尖”，而是“cross + text 让 top-100 质量分布更平”

- 在当前这组 `Beauty seed=2024` 结果里，`ID-only` 才是最 head-heavy 的分布
- `TF-IDF` 和 `MV` 都显著降低了 top-100 内部的头部垄断
- `MV` 相比 `TF-IDF` 还略微更平一些：
  - `Top-1 share`: `0.1041 -> 0.0969`
  - `Entropy`: `4.1998 -> 4.2367`
  - `Gini`: `0.3613 -> 0.3495`

所以如果写到 WSDM 里，更稳的说法不是预设“cross 一定提高 concentration”，而是：

> text-aware cross / alignment / multi-view 会系统性重排 full-ranking score mass，改变 top-100 内部的分配结构；在 Beauty 这组结果里，这种变化体现为更分散、更高熵的 candidate score profile。

### 2.3 MV 的增益更像“在更平的分布里把目标 item 往前拉”

从 `TF-IDF -> MV` 看：

- `Top-20` 的 concentration 指标几乎不变
- `Top-100` 的分布还略微更平

这意味着 MV 的改进不像是简单把少数 item 继续抬到极高分，而更像是在保留较宽候选覆盖的同时，重排局部相对顺序。

---

## 3. 图的读法

图文件：

- `ablation_study_doc/figures/beauty_mechanism_concentration_rank_curve_20260629.png`

推荐解读方式：

- 看 rank-1 到 rank-20 的平均 softmax share 曲线
- `ID-only` 会呈现更陡的头部下落
- `TF-IDF` 与 `MV` 曲线更平缓，且 `MV` 通常略低于 `TF-IDF` 的 rank-1 share

---

## 4. 当前还缺什么

### 已经有

- `score concentration`
- `entropy`
- `gini`
- `top-1 / top-5 share`

### 还没补上

- `head item 占比`
  - 当前已同步 `topk_indices`
  - 但本地还没把推荐 item index 和训练频次桶 join 起来
- 这一步补完后，就可以把“score concentration”和“popularity allocation”放在同一页解释
