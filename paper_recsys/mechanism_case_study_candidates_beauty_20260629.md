# Beauty 机制分析：Case Study 候选（2026-06-29）

> 数据源:  
> `ablation_study_doc/scores/beauty_{id_only,tfidf_v3,mv_v3}_seed2024_mechanism_phase_b_topk_indices.npy`  
> `ablation_study_doc/scores/beauty_*_positive_i.npy`

---

## 1. 当前能直接做什么

现有原材料已经足够先筛出一批 **MV 明显优于 ID / TF-IDF** 的 candidate user：

- `positive_i.npy` 给出每个 test user 的 ground-truth item index
- `topk_indices.npy` 给出每个模型的 top-100 推荐 item index

所以现在虽然还没有把 item index 反查成 title / text，但已经可以先选出一批值得人工看例子的 user。

---

## 2. 候选类型 A：MV 进入 Top-10，而 TF-IDF / ID 都没有

统计：

- 满足 `MV rank <= 10` 且 `TF-IDF rank > 10` 且 `ID rank > 10` 的用户数：`483`

前 12 个候选：

| user_idx | gt_item_idx | ID rank | TF-IDF rank | MV rank |
|---------:|------------:|--------:|------------:|--------:|
| 2060 | 239315 | 15 | 101 | 4 |
| 2138 | 235279 | 101 | 57 | 9 |
| 2210 | 79951 | 101 | 101 | 4 |
| 2370 | 243925 | 89 | 101 | 9 |
| 2801 | 230385 | 19 | 101 | 10 |
| 2961 | 238631 | 66 | 101 | 8 |
| 2994 | 2151 | 101 | 101 | 2 |
| 3071 | 31871 | 101 | 39 | 7 |
| 3075 | 13747 | 101 | 101 | 6 |
| 3196 | 186170 | 75 | 50 | 8 |
| 3305 | 233954 | 101 | 101 | 7 |
| 3537 | 203755 | 101 | 24 | 8 |

这类候选最适合讲下面这个故事：

> TF-IDF 或 ID-only 还没有把目标 item 拉进前 10，而 multi-view 已经能把它推进可见位置。

---

## 3. 候选类型 B：MV 相对 TF-IDF 提前至少 20 个 rank，且最终进 Top-20

统计：

- 满足 `MV rank <= 20` 且 `TF-IDF rank - MV rank >= 20` 的用户数：`979`

前 12 个候选：

| user_idx | gt_item_idx | ID rank | TF-IDF rank | MV rank |
|---------:|------------:|--------:|------------:|--------:|
| 278 | 231213 | 101 | 44 | 17 |
| 437 | 4744 | 66 | 101 | 12 |
| 738 | 2392 | 101 | 101 | 11 |
| 752 | 65074 | 101 | 101 | 16 |
| 823 | 246694 | 78 | 45 | 18 |
| 950 | 28144 | 101 | 101 | 11 |
| 1430 | 162676 | 101 | 101 | 18 |
| 1540 | 238606 | 26 | 101 | 12 |
| 1693 | 159825 | 72 | 101 | 19 |
| 1695 | 160675 | 47 | 64 | 14 |
| 1966 | 69708 | 101 | 101 | 12 |
| 2060 | 239315 | 15 | 101 | 4 |

这类候选更适合讲：

> MV 不是只做了边缘微调，而是把目标 item 从深后排显著抬升到前 20。

---

## 4. 当前 case study 还差哪一步

目前这份文件还只是 **candidate shortlist**，还不是最终论文里的案例图。

要变成真正可放论文的 case study，下一步需要：

1. 把 `gt_item_idx` 和 top-k item index 反查成 item title / metadata
2. 对每个候选 user 抽出 `ID / TF-IDF / LLM / MV` 的完整 top-k item 列表
3. 选 2 到 3 个最有叙事价值的例子，配上：
   - 目标 item 是否是 new / few / frequent
   - TF-IDF、single-view LLM、MV 的 rank 变化
   - 是否体现“语义相关但低频 item 被 MV 往前拉”

---

## 5. 推荐优先看哪些例子

如果只先人工看少量样本，建议优先看：

- `user_idx=2994, gt_item_idx=2151`  
  ID=101, TF-IDF=101, MV=2

- `user_idx=2210, gt_item_idx=79951`  
  ID=101, TF-IDF=101, MV=4

- `user_idx=2060, gt_item_idx=239315`  
  ID=15, TF-IDF=101, MV=4

- `user_idx=3305, gt_item_idx=233954`  
  ID=101, TF-IDF=101, MV=7

这几条的 rank jump 最明显，最适合作为第一批人工核查对象。
