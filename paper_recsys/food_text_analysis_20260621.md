# Food 数据集文本字段详细统计

> 生成日期: 2026-06-21  
> 数据源: `/home/charlie/project/RecSysDatasets/RecBole/ProcessedDatasets/Food/Food.zip`  
> 统计方式: 5090 原地读取，未 copy 到项目目录

---

## 1. Item 文件结构

RecBole 格式 `Food/Food.item`，共 **231,637** 条记录，8 个字段：

| 字段 | 类型 | 语义 | 文本用途 |
|------|------|------|----------|
| `item_id:token` | ID | 食谱 ID | — |
| **`name:token`** | 单 token 列 | **食谱标题** | ★ 主文本 |
| `submitted_timestamp:float` | 数值 | 提交时间 | — |
| `contributor_id:token` | ID | 贡献者 | — |
| `n_steps:float` | 数值 | 步骤数 | 可选 side feature |
| `minutes:float` | 数值 | 烹饪分钟数 | 可选 side feature |
| `nutrition:float_seq` | 7 维 float | 营养信息 | 非文本，数值 side feature |
| **`tags:token_seq`** | token 序列 | **分类/标签** | ★ 辅助文本 |

> RecBole 中 `name:token` 整列视为一个 token（含空格），实际词数需按空格 split 统计（见下文）。

---

## 2. 文本字段覆盖率

| 字段 | 非空 items | 空 items | 覆盖率 |
|------|-----------|---------|--------|
| **name** | 231,636 | 1 | **99.9996%** |
| **tags** | 231,528 | 109 | **99.95%** |
| name + tags 均有 | 231,527 | — | **99.95%** |
| nutrition | 231,637 | 0 | 100% |

### 与 .inter 交叉（有交互的 231,637 items）

| 检查项 | 结果 |
|--------|------|
| 有交互且有 name | 231,636 / 231,637 (**100.0%**) |
| 有交互且有 tags | 231,528 / 231,637 (**100.0%**) |
| inter 中 item 不在 item 表 | **0** |

### 分层 name 覆盖率

| Stratum | 有 name / 总数 | 覆盖率 |
|---------|---------------|--------|
| New \([1,3)\) | 137,072 / 137,073 | 100.0% |
| Few \([3,10)\) | 73,165 / 73,165 | 100.0% |
| Frequent \([10,\infty)\) | 21,399 / 21,399 | 100.0% |

**结论**: 文本字段几乎全覆盖，各 popularity 层无缺失，适合 MV 文本管线。

---

## 3. `name`（食谱标题）详细统计

### 长度分布

| 指标 | 字符数 | 词数（空格 split） |
|------|--------|-------------------|
| mean | 27.9 | 5.0 |
| median | 26 | 5 |
| p90 | 43 | 8 |
| max | 85 | 15 |

### 字符长度直方图

| 字符区间 | #items | 占比 |
|----------|--------|------|
| 1–20 | 61,697 | 26.6% |
| 21–40 | 139,621 | **60.3%** |
| 41–60 | 28,349 | 12.2% |
| 61–80 | 1,967 | 0.8% |
| 81+ | 2 | ~0% |

### 词数直方图

| 词数区间 | #items | 占比 |
|----------|--------|------|
| 1–3 | 84,811 | 36.6% |
| **4–6** | 118,480 | **51.1%** |
| 7–10 | 27,498 | 11.9% |
| 11–15 | 847 | 0.4% |

### 文本特征

| 特征 | 值 |
|------|-----|
| 全小写 | 100% |
| 含特殊字符 (`&%#` 等) | 0% |
| 唯一 name（小写归一） | 230,185 |
| 重复 name 种类 | 1,430 |
| 重复 name 涉及 items | 2,881 (**1.2%**) |

重复最高仅 3 次，例如 `banana chocolate chip muffins`、`broccoli cheese soup`。

### 样例

```
arriba   baked winter squash mexican style
a bit different  breakfast pizza
all in the kitchen  chili
```

---

## 4. `tags`（标签序列）详细统计

### 长度分布

| 指标 | 字符数 | tag 数（空格 split） |
|------|--------|---------------------|
| mean | 200.9 | 17.9 |
| median | 191 | 17 |
| p90 | 309 | 28 |
| max | 809 | 73 |

### 标签词表

| 指标 | 值 |
|------|-----|
| 唯一 tag 数 | **560** |
| 平均每 item tag 数 | 17.9 |

tags 为 **结构化菜谱元数据**（烹饪时间、菜系、场合、饮食限制等），非自由文本评论。

### Top-20 高频 tags

| Tag | 出现 item 数 |
|-----|-------------|
| preparation | 230,546 |
| time-to-make | 225,326 |
| course | 218,148 |
| main-ingredient | 170,446 |
| dietary | 165,091 |
| easy | 126,062 |
| occasion | 114,145 |
| cuisine | 91,165 |
| low-in-something | 85,776 |
| main-dish | 71,786 |
| equipment | 70,436 |
| 60-minutes-or-less | 69,990 |
| number-of-servings | 58,949 |
| meat | 56,042 |
| 30-minutes-or-less | 55,077 |
| vegetables | 53,814 |
| taste-mood | 52,143 |
| 4-hours-or-less | 49,497 |
| north-american | 48,479 |
| 3-steps-or-less | 44,933 |

常见前缀语义：`time-*`（耗时）、`low-*`（低卡/低脂等）、`north-american`（菜系）、`*-minutes-or-less`（快速食谱）。

---

## 5. 拼接文本（MV / TF-IDF / LLM 输入建议）

实际 embedding 通常拼接 **name + tags**：

| 指标 | name only | name + tags |
|------|-----------|-------------|
| 字符 mean / med / p90 | 28 / 26 / 43 | **230 / 220 / 339** |
| 字符 max | 85 | 823 |
| token mean / med / p90 | 5 / 5 / 8 | **22 / 21 / 32** |
| token max | 15 | 76 |

### 拼接 token 数分布

| Token 区间 | #items | 占比 |
|------------|--------|------|
| 1–20 | 104,286 | 45.0% |
| **21–40** | 123,445 | **53.3%** |
| 41–60 | 3,886 | 1.7% |
| 61+ | 20 | ~0% |

**结论**: 拼接后 median ≈ 21 tokens，p90 ≈ 32，远低于 LLM 上下文限制；TF-IDF 也足够。

---

## 6. 非文本字段（参考）

### `nutrition:float_seq`

- 7 维 float，100% 非空
- 样例: `51.5 0.0 13.0 0.0 2.0 0.0 4.0`（卡路里/脂肪/碳水等，无字段名标签）
- 可用于数值 side feature，**不建议**直接当文本 embedding 输入

### `n_steps` / `minutes`

| 字段 | mean | median | p90 |
|------|------|--------|-----|
| n_steps | 1.4 | 1 | 2 |
| minutes | 2.1 | 2 | 3 |

（RecBole 存为 float 字符串，实际为整数语义）

---

## 7. 与 Amazon Beauty 对比

| 指标 | Food `name` | Beauty `title` |
|------|-------------|----------------|
| 覆盖率 | ~100% | 99.8% (447 空) |
| 词数 mean / med | 5.0 / 5 | **10.3 / 9** |
| 词数 p90 | 8 | 15 |
| 字符 mean / med | 28 / 26 | **63 / 58** |
| 辅助文本 | tags (560 词表, ~18 tags/item) | brand + categories |
| 拼接后 token med | ~21 (name+tags) | title+categories 更长 |

Food 标题更短、更规范（全小写、无特殊字符）；tags 提供结构化语义，类似 Beauty 的 categories 但词表更小 (560 vs Amazon 类目树)。

---

## 8. MV 管线接入建议

| 视图 | 建议输入 | 说明 |
|------|----------|------|
| TF-IDF | `name` + `tags` 拼接 | 560 tag 词表 + 标题词，稀疏度适中 |
| LLM view 0 (identity) | `name` | 短标题，语义清晰 |
| LLM view 1 (function) | `tags` 中 `course`, `main-ingredient`, `preparation` | 功能/用途 |
| LLM view 2 (audience) | `tags` 中 `dietary`, `occasion`, `number-of-servings` | 受众/场合 |
| LLM view 3 (category) | `tags` 中 `cuisine`, `time-to-make`, `equipment` | 类别/属性 |

与 Beauty 4-view 拆分逻辑类似，可直接复用 `gen_text_emb_*` 脚本结构，替换 item 字段映射即可。

---

## 9. 数据路径

```
/home/charlie/project/RecSysDatasets/RecBole/ProcessedDatasets/Food/Food.zip
  ├── Food/Food.item   (231,637 rows)
  └── Food/Food.inter  (1,132,367 rows)
```

JSON 备份: `paper_recsys/food_text_stats_20260621.json`
