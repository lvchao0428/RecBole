# 5090 上 dataset/ 里的说明文字（2026-11-11）

这些 json / md 只在 5090 的 `dataset/` 里。`.gitignore` 忽略整个 `dataset/`，也忽略目录名以 `_4views` 结尾的路径，所以这里把文件名摊平，字节照抄。5090 磁盘上的原文件没有删。npy 和交互文件没有复制。

views.json 记的是四条 prompt 和 `view_*.npy` 的文件名。它们不是主表训练入口。冻结对照里 CKA 用过的特征目录没有从 json 钉死，这些文件只留下当时磁盘上的 prompt 文本。

| 5090 原路径 | 本目录文件 |
|---|---|
| `dataset/book-crossing/README.md` | `book-crossing__README.md` |
| `dataset/book-crossing/embed_backup_20260619_bc_aligned/qwen3_4views_prev/views.json` | `book-crossing__embed_backup_20260619_bc_aligned__qwen3_4views_prev__views.json` |
| `dataset/book-crossing/qwen3_4views/views.json` | `book-crossing__qwen3_4views__views.json` |
| `dataset/yelp_processed/README.md` | `yelp_processed__README.md` |
| `dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_ts/views.json` | `Amazon_Toys_and_Games__qwen2.5_7b_4views_ts__views.json` |
| `dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views/views.json` | `Amazon_Toys_and_Games__qwen2.5_7b_4views__views.json` |
| `dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_raw_4096d/views.json` | `Amazon_Toys_and_Games__qwen2.5_7b_4views_raw_4096d__views.json` |
| `dataset/Yelp/README.md` | `Yelp__README.md` |
| `dataset/Food/qwen2.5_7b_4views/views.json` | `Food__qwen2.5_7b_4views__views.json` |
| `dataset/Amazon_Beauty/qwen2.5_7b_4views_ts_early60/views.json` | `Amazon_Beauty__qwen2.5_7b_4views_ts_early60__views.json` |
| `dataset/Amazon_Beauty/qwen2.5_7b_4views/views.json` | `Amazon_Beauty__qwen2.5_7b_4views__views.json` |
| `dataset/Amazon_Beauty/qwen2.5_7b_4views_raw_4096d/views.json` | `Amazon_Beauty__qwen2.5_7b_4views_raw_4096d__views.json` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/qwen2.5_7b_4views_ts/views.json` | `Amazon_Grocery_and_Gourmet_Food__qwen2.5_7b_4views_ts__views.json` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/qwen2.5_7b_4views/views.json` | `Amazon_Grocery_and_Gourmet_Food__qwen2.5_7b_4views__views.json` |
| `dataset/Amazon_Grocery_and_Gourmet_Food/qwen2.5_7b_4views_raw_4096d/views.json` | `Amazon_Grocery_and_Gourmet_Food__qwen2.5_7b_4views_raw_4096d__views.json` |
