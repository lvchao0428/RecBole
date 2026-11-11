# 两边都留下的原文（2026-11-11）

这些路径上，本地和 5090 都有对方没有的句子或图，合成会改掉其中一边的主张。原路径继续放同步前本地的那一份字节。5090 的整份原文在本目录。没有选投稿版，没有改 `paper_recsys/main_wsdm2027_v3_0823.tex`。

| 原路径 | 本目录里的 5090 原文 | 为什么没有合成 |
|---|---|---|
| `paper_recsys/main_wsdm2027_v3.tex` | `main_wsdm2027_v3.tex.from_5090` | 本地抬头是 2026-08-20（0820zhidao）。5090 抬头是 2026-08-21 晚（0821zhidao derisk）。Toys 上 Last 的点估计，一边写成与 TF 约等，一边写成三个数据集都由 TF+MV 领先。冗余段落的问法也不一样。 |
| `paper_recsys/overleaf_deploy/main_wsdm2027_v3.tex` | 与上一行同一份字节（sha256 `1a3f8b3886c3d5862abb7becf7093a4414d397391ac71db0f461d0e2c724b08d`） | 5090 上 overleaf 这份与 `paper_recsys/main_wsdm2027_v3.tex` 字节相同，不另存第二份。本地 overleaf 路径仍与本地 `paper_recsys/main_wsdm2027_v3.tex` 字节相同。 |
| `paper_recsys/main_wsdm2027_v3_zh.tex` | `main_wsdm2027_v3_zh.tex.from_5090` | 中文对照稿跟着英文 v3 的 0820 / 0821 两套措辞走，同样没有合成。 |
| `paper_recsys/figures/plot_cka_heatmap.py` | `plot_cka_heatmap.py.from_5090` | 图题一句不同。本地是 “TF-IDF is geometrically distinct”。5090 是 “TF-IDF is complementary”。 |
| `paper_recsys/figures/fig_cka_heatmap.pdf` | `fig_cka_heatmap.pdf.from_5090` | 0823 用这个文件名。两份 pdf 字节不同（5090 sha256 `4da74b31a96396d3e81842260a98fa0ca4a22f32b8c073b90c01191939be8f58`）。没有证明哪一份是 0823 编译时嵌进去的，所以路径上仍是本地那份。 |
| `paper_recsys/figures/fig_cka_heatmap.png` | `fig_cka_heatmap.png.from_5090` | 与 pdf 同一对图，png 字节也不同。 |
| `paper_recsys/overleaf_deploy/figures/fig_cka_heatmap.pdf` | 与 `fig_cka_heatmap.pdf.from_5090` 字节相同 | 5090 overleaf 的这份 pdf 与 5090 `paper_recsys/figures/` 那份相同。 |
| `paper_recsys/figures/plot_protocol_contrast.py` | `plot_protocol_contrast.py.from_5090` | 这是 v1 脚本。0823 实际 include 的是 `figures/fig_protocol_contrast_v2.pdf`，对应只在本地的 `plot_protocol_contrast_v2.py`。v1 两边的排序标注不一样，没有当成主图入口去改。 |
| `paper_recsys/figures/fig_protocol_contrast.pdf` | `fig_protocol_contrast.pdf.from_5090` | v1 图的 pdf 字节不同。0823 用的是 v2 pdf，v2 只在本地。 |
| `paper_recsys/figures/fig_protocol_contrast.png` | `fig_protocol_contrast.png.from_5090` | 与上一行同一对 v1 图。 |
| `paper_recsys/overleaf_deploy/figures/fig_protocol_contrast.png` | `overleaf_fig_protocol_contrast.png.from_5090` | 5090 上 overleaf 这份 png 与 5090 `paper_recsys/figures/fig_protocol_contrast.png` 不是同一份字节，单独留下。overleaf 的 protocol / cka pdf 与 5090 `paper_recsys/figures/` 里同名 pdf 字节相同，见上表。 |

仓库根上的 `figures/fig_cka_heatmap.pdf` 和 `figures/fig_protocol_contrast.pdf` 只在 5090，已按原路径补上，不放进本目录。
