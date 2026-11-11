# WSDM 2027 稿本 2026-08-21 晚（0821zhidao derisk）

源文件：`main_wsdm2027_v3.tex` + `refs.bib` + `figures/fig_protocol_contrast.pdf`

**今晚改稿先读** `CHANGELOG_0821zhidao.md`（段落前后对照）和 `0821zhidao.txt`。

本机没有 `pdflatex`，PDF 需在 Overleaf 编译。老师：改完这些就提交；不要再让模型大范围润色。

## 相对 0820 晚多了什么

只改文字口径和图注释，主表格子未动。四类：

1. Early：写明前 60% / 次 10% / 再 10%，最后 20% unused；评价窗前移、horizon 长度不变。
2. RQ1/RQ2：items disjoint → interactions cutoff；Last 偏流行/niche、TF 均匀退化 → hypothesis / 我们没有直接统计。
3. RQ3：冗余已在 raw Qwen 输出（本提示设计），不是 encoder 本身；fail complementarity / predicted by → consistent with。
4. Fig.1：点估计 + 1 std 内 ≈。

## 不要做

重跑主实验、加数据集、全文润色、Early-Random、Toys/Grocery Early、Grocery LOO。
