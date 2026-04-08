# 论文变更总结（2026-04-08 老师指导意见落地）

对应文件：[main_original.tex](main_original.tex)  
依据：[zhidao_0408.txt](zhidao_0408.txt) 与修订规划（Table 2 数据对齐、结构节奏、Main track 气质）。

---

## 1. 数据与表述对齐（硬伤修复）

- **问题**：参数量说明中 Beauty 的 `|I|` 曾写为 12,102，与数据集统计表（Beauty 259,217 items）及约 67M 级参数量不一致。
- **修改**：
  - 在 **Parameter counts** 段落中，将 Amazon Beauty 的设定写为 `|I|=259,217`，与 Table 1 一致。
  - 在 **Model complexity**（`tab:params` 前文）中增加一句：约 66M 参数在 item embedding 表（`|I|×d`），其余约 1–3M 为 Transformer、文本投影与 cross 等，便于读者核对 67.2M–68.9M 的量级。

---

## 2. 结构：Prompt 模板位置调整

- **问题**：原 `\subsection{Prompt Templates for Text Embeddings}` 插在主结果与 Analysis 之间，打断「结果 → 分析」的阅读节奏。
- **修改**：
  - 删除 Section 5 末尾的独立 Prompt 小节。
  - 在 **§4.3 Implementation Details** 中，于「Training and evaluation protocol」之后、「Parameter counts」之前，新增 `\paragraph{Prompt templates.}`。
  - 内容压缩为一段：单视图模板、四路多视图模板列表、Qwen2.5-7B、SVD 至 `d_v=64` 与 per-view SE gating 的一句话说明。

---

## 3. 语气与小标题（减少 artifact 感）

- **Reproducibility 小标题**：`\paragraph{Reproducibility.}` 改为 `\paragraph{Training and evaluation protocol.}`（正文仍说明种子、指标与配置可复现，但不以「Reproducibility」为小节名）。
- **Table（`tab:params`）脚注**：删除表下「Values rounded from training logs (exact counts: …)」的精确参数/FLOPs 长脚注；主文仅保留四舍五入后的 M 级数字，精确值保留在匿名仓库即可。

---

## 4. 未改动的部分（按老师意见保持）

- **结论**：未加强措辞，维持克制表述（含 Beauty/Toys 上 cold-start HR 与 NDCG_new/MRR_new 的口径）。
- **图表**：Figure 2/3 灰度与可读性相关处理未改。

---

## 5. 文件清单

| 文件 | 变更类型 |
|------|----------|
| `paper_sigir/main_original.tex` | 按上表修改 |

如需与历史版本对比，可参考同目录下的备份稿（例如 `main_original_0408.tex`）。
