# 实验进展 2026-08-21

全文截止：WSDM 2027 full paper **Aug 24 23:59 AoE**（北京 **Aug 25 19:59**）。摘要已交。

**时间锚点：北京 18:37。** `0821zhidao` 最后一轮 derisk 已写入；段落前后对照见 `CHANGELOG_0821zhidao.md`。稿本：`VERSION_0821.md`。老师意见原文：`0821zhidao.txt`。

老师原话：已经可以提交；改完这些口径就交；**不要再让大模型全文润色**，最后一轮只人工逐句 diff。本轮只动点名的 protocol / claim / Fig.1 语气，**数字未改、没有重跑**。

---

## 现在（18:37）

| 块 | 状态 |
|---|---|
| 英文稿 `main_wsdm2027_v3.tex` | **0821 晚 derisk 已写入**（915 行） |
| 中文对照 `main_wsdm2027_v3_zh.tex` | 同一处已同步（919 行） |
| Fig.1 | 点估计 + 1 std 内 ≈；PDF 已重出 |
| 主表 / Early / 交集 / Hybrid / CKA 数字 | **未动** |
| RecBole GPU | 不需要跑实验 |
| Overleaf 编译 | **还没做**（本地/5090 都没有 texlive） |

**今晚改稿先读：** `0821zhidao.txt` → `CHANGELOG_0821zhidao.md` → `main_wsdm2027_v3.tex`。

---

## 今天做了什么

### 上午：从 5090 拉回夜间稿 + 新指导

- 02:17 英文/中文稿（摘要统一 TF / TF+LLM / TF+MV；`tab:shift`/`tab:bucket` 改回 all-target catalog context；交集附录仍在）。
- 08:38 `0821zhidao.txt`：可以提交，再做一轮文字 derisk，不要重跑、不要加故事。
- `final_draft_todolist_0820.md`：0820 十条对照（已全部 [x]）。

### 晚上：按 `0821zhidao` 四类全部改完

1. **Early 协议写死。** `[0.6,0.1,0.1]` 改为：前 60% 训练、随后 10% 验证、再随后 10% 测试，最后 20% unused；评价窗前移，valid/test 长度不变。§3.1 拆出独立 Early 段；Contribution 2、`tab:early`、RQ2 开篇、结论第 2 条同步。
2. **RQ1/RQ2 机制降成 hypothesis。** items temporally disjoint → 训练/评估 *交互* 被全局时间截断分开（warm 目标仍在 train catalog）。Last 偏流行/niche、TF-IDF 均匀退化：删掉当事实的句子，改成 one plausible hypothesis，并写明没有直接统计。claim 只保留：同 checkpoint 改 target selection 就能改 winner。
3. **RQ3 因果收口。** 冗余 = already present in the raw Qwen outputs under our prompt design，不是 originates in the encoder。fail complementarity / exactly predicted by → consistent with。Limitations 与 Future (2) 同步。
4. **Fig.1 winner 语气。** 柱仍是点估计；Toys Last：TF+MV ≈ TF > TF+LLM > ID；LOO：TF+LLM ≈ TF+MV ≈ TF。`tab:protocol` caption 与 RQ1 bullet 同步。

逐段前后对照：`CHANGELOG_0821zhidao.md`。

---

## 还剩什么（都不是新实验）

| 项 | 要不要做 |
|---|---|
| **Overleaf 编译**核 `tab:intersection` 表宽、总页数、Fig.1 注释是否重叠 | **要** |
| 作者通读一遍（老师：a fresh set of eyes） | **要** |
| 再让模型全文润色 / 重跑主实验 / 加数据集 | **不要** |
| Early-Random、Toys/Grocery Early、Grocery LOO | **不要** |

---

## 路径

| 用途 | 路径 |
|---|---|
| 今晚英文稿 | `paper_recsys/main_wsdm2027_v3.tex` |
| 中文对照 | `paper_recsys/main_wsdm2027_v3_zh.tex` |
| 段落前后对照 | `paper_recsys/CHANGELOG_0821zhidao.md` |
| 稿本说明 | `paper_recsys/VERSION_0821.md` |
| 老师意见 | `paper_recsys/0821zhidao.txt` |
| Fig.1 | `paper_recsys/figures/fig_protocol_contrast.pdf` |
| Overleaf | `paper_recsys/overleaf_deploy/`（应与 v3 对齐） |
