# 实验进展 2026-08-17

## 摘要截止时间
- WSDM 2027 Abstract deadline: **Aug 17, 2026 23:59 AoE = 北京时间 Aug 18 19:59**
- Full paper deadline: Aug 24, 2026 23:59 AoE

## 今日完成

### 1. 摘要定稿（0817老师指导）
- 摘要从三段式改为单段 compact paragraph（去掉段间空行，句子不变）
- 符合 ACM/WSDM conference paper abstract 的标准呈现方式
- 0816 指导全部 6 个疑问点已修复确认：
  - ✅ "universally" 改弱 → "often conducted"
  - ✅ ~25% drop 删除 → 纯定性表述
  - ✅ LOOV 数字一致性 → 四处引用均从 tab:loov 正确生成
  - ✅ "always beats" 改弱 → "consistently improves...on warm targets"
  - ✅ complementarity axis → 完全按老师措辞
  - ✅ 摘要结构 → Three findings emerge 模板

### 2. Early60-EQ 重训完成（5090）
- 9 个模型全部训练完成（TF/LLM/MV × 3 seeds × Beauty）
- 训练时间: 2026-08-16 13:55 → 2026-08-17 09:01（约 19 小时）
- 使用 `--save` 保存了所有 checkpoint（18 个 pth 文件）
- GTS-Last warm eval 正在执行中（`logs/early60eq_eval_0817.log`）

### 3. 论文同步
- `main_wsdm2027_v3.tex`、`refs.bib`、`0817zhidao.txt` 已同步到 5090
- 摘要可直接提交

### 4. Advisor Fixes 实验完成（0815-0816）
- **TF+LLM Hybrid** (Beauty/Toys × 3 seeds): Beauty .0203±.0001, Toys .0247±.0001
- **Pure LLM-only** (Beauty/Toys × 3 seeds): Beauty .0113, Toys .0058 (远低于 ID-only)
- **Early60-EQ retrain** (TF/LLM/MV × 3 seeds × Beauty): GTS-All MRR .0180/.0183/.0183

### 5. Raw 3584-d CKA Audit 完成
- Beauty: 平均 pairwise CKA = 0.961, concat ER = 1567/14336 (10.9%)
- Toys: 平均 pairwise CKA = 0.963, concat ER = 1622/14336 (11.3%)
- 结论：冗余完全来自 Qwen 编码器本身，非 SVD/whitening 引入

### 6. 论文 tab:early 更新
- 从 GTS-All 改为 GTS-Last warm
- Main window: TF .0051 / LLM .0059 / MV .0061
- Early60-EQ: TF .0036 / LLM .0038 / MV .0043
- MV 在两个窗口均领先（一致结论）

## 待完成

### ~~P0: GTS-Last warm eval~~ ✅ 已完成
- GTS-Last warm eval 已完成（`logs/early60eq_eval_0817.log` → `paper_recsys/early60eq_gts_last_eval.csv`）
- Early60-EQ GTS-Last warm MRR@10 (3-seed mean):
  - TF: .0036±.0007 | LLM: .0038±.0003 | **MV: .0043±.0006**
- `tab:early` 已从 GTS-All 更新为 GTS-Last warm
- 正文 RQ2 段落已同步修改

### P1: 论文数据现状
| 表格/图 | 数据状态 | 是否依赖新实验 |
|---------|---------|--------------|
| tab:protocol | ✅ 完整 | 否 |
| tab:ndcg_hit | ✅ 完整 | 否 |
| tab:perseed | ✅ 完整 | 否 |
| fig:protocol | ✅ 完整 | 否 |
| tab:early | ✅ **已更新为 GTS-Last warm** | 否 |
| tab:shift | ✅ 完整 | 否 |
| tab:bucket | ✅ 完整 | 否 |
| tab:cka | ✅ 完整 | 否 |
| tab:effrank | ✅ 完整 | 否 |
| tab:loov | ✅ 完整 | 否 |
| tab:cka-prepost | ✅ 完整 | 否 |
| fig:cka | ✅ 完整 | 否 |
| tab:cost | ✅ 完整 | 否 |
| Appendix tables | ✅ 完整 | 否 |

### P2: 摘要提交
- 摘要已定稿，不依赖正在跑的实验
- 可直接提交 CMT

### P3: Full Paper 补充实验（已部署，等 GPU 空闲自动启动）
| 实验 | 内容 | 预计时间 | 是否影响摘要 |
|------|------|---------|------------|
| Pure LLM 重训+保存 | Beauty/Toys × 3 seeds (带 --save) | ~4h | 否 |
| TF+LLM Hybrid 重训+保存 | Beauty/Toys × 3 seeds (带 --save) | ~4.5h | 否 |
| GTS-Last/Random eval | 上述 12 个新 checkpoint | ~1h | 否 |
| Grocery raw 3584-d CKA | Qwen hidden embedding 生成 + CKA/ER 审计 | ~35min | 否 |
| **总计** | | **~10h** | |

- 主脚本: `scripts/run_final_supplement_0817.sh`
- Watcher: `scripts/watch_gpu_and_run_0817.sh`（每 5 分钟检查显存，`<1GB` 时自动启动）
- 当前 5090 GPU 被 `qwen_contest_2026` 占用，实验排队中
- 说明: Qwen2.5-7B hidden_size = 3584（非 4096，文件夹为历史命名）

## 检查点文件列表（Early60-EQ）
```
# TF seed 42
saved/SASRecAlignV3-Aug-16-2026_13-55-52.pth  (Phase A, 520MB)
saved/SASRecAlignV3-Aug-16-2026_14-15-07.pth  (Phase B, 1032MB)

# LLM seed 42
saved/SASRecAlignV3-Aug-16-2026_15-48-59.pth  (Phase A, 780MB)
saved/SASRecAlignV3-Aug-16-2026_16-12-47.pth  (Phase B, 1293MB)

# MV seed 42
saved/SASRecAlignMultiViewV3-Aug-16-2026_19-08-08.pth  (Phase A, 653MB)
saved/SASRecAlignMultiViewV3-Aug-16-2026_20-19-39.pth  (Phase B, 1166MB)

# TF seed 2024
saved/SASRecAlignV3-Aug-17-2026_01-17-53.pth  (Phase A)
saved/SASRecAlignV3-Aug-17-2026_01-26-20.pth  (Phase B)

# LLM seed 2024
saved/SASRecAlignV3-Aug-17-2026_02-14-44.pth  (Phase A)
saved/SASRecAlignV3-Aug-17-2026_02-25-55.pth  (Phase B)

# MV seed 2024
saved/SASRecAlignMultiViewV3-Aug-17-2026_03-27-10.pth  (Phase A)
saved/SASRecAlignMultiViewV3-Aug-17-2026_03-41-32.pth  (Phase B)

# TF seed 2026
saved/SASRecAlignV3-Aug-17-2026_05-09-50.pth  (Phase A)
saved/SASRecAlignV3-Aug-17-2026_05-18-18.pth  (Phase B)

# LLM seed 2026
saved/SASRecAlignV3-Aug-17-2026_06-06-42.pth  (Phase A)
saved/SASRecAlignV3-Aug-17-2026_06-17-54.pth  (Phase B)

# MV seed 2026
saved/SASRecAlignMultiViewV3-Aug-17-2026_07-19-16.pth  (Phase A)
saved/SASRecAlignMultiViewV3-Aug-17-2026_07-33-40.pth  (Phase B)
```
