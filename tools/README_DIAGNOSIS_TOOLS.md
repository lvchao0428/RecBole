# Multi-View 诊断工具说明

## 工具列表

### 诊断脚本

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `check_diagnosis_setup.sh` | 验证环境和数据完整性 | 运行诊断前的检查 |
| `quick_diagnosis.sh` | 快速诊断（1分钟） | 获取初步判断 |
| `diagnose_multiview.sh` | 完整诊断套件（15分钟） | 深度分析问题 |

### Python分析工具

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `analyze_multiview_quality.py` | 分析multi-view embedding质量 | 数据集名称 | JSON报告 |
| `compare_datasets.py` | 对比Beauty和Toys数据集特征 | 无需参数 | JSON对比报告 |
| `analyze_training_logs.py` | 分析训练日志，检测过拟合 | 日志目录 | JSON分析结果 |
| `generate_tuning_suggestions.py` | 基于诊断生成调优建议 | 两个分析JSON | 配置变体 |

### 文档

| 文件 | 内容 |
|------|------|
| `MULTIVIEW_DIAGNOSTIC_GUIDE.md` | 完整的工具使用指南 |
| `RUN_DIAGNOSIS.md` | 详细的诊断流程和指标解读 |
| `README_DIAGNOSIS_TOOLS.md` | 本文件，工具索引 |

---

## 快速使用

### 1. 首次使用

```bash
cd /home/charlie/project/RecBole

# 检查环境
bash tools/check_diagnosis_setup.sh
```

### 2. 快速诊断

```bash
# 1分钟快速检查
bash tools/quick_diagnosis.sh
```

### 3. 完整诊断

```bash
# 15分钟深度分析
bash tools/diagnose_multiview.sh
```

### 4. 生成调优建议

```bash
python tools/generate_tuning_suggestions.py \
  --beauty_analysis multiview_diagnostics_*/multiview_quality_beauty.json \
  --toys_analysis multiview_diagnostics_*/multiview_quality_toys.json
```

---

## 工具详解

### check_diagnosis_setup.sh

**功能：** 验证所有必要文件和依赖是否就绪

**检查项：**
- ✓ Python依赖包（numpy, scipy, sklearn, pandas, yaml）
- ✓ Beauty数据集文件
- ✓ Toys数据集文件  
- ✓ Multi-view embeddings
- ✓ 配置文件

**使用：**
```bash
bash tools/check_diagnosis_setup.sh
```

**输出示例：**
```
✓ tools/analyze_multiview_quality.py
✓ numpy
✓ dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy
✓ dataset/Amazon_Beauty/qwen3_4views/
  ✓ View found: view_0.npy
  ...
```

---

### quick_diagnosis.sh

**功能：** 快速对比Beauty和Toys的基本统计信息

**输出关键指标：**
- Inter-item cosine similarity（物品间相似度）
- Feature variance（特征方差）
- 初步诊断建议

**使用：**
```bash
bash tools/quick_diagnosis.sh
```

**输出示例：**
```
Amazon_Beauty:
  Inter-item cosine: 0.3872 ± 0.1654

Amazon_Toys_and_Games:
  Inter-item cosine: 0.5341 ± 0.1423

⚠ WARNING: Toys items are more similar
  → Action: Increase temperature, alignment_weight
```

---

### diagnose_multiview.sh

**功能：** 运行完整诊断套件

**执行流程：**
1. 分析Beauty的multi-view质量
2. 分析Toys的multi-view质量
3. 对比数据集特征
4. 分析训练日志（如果存在）
5. 生成综合报告

**使用：**
```bash
bash tools/diagnose_multiview.sh
```

**输出目录：** `multiview_diagnostics_YYYYMMDD_HHMMSS/`

**包含文件：**
- `DIAGNOSIS_SUMMARY.md` - 诊断摘要
- `multiview_quality_beauty.json` - Beauty详细指标
- `multiview_quality_toys.json` - Toys详细指标
- `dataset_comparison_beauty_vs_toys.json` - 数据集对比
- `training_log_comparison.json` - 训练日志分析（如有）

---

### analyze_multiview_quality.py

**功能：** 深度分析single-view和multi-view embeddings

**分析内容：**
1. **基本统计：** 均值、方差、范数、稀疏度
2. **特征空间：** 有效秩、奇异值分解、物品间相似度
3. **View多样性：** 成对相似度、diversity score
4. **View vs LLM：** 各view与单LLM的相似度

**使用：**
```bash
python tools/analyze_multiview_quality.py \
  --dataset Amazon_Beauty \
  --output beauty_analysis.json

python tools/analyze_multiview_quality.py \
  --dataset Amazon_Toys_and_Games \
  --output toys_analysis.json
```

**关键输出指标：**
- `diversity.diversity_score` - View多样性评分（越高越好）
- `feature_space.llm.effective_rank` - 有效秩（越高越好）
- `feature_space.llm.mean_pairwise_cosine` - 物品间相似度（越低越好）
- `view_vs_llm` - 各view与单LLM的相似度

---

### compare_datasets.py

**功能：** 对比Beauty和Toys数据集的基本特征

**对比内容：**
1. **交互统计：** 用户数、物品数、密度
2. **文本统计：** 标题长度、词数
3. **Embedding分布：** 物品间相似度分布

**使用：**
```bash
python tools/compare_datasets.py
```

**输出：** `dataset_comparison_beauty_vs_toys.json`

---

### analyze_training_logs.py

**功能：** 分析训练日志，检测过拟合和收敛问题

**检测内容：**
- 最佳epoch位置
- 过拟合信号（best后性能下降）
- 改进趋势
- 稳定性

**使用：**
```bash
# 单个日志
python tools/analyze_training_logs.py \
  --log_dir saved/phase_runs_multiview_4views_toys

# 对比Beauty vs Toys
python tools/analyze_training_logs.py \
  --compare \
  --beauty_dir saved/phase_runs_multiview_4views \
  --toys_dir saved/phase_runs_multiview_4views_toys
```

**输出：** `training_log_analysis.json`

**关键指标：**
- `best_epoch` - 最佳验证性能的epoch
- `overfitting_signal` - 过拟合程度
- `improvement_trend` - 性能改进趋势

---

### generate_tuning_suggestions.py

**功能：** 基于诊断结果自动生成调优建议

**生成内容：**
1. 问题诊断（issue识别）
2. 严重程度（HIGH/MEDIUM）
3. 具体建议（actions）
4. 配置修改（config_changes）
5. 配置变体文件（*.yaml）

**使用：**
```bash
python tools/generate_tuning_suggestions.py \
  --beauty_analysis multiview_diagnostics_*/multiview_quality_beauty.json \
  --toys_analysis multiview_diagnostics_*/multiview_quality_toys.json \
  --base_config sasrec_align_multi_view_toys.yaml \
  --output_dir config_variants
```

**输出：**
- `tuning_suggestions.json` - 建议列表
- `config_variants/variant_1_*.yaml` - 配置变体1
- `config_variants/variant_2_*.yaml` - 配置变体2
- `config_variants/variant_combined_all_fixes.yaml` - 组合修复

---

## 典型工作流

### 工作流1：首次诊断

```bash
# 1. 检查环境
bash tools/check_diagnosis_setup.sh

# 2. 快速诊断
bash tools/quick_diagnosis.sh

# 3. 完整诊断
bash tools/diagnose_multiview.sh

# 4. 查看结果
cd multiview_diagnostics_*/
cat DIAGNOSIS_SUMMARY.md
```

### 工作流2：生成修复方案

```bash
# 5. 生成调优建议
python tools/generate_tuning_suggestions.py \
  --beauty_analysis multiview_diagnostics_*/multiview_quality_beauty.json \
  --toys_analysis multiview_diagnostics_*/multiview_quality_toys.json

# 6. 查看建议
cat tuning_suggestions.json

# 7. 测试配置变体
bash two_phase_run_multiview_split_toys.sh \
  --config_files config_variants/variant_1_*.yaml
```

### 工作流3：迭代优化

```bash
# 8. 修改配置后重新诊断
bash tools/diagnose_multiview.sh

# 9. 对比新旧结果
jq '.diversity.diversity_score' multiview_diagnostics_OLD/multiview_quality_toys.json
jq '.diversity.diversity_score' multiview_diagnostics_NEW/multiview_quality_toys.json

# 10. 继续调优直到满意
```

---

## 预期输出示例

### diversity_score 对比
```json
// Beauty (good multi-view)
{
  "diversity": {
    "diversity_score": 0.245,
    "mean_pairwise_cosine": 0.755
  }
}

// Toys (poor multi-view - 需要改进)
{
  "diversity": {
    "diversity_score": 0.098,  ← 太低！
    "mean_pairwise_cosine": 0.902  ← 太高！
  }
}
```

### 调优建议示例
```json
{
  "issue": "Low view diversity on Toys",
  "severity": "HIGH",
  "recommendation": "Views are too similar (diversity=0.098 vs Beauty=0.245)",
  "actions": [
    "Try different prompting strategies",
    "Increase SENet ratio from 4 to 8"
  ],
  "config_changes": {
    "text_view_senet_ratio": 8
  }
}
```

---

## 常见问题

**Q: 运行诊断需要多久？**
- 快速诊断：1分钟
- 完整诊断：10-15分钟
- 生成建议：1分钟

**Q: 需要什么依赖？**
- Python 3.7+
- numpy, scipy, scikit-learn, pandas, pyyaml

**Q: 如果缺少训练日志怎么办？**
- 可以跳过日志分析，其他工具仍然有效

**Q: 诊断结果如何解读？**
- 查看 `DIAGNOSIS_SUMMARY.md` 中的详细说明
- 参考 `../RUN_DIAGNOSIS.md` 中的指标解读

**Q: 如何应用修复建议？**
- 方法1：手动修改配置文件
- 方法2：使用自动生成的config variants
- 方法3：调整训练脚本的grid search范围

---

## 文件路径说明

**输入数据：**
```
dataset/Amazon_Beauty/
├── item_text_emb.qwen3.base.npy  (单LLM embedding)
└── qwen3_4views/
    ├── view_0.npy  (Multi-view view 0)
    ├── view_1.npy
    ├── view_2.npy
    ├── view_3.npy
    └── views.json  (view名称映射)

dataset/Amazon_Toys_and_Games/
└── (同上)
```

**输出结果：**
```
multiview_diagnostics_YYYYMMDD_HHMMSS/
├── DIAGNOSIS_SUMMARY.md
├── multiview_quality_beauty.json
├── multiview_quality_toys.json
├── dataset_comparison_beauty_vs_toys.json
└── training_log_comparison.json (如有日志)

config_variants/
├── variant_1_*.yaml
├── variant_2_*.yaml
└── variant_combined_all_fixes.yaml

tuning_suggestions.json
```

---

## 获取帮助

每个工具都支持 `--help` 参数：

```bash
python tools/analyze_multiview_quality.py --help
python tools/compare_datasets.py --help
python tools/analyze_training_logs.py --help
python tools/generate_tuning_suggestions.py --help
```

---

## 更新日志

- 2024-12-09: 初始版本，创建完整诊断工具套件
- 支持Beauty vs Toys对比分析
- 自动生成调优建议

---

## 相关文档

- 项目根目录：
  - `QUICK_START_DIAGNOSIS.md` - 快速开始指南
  - `MULTIVIEW_DIAGNOSIS_TOOLS.md` - 工具概览
  
- tools目录：
  - `MULTIVIEW_DIAGNOSTIC_GUIDE.md` - 完整使用指南
  - `RUN_DIAGNOSIS.md` - 详细操作说明
  - `README_DIAGNOSIS_TOOLS.md` - 本文件

选择合适的文档开始你的诊断之旅！
