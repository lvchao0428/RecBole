# Seed Stability数据核对与修正

## 问题诊断

用户指出：
1. Table 9 (Scale Law) 使用**Standard配置** ✅
2. 其他表格应该使用**Aggressive配置** ✅
3. Seed Stability (Table 12) caption说用aggressive，但基准数据可能不对？

## 数据核对

### 主表(Table 1)使用的Aggressive配置数据 (seed=2025)

**来源**: CSV lines 95-98, 63, 34

#### Toys (seed=2025 aggressive):
| Model | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | Source |
|-------|-------|---------|--------|-----------|--------|
| TF-IDF | 6.55 | 4.39 | 3.72 | 1.85 | line 95 |
| TF-IDF+LLM | 6.61 | 4.42 | 3.74 | 1.96 | line 96 |
| MV-7B | 6.92 | 4.51 | 3.76 | 1.99 | line 63 |

#### Beauty (seed=2025 aggressive):
| Model | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | Source |
|-------|-------|---------|--------|-----------|--------|
| TF-IDF | 5.63 | 3.73 | 3.15 | 1.68 | line 97 |
| TF-IDF+LLM | 5.74 | 3.77 | 3.17 | 1.65 | line 98 |
| MV-7B | 6.07 | 3.93 | 3.27 | 1.81 | line 34 |

---

### Seed=42实验数据 (aggressive配置)

**来源**: CSV lines 100-101, 107-108, 110, 112

#### Toys (seed=42):
| Model | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | Source |
|-------|-------|---------|--------|-----------|--------|
| TF-IDF | 6.51 | 4.35 | 3.69 | 1.91 | line 100 |
| TF-IDF+LLM | 6.60 | 4.40 | 3.72 | 1.91 | line 101 |
| MV-7B | 6.66 | 4.38 | 3.67 | 1.96 | line 112 |

#### Beauty (seed=42):
| Model | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | Source |
|-------|-------|---------|--------|-----------|--------|
| TF-IDF | 5.70 | 3.76 | 3.16 | 1.63 | line 107 |
| TF-IDF+LLM | 5.76 | 3.79 | 3.18 | 1.67 | line 108 |
| MV-7B | 6.03 | 3.89 | 3.24 | 1.75 | line 110 |

---

### Seed=2024实验数据 (aggressive配置)

**来源**: CSV lines 103, 105-106, 111, 113, 104

#### Toys (seed=2024):
| Model | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | Source |
|-------|-------|---------|--------|-----------|--------|
| TF-IDF | 6.55 | 4.39 | 3.72 | 1.82 | line 103 |
| TF-IDF+LLM | 6.59 | 4.41 | 3.73 | 1.88 | line 105 |
| MV-7B | 6.91 | 4.50 | 3.76 | 1.92 | line 106 |

#### Beauty (seed=2024):
| Model | HR@10 | NDCG@10 | MRR@10 | HR_new@10 | Source |
|-------|-------|---------|--------|-----------|--------|
| TF-IDF | 5.66 | 3.74 | 3.15 | 1.61 | line 111 |
| TF-IDF+LLM | 5.77 | 3.80 | 3.19 | 1.67 | line 113 |
| MV-7B | 6.00 | 3.87 | 3.21 | 1.81 | line 104 |

---

## 重新计算统计数据 (Aggressive配置)

### Beauty - 正确的3-seed统计

#### TF-IDF:
| Seed | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|------|-------|---------|--------|-----------|
| 42   | 5.70  | 3.76    | 3.16   | 1.63      |
| 2024 | 5.66  | 3.74    | 3.15   | 1.61      |
| 2025 | 5.63  | 3.73    | 3.15   | 1.68      |
| **Mean** | **5.66** | **3.74** | **3.15** | **1.64** |
| **Std**  | **0.04** | **0.02** | **0.01** | **0.04** |

#### TF-IDF+LLM:
| Seed | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|------|-------|---------|--------|-----------|
| 42   | 5.76  | 3.79    | 3.18   | 1.67      |
| 2024 | 5.77  | 3.80    | 3.19   | 1.67      |
| 2025 | 5.74  | 3.77    | 3.17   | 1.65      |
| **Mean** | **5.76** | **3.79** | **3.18** | **1.66** |
| **Std**  | **0.02** | **0.02** | **0.01** | **0.01** |

#### MV-7B:
| Seed | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|------|-------|---------|--------|-----------|
| 42   | 6.03  | 3.89    | 3.24   | 1.75      |
| 2024 | 6.00  | 3.87    | 3.21   | 1.81      |
| 2025 | 6.07  | 3.93    | 3.27   | 1.81      |
| **Mean** | **6.03** | **3.90** | **3.24** | **1.79** |
| **Std**  | **0.04** | **0.03** | **0.03** | **0.03** |

---

### Toys - 正确的3-seed统计

#### TF-IDF:
| Seed | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|------|-------|---------|--------|-----------|
| 42   | 6.51  | 4.35    | 3.69   | 1.91      |
| 2024 | 6.55  | 4.39    | 3.72   | 1.82      |
| 2025 | 6.55  | 4.39    | 3.72   | 1.85      |
| **Mean** | **6.54** | **4.38** | **3.71** | **1.86** |
| **Std**  | **0.02** | **0.02** | **0.02** | **0.05** |

#### TF-IDF+LLM:
| Seed | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|------|-------|---------|--------|-----------|
| 42   | 6.60  | 4.40    | 3.72   | 1.91      |
| 2024 | 6.59  | 4.41    | 3.73   | 1.88      |
| 2025 | 6.61  | 4.42    | 3.74   | 1.96      |
| **Mean** | **6.60** | **4.41** | **3.73** | **1.92** |
| **Std**  | **0.01** | **0.01** | **0.01** | **0.04** |

#### MV-7B:
| Seed | HR@10 | NDCG@10 | MRR@10 | HR_new@10 |
|------|-------|---------|--------|-----------|
| 42   | 6.66  | 4.38    | 3.67   | 1.96      |
| 2024 | 6.91  | 4.50    | 3.76   | 1.92      |
| 2025 | 6.92  | 4.51    | 3.76   | 1.99      |
| **Mean** | **6.83** | **4.46** | **3.73** | **1.96** |
| **Std**  | **0.15** | **0.07** | **0.05** | **0.04** |

---

## 对比：当前论文 vs 正确数据

### 当前论文中的数据 (可能有误)

我之前计算的seed stability统计与上面重新计算的一致。

### 验证结论

✅ **数据是正确的！**

所有seed实验（42, 2024, 2025）都使用了aggressive配置：
- `cold_start_align_boost: 2.5`
- `inference_cold_text_boost: 1.5`
- `λ=0.10, τ=0.05`

论文Table 12 (Seed Stability)的caption和数据都是正确的。

---

## 用户关注的"基准数据"问题

用户可能关注的是：
1. **主表baseline (TF-IDF, TF-IDF+LLM)** 使用的是aggressive配置 ✅
2. **Seed实验的baseline** 也使用aggressive配置 ✅
3. **Scale Law表格 (Table 9)** 使用standard配置 ✅

这是合理的设计：
- **Main results (Table 1)**: Aggressive - 展示最佳性能
- **Seed stability (Table 12)**: Aggressive - 验证主表的稳定性
- **Scale Law (Table 9)**: Standard - 公平对比LLM大小的效果

---

## SE-net和Cross+SE-net Ablation状态更新

用户已确认以下实验已用aggressive配置运行：

```bash
GPU_ID=4: beauty_train/two_phase_run_multiview_v2_stratified_no_senet.sh (RUNNING)
GPU_ID=5: beauty_train/two_phase_run_multiview_v2_stratified_no_cross_and_senet.sh (RUNNING)
GPU_ID=6: toy_train/two_phase_run_multiview_v2_toys_stratified_7b_nosenet.sh (RUNNING)
GPU_ID=7: toy_train/two_phase_run_multiview_v2_toys_stratified_7b_nosenet_nocross.sh (RUNNING)
```

**实验组合**:
1. No SE-net (Beauty)
2. No Cross + No SE-net (Beauty)
3. No SE-net (Toys)
4. No Cross + No SE-net (Toys)

这将补全ablation表格！

---

## 结论

✅ **Table 12 (Seed Stability) 数据正确**
- 所有seed实验使用aggressive配置
- 统计数据准确
- Caption说明准确

✅ **配置使用合理**
- Main + Seed stability: Aggressive
- Scale Law: Standard
- Ablation实验: 正在用Aggressive重做

✅ **论文数据一致性良好**

