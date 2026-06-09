# Experiment Changelog - 2026-01-10/11

## ⚠️ 重要发现：Burn-in 过拟合问题 (2026-01-11)

### 问题现象

从训练日志发现严重问题：

| 数据集 | 阶段 | 最终 Loss | Valid Score | 问题 |
|--------|------|-----------|-------------|------|
| **Beauty** | Burn-in (10ep) | 9,401 | **0.0224** | ✅ 快速收敛 |
| | Phase-A (20ep) | 9,791 | 0.0218 | ❌ **未超过 burn-in** |
| | Phase-B (24ep) | 7,818 | 0.0322 | ✅ 开始恢复 |
| **Toys** | Burn-in (10ep) | 10,523 | **0.0239** | ✅ 快速收敛 |
| | Phase-A (20ep) | 10,933 | 0.0233 | ❌ **Loss 反增，score 不涨** |

### 根因分析

```
10 epoch Burn-in:
  - ID backbone 过度拟合到"纯 ID 空间"
  - Item/Position embedding 被深度优化

Phase-A (freeze backbone):
  - Backbone 冻结，无法适应新引入的 text 特征
  - Text alignment loss 成为"额外负担"
  - Valid score 反映冻住的 backbone，所以不涨
```

### 解决方案

**折中方案**: `burn-in = 2 epochs` (原为 10)

- 2 epoch 足够初始化 ID embedding
- 不会过度拟合到纯 ID 空间
- Phase-A 有空间继续学习

---

## 8-GPU 实验设计 (更新版)

### 实验目标

1. **验证 burn-in 假设**: 0 vs 2 epochs
2. **TF-IDF+LLM > TF-IDF**: 恢复 LLM 优势
3. **Multi-View > TF-IDF+LLM**: 冷启动增强

### 实验配置

| GPU | 实验 | 模型 | Burn-in | Phase-A | Phase-B | 说明 |
|-----|------|------|---------|---------|---------|------|
| 0 | **burn0** | MV 7B | **0** | 20ep | 40ep | 无 burn-in 对照 |
| 1 | **burn2** | MV 7B | **2** | 20ep | 40ep | 折中方案测试 |
| 2 | **A** | LLM | 2 | 20ep | 40ep | TF-IDF+LLM 原版 |
| 3 | **H** | LLM | 2 | 20ep | 40ep | TF-IDF+LLM 最优 |
| 4 | **M1** | MV 7B | 2 | 20ep | 40ep | Multi-View 基础 |
| 5 | **M2** | MV 7B | 2 | 20ep | 40ep | Multi-View 冷启动 |
| 6 | **M3** | MV 7B | 2 | 20ep | 40ep | Multi-View 高对齐 |
| 7 | **M4** | MV 7B | 2 | 20ep | 40ep | Multi-View 激进 |
| **5090** | **M5** | MV 7B | 2 | **5ep** | **55ep** | 短 Phase-A 测试 (业界最小值) |

---

## 实验脚本

位置：`experiments/`

```
experiments/
├── grid_search_8gpu.sh          # 实验说明 (更新)
├── exp_burn0_no_burnin.sh       # GPU 0: 无 burn-in (新增)
├── exp_burn2_compromise.sh      # GPU 1: burn-in=2 (新增)
├── exp_A_original.sh            # GPU 2: LLM 原版 (burn-in: 10→2)
├── exp_H_optimal.sh             # GPU 3: LLM 最优 (burn-in: 10→2)
├── exp_M1_mv_base.sh            # GPU 4: MV 基础 (burn-in: 10→2)
├── exp_M2_mv_cold.sh            # GPU 5: MV 冷启动 (burn-in: 10→2)
├── exp_M3_mv_high_align.sh      # GPU 6: MV 高对齐 (burn-in: 10→2)
├── exp_M4_mv_aggressive.sh      # GPU 7: MV 激进 (burn-in: 10→2)
└── exp_M5_skip_phaseA.sh        # 5090: 最小 Phase-A (新增)

已删除:
├── exp_C_moderate.sh            # 移除
└── exp_D_high_tau.sh            # 移除
```

---

## 运行方式

```bash
# 手动启动每个实验，指定 GPU
bash experiments/exp_burn0_no_burnin.sh 0   # GPU 0: 无 burn-in
bash experiments/exp_burn2_compromise.sh 1  # GPU 1: burn-in=2
bash experiments/exp_A_original.sh 2        # GPU 2: LLM 原版
bash experiments/exp_H_optimal.sh 3         # GPU 3: LLM 最优
bash experiments/exp_M1_mv_base.sh 4        # GPU 4: MV 基础
bash experiments/exp_M2_mv_cold.sh 5        # GPU 5: MV 冷启动
bash experiments/exp_M3_mv_high_align.sh 6  # GPU 6: MV 高对齐
bash experiments/exp_M4_mv_aggressive.sh 7  # GPU 7: MV 激进

# 5090 专用实验
bash experiments/exp_M5_skip_phaseA.sh 0    # 5090: 最小 Phase-A
```

---

## 关键观察指标

训练过程中需要重点关注：

1. **Phase-A valid_score 趋势**
   - burn-in=0: Phase-A 是否能从零开始涨点
   - burn-in=2: Phase-A 是否能超过 burn-in 最终分数

2. **Loss 跳变**
   - Burn-in → Phase-A: loss 应小幅增加 (<10%)
   - Phase-A → Phase-B: loss 应正常重新下降

3. **最终指标**
   - MRR@10, MRR_new@10, MRR_few@10
   - 对比 burn-in=0 vs burn-in=2 vs 原来的 burn-in=10

---

## 预期结果

| 对比 | 预期 | 关键观察 |
|------|------|----------|
| burn0 vs burn2 | burn2 略优 | Phase-A 收敛速度 |
| burn2 vs burn10 | burn2 显著优 | Phase-A 是否能涨点 |
| TF-IDF+LLM vs TF-IDF | LLM > TF-IDF | 整体指标 |
| Multi-View vs TF-IDF+LLM | MV 冷启动更强 | MRR_new@10 |

---

## 代码修改记录

### 1. `scripts/two_phase_train.py` - `--gpu_id` 参数

```python
parser.add_argument("--gpu_id", type=str, default=None,
                    help="GPU ID to use (overrides YAML config)")
```

### 2. 所有实验脚本

```diff
- --backbone_burnin_epochs 10 \
+ --backbone_burnin_epochs 2 \
```

---

## 历史记录

### 2026-01-10
- 初始 8-GPU 实验设计
- 发现 TF-IDF+LLM balanced 配置表现差于 TF-IDF

### 2026-01-11
- 发现 burn-in 过拟合问题
- 修改方案：burn-in 从 10 改为 2
- 新增 exp_burn0, exp_burn2 对照实验
- 删除 exp_C, exp_D (相似配置精简)
