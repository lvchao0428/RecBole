# Experiment Changelog - 2026-01-10

## 8-GPU Grid Search 实验设计

### 背景
- TF-IDF+LLM 在 Balanced 配置下表现不如 TF-IDF（0110.csv）
- 原因分析：Balanced 配置 (tau=0.03, align=0.07, text_weight=0.8) 削弱了 LLM 特征
- 原版 stratified 配置 (tau=0.05, align=0.15, text_weight=1.0) TF-IDF+LLM > TF-IDF

### 实验目标
1. 找到让 **TF-IDF+LLM > TF-IDF** 的参数组合
2. 找到让 **Multi-View 7B > TF-IDF+LLM** 的配置（冷启动略强）
3. 为 14B/32B Scale Law 验证准备最佳基础配置

---

## 实验配置（8 GPU）

### TF-IDF+LLM 实验 (GPU 0-3)

| GPU | 实验 | tau | align | text_wt | cold_boost | 说明 |
|-----|------|-----|-------|---------|------------|------|
| 0 | **A: 原版** | 0.05 | 0.15 | 1.0 | 0 | 原版 stratified 基准 |
| 1 | **C: 中等** | 0.05 | 0.10 | 1.0 | 2.0 | 平衡配置 |
| 2 | **D: 高温** | 0.07 | 0.12 | 1.0 | 2.0 | 更平滑对比学习 |
| 3 | **H: 最优** | 0.05 | 0.12 | 1.0 | 2.5 | 预期最佳组合 |

### Multi-View 7B 实验 (GPU 4-7)

| GPU | 实验 | tau | align | text_wt | cold_boost | 说明 |
|-----|------|-----|-------|---------|------------|------|
| 4 | **M1: 基础** | 0.05 | 0.10 | 1.0 | 2.0 | Multi-View 基准 |
| 5 | **M2: 冷启动** | 0.05 | 0.12 | 1.0 | 3.0 | 增强冷启动 |
| 6 | **M3: 高对齐** | 0.05 | 0.15 | 0.9 | 2.0 | 强训练弱推理 |
| 7 | **M4: 激进** | 0.05 | 0.12 | 1.0 | 4.0 | 最大化冷启动 |

---

## 代码修改

### 1. `scripts/two_phase_train.py` - 新增 `--gpu_id` 参数

```python
# 新增参数
parser.add_argument("--gpu_id", type=str, default=None, 
                    help="GPU ID to use (overrides YAML config)")

# 应用到配置
if args.gpu_id is not None:
    dist_config_dict["gpu_id"] = args.gpu_id
```

### 2. 实验脚本

位置：`experiments/`

```
experiments/
├── grid_search_8gpu.sh      # 实验说明
├── exp_A_original.sh        # GPU 0: TF-IDF+LLM 原版
├── exp_C_moderate.sh        # GPU 1: TF-IDF+LLM 中等
├── exp_D_high_tau.sh        # GPU 2: TF-IDF+LLM 高温
├── exp_H_optimal.sh         # GPU 3: TF-IDF+LLM 最优
├── exp_M1_mv_base.sh        # GPU 4: Multi-View 基础
├── exp_M2_mv_cold.sh        # GPU 5: Multi-View 冷启动
├── exp_M3_mv_high_align.sh  # GPU 6: Multi-View 高对齐
└── exp_M4_mv_aggressive.sh  # GPU 7: Multi-View 激进
```

---

## 运行方式

```bash
# 手动启动（指定 GPU）
bash experiments/exp_A_original.sh 0      # GPU 0
bash experiments/exp_C_moderate.sh 1      # GPU 1
bash experiments/exp_D_high_tau.sh 2      # GPU 2
bash experiments/exp_H_optimal.sh 3       # GPU 3
bash experiments/exp_M1_mv_base.sh 4      # GPU 4
bash experiments/exp_M2_mv_cold.sh 5      # GPU 5
bash experiments/exp_M3_mv_high_align.sh 6  # GPU 6
bash experiments/exp_M4_mv_aggressive.sh 7  # GPU 7

# 使用默认 GPU（脚本中定义）
bash experiments/exp_A_original.sh        # 默认 GPU 0
```

---

## 待验证假设

1. **TF-IDF+LLM 需要更高的 alignment_weight (0.10-0.15)** 来增强 LLM 语义对齐
2. **temperature=0.05 比 0.03 更适合** LLM 特征
3. **text_weight=1.0** 能充分发挥 LLM 特征优势
4. **cold_start_align_boost** 能显著提升 new/few 指标

---

## 注意事项

⚠️ **cold_start_align_boost** 需要在 YAML 中配置，脚本使用的是 `sasrec_align_qwen3_stratified.yaml`（默认 cold_boost=0）

需要为每个实验创建专用 YAML 或使用 `sed` 修改：

```bash
# 示例：为 Exp C 创建 YAML
cp sasrec_align_qwen3_stratified.yaml experiments/yaml/exp_C_llm.yaml
sed -i 's/cold_start_align_boost: 0/cold_start_align_boost: 2.0/' experiments/yaml/exp_C_llm.yaml
```

---

## 预期结果

| 对比 | 预期 | 关键指标 |
|------|------|----------|
| TF-IDF+LLM vs TF-IDF | LLM > TF-IDF | mrr@10, ndcg@10, recall@10 |
| Multi-View vs TF-IDF+LLM | MV 冷启动更强 | MRR_new@10, NDCG_new@10 |
| 14B/32B vs 7B | Scale Law 成立 | 整体和低频指标 |
