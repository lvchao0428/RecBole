# WSDM 实验规范文档

> 创建: 2026-07-17  
> 最后更新: 2026-07-17  
> 适用范围: WSDM 2027 投稿 + RecSys 历史实验对照  
> 目的: 换新上下文后快速获取实验规范，避免重复沟通

---

## 一、评测协议

### 1.1 数据划分

| 项目 | 规范 | 说明 |
|------|------|------|
| 划分方式 | **Global time split** | 全局时间戳排序，设定 cutoff |
| 时间分位 | 80% / 10% / 10%（train / valid / test） | 论文中报告真实日期 |
| 用户准入 | train cutoff 前 ≥5 条历史 | 保证序列有意义 |
| Candidate 限制 | cutoff 前已出现的 warm items | 不推荐"未来商品" |
| Target 限制 | cutoff 前已出现的 item | warm-item evaluation |
| 排名方式 | **Full ranking** over all candidates | 非 sampled evaluation |

### 1.2 特征处理（Leakage-free）

| 特征 | 处理方式 | 泄露防护 |
|------|----------|----------|
| TF-IDF | vocabulary/IDF/SVD/center-whiten **仅在 train cutoff 前** item 上拟合 | 冻结后 apply to valid/test |
| LLM (Qwen2.5-7B) | 静态 encode 商品 title | title 视为商品首次出现时可用；不含交互信息 |
| Multi-view (Qwen 4-view) | 4 prompt × 同 encoder | 同上；view 为静态文本特征 |
| ID embedding | 随机初始化，训练学习 | 不涉及泄露 |

### 1.3 超参调优规范（0717 更新）

| 项目 | 规范 |
|------|------|
| 网格范围 | **很小且共享**：lr ∈ {1e-4, 5e-4, 1e-3}，dropout ∈ {0.1, 0.3, 0.5}，weight_decay ∈ {0, 1e-5, 1e-4} |
| 选参依据 | **仅用 valid set**；禁止根据 test 反复选配置 |
| 预算公平 | 所有模型同预算调参（如每模型最多 9 组） |
| 验证窗口 | 建议加一个**较早的 rolling validation window** 检查排序稳定性 |
| 报告 | 主表报 3 seeds 均值 ± std |

### 1.4 Item-frequency 分桶

| 分桶名称 | 定义（train cutoff 前） | 旧称（弃用） |
|----------|------------------------|-------------|
| **head** | 交互次数 ≥ 阈值（如 top 20%） | popular |
| **mid** | 中间层 | — |
| **low** (sparse-item) | 交互次数 ∈ [1, 阈值) | ~~cold-start~~ |

- 不再使用 "cold-start" 描述有训练交互的 item
- 可选增加 **item age** 分桶（首次出现时间距 cutoff）

### 1.5 评测指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 排序 | **MRR@10**（主指标）, NDCG@10 | |
| 覆盖 | Recall@10 (R@10) | |
| 分桶 | MRR_head, MRR_mid, MRR_low | 改名后 |
| | R_low@10 | 关注 sparse-item 覆盖 |
| 诊断 | valid→test gap | 泛化能力 |

---

## 二、模型结构规范

### 2.1 主方法（V4 最终版）

```
模型: SASRecAlignV3
Backbone: SASRec (2-layer Transformer, hidden_size=256, max_seq_len=50)
文本模式: text_mode=both (TF-IDF + LLM)
融合: concat+predictor (no-Cross)
  - text_concat_dim = TF(256) + LLM(256) = 512 → Linear(512, 256)
Align: per-source
  - align_proj_base: item_emb(256) → align_dim(128)
  - align_proj_llm: llm_proj(256) → align_dim(128)  
  - 2× InfoNCE losses (weight: align_weight=0.1)
训练: Two-phase optimization protocol
  - Phase-A: 冻结 SASRec backbone，训练 text projection + align heads (20 epochs)
  - Phase-B: 解冻全部参数联合微调 (30 epochs)
```

### 2.2 Ablation / Analysis 配置

| 配置 | 目的 | 与主方法差异 |
|------|------|------------|
| ID-only | 下界 baseline | 无文本 |
| TF-only | 强 baseline | text_mode=base, 1× InfoNCE |
| MV (4-view + TF) | Multi-view 分析 | text_mode=multiview, 5× InfoNCE, per-view align |
| +Cross | Cross 机制分析 | use_cross=True, DCN-V2 fusion |
| LLM +Cross | Cross 交互分析 | 主方法 + DCN-V2 |

### 2.3 已删除组件（不进入 V4）

| 组件 | 原因 |
|------|------|
| SENet (Squeeze-Excite) | 实证 marginal；增加复杂度 |
| cold_text_boost | 非稳定增益；与简化叙事冲突 |
| infer_boost | 同上 |
| 4×256 MV 扩维 | 老师明确反对；view 冗余不是容量问题 |

### 2.4 UniSRec Portability

```
模型: UniSRec (原版框架)
文本特征: 共享 Qwen2.5-7B 编码
配置:
  - Base: 原始 UniSRec
  - +AlignV3: 加入 per-source align
  - +MV-V3: 加入 multi-view align
论文定位: 跨 backbone portability check
命名: "UniSRec backbone with Qwen features"（非严格复现 original）
```

---

## 三、实验环境

### 3.1 硬件

| 机器 | GPU | 用途 |
|------|-----|------|
| 5090 | RTX 5090 | 主训练机 |
| log10 | RTX 4090 | 辅助 / UniSRec |
| logMac (mac128) | CPU | 数据分析 / 文档 |
| 本地 Mac | CPU | 开发 / 分析 / 论文写作 |

### 3.2 软件

| 项目 | 版本/路径 |
|------|----------|
| Python | 3.10 (anaconda3) |
| PyTorch | 2.x |
| RecBole | 自定义 fork (`exp1110` branch) |
| 代码路径 (5090) | `/home/charlie/project/RecBole` |
| 代码路径 (logMac) | `/Users/lvchao0428/project/ownRecBole/RecBole` |
| 代码路径 (本地) | `/Users/a58/project/RecBole` |
| Git remote | `git@github.com:lvchao0428/RecBole.git` |

### 3.3 数据集

| 数据集 | 路径 (5090) | #Items (train) | 特征文件 |
|--------|------------|:--------------:|---------|
| Amazon_Beauty | `dataset/Amazon_Beauty/` | ~12K | `item_text_emb.base.ts.npy`, `item_text_emb.qwen2.5_7b.base.ts.npy`, `qwen2.5_7b_4views_ts/` |
| Amazon_Toys | `dataset/Amazon_Toys_and_Games/` | ~11K | 同格式 |
| Amazon_Grocery | `dataset/Amazon_Grocery_and_Gourmet_Food/` | ~8K | 同格式 |

---

## 四、命名约定

### 4.1 日志命名

```
logs/{protocol}_{dataset}_{model}_{config}_{seed}.log

protocol: ts (time split), ps (per-source align)
dataset: beauty, toys, grocery
model: id_only, tfidf, llm, mv
config: nc (no-Cross), cross, noboost, boost
seed: seed42, seed2024, seed2025, seed2026
```

### 4.2 Checkpoint 命名

```
saved/{protocol}_{dataset}_{model}_{config}_{seed}/
```

### 4.3 YAML 配置命名

```
sasrec_align_{dataset}_{text_source}_{split_type}.yaml
  text_source: base (TF-IDF), qwen (LLM), multi_view
  split_type: stratified_ts (time split), stratified_v3_ts (V3 per-source)
```

---

## 五、引用规范

### 关键文件索引

| 文件 | 用途 |
|------|------|
| `paper_recsys/WSDM_convergence_tracker.md` | 版本迭代 + 截稿追踪 |
| `paper_recsys/WSDM_experiment_spec.md` | **本文档**：实验规范 |
| `paper_recsys/experiment_status_YYYYMMDD.md` | 每日进展 |
| `paper_recsys/0717zhidao.txt` | 老师 7/17 最新指导 |
| `paper_recsys/0712zhidao.txt` | 老师 7/12 指导 |
| `paper_recsys/0711zhidao.txt` | 老师 7/11 指导（RecSys 拒稿后方向调整）|
| `paper_recsys/response_to_0712zhidao.md` | 对 0712 指导的数据回应 |
| `scripts/diagnose_collapse.py` | 嵌入塌缩诊断工具 |

### 核心模型代码

| 文件 | 内容 |
|------|------|
| `recbole/model/sequential_recommender/sasrecalignv3.py` | 主方法 (TF/LLM single-view) |
| `recbole/model/sequential_recommender/sasrecalignmultiviewv3.py` | MV 模型 |
| `scripts/two_phase_train.py` | 两阶段训练入口 |

---

## 六、运维操作规范

### 6.1 5090 实验启动流程

```bash
# 1. SSH 进入 5090
ssh charlie@www.ultrapp.online

# 2. 进入项目目录并加载环境
cd /home/charlie/project/RecBole
source scripts/recbole_env.sh
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 确认 GPU 空闲
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# 4. 启动实验（nohup 后台）
nohup bash run_5090_xxx.sh > logs/xxx_nohup.log 2>&1 &
echo "PID=$!"

# 5. 验证启动
sleep 15 && tail -5 logs/xxx_nohup.log
nvidia-smi
```

### 6.2 实验脚本编写规范

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/scripts/recbole_env.sh"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 调用训练脚本
python scripts/two_phase_train.py \
    --model "SASRecAlignV3" \
    --dataset "Amazon_Beauty" \
    --config_files "$ROOT/sasrec_align_base_stratified_v3_ts.yaml" \
    --config_dict "{'learning_rate':0.0005}" \
    --gpu_id 0 \
    --min_train_interactions 5
```

### 6.3 故障排查清单

| 错误 | 原因 | 修复 |
|------|------|------|
| `ModuleNotFoundError: No module named 'recbole'` | 未 source 环境 | 添加 `source scripts/recbole_env.sh` + `export PYTHONPATH` |
| `python: command not found` | PATH 未设置 | 使用 `/home/charlie/anaconda3/bin/python` 或 source env |
| `CUDA out of memory` | batch 太大或多进程抢占 | 减 eval_batch_size、加 `torch.cuda.empty_cache()`、确认单进程 |
| `.log: 没有那个文件或目录` | logs 目录不存在 | `mkdir -p logs` |
| `set -e` 导致静默退出 | 脚本中某命令返回非 0 | 训练调用加 `|| true`，检查 exit code |

### 6.4 三端代码同步流程

```bash
# 方案 A: 通过 5090 push（推荐，5090 有 git push 权限）
# 1. 本地 commit
cd /Users/a58/project/RecBole
git add -A && git commit -m "描述"

# 2. rsync 到 5090
rsync -avz <files> charlie@www.ultrapp.online:/home/charlie/project/RecBole/

# 3. 5090 commit + push
ssh charlie@www.ultrapp.online "cd /home/charlie/project/RecBole && git add -A && git commit -m '描述' && git push origin exp1110"

# 4. logMac pull
ssh charlie@www.ultrapp.online "ssh mac128 'cd /Users/lvchao0428/project/ownRecBole/RecBole && git pull origin exp1110 --no-edit'"

# 5. 本地 fetch + reset
cd /Users/a58/project/RecBole && git fetch origin && git reset --hard origin/exp1110
```

### 6.5 进展检查命令（快速参考）

```bash
# GPU 状态
ssh charlie@www.ultrapp.online "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# 当前进程
ssh charlie@www.ultrapp.online "ps aux | grep -E 'two_phase|python.*scripts' | grep -v grep"

# 网格搜索进展
ssh charlie@www.ultrapp.online "tail -20 /home/charlie/project/RecBole/logs/shared_grid_nohup.log"

# 网格结果 CSV
ssh charlie@www.ultrapp.online "cat /home/charlie/project/RecBole/logs/grid_results_beauty_seed2025.csv"

# 具体实验日志
ssh charlie@www.ultrapp.online "tail -30 /home/charlie/project/RecBole/logs/grid_beauty_<TAG>_seed2025.log"
```

### 6.6 分析工具使用

| 工具 | 命令 | 需要 GPU | 预计耗时 |
|------|------|:--------:|:--------:|
| Distribution shift | `python scripts/analyze_distribution_shift.py --dataset Amazon_Beauty` | ❌ | 5s |
| View redundancy | `python scripts/analyze_view_redundancy.py --dataset Amazon_Beauty` | ❌ | 2s |
| SVD 谱图 | 先生成 json，再用 matplotlib 绘图 | ❌ | 10s |
| Rank-transition | `python scripts/analyze_rank_transition.py --ckpt_nocross ... --ckpt_cross ...` | ✅ | 5-30min |
| Leave-one-view-out | `python scripts/analyze_leave_one_view_out.py --ckpt ...` | ✅ | 10-60min |
| 嵌入塌缩诊断 | `python scripts/diagnose_collapse.py --mode A --dataset Amazon_Beauty` | ❌ (Mode A) | 30s |
| 共享小网格 | `bash run_5090_shared_grid.sh` | ✅ | 9-12h |

---

## 七、变更日志

| 日期 | 变更 |
|------|------|
| 2026-07-17 | 文档创建；整合 0711/0712/0717 老师建议为统一规范 |
| 2026-07-17 15:50 | 新增§六运维操作规范：启动流程、脚本编写、故障排查、三端同步、分析工具 |
