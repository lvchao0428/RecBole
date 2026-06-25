# 实验换机追踪备忘（2026-06-25）

> 最新更新: 2026-06-25 17:10 CST  
> 5090: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **状态: UniSRec ✅ · Grocery multiseed 9/12 完成 · seed42 TF-IDF 运行中**

---

## 1. 当前状态

| 实验 | 状态 | 备注 |
|------|------|------|
| UniSRec Beauty (3 configs) | ✅ | 6/25 06:36 |
| Grocery multiseed 2025/2026 | ✅ | 各 4 配置 |
| Grocery multiseed 42 | 🔄 | ID ✅ · TF-IDF 运行中 |
| GRU4Rec | 📋 | 待 pipeline 结束 |

**5090 GPU**: 训练中（Grocery TF-IDF seed=42）

---

## 2. 主文关键对比（Beauty · seed=2024）

| Model | MRR@10 | HR@10 | 角色 |
|-------|--------|-------|------|
| SASRec MV-Align (7B) | **3.18%** | 5.79% | 主方法 |
| UniSRec Base | 2.47% | **6.60%** | 外部 text baseline |

详见 [`experiment_results_all_20260625.md`](experiment_results_all_20260625.md)

---

## 3. 一键检查

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole
tail -20 logs/main_pipeline_20260625.log
pgrep -af run_5090 || echo "no training"
nvidia-smi
```

---

## 4. 文档同步

```bash
rsync -avz paper_recsys/*.md charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
```
