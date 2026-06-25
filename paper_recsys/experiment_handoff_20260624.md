# 实验换机追踪备忘（2026-06-24）

> 最新更新: 2026-06-24 20:18 CST  
> 5090: `charlie@www.ultrapp.online:/home/charlie/project/RecBole`  
> **状态: BC + Grocery seed=2024 全部完成，GPU 空闲**

---

## 1. 当前状态

| 实验 | 状态 | 完成时间 |
|------|------|----------|
| BC phaseb50 seed=2024 | ✅ 四配置 | 6/24 09:26 |
| Grocery seed=2024 | ✅ 四配置 | 6/24 17:59 |
| Food seed=2024 | ✅ 四配置 | 6/23 09:38 |
| Grocery multiseed | ⏸ 待启动（Go 条件满足） | — |

**5090 GPU**: 空闲，无 `run_5090` 进程。

---

## 2. 关键结果（test @10 · seed=2024）

### Grocery → **Go**（第三域主表）

| Model | MRR@10 | HR@10 | vs ID (MRR) | vs ID (HR) |
|-------|--------|-------|-------------|------------|
| ID | 2.08% | **6.36%** | — | — |
| TF-IDF | 2.93% | 5.77% | +41% | −9.3% |
| LLM | 2.95% | 5.83% | +42% | −8.3% |
| **MV** | **3.01%** | 6.02% | **+45%** | −5.3% |

**HR 分层**: new 1.56→1.32 (ID最高) · few 3.33→**3.19** (MV最高) · freq 11.67→11.09 (MV接近ID)

**vs Beauty**: MRR 阶梯同形；Overall HR 文本低于 ID（MRR–HR trade-off），写作时与 Beauty 主表口径对齐。

### BC → appendix

| Model | MRR@10 | vs ID |
|-------|--------|-------|
| ID | 1.95% | — |
| TF-IDF | 2.20% | +13% |
| LLM | 2.20% | +13% |
| MV | 2.27% | +16% |

详见 [`experiment_summary_20260624.md`](experiment_summary_20260624.md) · [`experiment_status_20260624.md`](experiment_status_20260624.md)

---

## 3. 一键检查

```bash
ssh charlie@www.ultrapp.online
cd /home/charlie/project/RecBole
pgrep -af run_5090 || echo "no training"
nvidia-smi
cat paper_recsys/experiment_status_20260624.md | head -60
```

---

## 4. 文档同步

```bash
rsync -avz paper_recsys/*.md charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/
```

---

## 5. 下一步

启动 **Grocery multiseed**（2024/2025/2026/42 × ID/TF-IDF/LLM/MV）。
