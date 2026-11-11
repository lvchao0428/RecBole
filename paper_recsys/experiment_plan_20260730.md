# 实验计划 — 2026-07-30（白天论文数据分析）

> 创建: 2026-07-29 21:10  
> 规范: [`WSDM_experiment_spec.md`](WSDM_experiment_spec.md)  
> Tracker: [`WSDM_convergence_tracker.md`](WSDM_convergence_tracker.md)

---

## 〇、过夜后台（勿跟进）

| 顺序 | 脚本 | 内容 | ETA |
|:----:|------|------|-----|
| 1 | `run_5090_unisrec_gts_resume.sh` | Toys UniSRec seed2024+2026 | ~00:30 |
| 2 | `run_5090_post_unisrec_queue.sh`（watcher 自动） | Warm-only 重评估 21 ckpt + 主表汇总 | ~04:00–06:00 |

查收：
```bash
ssh charlie@www.ultrapp.online 'cd /home/charlie/project/RecBole && tail -20 logs/unisrec_gts_queue.log; tail -20 logs/post_unisrec_queue.log; ls logs/unisrec_gts.DONE logs/post_unisrec.DONE 2>/dev/null; cat logs/unisrec_gts_results.csv; head -5 logs/warm_only_eval.csv'
```

---

## 一、白天待办（论文数据分析）

1. 查收 UniSRec Beauty+Toys 3-seed 完整表
2. 读 `logs/warm_only_eval.csv` / `paper_recsys/paper_tables_overnight.md`
3. 定稿主文 Table：GTS vs LOO（Beauty/Toys）+ UniSRec 一行
4. 定稿 warm-only vs all-target 对照（是否进主表）
5. 机制 Figure：CKA / eff-rank / LOVO / temporal shift（数据已齐）

---

## 二、已齐实验（勿再重跑）

Beauty/Toys GTS、Beauty/Toys Clean LOO、Early60、CKA/LOVO/shift、warm-target audit。
