# 明天换机器拉数据 — 从 5090

```bash
# 文档 + 结果表
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/RecBole/paper_recsys/ ./paper_recsys/
rsync -avz charlie@www.ultrapp.online:/home/charlie/project/RecBole/logs/{unisrec_gts_results.csv,warm_only_eval.csv,beauty_loo_results.csv,toys_loo_results.csv,wsdm_wrapup_toys_gts.csv,post_unisrec_queue.log,unisrec_gts_queue.log} ./logs/

# 完成标记
ssh charlie@www.ultrapp.online "ls /home/charlie/project/RecBole/logs/{unisrec_gts.DONE,post_unisrec.DONE} 2>/dev/null"
```

## 过夜队列
1. `run_5090_unisrec_gts_resume.sh` — Toys UniSRec seed2024+2026
2. watcher → `run_5090_post_unisrec_queue.sh` — warm-only 21 ckpt + `paper_tables_overnight.md`

## 白天分析入口
- `paper_recsys/experiment_plan_20260730.md`
- `paper_recsys/paper_tables_overnight.md`（跑完后生成）
- `logs/warm_only_eval.csv`
- `logs/unisrec_gts_results.csv`
