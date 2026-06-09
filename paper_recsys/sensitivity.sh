cd paper_sigir
# 默认：Hit vs NDCG 跷跷板
python scripts/plot_sensitivity.py
# 完整：tradeoff + pareto
python scripts/plot_sensitivity.py --mode full
# 仅 Pareto 图
python scripts/plot_sensitivity.py --mode pareto --metric-pair mrr
# 从 figures 目录运行（会调用 scripts 版本）
cd figures && python plot_sensitivity.py --mode full
