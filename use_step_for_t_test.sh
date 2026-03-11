# Step 1: 跑 4 次训练（2 模型 × 2 数据集）
bash run_peruser_significance.sh

# Step 2: 计算 paired t-test
# Beauty
python paper_sigir/compute_peruser_significance.py \
    --model_a saved/peruser/beauty_tfidf_llm_topk.npy \
    --model_b saved/peruser/beauty_mv_7b_topk.npy \
    --label_a 'TF-IDF+LLM' --label_b 'MV-Align(7B)'

# Toys
python paper_sigir/compute_peruser_significance.py \
    --model_a saved/peruser/toys_tfidf_llm_topk.npy \
    --model_b saved/peruser/toys_mv_7b_topk.npy \
    --label_a 'TF-IDF+LLM' --label_b 'MV-Align(7B)'
