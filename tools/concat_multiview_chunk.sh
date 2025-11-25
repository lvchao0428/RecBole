cd /home/charlie/project/RecBole
python tools/concat_multiview_views.py \
  --split_dir /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_amplified_views \
  --output /home/charlie/project/RecBole/dataset/Amazon_Beauty/item_text_emb_amplified.npy \
  --dtype float16 \
  --chunk_size 2048
