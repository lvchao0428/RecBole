import numpy as np
import pandas as pd

df = pd.read_csv("dataset/book-crossing/item_index_mapping.csv").sort_values("internal_item_id")
drop_second = df.duplicated("internal_item_id", keep="first")  # 第二次为 True，共 1 行

def fix(path):
    a = np.load(path)
    assert a.shape[0] == len(df) == 271380
    out = a[~drop_second.values]
    assert out.shape[0] == 271379
    np.save(path, out.astype(a.dtype))

for path in [
    "dataset/book-crossing/item_text_emb.qwen3.base.npy",
    "dataset/book-crossing/item_text_emb.qwen3.multiview.npy",
] + [f"dataset/book-crossing/qwen3_4views/view_{i}.npy" for i in range(4)]:
    fix(path)
