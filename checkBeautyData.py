from recbole.config.configurator import Config
from recbole.data.utils import create_dataset

# 无过滤配置
cfg1 = Config(model="BPR", dataset="Amazon_Beauty", config_file_list=[])
dataset1 = create_dataset(cfg1)
print(f"无config: {dataset1.num(dataset1.iid_field)} items")

# 使用你的config
cfg2 = Config(model="BPR", dataset="Amazon_Beauty", config_file_list=["sasrec_align_base.yaml"])
dataset2 = create_dataset(cfg2)
print(f"有config: {dataset2.num(dataset2.iid_field)} items")

# 检查原始.item文件
import pandas as pd
item_df = pd.read_csv(f"dataset/Amazon_Beauty/Amazon_Beauty.item", sep="\t")
print(f"原始.item文件: {len(item_df)} records")
