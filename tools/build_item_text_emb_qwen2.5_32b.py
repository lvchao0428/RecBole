#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build item text embeddings using Qwen2.5-32B-Instruct model.

该脚本是 build_item_text_emb_qwen3_hf.py 的封装，专为 Qwen2.5-32B-Instruct 优化。
非量化版本，使用 HuggingFace Transformers 加载。
建议使用 device_map=auto 进行多GPU推理。

默认模型路径: /data/model/Qwen2.5-32B-Instruct

Example:
  python tools/build_item_text_emb_qwen2.5_32b.py \
    --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
    --output dataset/Amazon_Beauty/item_text_emb.qwen2.5_32b.npy \
    --prompt_preset multiview \
    --output_mode concat \
    --device_map auto
"""

import sys
import os

# 确保可以导入同目录下的模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 设置默认模型路径
DEFAULT_MODEL_PATH = "/data/model/Qwen2.5-32B-Instruct"

def main():
    """封装 build_item_text_emb_qwen3_hf.py 的 main 函数，设置 Qwen2.5-32B 默认值"""
    # 检查是否指定了模型路径，如果没有则使用默认值
    if "--model_name_or_path" not in sys.argv:
        sys.argv.extend(["--model_name_or_path", DEFAULT_MODEL_PATH])
    
    # 导入并运行原始脚本
    from tools.build_item_text_emb_qwen3_hf import main as qwen3_main
    qwen3_main()


if __name__ == "__main__":
    main()

