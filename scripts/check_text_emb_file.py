#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查文本嵌入文件是否正确"""

import os
import sys
import numpy as np

def check_emb_file(filepath):
    """检查嵌入文件的基本信息"""
    print(f"\n检查文件: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ 文件不存在!")
        return False
    
    try:
        # 加载嵌入
        emb = np.load(filepath)
        
        print(f"✓ 文件存在")
        print(f"  Shape: {emb.shape}")
        print(f"  Dtype: {emb.dtype}")
        print(f"  Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        
        # 检查第一行（应该是padding，全零）
        first_row_zeros = np.all(emb[0] == 0)
        print(f"  第一行全零(padding): {'✓' if first_row_zeros else '❌'}")
        
        # 检查是否有全零行
        zero_rows = np.sum(np.all(emb == 0, axis=1))
        print(f"  全零行数: {zero_rows} / {emb.shape[0]} ({zero_rows/emb.shape[0]*100:.1f}%)")
        
        # 检查值范围
        non_zero_mask = emb != 0
        if np.any(non_zero_mask):
            print(f"  非零值范围: [{np.min(emb[non_zero_mask]):.4f}, {np.max(emb[non_zero_mask]):.4f}]")
            print(f"  非零值均值: {np.mean(emb[non_zero_mask]):.4f}")
            print(f"  非零值标准差: {np.std(emb[non_zero_mask]):.4f}")
        
        # 检查是否归一化
        norms = np.linalg.norm(emb[1:], axis=1)  # 跳过padding行
        norms_nonzero = norms[norms > 0]
        if len(norms_nonzero) > 0:
            is_normalized = np.allclose(norms_nonzero, 1.0, atol=1e-5)
            print(f"  是否L2归一化: {'✓' if is_normalized else '❌'}")
            print(f"  非零行L2范数: min={np.min(norms_nonzero):.4f}, max={np.max(norms_nonzero):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return False

def main():
    # 默认检查的文件
    default_files = [
        "dataset/Amazon_Beauty/item_text_emb.base.npy",
        "dataset/Amazon_Beauty/item_text_emb.qwen3.npy",
        "dataset/Amazon_Beauty/item_text_emb.npy"
    ]
    
    # 如果提供了命令行参数，使用参数
    if len(sys.argv) > 1:
        files_to_check = sys.argv[1:]
    else:
        files_to_check = default_files
    
    print("=== 文本嵌入文件检查 ===")
    
    # 检查每个文件
    found_any = False
    for filepath in files_to_check:
        if check_emb_file(filepath):
            found_any = True
    
    if not found_any:
        print("\n❌ 没有找到任何文本嵌入文件!")
        print("请确保已经生成了文本嵌入文件。")
        print("\n生成Base嵌入:")
        print("  python tools/build_item_text_emb_base.py --dataset Amazon_Beauty")
        print("\n生成Qwen3嵌入:")
        print("  python tools/build_item_text_emb_qwen3_hf.py --mapping dataset/Amazon_Beauty/item_index_mapping.csv")

if __name__ == "__main__":
    main()
