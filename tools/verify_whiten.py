#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 center + whiten 预处理的效果

用法：
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
"""

import argparse
import numpy as np
import os


def verify_whitening(emb_path: str):
    """验证embedding是否正确白化"""
    
    if not os.path.exists(emb_path):
        print(f"❌ 文件不存在: {emb_path}")
        return False
    
    # 加载embedding
    emb = np.load(emb_path)
    print(f"✅ 加载embedding: {emb.shape}, dtype={emb.dtype}")
    
    # 检查统计量文件
    stats_path = emb_path.replace('.npy', '_whiten_stats.npz')
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        print(f"✅ 找到统计量文件: {stats_path}")
        print(f"   - mean shape: {stats['mean'].shape}")
        if 'whiten_matrix' in stats:
            print(f"   - whiten_matrix shape: {stats['whiten_matrix'].shape}")
        else:
            print(f"   ⚠️  只有 center，无 whiten_matrix (可能使用了 --no_whiten)")
    else:
        print(f"⚠️  未找到统计量文件: {stats_path}")
    
    # 验证PAD为零向量
    pad_norm = np.linalg.norm(emb[0])
    if pad_norm < 1e-6:
        print(f"✅ PAD embedding 为零向量 (norm={pad_norm:.2e})")
    else:
        print(f"❌ PAD embedding 不为零 (norm={pad_norm:.2e})")
    
    # 检查非PAD embedding是否L2归一化
    norms = np.linalg.norm(emb[1:], axis=1)
    mean_norm = norms.mean()
    std_norm = norms.std()
    if abs(mean_norm - 1.0) < 0.01 and std_norm < 0.01:
        print(f"✅ Embedding已L2归一化 (mean={mean_norm:.4f}, std={std_norm:.4f})")
    else:
        print(f"⚠️  Embedding未充分L2归一化 (mean={mean_norm:.4f}, std={std_norm:.4f})")
    
    # 检查是否中心化（均值接近0）
    train_sample = emb[1:min(1000, len(emb))]  # 取前1000个非PAD样本
    mean_vec = train_sample.mean(axis=0)
    mean_abs = np.abs(mean_vec).mean()
    if mean_abs < 0.1:
        print(f"✅ Embedding已中心化 (mean_abs={mean_abs:.4f})")
    else:
        print(f"⚠️  Embedding未充分中心化 (mean_abs={mean_abs:.4f})")
    
    # 检查协方差矩阵是否接近单位矩阵（验证白化效果）
    sample_size = min(500, len(train_sample))
    sample = train_sample[:sample_size].astype(np.float64)
    cov = (sample.T @ sample) / sample_size
    
    # 对角线应接近1
    diag_vals = np.diag(cov)
    diag_mean = diag_vals.mean()
    diag_std = diag_vals.std()
    
    # 非对角线应接近0
    off_diag = cov - np.diag(diag_vals)
    off_diag_max = np.abs(off_diag).max()
    off_diag_mean = np.abs(off_diag).mean()
    
    print(f"\n📊 协方差矩阵分析 (sample_size={sample_size}):")
    print(f"   - 对角线均值: {diag_mean:.4f} (期望≈1.0)")
    print(f"   - 对角线标准差: {diag_std:.4f} (期望≈0)")
    print(f"   - 非对角线最大值: {off_diag_max:.4f} (期望≈0)")
    print(f"   - 非对角线平均值: {off_diag_mean:.4f} (期望≈0)")
    
    # 判断白化效果
    if abs(diag_mean - 1.0) < 0.15 and off_diag_max < 0.3:
        print(f"✅ 白化效果良好")
        return True
    elif abs(diag_mean - 1.0) < 0.3:
        print(f"⚠️  白化效果一般（可能样本量不足或未启用whiten）")
        return True
    else:
        print(f"❌ 未检测到白化效果")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证 center + whiten 预处理效果")
    parser.add_argument(
        "emb_path",
        help="Embedding文件路径 (*.npy)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("验证 Center + Whiten 预处理")
    print("=" * 60)
    print()
    
    success = verify_whitening(args.emb_path)
    
    print()
    print("=" * 60)
    if success:
        print("✅ 验证通过")
    else:
        print("❌ 验证失败")
    print("=" * 60)


if __name__ == "__main__":
    main()

