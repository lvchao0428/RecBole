#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 center + whiten 预处理的效果

用法：
  # 验证单个文件
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
  
  # 验证多个文件
  python tools/verify_whiten.py \
    dataset/Amazon_Beauty/item_text_emb.base.npy \
    dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
    dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
  
  # 验证分视图文件
  python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_0.npy
"""

import argparse
import numpy as np
import os
import sys

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def verify_whitening(emb_path: str):
    """验证embedding是否正确白化"""
    
    # 支持相对路径和绝对路径
    if not os.path.isabs(emb_path):
        # 如果是相对路径，尝试从项目根目录解析
        abs_path = os.path.join(ROOT_DIR, emb_path)
        if os.path.exists(abs_path):
            emb_path = abs_path
    
    if not os.path.exists(emb_path):
        print(f"❌ 文件不存在: {emb_path}")
        print(f"   提示: 请检查路径是否正确（支持相对路径和绝对路径）")
        return False
    
    # 加载embedding
    try:
        emb = np.load(emb_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    print(f"✅ 加载embedding: {os.path.basename(emb_path)}")
    print(f"   - 形状: {emb.shape}")
    print(f"   - 数据类型: {emb.dtype}")
    print(f"   - 文件大小: {os.path.getsize(emb_path) / 1024 / 1024:.2f} MB")
    
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
    # Note: Whitened embeddings are NOT L2-normalized by design
    norms = np.linalg.norm(emb[1:100].astype(np.float64), axis=1)  # Sample 100 items, use float64
    mean_norm = norms.mean()
    std_norm = norms.std()
    if abs(mean_norm - 1.0) < 0.01 and std_norm < 0.01:
        print(f"✅ Embedding已L2归一化 (mean={mean_norm:.4f}, std={std_norm:.4f})")
    else:
        print(f"ℹ️  Embedding未L2归一化 (mean={mean_norm:.4f}, std={std_norm:.4f})")
        print(f"   提示: Whitened embeddings通常不做L2归一化（这是正常的）")
    
    # 检查是否中心化（均值接近0）
    train_sample = emb[1:min(1000, len(emb))].astype(np.float64)  # 取前1000个非PAD样本
    mean_vec = train_sample.mean(axis=0)
    mean_abs = np.abs(mean_vec).mean()
    if mean_abs < 0.15:
        print(f"✅ Embedding已中心化 (mean_abs={mean_abs:.4f})")
    else:
        print(f"⚠️  Embedding未充分中心化 (mean_abs={mean_abs:.4f})")
    
    # 检查协方差矩阵是否接近单位矩阵（验证白化效果）
    sample_size = min(500, len(train_sample))
    sample = train_sample[:sample_size]
    
    # 重新中心化样本（以消除float16精度误差）
    sample_centered = sample - sample.mean(axis=0, keepdims=True)
    
    # 计算协方差矩阵
    cov = (sample_centered.T @ sample_centered) / sample_size
    
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
    # 更严格的判断：对角线应该非常接近1.0，非对角线应该接近0
    if abs(diag_mean - 1.0) < 0.1 and diag_std < 0.2 and off_diag_max < 0.3:
        print(f"✅ 白化效果良好")
        return True
    elif abs(diag_mean - 1.0) < 0.2 and off_diag_max < 0.4:
        print(f"⚠️  白化效果一般（可能样本量不足）")
        return True
    else:
        print(f"❌ 未检测到白化效果")
        print(f"   可能原因：")
        print(f"   1. 未启用whitening（使用了 --no_whiten）")
        print(f"   2. 训练集样本量太小")
        print(f"   3. 数据方差过小")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="验证 center + whiten 预处理效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证TF-IDF特征
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
  
  # 验证Qwen3单视图
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy
  
  # 验证Qwen3多视图
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
  
  # 验证单个视图
  python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_0.npy
  
  # 批量验证多个文件
  python tools/verify_whiten.py \\
    dataset/Amazon_Beauty/item_text_emb.base.npy \\
    dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
"""
    )
    parser.add_argument(
        "emb_paths",
        nargs='+',
        metavar='emb_path',
        help="Embedding文件路径 (*.npy)，支持多个文件"
    )
    args = parser.parse_args()
    
    overall_success = True
    
    for i, emb_path in enumerate(args.emb_paths):
        if len(args.emb_paths) > 1:
            print()
            print("=" * 70)
            print(f"[{i+1}/{len(args.emb_paths)}] 验证: {emb_path}")
            print("=" * 70)
        else:
            print("=" * 70)
            print("验证 Center + Whiten 预处理")
            print("=" * 70)
        print()
        
        success = verify_whitening(emb_path)
        overall_success = overall_success and success
        
        print()
        if success:
            print("✅ 当前文件验证通过")
        else:
            print("❌ 当前文件验证失败")
    
    # 总结
    if len(args.emb_paths) > 1:
        print()
        print("=" * 70)
        print(f"📊 验证总结: {len(args.emb_paths)} 个文件")
        print("=" * 70)
        if overall_success:
            print("✅ 全部验证通过")
        else:
            print("⚠️  部分文件验证失败，请检查上方详情")
        print("=" * 70)


if __name__ == "__main__":
    main()

