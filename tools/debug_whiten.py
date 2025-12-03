#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试白化实现 - 验证白化算法的正确性
"""

import numpy as np
import os
import sys

# 添加项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def test_whitening_implementation():
    """测试白化实现的正确性"""
    
    print("=" * 70)
    print("测试白化算法实现")
    print("=" * 70)
    print()
    
    # 1. 生成测试数据（有相关性的数据）
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # 生成有相关性的数据
    X_raw = np.random.randn(n_samples, n_features).astype(np.float64)
    # 添加相关性
    correlation_matrix = np.random.randn(n_features, n_features)
    correlation_matrix = np.dot(correlation_matrix, correlation_matrix.T)
    X_raw = np.dot(X_raw, correlation_matrix)
    
    print(f"✅ 生成测试数据: {X_raw.shape}")
    
    # 2. 验证原始数据有相关性
    cov_original = np.dot(X_raw.T, X_raw) / n_samples
    off_diag_orig = cov_original - np.diag(np.diag(cov_original))
    print(f"   原始数据非对角线最大值: {np.abs(off_diag_orig).max():.4f}")
    print(f"   原始数据非对角线均值: {np.abs(off_diag_orig).mean():.4f}")
    print()
    
    # 3. 应用当前的白化实现
    print("📊 测试当前白化实现...")
    
    # Center
    mean = X_raw.mean(axis=0, keepdims=True)
    X_centered = X_raw - mean
    
    # 计算协方差矩阵
    cov = np.dot(X_centered.T, X_centered) / len(X_centered)
    
    # SVD分解
    U, S, _ = np.linalg.svd(cov)
    
    # 白化矩阵（当前实现）
    whiten_matrix = np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-5)))
    
    # 应用白化
    X_whitened = np.dot(X_centered, whiten_matrix)
    
    # 验证白化后的协方差矩阵
    cov_whitened = np.dot(X_whitened.T, X_whitened) / len(X_whitened)
    
    diag_vals = np.diag(cov_whitened)
    diag_mean = diag_vals.mean()
    diag_std = diag_vals.std()
    
    off_diag = cov_whitened - np.diag(diag_vals)
    off_diag_max = np.abs(off_diag).max()
    off_diag_mean = np.abs(off_diag).mean()
    
    print(f"   白化后对角线均值: {diag_mean:.6f} (期望≈1.0)")
    print(f"   白化后对角线标准差: {diag_std:.6f} (期望≈0)")
    print(f"   白化后非对角线最大值: {off_diag_max:.6f} (期望≈0)")
    print(f"   白化后非对角线均值: {off_diag_mean:.6f} (期望≈0)")
    print()
    
    # 4. 判断白化效果
    is_good = (abs(diag_mean - 1.0) < 1e-6 and 
               diag_std < 1e-6 and 
               off_diag_max < 1e-6)
    
    if is_good:
        print("✅ 白化实现正确！协方差矩阵≈单位矩阵")
        return True
    else:
        print("❌ 白化实现有问题！协方差矩阵不是单位矩阵")
        print()
        print("📊 可能原因分析：")
        
        # 检查float64 vs float32 vs float16精度问题
        print("\n🔬 测试不同精度的影响...")
        
        for dtype_name, dtype in [('float32', np.float32), ('float16', np.float16)]:
            X_test = X_raw.astype(dtype)
            mean_test = X_test.mean(axis=0, keepdims=True).astype(np.float64)
            X_centered_test = (X_test - mean_test).astype(np.float64)
            
            cov_test = (X_centered_test.T @ X_centered_test) / len(X_centered_test)
            U_test, S_test, _ = np.linalg.svd(cov_test)
            whiten_matrix_test = U_test @ np.diag(1.0 / np.sqrt(S_test + 1e-5))
            
            X_whitened_test = X_centered_test @ whiten_matrix_test
            cov_whitened_test = (X_whitened_test.T @ X_whitened_test) / len(X_whitened_test)
            
            off_diag_test = cov_whitened_test - np.diag(np.diag(cov_whitened_test))
            print(f"   {dtype_name}: 非对角线最大值 = {np.abs(off_diag_test).max():.6f}")
        
        return False


def verify_embedding_file(emb_path: str):
    """验证实际的embedding文件"""
    
    print()
    print("=" * 70)
    print("验证实际Embedding文件")
    print("=" * 70)
    print()
    
    # 支持相对路径
    if not os.path.isabs(emb_path):
        abs_path = os.path.join(ROOT_DIR, emb_path)
        if os.path.exists(abs_path):
            emb_path = abs_path
    
    if not os.path.exists(emb_path):
        print(f"❌ 文件不存在: {emb_path}")
        return False
    
    # 加载embedding（转为float64避免精度问题）
    emb = np.load(emb_path).astype(np.float64)
    print(f"✅ 加载embedding: {os.path.basename(emb_path)}")
    print(f"   - 形状: {emb.shape}")
    print()
    
    # 加载统计量文件
    stats_path = emb_path.replace('.npy', '_whiten_stats.npz')
    if not os.path.exists(stats_path):
        print(f"❌ 未找到统计量文件: {stats_path}")
        return False
    
    stats = np.load(stats_path)
    mean = stats['mean'].astype(np.float64)
    
    if 'whiten_matrix' not in stats:
        print(f"⚠️  未找到whiten_matrix，可能使用了 --no_whiten")
        return False
    
    whiten_matrix = stats['whiten_matrix'].astype(np.float64)
    print(f"✅ 加载统计量: mean {mean.shape}, whiten_matrix {whiten_matrix.shape}")
    print()
    
    # 验证：重新应用白化变换，看结果是否一致
    print("📊 验证白化变换的一致性...")
    
    # 取样本（避免全量计算）
    sample_size = min(1000, len(emb) - 1)
    sample_ids = np.arange(1, sample_size + 1)  # 跳过PAD
    X_sample = emb[sample_ids]
    
    # 方法1：直接检查当前embedding的协方差
    cov_current = (X_sample.T @ X_sample) / len(X_sample)
    
    diag_vals = np.diag(cov_current)
    off_diag = cov_current - np.diag(diag_vals)
    
    print(f"   当前embedding的协方差矩阵:")
    print(f"   - 对角线均值: {diag_vals.mean():.6f}")
    print(f"   - 对角线标准差: {diag_vals.std():.6f}")
    print(f"   - 非对角线最大值: {np.abs(off_diag).max():.6f}")
    print(f"   - 非对角线均值: {np.abs(off_diag).mean():.6f}")
    print()
    
    # 方法2：验证whiten_matrix是否正确
    # 如果whiten_matrix正确，那么 W^T @ Cov_original @ W = I
    # 其中 Cov_original 是centered embeddings的协方差矩阵
    
    # 反推原始的centered embeddings（在应用whiten之前）
    # 如果 emb_whitened = emb_centered @ W，那么 emb_centered = emb_whitened @ W^(-1)
    # 但更简单的是：直接用mean去center当前的emb
    
    print("📊 检查whiten_matrix的正确性...")
    # 先去除center（反向操作）
    X_uncentered = X_sample + mean
    # 重新center
    X_recentered = X_uncentered - mean
    
    # 计算重新centered后的协方差
    cov_recentered = (X_recentered.T @ X_recentered) / len(X_recentered)
    
    # 应用whiten_matrix
    # 理论上：W^T @ Cov @ W = I
    result = whiten_matrix.T @ cov_recentered @ whiten_matrix
    
    diag_result = np.diag(result)
    off_diag_result = result - np.diag(diag_result)
    
    print(f"   W^T @ Cov @ W 的结果:")
    print(f"   - 对角线均值: {diag_result.mean():.6f} (期望≈1.0)")
    print(f"   - 对角线标准差: {diag_result.std():.6f} (期望≈0)")
    print(f"   - 非对角线最大值: {np.abs(off_diag_result).max():.6f} (期望≈0)")
    print(f"   - 非对角线均值: {np.abs(off_diag_result).mean():.6f} (期望≈0)")
    print()
    
    # 判断
    is_correct = (abs(diag_result.mean() - 1.0) < 1e-3 and 
                  np.abs(off_diag_result).max() < 1e-2)
    
    if is_correct:
        print("✅ whiten_matrix 是正确的（在float64精度下）")
        print()
        print("⚠️  但embedding文件本身的协方差不是单位矩阵")
        print("   可能原因：验证时使用的样本与训练时不同")
        print("   建议：使用训练集样本重新验证")
    else:
        print("❌ whiten_matrix 本身有问题")
    
    return is_correct


def main():
    # 1. 先测试白化算法实现
    test_passed = test_whitening_implementation()
    
    # 2. 如果算法实现正确，验证实际文件
    if len(sys.argv) > 1:
        for emb_path in sys.argv[1:]:
            verify_embedding_file(emb_path)
    
    print()
    print("=" * 70)
    print("调试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

