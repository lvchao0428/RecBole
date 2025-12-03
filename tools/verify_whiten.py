#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 center + whiten 预处理的效果（支持训练集专用验证）

用法：
  # 基础验证（随机采样）
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
  
  # 训练集专用验证（推荐）
  python tools/verify_whiten.py \
    dataset/Amazon_Beauty/item_text_emb.base.npy \
    --dataset Amazon_Beauty
  
  # 批量验证多个文件
  python tools/verify_whiten.py \
    dataset/Amazon_Beauty/item_text_emb.base.npy \
    dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
    --dataset Amazon_Beauty
"""

import argparse
import numpy as np
import os
import sys

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _load_train_item_ids(dataset_name, config_files, max_items):
    """加载训练集item IDs（参考 build_item_text_emb_base.py 的实现）"""
    try:
        from recbole.config.configurator import Config
        from recbole.data.utils import create_dataset, data_preparation
        
        print(f"[加载训练集] 正在加载数据集: {dataset_name}...")
        cfg = Config(model="BPR", dataset=dataset_name, config_file_list=config_files)
        dataset = create_dataset(cfg)
        
        print(f"[加载训练集] 准备数据划分...")
        train_data, valid_data, test_data = data_preparation(cfg, dataset)
        iid_field = cfg["ITEM_ID_FIELD"]
        
        try:
            train_iids = train_data.dataset.inter_feat[iid_field].numpy()
        except Exception:
            # Fallback: no split info available
            print(f"[WARN] 无法获取训练集划分，使用所有items")
            return None
        
        # 获取唯一的训练集item IDs
        train_iids_unique = np.unique(train_iids)
        train_iids_unique = train_iids_unique[(train_iids_unique > 0) & (train_iids_unique < max_items)]
        
        print(f"[加载训练集] 训练集items: {len(train_iids_unique)} / {max_items} items")
        return train_iids_unique
        
    except Exception as e:
        print(f"[WARN] 加载训练集失败: {e}")
        print(f"[WARN] 将使用随机采样验证（可能不准确）")
        return None


def verify_whitening(emb_path: str, dataset_name=None, config_files=None):
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
    has_whiten_matrix = False
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        print(f"✅ 找到统计量文件: {stats_path}")
        print(f"   - mean shape: {stats['mean'].shape}")
        if 'whiten_matrix' in stats:
            print(f"   - whiten_matrix shape: {stats['whiten_matrix'].shape}")
            has_whiten_matrix = True
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
    
    # 决定使用哪些items进行验证
    train_ids = None
    if dataset_name:
        train_ids = _load_train_item_ids(
            dataset_name, 
            config_files if config_files else [],
            len(emb)
        )
    
    if train_ids is not None and len(train_ids) > 0:
        # 使用训练集items
        print(f"\n📊 使用训练集items验证 ({len(train_ids)} items)")
        sample_indices = train_ids
        sample_size = min(2000, len(train_ids))  # 最多采样2000个
        if len(train_ids) > sample_size:
            # 随机采样
            np.random.seed(42)
            sample_indices = np.random.choice(train_ids, size=sample_size, replace=False)
    else:
        # Fallback: 随机采样（跳过PAD）
        print(f"\n⚠️  使用随机采样验证（可能不准确，建议指定 --dataset 参数）")
        available_indices = np.arange(1, len(emb))
        sample_size = min(2000, len(available_indices))
        np.random.seed(42)
        sample_indices = np.random.choice(available_indices, size=sample_size, replace=False)
    
    # 提取样本
    sample = emb[sample_indices].astype(np.float64)
    
    # 检查是否中心化（均值接近0）
    mean_vec = sample.mean(axis=0)
    mean_abs = np.abs(mean_vec).mean()
    if mean_abs < 0.2:
        print(f"✅ Embedding已中心化 (mean_abs={mean_abs:.4f})")
    else:
        print(f"⚠️  Embedding未充分中心化 (mean_abs={mean_abs:.4f})")
    
    # 检查协方差矩阵是否接近单位矩阵（验证白化效果）
    # 重新中心化样本（以消除float16精度误差）
    sample_centered = sample - sample.mean(axis=0, keepdims=True)
    
    # 计算协方差矩阵
    cov = (sample_centered.T @ sample_centered) / len(sample_centered)
    
    # 对角线应接近1
    diag_vals = np.diag(cov)
    diag_mean = diag_vals.mean()
    diag_std = diag_vals.std()
    
    # 非对角线应接近0
    off_diag_mask = ~np.eye(cov.shape[0], dtype=bool)
    off_diag_vals = cov[off_diag_mask]
    off_diag_max = np.abs(off_diag_vals).max()
    off_diag_mean = np.abs(off_diag_vals).mean()
    
    print(f"\n📊 协方差矩阵分析 (sample_size={len(sample_centered)}):")
    print(f"   - 对角线均值: {diag_mean:.4f} (期望≈1.0)")
    print(f"   - 对角线标准差: {diag_std:.4f} (期望≈0)")
    print(f"   - 非对角线最大值: {off_diag_max:.4f} (期望≈0)")
    print(f"   - 非对角线平均值: {off_diag_mean:.4f} (期望≈0)")
    
    # 判断白化效果
    if has_whiten_matrix:
        # 如果有白化矩阵，使用更严格的标准
        if abs(diag_mean - 1.0) < 0.1 and diag_std < 0.15 and off_diag_mean < 0.05:
            print(f"✅ 白化效果优秀")
            if train_ids is None:
                print(f"   💡 提示: 使用 --dataset 参数可获得更准确的验证结果")
            return True
        elif abs(diag_mean - 1.0) < 0.15 and diag_std < 0.2 and off_diag_mean < 0.08:
            print(f"✅ 白化效果良好")
            if train_ids is None:
                print(f"   💡 提示: 使用 --dataset 参数可获得更准确的验证结果")
            return True
        elif abs(diag_mean - 1.0) < 0.25 and off_diag_mean < 0.12:
            print(f"⚠️  白化效果一般")
            if train_ids is None:
                print(f"   可能原因: 验证时未使用训练集items（请添加 --dataset 参数）")
            else:
                print(f"   可能原因: 训练集样本量不足或数据方差较小")
            return True
        else:
            print(f"❌ 白化效果较差")
            print(f"   可能原因：")
            if train_ids is None:
                print(f"   1. 未使用训练集验证（请添加 --dataset 参数重新验证）")
            print(f"   2. 生成时未启用whitening（使用了 --no_whiten）")
            print(f"   3. 训练集样本量太小")
            print(f"   4. 数据方差过小或特征相关性过强")
            return False
    else:
        # 没有白化矩阵，只验证中心化
        if mean_abs < 0.2:
            print(f"✅ 中心化成功（未启用白化）")
            return True
        else:
            print(f"❌ 中心化不完全")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="验证 center + whiten 预处理效果（支持训练集专用验证）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础验证（随机采样，可能不准确）
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
  
  # 训练集专用验证（推荐，最准确）
  python tools/verify_whiten.py \\
    dataset/Amazon_Beauty/item_text_emb.base.npy \\
    --dataset Amazon_Beauty
  
  # 使用自定义配置验证
  python tools/verify_whiten.py \\
    dataset/Amazon_Beauty/item_text_emb.base.npy \\
    --dataset Amazon_Beauty \\
    --config recbole/properties/model/GRU4RecCPR.yaml
  
  # 批量验证多个文件
  python tools/verify_whiten.py \\
    dataset/Amazon_Beauty/item_text_emb.base.npy \\
    dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \\
    --dataset Amazon_Beauty
"""
    )
    parser.add_argument(
        "emb_paths",
        nargs='+',
        metavar='emb_path',
        help="Embedding文件路径 (*.npy)，支持多个文件"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="数据集名称（用于加载训练集item IDs进行准确验证），例如: Amazon_Beauty"
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=[],
        help="可选的YAML配置文件"
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
        
        success = verify_whitening(emb_path, args.dataset, args.config)
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
