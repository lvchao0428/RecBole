#!/usr/bin/env python3
"""
验证所有数据集的文本特征文件

检查项：
1. 特征文件是否存在
2. 特征维度是否正确（256维）
3. Center+Whiten 统计文件是否存在
4. 多视图分视图文件是否存在且维度正确（64维）
"""

import os
import sys
import numpy as np
from pathlib import Path
import json

# 数据集配置
DATASETS = {
    'Amazon_Beauty': 'dataset/Amazon_Beauty',
    'Amazon_Toys_and_Games': 'dataset/Amazon_Toys_and_Games',
    'Amazon_Video_Games': 'dataset/Amazon_Video_Games',
    'yelp': 'dataset/yelp',
}

# 预期的特征文件
EXPECTED_FILES = {
    'TF-IDF': {
        'embedding': 'item_text_emb.base.npy',
        'stats': 'item_text_emb.base_whiten_stats.npz',
        'expected_dim': 256,
    },
    'Qwen3 单视图': {
        'embedding': 'item_text_emb.qwen3.base.npy',
        'stats': 'item_text_emb.qwen3.base_whiten_stats.npz',
        'expected_dim': 256,
    },
    'Qwen3 多视图': {
        'embedding': 'item_text_emb.qwen3.multiview.npy',
        'stats': 'item_text_emb.qwen3.multiview_whiten_stats.npz',
        'expected_dim': 256,
    },
}

# 多视图配置
MULTIVIEW_CONFIG = {
    'dir': 'qwen3_4views',
    'num_views': 4,
    'view_dim': 64,
    'view_names': ['Identity', 'Function', 'Audience', 'Category'],
}


def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)


def check_embedding_shape(filepath, expected_dim):
    """检查特征文件的维度"""
    try:
        emb = np.load(filepath)
        if emb.ndim != 2:
            return False, f"维度错误：{emb.shape}（应为 2D）"
        if emb.shape[1] != expected_dim:
            return False, f"特征维度错误：{emb.shape[1]}（应为 {expected_dim}）"
        return True, f"{emb.shape}"
    except Exception as e:
        return False, f"加载失败：{str(e)}"


def check_stats_file(filepath):
    """检查 center+whiten 统计文件"""
    try:
        stats = np.load(filepath)
        if 'center' not in stats or 'whiten' not in stats:
            return False, "缺少 'center' 或 'whiten' 字段"
        center_shape = stats['center'].shape
        whiten_shape = stats['whiten'].shape
        if len(center_shape) != 1:
            return False, f"center 维度错误：{center_shape}"
        if len(whiten_shape) != 2:
            return False, f"whiten 维度错误：{whiten_shape}"
        if center_shape[0] != whiten_shape[0] or whiten_shape[0] != whiten_shape[1]:
            return False, f"维度不匹配：center{center_shape}, whiten{whiten_shape}"
        return True, f"center{center_shape}, whiten{whiten_shape}"
    except Exception as e:
        return False, f"加载失败：{str(e)}"


def check_multiview_splits(dataset_dir):
    """检查多视图分视图文件"""
    multiview_dir = os.path.join(dataset_dir, MULTIVIEW_CONFIG['dir'])
    
    results = []
    
    # 检查目录是否存在
    if not os.path.exists(multiview_dir):
        return False, f"目录不存在：{multiview_dir}"
    
    # 检查 views.json
    views_json = os.path.join(multiview_dir, 'views.json')
    if not os.path.exists(views_json):
        results.append(f"  ⚠️  views.json 不存在")
    else:
        try:
            with open(views_json, 'r') as f:
                views_meta = json.load(f)
            results.append(f"  ✅ views.json: {len(views_meta.get('views', []))} 个视图")
        except Exception as e:
            results.append(f"  ❌ views.json 读取失败：{str(e)}")
    
    # 检查每个视图文件
    all_ok = True
    for i in range(MULTIVIEW_CONFIG['num_views']):
        view_file = os.path.join(multiview_dir, f'view_{i}.npy')
        if not os.path.exists(view_file):
            results.append(f"  ❌ view_{i}.npy 不存在")
            all_ok = False
        else:
            try:
                view_emb = np.load(view_file)
                if view_emb.shape[1] != MULTIVIEW_CONFIG['view_dim']:
                    results.append(f"  ❌ view_{i}.npy 维度错误：{view_emb.shape[1]}（应为 {MULTIVIEW_CONFIG['view_dim']}）")
                    all_ok = False
                else:
                    view_name = MULTIVIEW_CONFIG['view_names'][i] if i < len(MULTIVIEW_CONFIG['view_names']) else f"View{i}"
                    results.append(f"  ✅ view_{i}.npy ({view_name}): {view_emb.shape}")
            except Exception as e:
                results.append(f"  ❌ view_{i}.npy 读取失败：{str(e)}")
                all_ok = False
    
    return all_ok, "\n".join(results)


def verify_dataset(dataset_name, dataset_dir):
    """验证单个数据集的所有特征文件"""
    print(f"\n{'='*60}")
    print(f"📊 检查数据集: {dataset_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 数据集目录不存在：{dataset_dir}")
        return False
    
    all_ok = True
    
    # 检查每种特征类型
    for feature_type, config in EXPECTED_FILES.items():
        print(f"\n🔍 {feature_type}:")
        
        # 检查特征文件
        emb_file = os.path.join(dataset_dir, config['embedding'])
        if not check_file_exists(emb_file):
            print(f"  ❌ 特征文件不存在：{config['embedding']}")
            all_ok = False
        else:
            ok, msg = check_embedding_shape(emb_file, config['expected_dim'])
            if ok:
                print(f"  ✅ 特征文件：{config['embedding']} - shape={msg}")
            else:
                print(f"  ❌ 特征文件：{config['embedding']} - {msg}")
                all_ok = False
        
        # 检查统计文件
        stats_file = os.path.join(dataset_dir, config['stats'])
        if not check_file_exists(stats_file):
            print(f"  ⚠️  统计文件不存在：{config['stats']}")
            # 不标记为错误，因为可能是旧版本生成的
        else:
            ok, msg = check_stats_file(stats_file)
            if ok:
                print(f"  ✅ 统计文件：{config['stats']} - {msg}")
            else:
                print(f"  ❌ 统计文件：{config['stats']} - {msg}")
                all_ok = False
    
    # 检查多视图分视图
    print(f"\n🔍 多视图分视图:")
    ok, msg = check_multiview_splits(dataset_dir)
    if ok:
        print(msg)
    else:
        print(f"  ❌ {msg}")
        all_ok = False
    
    # 总结
    if all_ok:
        print(f"\n✅ {dataset_name} 所有检查通过！")
    else:
        print(f"\n⚠️  {dataset_name} 存在问题，请检查上述错误。")
    
    return all_ok


def main():
    """主函数"""
    print("="*60)
    print("文本特征验证工具")
    print("="*60)
    print(f"工作目录: {os.getcwd()}")
    print(f"检查数据集: {', '.join(DATASETS.keys())}")
    
    all_datasets_ok = True
    summary = []
    
    for dataset_name, dataset_dir in DATASETS.items():
        ok = verify_dataset(dataset_name, dataset_dir)
        summary.append((dataset_name, ok))
        if not ok:
            all_datasets_ok = False
    
    # 打印总结
    print(f"\n{'='*60}")
    print("📋 验证总结")
    print(f"{'='*60}")
    for dataset_name, ok in summary:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {dataset_name:30s} {status}")
    
    print(f"\n{'='*60}")
    if all_datasets_ok:
        print("🎉 所有数据集验证通过！")
        print("\n下一步:")
        print("  1. 运行 TF-IDF 实验：bash two_phase_run_tfidf_*.sh")
        print("  2. 运行多视图实验：bash two_phase_run_multiview_split_*.sh")
    else:
        print("⚠️  部分数据集验证失败，请检查上述错误。")
        print("\n建议:")
        print("  1. 运行对应的生成脚本：bash tools/gen_text_emb_*_full_fast.sh")
        print("  2. 检查错误日志")
    print(f"{'='*60}\n")
    
    return 0 if all_datasets_ok else 1


if __name__ == '__main__':
    sys.exit(main())

