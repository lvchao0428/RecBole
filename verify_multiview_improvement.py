#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证多视图模型改进是否生效

检查项：
1. 模型能否正确初始化
2. 投影维度是否正确（1280→512 或 1024→512）
3. 融合维度是否为768
4. Base特征是否被正确加载

用法：
  python verify_multiview_improvement.py
"""

import os
import sys
import torch

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config import Config
from recbole.data import create_dataset
from recbole.model.sequential_recommender.sasrecalignmultiview import SASRecAlignMultiView


def verify_model():
    print("=" * 70)
    print("验证多视图模型改进")
    print("=" * 70)
    print()
    
    # 创建配置
    print("[1/5] 加载配置...")
    try:
        config = Config(
            model='SASRecAlignMultiView',
            dataset='Amazon_Beauty',
            config_file_list=['sasrec_align_multi_view.yaml']
        )
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 创建数据集
    print()
    print("[2/5] 加载数据集...")
    try:
        dataset = create_dataset(config)
        print(f"✅ 数据集加载成功: {dataset.num(dataset.iid_field)} items")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False
    
    # 创建模型
    print()
    print("[3/5] 初始化模型...")
    try:
        model = SASRecAlignMultiView(config, dataset)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查特征加载
    print()
    print("[4/5] 检查特征加载...")
    has_base = model.item_text_emb_base is not None
    has_views = len(model.text_view_buffer_names) > 0
    
    if has_base:
        base_shape = model.item_text_emb_base.shape
        print(f"✅ Base特征已加载: {base_shape}")
    else:
        print(f"⚠️  Base特征未加载（可能未配置路径）")
    
    if has_views:
        num_views = len(model.text_view_buffer_names)
        first_view = model._get_view_buffer(0)
        print(f"✅ 多视图特征已加载: {num_views} views, shape={first_view.shape}")
    else:
        print(f"❌ 多视图特征未加载")
        return False
    
    # 检查投影维度
    print()
    print("[5/5] 检查网络维度...")
    
    # 投影层
    proj_layer = model.multiview_concat_proj
    proj_in = proj_layer.in_features
    proj_out = proj_layer.out_features
    
    expected_in = 1280 if has_base else 1024
    expected_out = 512
    
    if proj_in == expected_in and proj_out == expected_out:
        print(f"✅ 投影层维度正确: [{proj_in} → {proj_out}]")
    else:
        print(f"❌ 投影层维度错误: [{proj_in} → {proj_out}]")
        print(f"   期望: [{expected_in} → {expected_out}]")
        return False
    
    # 融合网络
    if model.use_cross and model.item_fusion_cross is not None:
        fusion_in = model.item_fusion_cross.input_dim
        expected_fusion_in = 768  # 256 + 512
        
        if fusion_in == expected_fusion_in:
            print(f"✅ 融合网络维度正确: {fusion_in}")
        else:
            print(f"❌ 融合网络维度错误: {fusion_in}")
            print(f"   期望: {expected_fusion_in}")
            return False
        
        # 估算参数量
        cross_params = sum(p.numel() for p in model.item_fusion_cross.parameters())
        print(f"✅ Cross Network参数量: {cross_params / 1e6:.2f}M")
    
    # 总结
    print()
    print("=" * 70)
    print("✅ 验证通过！改进已生效")
    print("=" * 70)
    print()
    print("架构总结:")
    print(f"  - Base特征: {'启用' if has_base else '未启用'}")
    print(f"  - 多视图数量: {num_views if has_views else 0}")
    print(f"  - 投影维度: {proj_in} → {proj_out}")
    print(f"  - 融合维度: {fusion_in if model.use_cross else 'N/A'}")
    print(f"  - 与双路模型对齐: {'是' if fusion_in == 768 else '否'}")
    print()
    
    return True


if __name__ == "__main__":
    success = verify_model()
    sys.exit(0 if success else 1)

