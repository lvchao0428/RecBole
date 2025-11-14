#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速检查文本特征是否生效的脚本"""

import os
import sys

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from recbole.quick_start import load_data_and_model
from recbole.config import Config
from recbole.data.utils import create_dataset

def check_text_features(config_files):
    """检查文本特征配置和实际加载情况"""
    
    # 加载配置
    config = Config(model='SASRec_Align', dataset='Amazon_Beauty', config_file_list=config_files)
    
    print("\n=== 配置检查 ===")
    print(f"disable_text_feature: {config['disable_text_feature'] if 'disable_text_feature' in config else False}")
    print(f"use_llm: {config['use_llm'] if 'use_llm' in config else False}")
    print(f"use_cross: {config['use_cross'] if 'use_cross' in config else False}")
    print(f"text_weight: {config['text_weight'] if 'text_weight' in config else 1.0}")
    print(f"text_gate_init: {config['text_gate_init'] if 'text_gate_init' in config else 0.5}")
    
    # 创建数据集和模型
    dataset = create_dataset(config)
    from recbole.model.sequential_recommender.sasrec_align import SASRecAlign
    model = SASRecAlign(config, dataset)
    
    print("\n=== 模型架构检查 ===")
    print(f"模型类型: {model.__class__.__name__}")
    
    # 检查文本相关模块
    text_modules = [
        'item_text_proj',
        'item_concat_predictor', 
        'item_fusion_predictor',
        'text_cross',
        'text_deep',
        'text_predictor'
    ]
    
    for module in text_modules:
        exists = hasattr(model, module) and getattr(model, module) is not None
        print(f"{module}: {'✓ 存在' if exists else '✗ 不存在'}")
    
    # 检查文本嵌入
    print("\n=== 文本嵌入检查 ===")
    if hasattr(model, 'item_text_emb_base'):
        if model.item_text_emb_base is not None:
            print(f"Base嵌入: shape={model.item_text_emb_base.shape}, "
                  f"dtype={model.item_text_emb_base.dtype}")
            # 检查是否全零
            if torch.all(model.item_text_emb_base == 0):
                print("  ⚠️  警告: Base嵌入全为零!")
        else:
            print("Base嵌入: None (未加载)")
    
    if hasattr(model, 'item_text_emb_llm'):
        if model.item_text_emb_llm is not None:
            print(f"LLM嵌入: shape={model.item_text_emb_llm.shape}, "
                  f"dtype={model.item_text_emb_llm.dtype}")
        else:
            print("LLM嵌入: None (未加载)")
    
    # 检查文本模式
    print(f"\n文本模式 (_text_mode): {getattr(model, '_text_mode', 'unknown')}")
    
    # 检查参数数量
    print("\n=== 参数统计 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print("\n=== 前向传播测试 ===")
    try:
        # 创建一个小批量测试
        with torch.no_grad():
            # 模拟输入
            batch_size = 2
            seq_len = 10
            item_seq = torch.randint(1, 100, (batch_size, seq_len))
            item_seq_len = torch.tensor([seq_len, seq_len])
            
            # 获取融合后的item embeddings
            test_items = torch.tensor([1, 2, 3, 4, 5])
            fused_emb = model._get_fused_item_embeddings(test_items)
            
            # 对比原始embeddings
            orig_emb = model.item_embedding(test_items)
            
            # 计算差异
            diff = torch.norm(fused_emb - orig_emb, dim=1).mean()
            print(f"融合嵌入与原始嵌入的平均差异: {diff:.6f}")
            
            if diff < 1e-6:
                print("  ⚠️  警告: 融合嵌入与原始嵌入几乎相同，文本特征可能未生效!")
            else:
                print("  ✓ 文本特征已融合到item嵌入中")
                
            # 检查门控值
            if hasattr(model, 'text_gate_param'):
                alpha = torch.sigmoid(model.text_gate_param).item()
                effective_weight = alpha * model.text_weight
                print(f"\n门控参数 alpha: {alpha:.6f}")
                print(f"文本权重 text_weight: {model.text_weight}")
                print(f"有效文本权重: {effective_weight:.6f}")
                
    except Exception as e:
        print(f"前向传播测试失败: {e}")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='+', required=True, help='配置文件列表')
    args = parser.parse_args()
    
    check_text_features(args.config)
