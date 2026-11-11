#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简化版：快速检查文本特征是否生效"""

import os
import sys

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.model.sequential_recommender.sasrec_align import SASRecAlign

def main():
    # 获取命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python simple_check_text.py <config_file1> [config_file2] ...")
        sys.exit(1)
    
    config_files = sys.argv[1:]
    print(f"配置文件: {config_files}")
    
    try:
        # 创建配置
        config = Config(
            model='SASRec_Align', 
            dataset='Amazon_Beauty', 
            config_file_list=config_files
        )
        
        # 创建数据集
        dataset = create_dataset(config)
        
        # 创建模型
        model = SASRecAlign(config, dataset)
        
        print("\n=== 关键信息 ===")
        
        # 1. 文本特征是否禁用
        disable_text = config.get('disable_text_feature', False) if hasattr(config, 'get') else \
                      (config['disable_text_feature'] if 'disable_text_feature' in config else False)
        print(f"1. 文本特征禁用: {disable_text}")
        
        # 2. 文本模式
        text_mode = getattr(model, '_text_mode', 'unknown')
        print(f"2. 文本模式: {text_mode}")
        
        # 3. 文本嵌入是否存在
        has_base = hasattr(model, 'item_text_emb_base') and model.item_text_emb_base is not None
        has_llm = hasattr(model, 'item_text_emb_llm') and model.item_text_emb_llm is not None
        print(f"3. Base嵌入存在: {has_base}")
        print(f"   LLM嵌入存在: {has_llm}")
        
        if has_base and model.item_text_emb_base is not None:
            print(f"   Base嵌入shape: {model.item_text_emb_base.shape}")
        
        # 4. 文本相关模块
        print(f"4. 文本投影层存在: {hasattr(model, 'item_text_proj') and model.item_text_proj is not None}")
        print(f"   拼接预测器存在: {hasattr(model, 'item_concat_predictor') and model.item_concat_predictor is not None}")
        
        # 5. 有效文本权重
        if hasattr(model, 'text_gate_param') and hasattr(model, 'text_weight'):
            alpha = torch.sigmoid(model.text_gate_param).item()
            weight = model.text_weight
            effective = alpha * weight
            print(f"5. 门控值(alpha): {alpha:.4f}")
            print(f"   文本权重: {weight}")
            print(f"   有效权重: {effective:.4f}")
        
        # 6. 快速测试融合效果
        print("\n=== 融合测试 ===")
        with torch.no_grad():
            # 测试几个item的融合嵌入
            test_ids = torch.tensor([1, 2, 3, 4, 5])
            
            # 获取原始嵌入
            orig_emb = model.item_embedding(test_ids)
            
            # 获取融合嵌入
            fused_emb = model._get_fused_item_embeddings(test_ids)
            
            # 计算差异
            diff = torch.norm(fused_emb - orig_emb, dim=1).mean().item()
            
            print(f"原始嵌入与融合嵌入的平均L2差异: {diff:.6f}")
            
            if diff < 1e-6:
                print("⚠️  警告: 差异接近0，文本特征可能未生效!")
            else:
                print("✓ 文本特征已融入item嵌入")
                
                # 计算相对差异
                orig_norm = torch.norm(orig_emb, dim=1).mean().item()
                rel_diff = diff / orig_norm if orig_norm > 0 else 0
                print(f"相对差异: {rel_diff:.4f} ({rel_diff*100:.2f}%)")
        
        # 7. 参数统计
        print("\n=== 参数统计 ===")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params:,}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
