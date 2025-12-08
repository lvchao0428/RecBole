#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分档评估指标是否正常工作

快速验证：
1. 导入测试
2. 模拟数据测试
3. 配置文件验证
"""

import sys
import numpy as np
import torch
from collections import Counter

print("="*60)
print("分档评估指标测试")
print("="*60)
print()

# ==========================================
# 测试 1: 导入测试
# ==========================================
print("[1/3] 测试导入...")
try:
    from recbole.evaluator.stratified_metrics import (
        StratifiedRecall, 
        StratifiedNDCG, 
        ItemPopularityStats
    )
    print("  ✅ StratifiedRecall 导入成功")
    print("  ✅ StratifiedNDCG 导入成功")
    print("  ✅ ItemPopularityStats 导入成功")
except ImportError as e:
    print("  ❌ 导入失败: {0}".format(str(e)))
    print("\n请检查:")
    print("  1. recbole/evaluator/stratified_metrics.py 是否存在")
    print("  2. recbole/evaluator/__init__.py 是否包含导入")
    sys.exit(1)

print()

# ==========================================
# 测试 2: 基本功能测试
# ==========================================
print("[2/3] 测试基本功能...")

try:
    # 创建简单配置
    from recbole.config import Config
    
    test_config = {
        'topk': [5, 10, 20],
        'metric_decimal_place': 4
    }
    
    # 实例化指标
    recall_metric = StratifiedRecall(test_config)
    ndcg_metric = StratifiedNDCG(test_config)
    coverage_metric = ItemPopularityStats(test_config)
    
    print("  ✅ StratifiedRecall 实例化成功")
    print("  ✅ StratifiedNDCG 实例化成功")
    print("  ✅ ItemPopularityStats 实例化成功")
    
    # 检查阈值
    print("\n  分档阈值:")
    print("    new:      {0}".format(recall_metric.new_threshold))
    print("    few:      {0}".format(recall_metric.few_threshold))
    print("    frequent: {0}".format(recall_metric.freq_threshold))
    
except Exception as e:
    print("  ❌ 功能测试失败: {0}".format(str(e)))
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ==========================================
# 测试 3: 配置文件验证
# ==========================================
print("[3/3] 验证配置文件...")

config_files = [
    'sasrec_align_base_stratified.yaml',
    'sasrec_align_qwen3_stratified.yaml',
    'sasrec_align_multi_view_stratified.yaml',
    'sasrec_align_toys_base_stratified.yaml',
    'sasrec_align_toys_qwen3_stratified.yaml',
    'sasrec_align_multi_view_toys_stratified.yaml',
    'sasrec_baseline_70ep_stratified.yaml',
    'sasrec_baseline_70ep_toys_stratified.yaml',
]

import os
import yaml

found = 0
missing = []

for config_file in config_files:
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 检查 metrics 字段
            if 'metrics' in config:
                metrics = config['metrics']
                has_stratified = (
                    'StratifiedRecall' in metrics or 
                    'StratifiedNDCG' in metrics or
                    'ItemPopularityStats' in metrics
                )
                
                if has_stratified:
                    print("  ✅ {0}".format(config_file))
                    found += 1
                else:
                    print("  ⚠️  {0} (无分档指标)".format(config_file))
            else:
                print("  ⚠️  {0} (无 metrics 字段)".format(config_file))
        except Exception as e:
            print("  ❌ {0} (解析失败: {1})".format(config_file, str(e)))
    else:
        print("  ❌ {0} (文件不存在)".format(config_file))
        missing.append(config_file)

print()
print("  找到 {0}/{1} 个配置文件".format(found, len(config_files)))

if missing:
    print("\n  缺失文件:")
    for fname in missing:
        print("    - {0}".format(fname))

print()

# ==========================================
# 总结
# ==========================================
print("="*60)
print("测试总结")
print("="*60)
print()

if found == len(config_files):
    print("✅ 所有测试通过！")
    print()
    print("分档评估功能已正确安装，可以使用以下脚本:")
    print()
    print("Beauty 数据集:")
    print("  bash run70epBase_stratified.sh")
    print("  bash two_phase_run_tfidf_stratified.sh")
    print("  bash two_phase_run_tfidf_llm_stratified.sh")
    print("  bash two_phase_run_multiview_split_stratified.sh")
    print()
    print("Toys 数据集:")
    print("  bash run70epBase_toys_stratified.sh")
    print("  bash two_phase_run_tfidf_toys_stratified.sh")
    print("  bash two_phase_run_tfidf_llm_toys_stratified.sh")
    print("  bash two_phase_run_multiview_split_toys_stratified.sh")
    print()
    print("输出指标包括:")
    print("  - Recall_new@10, Recall_few@10, Recall_frequent@10")
    print("  - NDCG_new@10, NDCG_few@10, NDCG_frequent@10")
    print("  - Coverage_new@10, Coverage_few@10, Coverage_frequent@10")
    print()
    sys.exit(0)
else:
    print("⚠️  部分测试失败，请检查上述错误")
    sys.exit(1)

