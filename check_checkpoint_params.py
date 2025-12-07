#!/usr/bin/env python3
"""
检查 Phase A checkpoint 中保存的网格参数
"""

import torch
import os
from pathlib import Path

# Checkpoint 目录
CHECKPOINT_DIR = "./saved/phase_runs_multiview_4views"

print("=" * 70)
print("检查 Phase A Checkpoint 中的网格参数")
print("=" * 70)
print()

checkpoint_dir = Path(CHECKPOINT_DIR)

if not checkpoint_dir.exists():
    print(f"❌ 目录不存在: {CHECKPOINT_DIR}")
    exit(1)

# 查找所有 .pth 文件
ckpt_files = sorted(checkpoint_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime)

if not ckpt_files:
    print(f"❌ 未找到 checkpoint 文件")
    exit(1)

print(f"找到 {len(ckpt_files)} 个 checkpoint 文件")
print()

# 分析每个 checkpoint
results = []

for i, ckpt_path in enumerate(ckpt_files, 1):
    try:
        # 加载 checkpoint
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        
        # 提取关键信息
        config = ckpt.get('config', {})
        
        info = {
            'filename': ckpt_path.name,
            'path': str(ckpt_path),
            'alignment_weight': config.get('alignment_weight', None),
            'temperature': config.get('temperature', None),
            'freeze_backbone': config.get('freeze_backbone', None),
            'best_valid_score': ckpt.get('best_valid_score', None),
            'epoch': ckpt.get('epoch', None),
            'model': config.get('model', None),
            'dataset': config.get('dataset', None),
        }
        
        results.append(info)
        
        # 打印详细信息
        print(f"[{i}] {info['filename']}")
        print(f"    路径: {info['path']}")
        print(f"    模型: {info['model']}")
        print(f"    数据集: {info['dataset']}")
        print(f"    Freeze Backbone: {info['freeze_backbone']}")
        print(f"    ✅ Alignment Weight: {info['alignment_weight']}")
        print(f"    ✅ Temperature: {info['temperature']}")
        print(f"    Best Valid Score: {info['best_valid_score']}")
        print(f"    Epoch: {info['epoch']}")
        print()
        
    except Exception as e:
        print(f"[{i}] {ckpt_path.name}")
        print(f"    ❌ 加载失败: {str(e)}")
        print()

print("=" * 70)
print("总结")
print("=" * 70)
print()

# 统计 Phase A checkpoint
phase_a_ckpts = [r for r in results if r.get('freeze_backbone') is True]
phase_b_ckpts = [r for r in results if r.get('freeze_backbone') is False]
unknown_ckpts = [r for r in results if r.get('freeze_backbone') is None]

print(f"Phase A checkpoint (freeze_backbone=True): {len(phase_a_ckpts)} 个")
print(f"Phase B checkpoint (freeze_backbone=False): {len(phase_b_ckpts)} 个")
print(f"未知类型: {len(unknown_ckpts)} 个")
print()

if phase_a_ckpts:
    print("=" * 70)
    print("Phase A 网格参数对比")
    print("=" * 70)
    print()
    
    # 按 valid score 排序
    phase_a_sorted = sorted(
        [r for r in phase_a_ckpts if r.get('best_valid_score') is not None],
        key=lambda x: x['best_valid_score'],
        reverse=True
    )
    
    if phase_a_sorted:
        print(f"{'排名':<6} {'Valid Score':<15} {'Align Weight':<15} {'Temperature':<15} {'文件名'}")
        print("-" * 70)
        
        for rank, info in enumerate(phase_a_sorted, 1):
            print(f"{rank:<6} {info['best_valid_score']:<15.6f} "
                  f"{info['alignment_weight']:<15} {info['temperature']:<15} "
                  f"{info['filename']}")
        
        print()
        print("=" * 70)
        print("✅ 最佳网格参数组合")
        print("=" * 70)
        best = phase_a_sorted[0]
        print(f"文件: {best['filename']}")
        print(f"Alignment Weight: {best['alignment_weight']}")
        print(f"Temperature: {best['temperature']}")
        print(f"Best Valid Score: {best['best_valid_score']:.6f}")
        print(f"完整路径: {best['path']}")
        print()
        
        # 生成使用该 checkpoint 的脚本
        script_path = "./use_best_phase_a.sh"
        with open(script_path, 'w') as f:
            f.write(f"""#!/usr/bin/env bash
# 使用最佳 Phase A checkpoint 启动 Phase B
# 最佳参数: alignment_weight={best['alignment_weight']}, temperature={best['temperature']}
# Valid Score: {best['best_valid_score']:.6f}

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BEST_CHECKPOINT="{best['path']}"

echo "使用最佳 Phase A checkpoint:"
echo "  文件: {best['filename']}"
echo "  Alignment Weight: {best['alignment_weight']}"
echo "  Temperature: {best['temperature']}"
echo "  Valid Score: {best['best_valid_score']:.6f}"
echo ""

python scripts/two_phase_train.py \\
  --model SASRecAlignMultiView \\
  --dataset Amazon_Beauty \\
  --config_files "sasrec_align_multi_view.yaml" \\
  --only_phase_b \\
  --resume_from "$BEST_CHECKPOINT" \\
  --phase_b_alignment_weight {best['alignment_weight']} \\
  --phase_b_text_gate_reg_l2 0.05 \\
  --phase_b_text_weight 0.8 \\
  --phase_b_epochs 40 \\
  --lr_text_head 1e-3 \\
  --lr_dnn_cross 5e-4 \\
  --backbone_lr_scale 0.1 \\
  --checkpoint_dir ./saved/phase_runs_multiview_4views \\
  --seed 2025 \\
  --variant_features "sasrec,multiview,4views,per_view_align,qwen3,phase_b_only" \\
  --watchdog_disable \\
  --save

echo ""
echo "✅ Phase B 完成！"
""")
        
        os.chmod(script_path, 0o755)
        
        print(f"✅ 已生成启动脚本: {script_path}")
        print(f"   直接运行: bash {script_path}")
        print()
    else:
        print("⚠️  Phase A checkpoint 没有 best_valid_score 信息")
        print()
else:
    print("❌ 未找到 Phase A checkpoint (freeze_backbone=True)")
    print()
    print("提示：可能 checkpoint 文件中没有保存 freeze_backbone 配置")
    print()

print("=" * 70)
print("结论")
print("=" * 70)
print()
if phase_a_ckpts and any(r.get('alignment_weight') is not None for r in phase_a_ckpts):
    print("✅ 是的！Checkpoint 中保存了网格参数：")
    print("   - alignment_weight")
    print("   - temperature")
    print("   - best_valid_score")
    print("   - 以及完整的 config 配置")
    print()
    print("这意味着：")
    print("1. 可以从 checkpoint 中读取最佳的网格参数组合")
    print("2. Phase B 可以继承 Phase A 的最佳参数")
    print("3. 整个实验是完全可复现的")
else:
    print("⚠️  未能从 checkpoint 中提取网格参数")
    print("   可能需要检查 checkpoint 文件格式")

print()
print("=" * 70)

