#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证白化修复是否成功

用法：
  python tools/verify_whiten_fix.py
"""

import os
import sys

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def check_code_fix():
    """检查代码是否已修复"""
    qwen3_file = os.path.join(ROOT_DIR, "tools/build_item_text_emb_qwen3_hf.py")
    
    with open(qwen3_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    fixes = []
    
    # 检查修复点1: 白化后不应该有 L2 归一化
    if "We MUST L2 normalize even after whitening" in content:
        issues.append("❌ 修复点1未应用: 仍然包含错误的注释")
    elif "Do NOT L2 normalize after whitening!" in content:
        fixes.append("✅ 修复点1已应用: 移除了白化后的 L2 归一化")
    else:
        issues.append("⚠️  修复点1状态不明确")
    
    # 检查修复点2: SVD 投影应该设置 normalize=False
    if "normalize=True,  # Re-normalize after whitening+projection" in content:
        issues.append("❌ 修复点2未应用: SVD 投影仍然在归一化")
    elif "normalize=False,  # Preserve variance for subsequent whitening" in content:
        fixes.append("✅ 修复点2已应用: SVD 投影不再归一化")
    else:
        issues.append("⚠️  修复点2状态不明确")
    
    # 检查是否还有错误的 L2 归一化代码
    if "emb_whitened[1:] = emb_whitened[1:] / np.clip(norms, 1e-8, None)" in content:
        issues.append("❌ 检测到白化后的 L2 归一化代码未删除")
    
    print("=" * 70)
    print("白化修复代码检查")
    print("=" * 70)
    print()
    
    if fixes:
        for fix in fixes:
            print(fix)
        print()
    
    if issues:
        for issue in issues:
            print(issue)
        print()
        print("⚠️  请检查代码修复是否完整应用")
        return False
    else:
        print("✅ 所有修复已正确应用！")
        print()
        print("下一步:")
        print("  1. 重新生成 Qwen3 特征:")
        print("     bash tools/regen_qwen3_beauty.sh")
        print()
        print("  2. 验证白化效果:")
        print("     python tools/verify_whiten.py \\")
        print("       dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \\")
        print("       --dataset Amazon_Beauty")
        print()
        return True


def main():
    success = check_code_fix()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

