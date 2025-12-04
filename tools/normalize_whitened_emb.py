#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process whitened embeddings with L2 normalization.

Whitening decorrelates features but doesn't normalize the magnitude.
This script adds L2 normalization to whitened embeddings to:
1. Restore norm ≈ 1 for better compatibility with contrastive learning
2. Maintain the whitening property (decorrelation)

Usage:
    python tools/normalize_whitened_emb.py \
        --input dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
        --output dataset/Amazon_Beauty/item_text_emb.qwen3.base.normed.npy
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="L2-normalize whitened embeddings")
    parser.add_argument("--input", required=True, help="Path to whitened .npy file")
    parser.add_argument("--output", required=True, help="Path to output normalized .npy file")
    parser.add_argument("--keep_dtype", action="store_true", help="Keep original dtype (default: convert to float32)")
    args = parser.parse_args()

    print(f"Loading whitened embeddings from: {args.input}")
    emb = np.load(args.input)
    orig_dtype = emb.dtype
    print(f"  Shape: {emb.shape}, dtype: {orig_dtype}")

    # Convert to float32 for stable normalization
    if orig_dtype != np.float32:
        emb = emb.astype(np.float32)

    # Check PAD row
    pad_norm = np.linalg.norm(emb[0])
    print(f"  PAD row (row 0) norm: {pad_norm:.6f}")

    # Compute norms before normalization
    norms_before = np.linalg.norm(emb[1:], axis=1)
    print(f"  Norms before L2-norm: mean={norms_before.mean():.4f}, std={norms_before.std():.4f}, "
          f"min={norms_before.min():.4f}, max={norms_before.max():.4f}")

    # L2-normalize non-PAD rows
    eps = 1e-8
    norms = np.linalg.norm(emb[1:], axis=1, keepdims=True)
    emb[1:] = emb[1:] / np.maximum(norms, eps)

    # Verify normalization
    norms_after = np.linalg.norm(emb[1:], axis=1)
    print(f"  Norms after L2-norm: mean={norms_after.mean():.4f}, std={norms_after.std():.4f}, "
          f"min={norms_after.min():.4f}, max={norms_after.max():.4f}")

    # Ensure PAD row is still zeros
    emb[0, :] = 0.0

    # Optionally convert back to original dtype
    if args.keep_dtype and orig_dtype != np.float32:
        emb = emb.astype(orig_dtype)
        print(f"  Converted back to {orig_dtype}")

    # Save
    np.save(args.output, emb)
    print(f"✅ Saved L2-normalized embeddings to: {args.output}")
    print(f"   Shape: {emb.shape}, dtype: {emb.dtype}")


if __name__ == "__main__":
    main()

