#!/usr/bin/env bash
#
# Quick Multi-View Diagnosis
#
# Fast diagnostic script that runs essential checks only.
# Use this for quick feedback before running full diagnostic suite.
#
# Usage:
#   bash tools/quick_diagnosis.sh

#set -euo pipefail

cd /home/charlie/project/RecBole
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo ""
echo "================================================================================"
echo "QUICK MULTI-VIEW DIAGNOSIS"
echo "================================================================================"
echo ""

# Quick check function
quick_check() {
  local dataset=$1
  local emb_dir="/home/charlie/project/RecBole/dataset/$dataset"
  
  echo "Checking $dataset..."
  
  # Check if embeddings exist
  if [ ! -f "$emb_dir/item_text_emb.qwen3.base.npy" ]; then
    echo "  ✗ Missing: item_text_emb.qwen3.base.npy"
    return 1
  else
    echo "  ✓ Single LLM embedding found"
  fi
  
  if [ ! -d "$emb_dir/qwen3_4views" ]; then
    echo "  ✗ Missing: qwen3_4views directory"
    return 1
  else
    echo "  ✓ Multi-view directory found"
    
    # Check individual views
    for view_idx in 0 1 2 3; do
      if [ ! -f "$emb_dir/qwen3_4views/view_${view_idx}.npy" ]; then
        echo "    ✗ Missing view: view_${view_idx}.npy"
      else
        echo "    ✓ View found: view_${view_idx}.npy"
      fi
    done
  fi
  
  echo ""
}

# Check both datasets
quick_check "Amazon_Beauty"
quick_check "Amazon_Toys_and_Games"

# Quick embedding comparison
echo "================================================================================"
echo "QUICK EMBEDDING STATISTICS"
echo "================================================================================"
echo ""

python3 << 'PYTHON_SCRIPT'
import numpy as np
from pathlib import Path

def quick_stats(dataset_name):
    base_dir = Path("/home/charlie/project/RecBole/dataset") / dataset_name
    
    # Load LLM embedding (qwen3 base)
    llm_path = base_dir / "item_text_emb.qwen3.base.npy"
    if not llm_path.exists():
        print(f"⚠ {dataset_name}: LLM embedding not found")
        return
    
    llm_emb = np.load(llm_path)
    
    # Convert float16 to float32 for numerical stability
    if llm_emb.dtype == np.float16:
        llm_emb = llm_emb.astype(np.float32)
    
    # Quick statistics
    print(f"{dataset_name}:")
    print(f"  Shape: {llm_emb.shape}")
    print(f"  Dtype: {llm_emb.dtype}")
    print(f"  Mean: {llm_emb.mean():.6f}")
    print(f"  Std:  {llm_emb.std():.6f}")
    print(f"  Norm: {np.mean(np.linalg.norm(llm_emb, axis=1)):.4f}")
    
    # Sample inter-item similarity
    from sklearn.metrics.pairwise import cosine_similarity
    n_samples = min(len(llm_emb), 200)
    sampled = llm_emb[:n_samples]
    cos_sim = cosine_similarity(sampled)
    
    # Remove diagonal
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    similarities = cos_sim[mask]
    
    # Filter nan/inf
    similarities = similarities[~np.isnan(similarities) & ~np.isinf(similarities)]
    
    if len(similarities) > 0:
        print(f"  Inter-item cosine: {similarities.mean():.4f} ± {similarities.std():.4f}")
        print(f"  Max similarity: {similarities.max():.4f}")
    else:
        print(f"  Inter-item cosine: Could not compute (numerical issues)")
    print()

quick_stats("Amazon_Beauty")
quick_stats("Amazon_Toys_and_Games")

print("="*80)
print("INTERPRETATION:")
print("="*80)
print()
print("If Toys shows:")
print("  • Higher inter-item cosine → items are more similar")
print("    → Multi-view may struggle to find discriminative features")
print("    → Action: Increase temperature, alignment_weight")
print()
print("  • Lower std → less feature variance")
print("    → Multi-view views might be redundant")
print("    → Action: Use more diverse prompts, increase SENet ratio")
print()
print("  • Similar statistics to Beauty → problem likely in training")
print("    → Action: Check overfitting, adjust regularization")
print()
print("="*80)
print()
print("For full analysis, run: bash tools/diagnose_multiview.sh")
print("="*80)
print()

PYTHON_SCRIPT

echo ""
