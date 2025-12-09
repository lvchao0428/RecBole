#!/usr/bin/env bash
#
# Check Diagnosis Setup
# 
# Verifies that all necessary files exist and tools are ready to run.
#
# Usage:
#   bash tools/check_diagnosis_setup.sh

set -euo pipefail

echo ""
echo "================================================================================"
echo "DIAGNOSIS SETUP CHECK"
echo "================================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Function to check file existence
check_file() {
  local file=$1
  local type=$2  # "required" or "optional"
  
  if [ -f "$file" ]; then
    echo "✓ $file"
    return 0
  else
    if [ "$type" = "required" ]; then
      echo "✗ $file (REQUIRED - MISSING)"
      ((ERRORS++))
      return 1
    else
      echo "⚠ $file (optional - missing)"
      ((WARNINGS++))
      return 0
    fi
  fi
}

# Function to check directory
check_dir() {
  local dir=$1
  local type=$2
  
  if [ -d "$dir" ]; then
    echo "✓ $dir/"
    return 0
  else
    if [ "$type" = "required" ]; then
      echo "✗ $dir/ (REQUIRED - MISSING)"
      ((ERRORS++))
      return 1
    else
      echo "⚠ $dir/ (optional - missing)"
      ((WARNINGS++))
      return 0
    fi
  fi
}

# Check diagnostic tools
echo "1. Checking Diagnostic Tools..."
echo "----------------------------------------"
check_file "tools/analyze_multiview_quality.py" "required"
check_file "tools/compare_datasets.py" "required"
check_file "tools/analyze_training_logs.py" "required"
check_file "tools/generate_tuning_suggestions.py" "required"
check_file "tools/diagnose_multiview.sh" "required"
check_file "tools/quick_diagnosis.sh" "required"
check_file "tools/MULTIVIEW_DIAGNOSTIC_GUIDE.md" "required"
check_file "tools/RUN_DIAGNOSIS.md" "required"
echo ""

# Check Python dependencies
echo "2. Checking Python Dependencies..."
echo "----------------------------------------"
python3 -c "import numpy; print('✓ numpy')" 2>/dev/null || { echo "✗ numpy (pip install numpy)"; ((ERRORS++)); }
python3 -c "import scipy; print('✓ scipy')" 2>/dev/null || { echo "✗ scipy (pip install scipy)"; ((ERRORS++)); }
python3 -c "import sklearn; print('✓ scikit-learn')" 2>/dev/null || { echo "✗ scikit-learn (pip install scikit-learn)"; ((ERRORS++)); }
python3 -c "import pandas; print('✓ pandas')" 2>/dev/null || { echo "✗ pandas (pip install pandas)"; ((ERRORS++)); }
python3 -c "import yaml; print('✓ pyyaml')" 2>/dev/null || { echo "✗ pyyaml (pip install pyyaml)"; ((ERRORS++)); }
echo ""

# Check Beauty dataset
echo "3. Checking Beauty Dataset Files..."
echo "----------------------------------------"
BEAUTY_DIR="dataset/Amazon_Beauty"
check_file "$BEAUTY_DIR/Amazon_Beauty.inter" "required"
check_file "$BEAUTY_DIR/Amazon_Beauty.item" "required"
check_file "$BEAUTY_DIR/item_text_emb.base.npy" "required"
check_file "$BEAUTY_DIR/item_text_emb.qwen3.base.npy" "required"
check_dir "$BEAUTY_DIR/qwen3_4views" "required"

if [ -d "$BEAUTY_DIR/qwen3_4views" ]; then
  check_file "$BEAUTY_DIR/qwen3_4views/view_0.npy" "required"
  check_file "$BEAUTY_DIR/qwen3_4views/view_1.npy" "required"
  check_file "$BEAUTY_DIR/qwen3_4views/view_2.npy" "required"
  check_file "$BEAUTY_DIR/qwen3_4views/view_3.npy" "required"
  check_file "$BEAUTY_DIR/qwen3_4views/views.json" "required"
fi
echo ""

# Check Toys dataset
echo "4. Checking Toys Dataset Files..."
echo "----------------------------------------"
TOYS_DIR="dataset/Amazon_Toys_and_Games"
check_file "$TOYS_DIR/Amazon_Toys_and_Games.inter" "required"
check_file "$TOYS_DIR/Amazon_Toys_and_Games.item" "required"
check_file "$TOYS_DIR/item_text_emb.base.npy" "required"
check_file "$TOYS_DIR/item_text_emb.qwen3.base.npy" "required"
check_dir "$TOYS_DIR/qwen3_4views" "required"

if [ -d "$TOYS_DIR/qwen3_4views" ]; then
  check_file "$TOYS_DIR/qwen3_4views/view_0.npy" "required"
  check_file "$TOYS_DIR/qwen3_4views/view_1.npy" "required"
  check_file "$TOYS_DIR/qwen3_4views/view_2.npy" "required"
  check_file "$TOYS_DIR/qwen3_4views/view_3.npy" "required"
  check_file "$TOYS_DIR/qwen3_4views/views.json" "required"
fi
echo ""

# Check training logs (optional)
echo "5. Checking Training Logs (optional)..."
echo "----------------------------------------"
check_dir "saved/phase_runs_multiview_4views" "optional"
check_dir "saved/phase_runs_multiview_4views_toys" "optional"
echo ""

# Check configuration files
echo "6. Checking Configuration Files..."
echo "----------------------------------------"
check_file "sasrec_align_multi_view.yaml" "required"
check_file "sasrec_align_multi_view_toys.yaml" "required"
check_file "sasrec_align_qwen3.yaml" "required"
check_file "sasrec_align_toys_qwen3.yaml" "required"
check_file "two_phase_run_multiview_split.sh" "required"
check_file "two_phase_run_multiview_split_toys.sh" "required"
check_file "two_phase_run_tfidf_llm.sh" "required"
check_file "two_phase_run_tfidf_llm_toys.sh" "required"
echo ""

# Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo "✓ All checks passed! You're ready to run diagnostics."
  echo ""
  echo "Next steps:"
  echo "  1. Quick check:  bash tools/quick_diagnosis.sh"
  echo "  2. Full diagnosis: bash tools/diagnose_multiview.sh"
  echo ""
  exit 0
elif [ $ERRORS -eq 0 ]; then
  echo "✓ All required files found ($WARNINGS warnings)"
  echo ""
  echo "Warnings are for optional files (e.g., training logs)."
  echo "You can proceed with diagnostics."
  echo ""
  echo "Next steps:"
  echo "  1. Quick check:  bash tools/quick_diagnosis.sh"
  echo "  2. Full diagnosis: bash tools/diagnose_multiview.sh"
  echo ""
  exit 0
else
  echo "✗ Found $ERRORS error(s) and $WARNINGS warning(s)"
  echo ""
  echo "Please fix the errors above before running diagnostics."
  echo ""
  
  if [ $ERRORS -gt 0 ]; then
    echo "Common fixes:"
    echo "  - Missing Python packages: pip install numpy scipy scikit-learn pandas pyyaml"
    echo "  - Missing embeddings: Run embedding generation scripts first"
    echo "  - Missing multi-view: Run tools/gen_multiview_4views_*.sh"
  fi
  echo ""
  exit 1
fi
