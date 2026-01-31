#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer Boost Sweep - 加载 checkpoint 并在 validation 上扫描 infer_boost

用途：
  - 加载已训练好的 checkpoint
  - 跳过训练，直接在 validation set 上进行评估
  - 网格搜索 infer_boost 参数
  - 找到最优的 infer_boost 后，在 test set 上报告最终结果

使用方法：
  python scripts/infer_boost_sweep.py \
    --model SASRecAlignMultiViewV3 \
    --dataset Amazon_Beauty \
    --config_files "sasrec_align_multi_view_v3_stratified.yaml" \
    --checkpoint "/path/to/checkpoint.pth" \
    --infer_boost_grid "0.0,0.5,0.8,1.0,1.2,1.5,2.0" \
    --valid_metric "MRR@10"

设计思路：
  凡是用了文本的模型（TF-IDF、TF-IDF+LLM、MV-Align 这些），都允许在 validation
  （最好是 full-ranking 的 val）上调一个 infer_boost(γ)，然后固定 γ* 去跑 test。
  ID-only 当然就是 γ=0。这样主表就不会再出现有的模型带旋钮，有的没带的不公平。
"""

import argparse
import ast
import json
import os
import sys
from datetime import datetime
from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    init_logger,
    init_seed,
    get_model,
    get_trainer,
    set_color,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer Boost Sweep on Validation Set")
    
    # Model and dataset
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
    parser.add_argument("--config_files", type=str, default=None, help="Config yaml files (space-separated)")
    parser.add_argument("--config_dict", type=str, default=None, help="JSON/Python dict to override config")
    
    # Checkpoint
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to checkpoint file")
    
    # GPU
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    
    # Infer boost grid
    parser.add_argument(
        "--infer_boost_grid", 
        type=str, 
        default="0.0,0.5,0.8,1.0,1.2,1.5,2.0",
        help="Comma-separated infer_boost values to sweep"
    )
    
    # Fine-grained sweep
    parser.add_argument(
        "--fine_sweep_range",
        type=float,
        default=0.2,
        help="Range around best coarse value for fine sweep (e.g., 0.2 means ±0.2)"
    )
    parser.add_argument(
        "--fine_sweep_step",
        type=float,
        default=0.1,
        help="Step size for fine sweep"
    )
    parser.add_argument(
        "--skip_fine_sweep",
        action="store_true",
        help="Skip fine-grained sweep, only do coarse sweep"
    )
    
    # Validation metric
    parser.add_argument("--valid_metric", type=str, default="MRR@10", help="Metric for selection")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./run_metrics/infer_boost_sweep", help="Output directory")
    parser.add_argument("--variant_label", type=str, default=None, help="Variant label for logging")
    
    # Seed
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    
    return parser.parse_args()


def split_config_files(config_files):
    if not config_files:
        return None
    return config_files.strip().split(" ")


def load_checkpoint_and_model(config, checkpoint_path, device):
    """Load model and restore weights from checkpoint."""
    logger = getLogger()
    
    # Create dataset
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # Create dataloaders
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Create model
    model = get_model(config["model"])(config, train_data._dataset).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    # Load other parameters if available
    if "other_parameter" in ckpt and hasattr(model, "load_other_parameter"):
        model.load_other_parameter(ckpt.get("other_parameter"))
    
    logger.info(set_color("Checkpoint loaded successfully", "green"))
    
    return model, dataset, train_data, valid_data, test_data


def evaluate_with_infer_boost(model, trainer, data, infer_boost_value, logger):
    """Set infer_boost and evaluate on given data."""
    # Set infer_boost on model
    if hasattr(model, "infer_boost"):
        model.infer_boost = infer_boost_value
    else:
        logger.warning(f"Model does not have infer_boost attribute, skipping...")
        return None
    
    logger.info(f"  Evaluating with infer_boost={infer_boost_value:.2f}")
    
    # Evaluate (no training, just inference)
    result = trainer.evaluate(data, load_best_model=False, show_progress=False)
    
    return result


def lookup_metric(metric_name, result_dict):
    """Lookup metric value from result dict (case-insensitive)."""
    if not result_dict:
        return None
    metric_name_lower = metric_name.lower()
    for key, value in result_dict.items():
        if key.lower() == metric_name_lower:
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
    return None


def format_results_table(results, metric_name):
    """Format results as a table string."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{'infer_boost':>12} | {metric_name:>15} | {'Status':>10}")
    lines.append("-" * 60)
    
    best_value = max(r["metric_value"] for r in results if r["metric_value"] is not None)
    
    for r in results:
        value = r["metric_value"]
        if value is not None:
            status = "★ BEST" if value == best_value else ""
            lines.append(f"{r['infer_boost']:>12.2f} | {value:>15.6f} | {status:>10}")
        else:
            lines.append(f"{r['infer_boost']:>12.2f} | {'N/A':>15} | ")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    args = parse_args()
    
    # Parse infer_boost grid
    try:
        infer_boost_grid = [float(x.strip()) for x in args.infer_boost_grid.split(",") if x.strip()]
    except ValueError as e:
        print(f"Error parsing infer_boost_grid: {e}")
        sys.exit(1)
    
    if len(infer_boost_grid) == 0:
        print("Error: infer_boost_grid is empty")
        sys.exit(1)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Parse config_dict
    user_config_dict = {}
    if args.config_dict:
        try:
            user_config_dict = ast.literal_eval(args.config_dict)
        except Exception as e:
            print(f"Warning: Failed to parse config_dict: {e}")
    
    # Add gpu_id to config
    user_config_dict["gpu_id"] = args.gpu_id
    
    # Build config
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=split_config_files(args.config_files),
        config_dict=user_config_dict,
    )
    
    # Initialize
    init_seed(args.seed, config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    
    logger.info("=" * 60)
    logger.info(set_color("Infer Boost Sweep", "yellow"))
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Coarse Grid: {infer_boost_grid}")
    logger.info(f"Valid Metric: {args.valid_metric}")
    if args.variant_label:
        logger.info(f"Variant: {args.variant_label}")
    logger.info("=" * 60)
    
    device = config["device"]
    
    # Load model and data
    model, dataset, train_data, valid_data, test_data = load_checkpoint_and_model(
        config, args.checkpoint, device
    )
    
    # Create trainer (for evaluation only)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    
    # ========== Phase 1: Coarse sweep on validation set ==========
    logger.info("")
    logger.info(set_color("[Phase 1] Coarse Sweep on Validation Set", "cyan"))
    logger.info("-" * 60)
    
    coarse_results = []
    for boost_val in infer_boost_grid:
        result = evaluate_with_infer_boost(model, trainer, valid_data, boost_val, logger)
        metric_value = lookup_metric(args.valid_metric, result)
        coarse_results.append({
            "infer_boost": boost_val,
            "result": result,
            "metric_value": metric_value,
        })
        if metric_value is not None:
            logger.info(f"    {args.valid_metric} = {metric_value:.6f}")
    
    # Find best coarse value
    valid_results = [r for r in coarse_results if r["metric_value"] is not None]
    if not valid_results:
        logger.error("No valid results from coarse sweep!")
        sys.exit(1)
    
    best_coarse = max(valid_results, key=lambda r: r["metric_value"])
    logger.info("")
    logger.info(f"Best coarse: infer_boost={best_coarse['infer_boost']:.2f}, {args.valid_metric}={best_coarse['metric_value']:.6f}")
    
    # ========== Phase 2: Fine sweep around best (optional) ==========
    fine_results = []
    if not args.skip_fine_sweep:
        logger.info("")
        logger.info(set_color("[Phase 2] Fine Sweep around Best Value", "cyan"))
        logger.info("-" * 60)
        
        # Generate fine grid around best coarse value
        best_val = best_coarse["infer_boost"]
        fine_min = max(0.0, best_val - args.fine_sweep_range)
        fine_max = best_val + args.fine_sweep_range
        
        fine_grid = []
        current = fine_min
        while current <= fine_max + 1e-6:
            # Skip values already in coarse grid
            if not any(abs(current - c) < 1e-6 for c in infer_boost_grid):
                fine_grid.append(round(current, 2))
            current += args.fine_sweep_step
        
        if fine_grid:
            logger.info(f"Fine grid: {fine_grid}")
            for boost_val in fine_grid:
                result = evaluate_with_infer_boost(model, trainer, valid_data, boost_val, logger)
                metric_value = lookup_metric(args.valid_metric, result)
                fine_results.append({
                    "infer_boost": boost_val,
                    "result": result,
                    "metric_value": metric_value,
                })
                if metric_value is not None:
                    logger.info(f"    {args.valid_metric} = {metric_value:.6f}")
        else:
            logger.info("No additional fine grid points needed.")
    
    # ========== Combine and find overall best ==========
    all_results = coarse_results + fine_results
    valid_all = [r for r in all_results if r["metric_value"] is not None]
    best_overall = max(valid_all, key=lambda r: r["metric_value"])
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(set_color("[Summary] All Sweep Results", "yellow"))
    logger.info(format_results_table(all_results, args.valid_metric))
    
    logger.info("")
    logger.info(set_color(f"[Best] infer_boost = {best_overall['infer_boost']:.2f}", "green"))
    logger.info(set_color(f"[Best] Validation {args.valid_metric} = {best_overall['metric_value']:.6f}", "green"))
    
    # ========== Phase 3: Evaluate on Test Set with Best infer_boost ==========
    logger.info("")
    logger.info(set_color("[Phase 3] Test Set Evaluation with Best infer_boost", "cyan"))
    logger.info("-" * 60)
    
    test_result = evaluate_with_infer_boost(
        model, trainer, test_data, best_overall["infer_boost"], logger
    )
    
    logger.info("")
    logger.info(set_color("[Test Results]", "yellow"))
    logger.info(json.dumps(test_result, indent=2, default=str))
    
    # ========== Save results ==========
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_str = args.variant_label.replace(" ", "_").replace(",", "_") if args.variant_label else args.model
    
    output_file = os.path.join(
        args.output_dir,
        f"{timestamp}_{variant_str}_infer_boost_sweep.json"
    )
    
    output_data = {
        "timestamp": timestamp,
        "model": args.model,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "variant_label": args.variant_label,
        "valid_metric": args.valid_metric,
        "coarse_grid": infer_boost_grid,
        "best_infer_boost": best_overall["infer_boost"],
        "best_valid_score": best_overall["metric_value"],
        "all_sweep_results": [
            {
                "infer_boost": r["infer_boost"],
                "valid_metric_value": r["metric_value"],
            }
            for r in all_results
        ],
        "test_result": test_result,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info("")
    
    # Print final summary for easy copy-paste
    logger.info("=" * 60)
    logger.info(set_color("[Final Summary]", "yellow"))
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Variant: {args.variant_label or 'N/A'}")
    logger.info(f"  Best infer_boost: {best_overall['infer_boost']:.2f}")
    logger.info(f"  Valid {args.valid_metric}: {best_overall['metric_value']:.6f}")
    test_metric = lookup_metric(args.valid_metric, test_result)
    if test_metric:
        logger.info(f"  Test {args.valid_metric}: {test_metric:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
