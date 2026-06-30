#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Test Script for RecBole Models

This script loads a trained model checkpoint and runs evaluation with 
specified metrics, without requiring the original training configuration.

Usage:
    python scripts/standalone_test.py --model_file saved/model.pth [--metrics "Recall,NDCG,MRR"]
    
Example:
    # Test with new stratified metrics
    python scripts/standalone_test.py \
        --model_file saved/phase_runs_stratified/SASRecAlign-Amazon_Beauty-xxx.pth \
        --metrics "Recall,MRR,NDCG,Hit,StratifiedRecall,StratifiedNDCG,StratifiedMRR,StratifiedHit,ItemPopularityStats"
"""

import argparse
import os
import sys
from collections import OrderedDict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
from logging import getLogger

if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = np.int_
if not hasattr(np, "float"):
    np.float = np.float64

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.evaluator import Evaluator, Collector


def load_model_for_test(model_file, override_metrics=None, override_topk=None, device=None):
    """
    Load a saved model checkpoint with optional metric overrides.
    
    Args:
        model_file: Path to the saved .pth checkpoint
        override_metrics: List of metrics to use (overrides checkpoint config)
        override_topk: List of topk values (e.g., [5, 10, 20])
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        config, model, dataset, train_data, valid_data, test_data
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_file}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    config = checkpoint["config"]

    # Some older checkpoints rely on model/property defaults that are not
    # materialized in the serialized config. Fill the common init default so
    # eval-only reconstruction does not fail before loading weights.
    try:
        init_range = config["initializer_range"]
    except Exception:
        init_range = None
    if init_range is None:
        config["initializer_range"] = 0.02
    
    # Override metrics if specified
    if override_metrics:
        print(f"Overriding metrics: {override_metrics}")
        config["metrics"] = override_metrics
    
    if override_topk:
        print(f"Overriding topk: {override_topk}")
        config["topk"] = override_topk
    
    if device:
        config["device"] = device
    
    # Initialize
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    
    logger.info(f"Config metrics: {config['metrics']}")
    logger.info(f"Config topk: {config['topk']}")
    
    # Create dataset and dataloaders
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Create model and load weights
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    
    logger.info(f"Model loaded successfully: {config['model']}")
    
    return config, model, dataset, train_data, valid_data, test_data


def run_evaluation(config, model, train_data, test_data, show_progress=True):
    """
    Run evaluation on test data using the specified metrics.
    
    Args:
        config: RecBole config
        model: Loaded model
        train_data: Training dataloader (needed for item_counter)
        test_data: Test dataloader
        show_progress: Whether to show progress bar
    
    Returns:
        OrderedDict of evaluation results
    """
    logger = getLogger()
    
    # Create evaluator and collector with updated metrics
    evaluator = Evaluator(config)
    eval_collector = Collector(config)
    
    # Collect data statistics (needed for stratified metrics)
    eval_collector.data_collect(train_data)
    
    # Get trainer for evaluation
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    
    # Override trainer's evaluator and collector
    trainer.evaluator = evaluator
    trainer.eval_collector = eval_collector
    
    # Run evaluation
    logger.info(f"\n{'='*60}")
    logger.info("Running evaluation on test set...")
    logger.info(f"{'='*60}\n")
    
    result = trainer.evaluate(
        test_data, 
        load_best_model=False,  # Model already loaded
        show_progress=show_progress
    )
    
    return result


def format_results(result, group_by_metric=True):
    """Format evaluation results for display."""
    output = []
    output.append("\n" + "="*70)
    output.append("EVALUATION RESULTS")
    output.append("="*70)
    
    if group_by_metric:
        # Group by metric type
        standard_metrics = {}
        stratified_metrics = {}
        coverage_metrics = {}
        
        for key, value in result.items():
            if key.startswith(('Recall_', 'NDCG_', 'MRR_', 'Hit_')) and '@' in key:
                # Stratified metrics like Recall_new@10
                stratified_metrics[key] = value
            elif key.startswith('Coverage_'):
                coverage_metrics[key] = value
            else:
                standard_metrics[key] = value
        
        # Print standard metrics
        output.append("\n📊 Standard Metrics:")
        output.append("-" * 50)
        for key, value in sorted(standard_metrics.items()):
            output.append(f"  {key:25s}: {value:.4f}")
        
        # Print stratified metrics
        if stratified_metrics:
            output.append("\n📈 Stratified Metrics (by ground truth item stratum):")
            output.append("-" * 50)
            
            # Group by stratum
            for stratum in ['new', 'few', 'frequent']:
                stratum_results = {k: v for k, v in stratified_metrics.items() if f'_{stratum}@' in k}
                if stratum_results:
                    output.append(f"  [{stratum.upper()}] (training interactions: " + 
                                  ("1-2" if stratum == 'new' else "3-9" if stratum == 'few' else "10+") + ")")
                    for key, value in sorted(stratum_results.items()):
                        output.append(f"    {key:23s}: {value:.4f}")
        
        # Print coverage metrics
        if coverage_metrics:
            output.append("\n🎯 Coverage Metrics (recommendation diversity):")
            output.append("-" * 50)
            for key, value in sorted(coverage_metrics.items()):
                output.append(f"  {key:25s}: {value:.4f}")
    else:
        # Simple list
        for key, value in result.items():
            output.append(f"  {key:30s}: {value:.4f}")
    
    output.append("\n" + "="*70 + "\n")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Standalone test script for RecBole models")
    parser.add_argument(
        "--model_file", 
        type=str, 
        required=True,
        help="Path to saved model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="Recall,MRR,NDCG,Hit,Precision,StratifiedRecall,StratifiedNDCG,StratifiedMRR,StratifiedHit,ItemPopularityStats",
        help="Comma-separated list of metrics to evaluate"
    )
    parser.add_argument(
        "--topk",
        type=str,
        default=None,
        help="Comma-separated list of topk values (e.g., '5,10,20')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save results to file"
    )
    
    args = parser.parse_args()
    
    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(",")]
    
    # Parse topk
    topk = None
    if args.topk:
        topk = [int(k.strip()) for k in args.topk.split(",")]
    
    # Load model
    config, model, dataset, train_data, valid_data, test_data = load_model_for_test(
        args.model_file,
        override_metrics=metrics,
        override_topk=topk,
        device=args.device
    )
    
    # Run evaluation
    result = run_evaluation(
        config, model, train_data, test_data,
        show_progress=not args.no_progress
    )
    
    # Format and print results
    formatted = format_results(result)
    print(formatted)
    
    # Optionally save to file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(formatted)
            f.write("\n\nRaw results:\n")
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
        print(f"Results saved to: {args.output_file}")
    
    return result


if __name__ == "__main__":
    main()

