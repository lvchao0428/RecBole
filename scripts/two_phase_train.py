#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-phase training orchestrator for RecBole models (e.g., SASRec_Align).

Phase-A: warmup for text alignment/fusion with backbone frozen
Phase-B: joint fine-tuning with backbone unfrozen and grouped learning rates
"""
import argparse
import copy
import json
import os
from datetime import datetime
from logging import getLogger

import psutil
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.quick_start.quick_start import load_data_and_model  # optional utility
from recbole.utils import (
    init_logger,
    init_seed,
    get_model,
    get_trainer,
    set_color,
    get_flops,
    get_environment,
)


def _split_config_files(config_files: str | None):
    return config_files.strip().split(" ") if config_files else None


def _build_and_prepare(config: Config):
    """Build logger, dataset, dataloaders, model, trainer for a single phase."""
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    return logger, dataset, train_data, valid_data, test_data, model, trainer


def _cfg_val(cfg: Config, key: str, default=None):
    try:
        return cfg[key]
    except KeyError:
        return default


def _log_and_dump_summary(config: Config, model, res_dict: dict, label: str):
    """Log grouped hyperparameters, resource usage, and dump to file."""
    hyper_groups = {
        "training": ["epochs", "train_batch_size", "learning_rate", "eval_step", "stopping_step", "weight_decay", "label_smoothing"],
        "lr_groups": ["lr_text_head", "lr_dnn_cross", "lr_backbone"],
        "model": ["n_layers", "n_heads", "hidden_size", "inner_size", "hidden_dropout_prob", "attn_dropout_prob", "cross_dropout_prob", "token_dropout_prob", "text_gate_reg_l2", "text_gate_reg_entropy"],
        "text_align": ["alignment_weight", "temperature", "text_weight", "text_tail_threshold", "text_gate_init", "use_cross", "use_seq_text_cross", "use_llm", "use_align", "disable_text_feature", "fuse_text_feature"],
    }
    summary = {}
    for group, keys in hyper_groups.items():
        summary[group] = {}
        for key in keys:
            val = _cfg_val(config, key, None)
            if val is not None:
                summary[group][key] = val

    # Resource usage
    proc = psutil.Process(os.getpid())
    cpu_mem_mb = proc.memory_info().rss / (1024 ** 2)
    gpu_info = {}
    if torch.cuda.is_available():
        device = torch.device(config["device"]) if "device" in config else torch.device("cuda")
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        gpu_info = {
            "memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
            "memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        }
    else:
        gpu_info = {"memory_allocated_mb": 0.0, "memory_reserved_mb": 0.0}

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    payload = {
        "label": label,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model": config["model"],
        "dataset": config["dataset"],
        "config_groups": summary,
        "resource": {
            "cpu_memory_mb": cpu_mem_mb,
            "gpu": gpu_info,
            "trainable_params": trainable_params,
            "total_params": total_params,
        },
        "results": res_dict,
    }

    logger = getLogger()
    logger.info(set_color("[Run Summary JSON]", "blue") + f" {json.dumps(payload, ensure_ascii=False)}")

    output_dir = os.path.join(os.getcwd(), "run_metrics")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{payload['timestamp']}_{config['model']}_{label}.txt"
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def _train_and_eval_phase(logger, trainer, train_data, valid_data, test_data, saved=True):
    """Run one training phase and return results and saved model path."""
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=trainer.config["show_progress"]
    )
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=trainer.config["show_progress"]
    )
    env_tb = get_environment(trainer.config)
    logger.info("The running environment of this training is as follows:\n" + env_tb.draw())
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    return {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "saved_model_file": trainer.saved_model_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Two-phase training orchestrator")
    parser.add_argument("--model", "-m", type=str, required=True, help="model name, e.g., SASRec_Align")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="dataset name")
    parser.add_argument("--config_files", type=str, default=None, help="space-separated config yaml files")

    # Phase controls
    parser.add_argument("--phase_a_epochs", type=int, default=8, help="epochs for Phase-A")
    parser.add_argument("--phase_b_epochs", type=int, default=40, help="epochs for Phase-B")
    parser.add_argument("--phase_a_alignment_weight", type=float, default=None, help="override Phase-A alignment_weight (disabled when using grid)")
    parser.add_argument("--phase_b_alignment_weight", type=float, default=None, help="override Phase-B alignment_weight")
    parser.add_argument("--phase_a_text_gate_reg_l2", type=float, default=None, help="override Phase-A text_gate_reg_l2")
    parser.add_argument("--phase_b_text_gate_reg_l2", type=float, default=None, help="override Phase-B text_gate_reg_l2")
    parser.add_argument("--phase_b_text_weight", type=float, default=None, help="override Phase-B text_weight")

    # LR grouping (used by models that implement get_optimizer_grouped_parameters)
    parser.add_argument("--lr_text_head", type=float, default=None, help="LR for text projection head")
    parser.add_argument("--lr_dnn_cross", type=float, default=None, help="LR for fusion DNN/Cross")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1, help="lr_backbone = lr_text_head * scale")

    # Misc
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="override checkpoint_dir")
    parser.add_argument("--seed", type=int, default=None, help="override seed")
    parser.add_argument("--save", action="store_true", help="save checkpoints")
    # Manual switching
    parser.add_argument("--only_phase_a", action="store_true", help="run Phase-A only")
    parser.add_argument("--only_phase_b", action="store_true", help="run Phase-B only")
    parser.add_argument("--resume_from", type=str, default=None, help="resume Phase-B from Phase-A checkpoint path")
    # Phase-A grid & gating
    parser.add_argument("--phase_a_grid", action="store_true", help="run Phase-A grid over alignment_weight and temperature")
    parser.add_argument("--align_grid", type=str, default="0.05,0.1,0.2", help="comma-separated alignment_weight grid")
    parser.add_argument("--tau_grid", type=str, default="0.05,0.07", help="comma-separated temperature grid")
    parser.add_argument("--phase_a_eval_step", type=int, default=2, help="validation interval (epochs) in Phase-A")
    parser.add_argument("--phase_a_valid_metric", type=str, default="NDCG@10", help="validation metric used for Phase-A gating")
    parser.add_argument("--ndcg_baseline", type=float, default=None, help="strong baseline NDCG@10 for Phase-A gating")
    parser.add_argument("--ndcg_gain_threshold", type=float, default=0.01, help="required relative gain over baseline, e.g., 0.01 for +1%")
    parser.add_argument("--phase_a_auto_to_b", action="store_true", help="if pass condition met, continue to Phase-B automatically")
    parser.add_argument("--phase_a_require_pass_for_b", action="store_true", help="only enter Phase-B if pass condition is met")
    # Optional ID-only burn-in before Phase-A
    parser.add_argument("--backbone_burnin_epochs", type=int, default=0, help="ID-only burn-in epochs before Phase-A")
    parser.add_argument("--burnin_eval_step", type=int, default=2, help="validation interval (epochs) in burn-in stage")
    args, _ = parser.parse_known_args()

    if args.only_phase_a and args.only_phase_b:
        raise ValueError("only_phase_a and only_phase_b cannot be used together.")

    # Phase-A config
    res_a = None
    phase_a_ckpt = None
    phase_a_passed = False
    burnin_ckpt = None

    # Optional: short ID-only burn-in to stabilize backbone before freezing in Phase-A
    if not args.only_phase_b and args.backbone_burnin_epochs and args.backbone_burnin_epochs > 0:
        burnin_dict = {
            "freeze_backbone": False,
            "epochs": int(args.backbone_burnin_epochs),
            "eval_step": int(args.burnin_eval_step),
            "valid_metric": args.phase_a_valid_metric,
            # Disable text/align in burn-in
            "disable_text_feature": True,
            "use_align": False,
            "fuse_text_feature": False,
            "use_llm": False,
        }
        if args.checkpoint_dir:
            burnin_dict["checkpoint_dir"] = args.checkpoint_dir
        if args.seed is not None:
            burnin_dict["seed"] = int(args.seed)

        config_burn = Config(
            model=args.model,
            dataset=args.dataset,
            config_file_list=_split_config_files(args.config_files),
            config_dict=burnin_dict,
        )
        logger_burn, dataset_burn, train_burn, valid_burn, test_burn, model_burn, trainer_burn = _build_and_prepare(config_burn)
        logger_burn.info(set_color("[Burn-in] ID-only warmup", "cyan") + f": epochs={burnin_dict['epochs']}, eval_step={burnin_dict['eval_step']}")
        res_burn = _train_and_eval_phase(logger_burn, trainer_burn, train_burn, valid_burn, test_burn, saved=True)
        burnin_ckpt = res_burn.get("saved_model_file")
        if burnin_ckpt:
            logger_burn.info(set_color("[Burn-in] Saved checkpoint", "green") + f": {burnin_ckpt}")
    if not args.only_phase_b:
        if args.phase_a_grid:
            # Parse grids
            try:
                align_list = [float(x) for x in args.align_grid.split(",") if x.strip() != ""]
                tau_list = [float(x) for x in args.tau_grid.split(",") if x.strip() != ""]
            except Exception:
                raise ValueError("Failed to parse align_grid or tau_grid; use comma-separated floats, e.g., '0.05,0.1,0.2'")
            if len(align_list) == 0 or len(tau_list) == 0:
                raise ValueError("Empty grid for alignment_weight or temperature.")
            if args.ndcg_baseline is None:
                getLogger().warning("ndcg_baseline not provided; pass gating will be disabled. "
                                    "If you need early finish on reaching gain threshold, please specify --ndcg_baseline as your strong baseline.")
            ndcg_target = None if args.ndcg_baseline is None else args.ndcg_baseline * (1.0 + float(args.ndcg_gain_threshold))

            best_tuple = None  # (ndcg10, alignment_weight, temperature, res, ckpt)
            for aw in align_list:
                for tau in tau_list:
                    phase_a_dict = {
                        "freeze_backbone": True,
                        "epochs": int(args.phase_a_epochs),
                        "eval_step": int(args.phase_a_eval_step),
                        "valid_metric": args.phase_a_valid_metric,
                        "alignment_weight": float(aw),
                        "temperature": float(tau),
                    }
                    if args.phase_a_text_gate_reg_l2 is not None:
                        phase_a_dict["text_gate_reg_l2"] = float(args.phase_a_text_gate_reg_l2)
                    if args.lr_text_head is not None:
                        phase_a_dict["lr_text_head"] = float(args.lr_text_head)
                    if args.lr_dnn_cross is not None:
                        phase_a_dict["lr_dnn_cross"] = float(args.lr_dnn_cross)
                    if args.checkpoint_dir:
                        phase_a_dict["checkpoint_dir"] = args.checkpoint_dir
                    if args.seed is not None:
                        phase_a_dict["seed"] = int(args.seed)

                    config_a = Config(
                        model=args.model,
                        dataset=args.dataset,
                        config_file_list=_split_config_files(args.config_files),
                        config_dict=phase_a_dict,
                    )
                    logger_a, dataset_a, train_a, valid_a, test_a, model_a, trainer_a = _build_and_prepare(config_a)
                    logger_a.info(set_color("[Phase-A:grid] setting", "cyan") + f": alignment_weight={aw}, temperature={tau}")
                    # Load burn-in checkpoint if available
                    if burnin_ckpt and os.path.exists(burnin_ckpt):
                        try:
                            ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"])
                        except Exception:
                            ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"], weights_only=False)
                        model_a.load_state_dict(ckpt_b["state_dict"], strict=False)
                        model_a.load_other_parameter(ckpt_b.get("other_parameter"))
                        logger_a.info(set_color("[Phase-A:grid] Loaded burn-in checkpoint (strict=False)", "green") + f": {burnin_ckpt}")
                    # Optional: early finish inside Phase-A when reaching ndcg_target
                    callback_fn = None
                    if ndcg_target is not None:
                        def _gate_cb(epoch_idx, valid_score, _trainer=trainer_a, _logger=logger_a, _target=ndcg_target):
                            try:
                                if valid_score >= _target:
                                    # Save current checkpoint and stop this phase immediately
                                    _trainer._save_checkpoint(epoch_idx, verbose=True)
                                    _logger.info(set_color("[Phase-A:grid] Early PASS gate", "green") + f": epoch={epoch_idx}, valid={valid_score:.6f} >= target={_target:.6f}")
                                    raise StopIteration  # break fit()
                            except StopIteration:
                                raise
                            except Exception as e:
                                _logger.warning(f"[Phase-A:grid] gate callback error ignored: {e}")
                        
                        # If ndcg_target is set, we must manually check early stopping because RecBole Trainer 
                        # doesn't support custom per-epoch callbacks easily in older versions.
                        # However, we can inject it via callback_fn if the trainer supports it.
                        # But standard RecBole trainer.fit() signature is:
                        # fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None)
                        # Let's ensure we pass it correctly.
                        callback_fn = _gate_cb
                    else:
                        callback_fn = None
                    # Run training phase with optional early gate
                    try:
                        # Verify freeze status before training starts
                        if phase_a_dict.get("freeze_backbone", False):
                            logger_a.info(set_color("=== [Phase-A] Parameter Freeze Check ===", "red"))
                            frozen_params = []
                            active_params = []
                            for name, param in model_a.named_parameters():
                                if not param.requires_grad:
                                    frozen_params.append(name)
                                else:
                                    active_params.append(name)
                            
                            # Heuristic check for backbone parts
                            backbone_keywords = ["item_embedding", "position_embedding", "trm_encoder", "LayerNorm"]
                            backbone_frozen = all(any(k in name for k in backbone_keywords) for name in frozen_params if any(k in name for k in backbone_keywords))
                            
                            if backbone_frozen:
                                logger_a.info(set_color(f"SUCCESS: Backbone parameters are FROZEN. Total frozen: {len(frozen_params)}", "green"))
                                logger_a.info(f"Frozen examples: {frozen_params[:3]} ...")
                            else:
                                logger_a.warning(set_color("WARNING: Backbone parameters might NOT be fully frozen!", "red"))
                            
                            logger_a.info(set_color(f"Active parameters (training targets): {len(active_params)}", "yellow"))
                            logger_a.info(f"Active examples: {active_params[:5]} ...")
                            logger_a.info(set_color("========================================", "red"))

                        best_valid_score, best_valid_result = trainer_a.fit(
                            train_a, valid_a, saved=args.save, show_progress=trainer_a.config["show_progress"], callback_fn=callback_fn
                        )
                    except StopIteration:
                        # retrieve best fields from trainer after early stop
                        best_valid_score = trainer_a.best_valid_score
                        best_valid_result = trainer_a.best_valid_result
                        
                        # CRITICAL FIX: Save checkpoint immediately upon early stop if callback didn't
                        # (Though callback usually saves it, trainer state might lag)
                        if args.save and not os.path.exists(trainer_a.saved_model_file):
                             trainer_a._save_checkpoint(trainer_a.start_epoch + trainer_a.cur_step * trainer_a.eval_step)
                    
                    # Evaluate on test (load best)
                    test_result = trainer_a.evaluate(test_a, load_best_model=args.save, show_progress=trainer_a.config["show_progress"])
                    res = {
                        "best_valid_score": best_valid_score,
                        "best_valid_result": best_valid_result,
                        "test_result": test_result,
                        "saved_model_file": trainer_a.saved_model_file,
                    }
                    _log_and_dump_summary(
                        config_a,
                        model_a,
                        {
                            "best_valid_result": res["best_valid_result"],
                            "test_result": res["test_result"],
                        },
                        f"Phase-A_aw{aw}_tau{tau}",
                    )
                    ckpt_path = res["saved_model_file"] if args.save else None
                    if ckpt_path:
                        logger_a.info(set_color("[Phase-A:grid] Saved checkpoint", "green") + f": {ckpt_path}")
                    # Extract NDCG@10
                    valid_dict = res.get("best_valid_result") or {}
                    ndcg10 = None
                    for key in ["NDCG@10", "ndcg@10", "NDCG@10(Avg)"]:
                        if key in valid_dict:
                            ndcg10 = float(valid_dict[key])
                            break
                    logger_a.info(set_color("[Phase-A:grid] NDCG@10", "yellow") + f": {ndcg10}")
                    if ndcg10 is not None:
                        if best_tuple is None or ndcg10 > best_tuple[0]:
                            best_tuple = (ndcg10, aw, tau, res, ckpt_path)
                        if ndcg_target is not None and ndcg10 >= ndcg_target:
                            phase_a_passed = True
                            phase_a_ckpt = ckpt_path
                            res_a = res
                            logger_a.info(set_color("[Phase-A:grid] PASS gate reached", "green") + f": ndcg@10={ndcg10:.6f} >= target={ndcg_target:.6f}")
                            break
                if phase_a_passed:
                    break
            if not phase_a_passed and best_tuple is not None:
                # choose the best combination anyway
                phase_a_ckpt = best_tuple[4]
                res_a = best_tuple[3]
                logger = getLogger()
                logger.info(set_color("[Phase-A:grid] Best combo", "blue") + f": ndcg@10={best_tuple[0]:.6f}, alignment_weight={best_tuple[1]}, temperature={best_tuple[2]}")
                if ndcg_target is not None:
                    logger.info(set_color("[Phase-A:grid] Gate not reached", "red") + f": best_ndcg@10={best_tuple[0]:.6f} < target={ndcg_target:.6f}")
        else:
            phase_a_dict = {
                "freeze_backbone": True,
                "epochs": int(args.phase_a_epochs),
                "eval_step": int(args.phase_a_eval_step),
                "valid_metric": args.phase_a_valid_metric,
            }
            if args.phase_a_alignment_weight is not None:
                phase_a_dict["alignment_weight"] = float(args.phase_a_alignment_weight)
            if args.phase_a_text_gate_reg_l2 is not None:
                phase_a_dict["text_gate_reg_l2"] = float(args.phase_a_text_gate_reg_l2)
            # Optional user overrides
            if args.lr_text_head is not None:
                phase_a_dict["lr_text_head"] = float(args.lr_text_head)
            if args.lr_dnn_cross is not None:
                phase_a_dict["lr_dnn_cross"] = float(args.lr_dnn_cross)
            if args.checkpoint_dir:
                phase_a_dict["checkpoint_dir"] = args.checkpoint_dir
            if args.seed is not None:
                phase_a_dict["seed"] = int(args.seed)

            config_a = Config(
                model=args.model,
                dataset=args.dataset,
                config_file_list=_split_config_files(args.config_files),
                config_dict=phase_a_dict,
            )
            logger_a, dataset_a, train_a, valid_a, test_a, model_a, trainer_a = _build_and_prepare(config_a)
            logger_a.info(set_color("[Phase-A] freeze_backbone", "cyan") + f": {config_a['freeze_backbone']}")
            if "lr_text_head" in config_a and "lr_dnn_cross" in config_a:
                logger_a.info(
                    set_color("[Phase-A] lr groups", "cyan")
                    + f": lr_text_head={config_a['lr_text_head']}, lr_dnn_cross={config_a['lr_dnn_cross']}"
                )
            # Load burn-in checkpoint if available
            if burnin_ckpt and os.path.exists(burnin_ckpt):
                try:
                    ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"])
                except Exception:
                    ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"], weights_only=False)
                model_a.load_state_dict(ckpt_b["state_dict"], strict=False)
                model_a.load_other_parameter(ckpt_b.get("other_parameter"))
                logger_a.info(set_color("[Phase-A] Loaded burn-in checkpoint (strict=False)", "green") + f": {burnin_ckpt}")
            # If baseline is provided, compute target for early gate
            ndcg_target = None if args.ndcg_baseline is None else args.ndcg_baseline * (1.0 + float(args.ndcg_gain_threshold))
            callback_fn = None
            if ndcg_target is not None:
                def _gate_cb(epoch_idx, valid_score, _trainer=trainer_a, _logger=logger_a, _target=ndcg_target):
                    try:
                        if valid_score >= _target:
                            _trainer._save_checkpoint(epoch_idx, verbose=True)
                            _logger.info(set_color("[Phase-A] Early PASS gate", "green") + f": epoch={epoch_idx}, valid={valid_score:.6f} >= target={_target:.6f}")
                            raise StopIteration
                    except StopIteration:
                        raise
                    except Exception as e:
                        _logger.warning(f"[Phase-A] gate callback error ignored: {e}")
            # Run training with optional early gate
            try:
                # Verify freeze status before training starts
                if phase_a_dict.get("freeze_backbone", False):
                    logger_a.info(set_color("=== [Phase-A] Parameter Freeze Check ===", "red"))
                    frozen_params = []
                    active_params = []
                    for name, param in model_a.named_parameters():
                        if not param.requires_grad:
                            frozen_params.append(name)
                        else:
                            active_params.append(name)
                    
                    backbone_keywords = ["item_embedding", "position_embedding", "trm_encoder", "LayerNorm"]
                    # Check if typical backbone layers appear in frozen list
                    backbone_frozen_count = sum(1 for name in frozen_params if any(k in name for k in backbone_keywords))
                    
                    if backbone_frozen_count > 0:
                        logger_a.info(set_color(f"SUCCESS: Found {backbone_frozen_count} FROZEN backbone parameters.", "green"))
                        logger_a.info(f"Frozen examples: {frozen_params[:3]} ...")
                    else:
                        logger_a.warning(set_color("WARNING: Backbone parameters might NOT be frozen!", "red"))
                    
                    logger_a.info(set_color(f"Active parameters (training targets): {len(active_params)}", "yellow"))
                    logger_a.info(f"Active examples: {active_params[:5]} ...")
                    logger_a.info(set_color("========================================", "red"))

                best_valid_score, best_valid_result = trainer_a.fit(
                    train_a, valid_a, saved=args.save, show_progress=trainer_a.config["show_progress"], callback_fn=callback_fn
                )
            except StopIteration:
                best_valid_score = trainer_a.best_valid_score
                best_valid_result = trainer_a.best_valid_result
                
                # CRITICAL FIX: Ensure checkpoint exists if early stopped
                if args.save and not os.path.exists(trainer_a.saved_model_file):
                     trainer_a._save_checkpoint(trainer_a.start_epoch + trainer_a.cur_step * trainer_a.eval_step)
                     
            test_result = trainer_a.evaluate(test_a, load_best_model=args.save, show_progress=trainer_a.config["show_progress"])
            res_a = {
                "best_valid_score": best_valid_score,
                "best_valid_result": best_valid_result,
                "test_result": test_result,
                "saved_model_file": trainer_a.saved_model_file,
            }
            _log_and_dump_summary(
                config_a,
                model_a,
                {
                    "best_valid_result": res_a["best_valid_result"],
                    "test_result": res_a["test_result"],
                },
                "Phase-A",
            )
            phase_a_ckpt = res_a["saved_model_file"] if args.save else None
            if phase_a_ckpt:
                logger_a.info(set_color("[Phase-A] Saved checkpoint", "green") + f": {phase_a_ckpt}")
            # Gate check when baseline provided
            if args.ndcg_baseline is not None:
                ndcg_target = args.ndcg_baseline * (1.0 + float(args.ndcg_gain_threshold))
                ndcg10 = None
                for key in ["NDCG@10", "ndcg@10", "NDCG@10(Avg)"]:
                    if key in (res_a.get("best_valid_result") or {}):
                        ndcg10 = float(res_a["best_valid_result"][key])
                        break
                if ndcg10 is not None and ndcg10 >= ndcg_target:
                    phase_a_passed = True
                    logger_a.info(set_color("[Phase-A] PASS gate reached", "green") + f": ndcg@10={ndcg10:.6f} >= target={ndcg_target:.6f}")

    # Phase-B config (unfreeze + grouped LR; lr_backbone = lr_text_head * scale)
    if not args.only_phase_a:
        # If Phase-A grid is used and require pass to enter B, honor the gate
        if args.phase_a_grid and args.phase_a_require_pass_for_b and not phase_a_passed:
            getLogger().warning("[Two-Phase] Phase-B is skipped because Phase-A gate was not reached (require_pass_for_b).")
            return
        # If only_phase_b is False but user doesn't want auto transition, they can call with only_phase_b separately.
        if args.phase_a_grid and not args.phase_a_auto_to_b and not args.only_phase_b and phase_a_ckpt:
            getLogger().info(set_color("[Two-Phase] Phase-A finished. Not auto-continuing to Phase-B (phase_a_auto_to_b is False).", "yellow"))
            return

        phase_b_dict = {
            "freeze_backbone": False,
            "epochs": int(args.phase_b_epochs),
        }
        if args.phase_b_alignment_weight is not None:
            phase_b_dict["alignment_weight"] = float(args.phase_b_alignment_weight)
        if args.phase_b_text_gate_reg_l2 is not None:
            phase_b_dict["text_gate_reg_l2"] = float(args.phase_b_text_gate_reg_l2)
        if args.phase_b_text_weight is not None:
            phase_b_dict["text_weight"] = float(args.phase_b_text_weight)
        # Carry over LR groups if provided; compute lr_backbone from lr_text_head when available
        if args.lr_text_head is not None:
            phase_b_dict["lr_text_head"] = float(args.lr_text_head)
            phase_b_dict["lr_backbone"] = float(args.lr_text_head) * float(args.backbone_lr_scale)
        if args.lr_dnn_cross is not None:
            phase_b_dict["lr_dnn_cross"] = float(args.lr_dnn_cross)
        if args.checkpoint_dir:
            phase_b_dict["checkpoint_dir"] = args.checkpoint_dir
        if args.seed is not None:
            phase_b_dict["seed"] = int(args.seed)

        config_b = Config(
            model=args.model,
            dataset=args.dataset,
            config_file_list=_split_config_files(args.config_files),
            config_dict=phase_b_dict,
        )
        logger_b, dataset_b, train_b, valid_b, test_b, model_b, trainer_b = _build_and_prepare(config_b)
        logger_b.info(
            set_color("[Phase-B] freeze_backbone", "cyan") + f": {config_b['freeze_backbone']}"
        )
        if "lr_text_head" in config_b:
            logger_b.info(
                set_color("[Phase-B] lr groups", "cyan")
                + f": lr_text_head={config_b['lr_text_head']}, lr_dnn_cross={config_b['lr_dnn_cross']}, "
                  f"lr_backbone={config_b['lr_backbone']}"
            )

        # Determine checkpoint to resume Phase-B
        resume_path = args.resume_from if args.resume_from else phase_a_ckpt
        if resume_path and os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location=config_b["device"])
            except Exception:
                ckpt = torch.load(resume_path, map_location=config_b["device"], weights_only=False)
            model_b.load_state_dict(ckpt["state_dict"])
            model_b.load_other_parameter(ckpt.get("other_parameter"))
            logger_b.info(set_color("[Phase-B] Loaded checkpoint", "green") + f": {resume_path}")
        else:
            logger_b.warning("[Phase-B] No valid checkpoint provided; training from scratch weights.")

        res_b = _train_and_eval_phase(logger_b, trainer_b, train_b, valid_b, test_b, saved=args.save)

        # Final summary
        pa_best = res_a["best_valid_result"] if res_a else None
        logger_b.info(set_color("[Two-Phase] Summary", "yellow") + f":\n"
                      f"Phase-A best_valid={pa_best}\n"
                      f"Phase-B best_valid={res_b['best_valid_result']}\n"
                      f"Phase-B test={res_b['test_result']}")

        payload = {
            "best_valid_result": res_a["best_valid_result"] if res_a else None,
            "phase_b_result": res_b,
        }
        _log_and_dump_summary(config_b, model_b, payload, "Phase-B")


if __name__ == "__main__":
    main()


