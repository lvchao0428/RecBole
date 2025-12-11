#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-phase training orchestrator for RecBole models (e.g., SASRec_Align).

Phase-A: warmup for text alignment/fusion with backbone frozen
Phase-B: joint fine-tuning with backbone unfrozen and grouped learning rates
"""
import argparse
import atexit
import copy
import gc
import json
import os
import subprocess
import threading
from datetime import datetime
from logging import getLogger

import csv

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


def _collect_resource_usage(device=None):
    """Sample current CPU/GPU memory usage for diagnostics."""
    usage = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gpu_device": "cpu-only",
    }
    proc = psutil.Process(os.getpid())
    with proc.oneshot():
        mem = proc.memory_info()
        usage["cpu_rss_mb"] = mem.rss / (1024 ** 2)
        usage["cpu_vms_mb"] = mem.vms / (1024 ** 2)
        usage["num_threads"] = proc.num_threads()
    usage["cpu_percent"] = proc.cpu_percent(interval=None)

    if torch.cuda.is_available():
        device = device or torch.device(torch.cuda.current_device())
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        usage["gpu_alloc_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        usage["gpu_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        usage["gpu_max_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        usage["gpu_max_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        usage["gpu_device"] = torch.cuda.get_device_name(device)
    else:
        usage["gpu_alloc_mb"] = 0.0
        usage["gpu_reserved_mb"] = 0.0
        usage["gpu_max_alloc_mb"] = 0.0
        usage["gpu_max_reserved_mb"] = 0.0
    return usage


def _log_gpu_snapshot(tag: str, device=None, include_nvidia=True):
    """Print CPU+GPU utilization; helps diagnose OOM/kills."""
    logger = getLogger()
    usage = _collect_resource_usage(device)
    cpu_msg = (
        f"CPU rss={usage['cpu_rss_mb']:.1f}MB vms={usage['cpu_vms_mb']:.1f}MB "
        f"threads={usage['num_threads']}"
    )
    gpu_msg = (
        f"{usage['gpu_device']} alloc={usage['gpu_alloc_mb']:.1f}MB "
        f"reserved={usage['gpu_reserved_mb']:.1f}MB "
        f"max_alloc={usage['gpu_max_alloc_mb']:.1f}MB"
    )
    logger.info(set_color(f"[RES:{tag}]", "blue") + f" {cpu_msg}; GPU {gpu_msg}")

    if include_nvidia and torch.cuda.is_available():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(set_color(f"[RES:{tag}] nvidia-smi", "blue") + f" {result.stdout.strip()}")
        except FileNotFoundError:
            logger.warning(set_color(f"[RES:{tag}] nvidia-smi not found", "red"))
        except subprocess.CalledProcessError as exc:
            logger.warning(set_color(f"[RES:{tag}] nvidia-smi failed", "red") + f": {exc}")
    return usage


class ResourceWatchdog:
    """Background monitor that logs resource snapshots periodically."""

    def __init__(
        self,
        interval: int = 30,
        device=None,
        include_nvidia: bool = False,
        log_path: str | None = None,
        cpu_threshold_gb: float = 0.0,
        gpu_threshold_gb: float = 0.0,
    ):
        self.interval = max(1, int(interval))
        self.device = device
        self.include_nvidia = include_nvidia
        self.log_path = log_path
        self.cpu_threshold_mb = max(0.0, cpu_threshold_gb) * 1024.0
        self.gpu_threshold_mb = max(0.0, gpu_threshold_gb) * 1024.0
        self._thread = None
        self._stop_event = threading.Event()
        self.logger = getLogger()

    def start(self):
        if self._thread is not None:
            return
        if self.log_path:
            log_dir = os.path.dirname(os.path.abspath(self.log_path))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        self.logger.info(
            set_color("[Watchdog] start", "blue")
            + f": interval={self.interval}s, log_path={self.log_path or 'stderr'}"
        )
        self._thread = threading.Thread(
            target=self._run, name="ResourceWatchdog", daemon=True
        )
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 2)
        self.logger.info(set_color("[Watchdog] stop", "blue"))
        self._thread = None
        self._stop_event.clear()

    def _emit(self, usage: dict):
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(usage, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.logger.warning(
                set_color("[Watchdog] write failed", "red")
                + f": {exc} path={self.log_path}"
            )

    def _check_threshold(self, usage: dict):
        if self.cpu_threshold_mb and usage["cpu_rss_mb"] >= self.cpu_threshold_mb:
            self.logger.warning(
                set_color("[Watchdog] CPU RSS high", "red")
                + f": {usage['cpu_rss_mb'] / 1024:.2f} GB >= {self.cpu_threshold_mb / 1024:.2f} GB"
            )
        if self.gpu_threshold_mb and usage["gpu_alloc_mb"] >= self.gpu_threshold_mb:
            self.logger.warning(
                set_color("[Watchdog] GPU alloc high", "red")
                + f": {usage['gpu_alloc_mb'] / 1024:.2f} GB >= {self.gpu_threshold_mb / 1024:.2f} GB"
            )

    def _run(self):
        while not self._stop_event.is_set():
            usage = _log_gpu_snapshot("watchdog", self.device, include_nvidia=self.include_nvidia)
            self._emit(usage)
            self._check_threshold(usage)
            if self._stop_event.wait(self.interval):
                break


def _release_phase_resources(tag: str, *objects):
    """Release references, run GC, and clear CUDA cache to avoid OOM."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log_gpu_snapshot(f"{tag}-post-free")


def _resolve_device(config: Config | dict | None):
    """Best-effort fetch of device from Config/dict; fallback to cuda/cpu."""
    if config is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if isinstance(config, dict):
            return config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if "device" in config:
            return config["device"]
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


PROGRESS_HEADERS = ["timestamp", "phase", "alignment_weight", "temperature", "epoch", "metric", "score", "event"]


class PhaseProgressLogger:
    def __init__(self, label: str, headers=None):
        headers = headers or PROGRESS_HEADERS
        self.headers = headers
        os.makedirs("run_metrics", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join("run_metrics", f"{timestamp}_{label}.csv")
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.headers)

    def log(self, event: str, **fields):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
        }
        row.update(fields)
        self._writer.writerow([row.get(col, "") for col in self.headers])
        self._file.flush()

    def close(self):
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()


def _combine_callbacks(*callbacks):
    callbacks = [cb for cb in callbacks if cb is not None]
    if not callbacks:
        return None

    def _combined(epoch_idx, valid_score):
        for cb in callbacks:
            cb(epoch_idx, valid_score)

    return _combined


def _make_phase_progress_callback(logger: PhaseProgressLogger | None, phase: str, metric: str | None, **fields):
    if logger is None:
        return None
    metric = metric or ""

    def _cb(epoch_idx, valid_score):
        logger.log(
            "eval",
            phase=phase,
            metric=metric,
            epoch=epoch_idx,
            score=float(valid_score) if valid_score is not None else "",
            **fields,
        )

    return _cb


def _lookup_metric_score(metric_name: str | None, metric_dict: dict | None):
    if not metric_name or not metric_dict:
        return None
    metric_name = metric_name.lower()
    for key, value in metric_dict.items():
        if key.lower() == metric_name:
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
    return None


DEFAULT_METRIC_ORDER = [
    "recall@5",
    "recall@10",
    "recall@20",
    "mrr@5",
    "mrr@10",
    "mrr@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "hit@5",
    "hit@10",
    "hit@20",
    "precision@5",
    "precision@10",
    "precision@20",
    # Frequency-stratified metrics
    "Recall_new@5",
    "Recall_new@10",
    "Recall_new@20",
    "Recall_few@5",
    "Recall_few@10",
    "Recall_few@20",
    "Recall_frequent@5",
    "Recall_frequent@10",
    "Recall_frequent@20",
    "NDCG_new@5",
    "NDCG_new@10",
    "NDCG_new@20",
    "NDCG_few@5",
    "NDCG_few@10",
    "NDCG_few@20",
    "NDCG_frequent@5",
    "NDCG_frequent@10",
    "NDCG_frequent@20",
    "Coverage_new@5",
    "Coverage_new@10",
    "Coverage_new@20",
    "Coverage_few@5",
    "Coverage_few@10",
    "Coverage_few@20",
    "Coverage_frequent@5",
    "Coverage_frequent@10",
    "Coverage_frequent@20",
]


def _coerce_metric_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _sanitize_metrics(metrics: dict | None):
    if not metrics:
        return None
    clean = {}
    for key, val in metrics.items():
        clean[key] = _coerce_metric_value(val)
    return clean


def _format_metric_csv_line(metrics: dict | None, metric_order=None):
    metric_order = metric_order or DEFAULT_METRIC_ORDER
    header = ",".join(metric_order)
    if not metrics:
        return header, ",".join("" for _ in metric_order)
    values = []
    for key in metric_order:
        if key not in metrics:
            values.append("")
            continue
        val = _coerce_metric_value(metrics[key])
        values.append(f"{val:.4f}" if isinstance(val, float) else str(val))
    return header, ",".join(values)


def _split_variant_tokens(raw: str | None):
    if not raw:
        return []
    normalized = raw.replace("+", ",")
    tokens = [tok.strip() for tok in normalized.split(",")]
    return [tok for tok in tokens if tok]


def _build_variant_label_from_args(args):
    raw_label = getattr(args, "variant_label", None)
    if raw_label:
        tokens = _split_variant_tokens(raw_label)
        return " + ".join(tokens) if tokens else raw_label.strip()
    raw_features = getattr(args, "variant_features", None)
    if raw_features:
        tokens = _split_variant_tokens(raw_features)
        if tokens:
            return " + ".join(tokens)
    return None


def _log_grouped_two_phase_summary(res_a: dict | None, res_b: dict | None, variant_label: str | None = None):
    logger = getLogger()
    sep = "=" * 80
    logger.info(sep)
    logger.info(set_color("[Two-Phase] Grouped Summary", "yellow"))
    logger.info("-" * 80)

    phase_a_best = _sanitize_metrics((res_a or {}).get("best_valid_result") if res_a else None)
    phase_b_best = _sanitize_metrics((res_b or {}).get("best_valid_result") if res_b else None)
    phase_b_test = _sanitize_metrics((res_b or {}).get("test_result") if res_b else None)

    if phase_a_best:
        logger.info("[Phase-A] best_valid metrics:")
        logger.info(json.dumps(phase_a_best, ensure_ascii=False, indent=2))
    else:
        logger.info("[Phase-A] best_valid metrics: (not available)")

    if phase_b_best:
        logger.info("[Phase-B] best_valid metrics:")
        logger.info(json.dumps(phase_b_best, ensure_ascii=False, indent=2))
    else:
        logger.info("[Phase-B] best_valid metrics: (not available)")

    if phase_b_test:
        logger.info("[Phase-B] test metrics:")
        logger.info(json.dumps(phase_b_test, ensure_ascii=False, indent=2))
    else:
        logger.info("[Phase-B] test metrics: (not available)")

    header, values = _format_metric_csv_line(phase_b_test or phase_b_best)
    if variant_label:
        logger.info(variant_label)
    logger.info("")
    logger.info(header)
    logger.info(values)
    logger.info(sep)


def _train_and_eval_phase(
    logger,
    trainer,
    train_data,
    valid_data,
    test_data,
    saved=True,
    phase_label="Phase",
    progress_logger: PhaseProgressLogger | None = None,
    progress_metric: str | None = None,
    progress_fields: dict | None = None,
    callback_fn=None,
):
    """Run one training phase and return results and saved model path."""
    device = _resolve_device(trainer.config)
    _log_gpu_snapshot("before-fit", device)
    metric_name = progress_metric or (trainer.config["valid_metric"] if "valid_metric" in trainer.config else None)
    progress_cb = _make_phase_progress_callback(
        progress_logger,
        phase_label,
        metric_name,
        **(progress_fields or {}),
    )
    combined_callback = _combine_callbacks(callback_fn, progress_cb)
    try:
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=saved,
            show_progress=trainer.config["show_progress"],
            callback_fn=combined_callback,
        )
    except RuntimeError as err:
        if "CUDA" in str(err) or "cuda" in str(err):
            logger.error(set_color("[GPU] RuntimeError during fit()", "red") + f": {err}")
            _log_gpu_snapshot("fit-exception", device)
            try:
                logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
            except Exception:
                pass
        raise

    _log_gpu_snapshot("before-eval", device)
    try:
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=trainer.config["show_progress"]
        )
    except RuntimeError as err:
        if "CUDA" in str(err) or "cuda" in str(err):
            logger.error(set_color("[GPU] RuntimeError during evaluate()", "red") + f": {err}")
            _log_gpu_snapshot("eval-exception", device)
            try:
                logger.error(torch.cuda.memory_summary(device=device, abbreviated=True))
            except Exception:
                pass
        raise
    env_tb = get_environment(trainer.config)
    logger.info("The running environment of this training is as follows:\n" + env_tb.draw())
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    if progress_logger:
        best_metric_val = _lookup_metric_score(metric_name, best_valid_result)
        test_metric_val = _lookup_metric_score(metric_name, test_result)
        base_fields = progress_fields or {}
        progress_logger.log(
            "best_valid",
            phase=phase_label,
            metric=metric_name,
            score=best_metric_val if best_metric_val is not None else "",
            **base_fields,
        )
        progress_logger.log(
            "test_result",
            phase=phase_label,
            metric=metric_name,
            score=test_metric_val if test_metric_val is not None else "",
            **base_fields,
        )
    result = {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "saved_model_file": trainer.saved_model_file,
    }
    _log_gpu_snapshot("phase-end", device)
    return result


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
    parser.add_argument("--variant_label", type=str, default=None, help="custom label printed in final metric table")
    parser.add_argument("--variant_features", type=str, default=None, help="comma/plus separated feature tokens to auto-build label (e.g., 'sasrec,tfidf,llm')")
    # Manual switching
    parser.add_argument("--only_phase_a", action="store_true", help="run Phase-A only")
    parser.add_argument("--only_phase_b", action="store_true", help="run Phase-B only")
    parser.add_argument("--resume_from", type=str, default=None, help="resume Phase-B from Phase-A checkpoint path")
    # Phase-A grid & gating
    parser.add_argument("--phase_a_grid", action="store_true", help="run Phase-A grid over alignment_weight and temperature")
    parser.add_argument("--align_grid", type=str, default="0.05,0.1,0.2", help="comma-separated alignment_weight grid")
    parser.add_argument("--tau_grid", type=str, default="0.05,0.07", help="comma-separated temperature grid")
    parser.add_argument("--phase_a_eval_step", type=int, default=2, help="validation interval (epochs) in Phase-A")
    parser.add_argument("--phase_a_valid_metric", type=str, default="NDCG@10", help="validation metric used for Phase-A selection")
    parser.add_argument("--metric_baseline", type=float, default=None, help="strong baseline for phase_a_valid_metric (e.g., 0.0272 for Recall@10)")
    parser.add_argument("--metric_gain_threshold", type=float, default=0.01, help="required relative gain over baseline, e.g., 0.01 for +1%")
    parser.add_argument("--phase_a_auto_to_b", action="store_true", help="if pass condition met, continue to Phase-B automatically")
    parser.add_argument("--phase_a_require_pass_for_b", action="store_true", help="only enter Phase-B if pass condition is met")
    # Optional ID-only burn-in before Phase-A
    parser.add_argument("--backbone_burnin_epochs", type=int, default=0, help="ID-only burn-in epochs before Phase-A")
    parser.add_argument("--burnin_eval_step", type=int, default=2, help="validation interval (epochs) in burn-in stage")
    # Resource watchdog
    parser.add_argument("--watchdog_interval", type=int, default=0, help="enable periodic resource logging every N seconds (0 disables)")
    parser.add_argument("--watchdog_log", type=str, default=None, help="path to append watchdog JSON lines (default run_metrics/resource_watchdog.log)")
    parser.add_argument("--watchdog_cpu_gb", type=float, default=0.0, help="optional CPU RSS threshold in GB for warning")
    parser.add_argument("--watchdog_gpu_gb", type=float, default=0.0, help="optional GPU alloc threshold in GB for warning")
    parser.add_argument("--watchdog_disable", action="store_true", help="disable resource watchdog regardless of other settings")
    args, _ = parser.parse_known_args()
    args.variant_label = _build_variant_label_from_args(args)

    watchdog = None
    if (not args.watchdog_disable) and args.watchdog_interval and args.watchdog_interval > 0:
        wd_log = args.watchdog_log or os.path.join("run_metrics", "resource_watchdog.log")
        wd_device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else None
        watchdog = ResourceWatchdog(
            interval=args.watchdog_interval,
            device=wd_device,
            include_nvidia=False,
            log_path=wd_log,
            cpu_threshold_gb=args.watchdog_cpu_gb,
            gpu_threshold_gb=args.watchdog_gpu_gb,
        )
        watchdog.start()
        atexit.register(lambda wd=watchdog: wd and wd.stop())
    elif args.watchdog_disable:
        getLogger().info("[Watchdog] disabled via --watchdog_disable")

    if args.only_phase_a and args.only_phase_b:
        raise ValueError("only_phase_a and only_phase_b cannot be used together.")

    # Phase-A config
    res_a = None
    phase_a_ckpt = None
    phase_a_passed = False
    burnin_ckpt = None
    phase_a_pass_record = None
    phase_a_pass_events = []
    phase_a_progress_logger = None
    phase_a_progress_path = None
    phase_b_progress_logger = None
    phase_b_progress_path = None

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
        _release_phase_resources("Burn-in", model_burn, trainer_burn, dataset_burn, train_burn, valid_burn, test_burn)
    if not args.only_phase_b:
        phase_a_progress_logger = PhaseProgressLogger("phase_a_progress")
        phase_a_progress_path = phase_a_progress_logger.path
        if args.phase_a_grid:
            # Parse grids
            try:
                align_list = [float(x) for x in args.align_grid.split(",") if x.strip() != ""]
                tau_list = [float(x) for x in args.tau_grid.split(",") if x.strip() != ""]
            except Exception:
                raise ValueError("Failed to parse align_grid or tau_grid; use comma-separated floats, e.g., '0.05,0.1,0.2'")
            if len(align_list) == 0 or len(tau_list) == 0:
                raise ValueError("Empty grid for alignment_weight or temperature.")
            # Phase-A grid: always run all combinations, select best by phase_a_valid_metric
            metric_target = None
            if args.metric_baseline is not None:
                metric_target = args.metric_baseline * (1.0 + float(args.metric_gain_threshold))
                getLogger().info(
                    f"[Phase-A:grid] Target threshold for {args.phase_a_valid_metric}: "
                    f"{metric_target:.6f} (baseline={args.metric_baseline:.6f}, gain={args.metric_gain_threshold})"
                )

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
                    progress_cb = _make_phase_progress_callback(
                        phase_a_progress_logger,
                        "Phase-A",
                        args.phase_a_valid_metric,
                        alignment_weight=aw,
                        temperature=tau,
                    )
                    # Load burn-in checkpoint if available
                    if burnin_ckpt and os.path.exists(burnin_ckpt):
                        try:
                            ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"])
                        except Exception:
                            ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"], weights_only=False)
                        
                        # Check for dimension mismatches (e.g., multiview with/without base features)
                        mismatched_keys = []
                        current_state = model_a.state_dict()
                        for key, ckpt_tensor in ckpt_b["state_dict"].items():
                            if key in current_state:
                                if ckpt_tensor.shape != current_state[key].shape:
                                    mismatched_keys.append(
                                        f"{key}: ckpt{list(ckpt_tensor.shape)} vs model{list(current_state[key].shape)}"
                                    )
                        
                        if mismatched_keys:
                            logger_a.warning(set_color("[Phase-A:grid] Dimension mismatch in burn-in checkpoint:", "yellow"))
                            for mk in mismatched_keys:
                                logger_a.warning(f"  - {mk}")
                            logger_a.warning(set_color("[Phase-A:grid] Skipping mismatched layers, loading compatible ones only", "yellow"))
                            
                            # Filter out mismatched keys
                            filtered_state = {k: v for k, v in ckpt_b["state_dict"].items() 
                                            if k in current_state and v.shape == current_state[k].shape}
                            model_a.load_state_dict(filtered_state, strict=False)
                        else:
                            model_a.load_state_dict(ckpt_b["state_dict"], strict=False)
                        
                        model_a.load_other_parameter(ckpt_b.get("other_parameter"))
                        logger_a.info(set_color("[Phase-A:grid] Loaded burn-in checkpoint (strict=False)", "green") + f": {burnin_ckpt}")
                    # Phase-A grid: always run all combinations without early stopping for fair comparison
                    combined_cb = progress_cb
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

                        _log_gpu_snapshot("Phase-A:grid-before-fit", config_a["device"])
                        try:
                            best_valid_score, best_valid_result = trainer_a.fit(
                                train_a,
                                valid_a,
                                saved=args.save,
                                show_progress=trainer_a.config["show_progress"],
                                callback_fn=combined_cb,
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
                        # Extract validation metric (use args.phase_a_valid_metric)
                        valid_dict = res.get("best_valid_result") or {}
                        metric_value = _lookup_metric_score(args.phase_a_valid_metric, valid_dict)
                        if phase_a_progress_logger:
                            phase_a_progress_logger.log(
                                "combo_best_valid",
                                phase="Phase-A",
                                alignment_weight=aw,
                                temperature=tau,
                                metric=args.phase_a_valid_metric.lower(),
                                score=metric_value if metric_value is not None else "",
                            )
                            combo_test_metric = _lookup_metric_score(args.phase_a_valid_metric, res.get("test_result"))
                            phase_a_progress_logger.log(
                                "combo_test_result",
                                phase="Phase-A",
                                alignment_weight=aw,
                                temperature=tau,
                                metric=args.phase_a_valid_metric.lower(),
                                score=combo_test_metric if combo_test_metric is not None else "",
                            )
                        logger_a.info(set_color(f"[Phase-A:grid] {args.phase_a_valid_metric}", "yellow") + f": {metric_value}")
                        _log_gpu_snapshot("Phase-A:grid-phase-end", config_a["device"])
                        if metric_value is not None:
                            # Track best combination across all grid cells
                            if best_tuple is None or metric_value > best_tuple[0]:
                                best_tuple = (metric_value, aw, tau, res, ckpt_path)
                                logger_a.info(set_color(f"[Phase-A:grid] New best: {args.phase_a_valid_metric}={metric_value:.6f}", "green") + f" (aw={aw}, tau={tau})")
                    finally:
                        _release_phase_resources("Phase-A:grid-loop", model_a, trainer_a, dataset_a, train_a, valid_a, test_a)
            
            # After grid search complete, select best combination
            if best_tuple is not None:
                phase_a_ckpt = best_tuple[4]
                res_a = best_tuple[3]
                logger = getLogger()
                logger.info(set_color("[Phase-A:grid] Grid complete. Best combo", "blue") + f": {args.phase_a_valid_metric}={best_tuple[0]:.6f}, alignment_weight={best_tuple[1]}, temperature={best_tuple[2]}")
                if metric_target is not None:
                    status = "✅ PASS" if best_tuple[0] >= metric_target else "⚠️ below"
                    logger.info(f"[Phase-A:grid] Target threshold: {metric_target:.6f} → {status}")
                phase_a_pass_record = {
                    "alignment_weight": best_tuple[1],
                    "temperature": best_tuple[2],
                    "metric_value": best_tuple[0],
                    "metric_name": args.phase_a_valid_metric,
                }
                phase_a_passed = True  # Always pass after completing grid search
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
            progress_cb = _make_phase_progress_callback(
                phase_a_progress_logger,
                "Phase-A",
                args.phase_a_valid_metric,
                alignment_weight=phase_a_dict.get("alignment_weight"),
                temperature=phase_a_dict.get("temperature"),
                )
            # Load burn-in checkpoint if available
            if burnin_ckpt and os.path.exists(burnin_ckpt):
                try:
                    ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"])
                except Exception:
                    ckpt_b = torch.load(burnin_ckpt, map_location=config_a["device"], weights_only=False)
                
                # Check for dimension mismatches (e.g., multiview with/without base features)
                mismatched_keys = []
                current_state = model_a.state_dict()
                for key, ckpt_tensor in ckpt_b["state_dict"].items():
                    if key in current_state:
                        if ckpt_tensor.shape != current_state[key].shape:
                            mismatched_keys.append(
                                f"{key}: ckpt{list(ckpt_tensor.shape)} vs model{list(current_state[key].shape)}"
                            )
                
                if mismatched_keys:
                    logger_a.warning(set_color("[Phase-A] Dimension mismatch in burn-in checkpoint:", "yellow"))
                    for mk in mismatched_keys:
                        logger_a.warning(f"  - {mk}")
                    logger_a.warning(set_color("[Phase-A] Skipping mismatched layers, loading compatible ones only", "yellow"))
                    
                    # Filter out mismatched keys
                    filtered_state = {k: v for k, v in ckpt_b["state_dict"].items() 
                                    if k in current_state and v.shape == current_state[k].shape}
                    model_a.load_state_dict(filtered_state, strict=False)
                else:
                    model_a.load_state_dict(ckpt_b["state_dict"], strict=False)
                
                model_a.load_other_parameter(ckpt_b.get("other_parameter"))
                logger_a.info(set_color("[Phase-A] Loaded burn-in checkpoint (strict=False)", "green") + f": {burnin_ckpt}")
            # Phase-A non-grid: run full training without early stopping
            # Log target if provided for reference only
            if args.metric_baseline is not None:
                metric_target = args.metric_baseline * (1.0 + float(args.metric_gain_threshold))
                logger_a.info(f"[Phase-A] Reference target for {args.phase_a_valid_metric}: {metric_target:.6f}")
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

                combined_cb = progress_cb
                best_valid_score, best_valid_result = trainer_a.fit(
                    train_a,
                    valid_a,
                    saved=args.save,
                    show_progress=trainer_a.config["show_progress"],
                    callback_fn=combined_cb,
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
            # Phase-A complete - always mark as passed (no gate check needed)
            phase_a_passed = True
            if args.metric_baseline is not None:
                metric_target = args.metric_baseline * (1.0 + float(args.metric_gain_threshold))
                metric_value = _lookup_metric_score(args.phase_a_valid_metric, res_a.get("best_valid_result"))
                if metric_value is not None:
                    status = "✅ above" if metric_value >= metric_target else "⚠️ below"
                    logger_a.info(f"[Phase-A] Final {args.phase_a_valid_metric}={metric_value:.6f} (target={metric_target:.6f}) → {status}")
            _release_phase_resources("Phase-A", model_a, trainer_a, dataset_a, train_a, valid_a, test_a)
    if phase_a_progress_logger:
        phase_a_progress_logger.close()

    # Phase-B config (unfreeze + grouped LR; lr_backbone = lr_text_head * scale)
    if not args.only_phase_a:
        # Check if we should skip Phase-B
        if args.phase_a_require_pass_for_b and not phase_a_passed:
            getLogger().warning("[Two-Phase] Phase-B is skipped (phase_a_require_pass_for_b and not passed).")
            return
        # Auto-transition to Phase-B when phase_a_auto_to_b is True (default behavior)
        if phase_a_ckpt:
            getLogger().info(set_color("[Two-Phase] Phase-A finished. Auto-continuing to Phase-B...", "green"))

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
        payload = {
            "best_valid_result": res_a["best_valid_result"] if res_a else None,
            "phase_b_result": res_b,
        }
        _log_and_dump_summary(config_b, model_b, payload, "Phase-B")
        _log_grouped_two_phase_summary(res_a, res_b, variant_label=args.variant_label)
        _release_phase_resources("Phase-B", model_b, trainer_b, dataset_b, train_b, valid_b, test_b)


if __name__ == "__main__":
    main()


