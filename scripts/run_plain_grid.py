#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a small grid for pure-text (Plain) fusion on top of SASRecAlign.
- Only varies text_weight (default: 0.4, 0.8)
- Forces pure text setting: use_align=False, alignment_weight=0.0, use_cross=False
- Leaves all other settings to the provided YAML
"""
import argparse
import os
from typing import List, Dict, Any
from copy import deepcopy

# Ensure project root is on sys.path when running this script directly
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recbole.quick_start import run_recbole


def pick_metric(result: Dict[str, Any]) -> float:
    """Pick a validation metric (higher better). Prefer MRR@10 > NDCG@10 > Recall@10 > Hit@10."""
    if result is None:
        return float("-inf")
    for k in ["MRR@10", "NDCG@10", "Recall@10", "Hit@10"]:
        if k in result:
            return float(result[k])
    # fallback: if empty, return -inf
    return float("-inf")


def main():
    parser = argparse.ArgumentParser(description="Plain-text fusion grid for SASRecAlign")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., Amazon_Beauty)")
    parser.add_argument("--config_file", required=True, help="YAML config file path (e.g., sasrec_base_plain.yaml)")
    parser.add_argument("--model", default="SASRec_Align", help="Model name (default: SASRec_Align)")
    parser.add_argument(
        "--text_weights", type=float, nargs="+", default=[0.4, 0.8], help="Values for text_weight grid"
    )
    parser.add_argument(
        "--text_gate_init_grid", type=float, nargs="+", default=[0.3, 0.5], help="Values for text_gate_init"
    )
    parser.add_argument(
        "--text_gate_reg_entropy_grid", type=float, nargs="+", default=[0.0, 5e-4], help="Values for text_gate_reg_entropy"
    )
    parser.add_argument(
        "--text_gate_reg_l2_grid", type=float, nargs="+", default=[0.0, 1e-5], help="Values for text_gate_reg_l2"
    )
    parser.add_argument(
        "--text_tail_threshold_grid", type=int, nargs="+", default=[0, 5], help="Values for text_tail_threshold"
    )
    # Optional quick training overrides (set if you want to shorten runtime; keep None to leave YAML as-is)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval_step", type=int, default=None)
    parser.add_argument("--stopping_step", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Override eval_batch_size for faster eval")
    # Shared-HP workflow (optional): grid on one family, reuse best on the other, then micro-tune 1–2 points
    parser.add_argument("--shared_hp", action="store_true", default=False, help="Enable Shared-HP workflow for base/llm")
    parser.add_argument("--base_config_file", type=str, default=None, help="YAML config for base (TF-IDF)")
    parser.add_argument("--llm_config_file", type=str, default=None, help="YAML config for llm (Qwen3)")
    parser.add_argument("--shared_hp_from", choices=["base", "llm"], default="base", help="Which family to grid first")
    args = parser.parse_args()

    # If Shared-HP mode, require both config files; else require single config_file
    if args.shared_hp:
        assert args.base_config_file and args.llm_config_file, "In --shared_hp mode, --base_config_file and --llm_config_file are required"
        assert os.path.exists(args.base_config_file), f"base_config_file not found: {args.base_config_file}"
        assert os.path.exists(args.llm_config_file), f"llm_config_file not found: {args.llm_config_file}"
    else:
        assert os.path.exists(args.config_file), f"config_file not found: {args.config_file}"

    def run_one(config_file: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        return run_recbole(
            model=args.model,
            dataset=args.dataset,
            config_file_list=[config_file],
            config_dict=overrides,
            saved=True,
        )

    def build_overrides(base_overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(base_overrides)
        if args.epochs is not None:
            cfg["epochs"] = int(args.epochs)
        if args.eval_step is not None:
            cfg["eval_step"] = int(args.eval_step)
        if args.stopping_step is not None:
            cfg["stopping_step"] = int(args.stopping_step)
        if args.eval_batch_size is not None:
            cfg["eval_batch_size"] = int(args.eval_batch_size)
        return cfg

    # Pure text fixed base overrides
    pure_text_base = {
        "use_align": False,
        "alignment_weight": 0.0,
        "use_cross": False,
    }

    if not args.shared_hp:
        trials = []
        for tw in args.text_weights:
            for ginit in args.text_gate_init_grid:
                for gent in args.text_gate_reg_entropy_grid:
                    for gl2 in args.text_gate_reg_l2_grid:
                        for ttail in args.text_tail_threshold_grid:
                            overrides = {
                                **pure_text_base,
                                "text_weight": float(tw),
                                "text_gate_init": float(ginit),
                                "text_gate_reg_entropy": float(gent),
                                "text_gate_reg_l2": float(gl2),
                                "text_tail_threshold": int(ttail),
                            }
                            cfg = build_overrides(overrides)
                            print(f"[Run] tw={tw} ginit={ginit} gent={gent} gl2={gl2} ttail={ttail}")
                            res = run_one(args.config_file, cfg)
                            trials.append(
                                {
                                    "text_weight": tw,
                                    "text_gate_init": ginit,
                                    "text_gate_reg_entropy": gent,
                                    "text_gate_reg_l2": gl2,
                                    "text_tail_threshold": ttail,
                                    "best_valid_score": res.get("best_valid_score"),
                                    "best_valid_result": res.get("best_valid_result"),
                                    "test_result": res.get("test_result"),
                                }
                            )
                            print(f"[Result] tw={tw} ginit={ginit} gent={gent} gl2={gl2} ttail={ttail} | valid={res.get('best_valid_result')} | test={res.get('test_result')}")

        best = None
        best_score = float("-inf")
        for t in trials:
            score = pick_metric(t["best_valid_result"])
            if score > best_score:
                best = t
                best_score = score
        print("\n================ Best by validation ================")
        if best is None:
            print("No successful runs.")
            return
        print(f"text_weight={best['text_weight']}")
        print(f"text_gate_init={best['text_gate_init']} text_gate_reg_entropy={best['text_gate_reg_entropy']} text_gate_reg_l2={best['text_gate_reg_l2']} text_tail_threshold={best['text_tail_threshold']}")
        print(f"valid={best['best_valid_result']}")
        print(f"test={best['test_result']}")
        print("====================================================")
        return

    # Shared-HP workflow
    family_first = args.base_config_file if args.shared_hp_from == "base" else args.llm_config_file
    family_second = args.llm_config_file if args.shared_hp_from == "base" else args.base_config_file
    trials_first = []
    # Grid on the first family
    for tw in args.text_weights:
        for ginit in args.text_gate_init_grid:
            for gent in args.text_gate_reg_entropy_grid:
                for gl2 in args.text_gate_reg_l2_grid:
                    for ttail in args.text_tail_threshold_grid:
                        overrides = {
                            **pure_text_base,
                            "text_weight": float(tw),
                            "text_gate_init": float(ginit),
                            "text_gate_reg_entropy": float(gent),
                            "text_gate_reg_l2": float(gl2),
                            "text_tail_threshold": int(ttail),
                        }
                        cfg = build_overrides(overrides)
                        print(f"[Shared-HP:Grid:{args.shared_hp_from}] tw={tw} ginit={ginit} gent={gent} gl2={gl2} ttail={ttail}")
                        res = run_one(family_first, cfg)
                        trials_first.append(
                            {
                                "text_weight": tw,
                                "text_gate_init": ginit,
                                "text_gate_reg_entropy": gent,
                                "text_gate_reg_l2": gl2,
                                "text_tail_threshold": ttail,
                                "best_valid_score": res.get("best_valid_score"),
                                "best_valid_result": res.get("best_valid_result"),
                                "test_result": res.get("test_result"),
                            }
                        )
                        print(f"[Result:{args.shared_hp_from}] valid={res.get('best_valid_result')} | test={res.get('test_result')}")

    # Pick shared best
    shared_best = None
    shared_score = float("-inf")
    for t in trials_first:
        score = pick_metric(t["best_valid_result"])
        if score > shared_score:
            shared_best = t
            shared_score = score
    if shared_best is None:
        print("[Shared-HP] No successful runs in the first family.")
        return
    print("\n[Shared-HP] Best (first family):", shared_best)

    # Apply shared best to the second family
    shared_overrides = {
        **pure_text_base,
        "text_weight": float(shared_best["text_weight"]),
        "text_gate_init": float(shared_best["text_gate_init"]),
        "text_gate_reg_entropy": float(shared_best["text_gate_reg_entropy"]),
        "text_gate_reg_l2": float(shared_best["text_gate_reg_l2"]),
        "text_tail_threshold": int(shared_best["text_tail_threshold"]),
    }
    cfg_shared = build_overrides(shared_overrides)
    print(f"[Shared-HP:Apply to second] {cfg_shared}")
    res_second_shared = run_one(family_second, cfg_shared)
    print(f"[Shared-HP:Second] valid={res_second_shared.get('best_valid_result')} | test={res_second_shared.get('test_result')}")

    # Micro-tune 1–2 points on the second family (adjust text_weight ±0.1 around best)
    tw_best = float(shared_best["text_weight"])
    tw_candidates = sorted({max(0.1, round(tw_best - 0.1, 2)), round(tw_best, 2), round(tw_best + 0.1, 2)})
    fine_trials = []
    for tw in tw_candidates:
        overrides_ft = deepcopy(shared_overrides)
        overrides_ft["text_weight"] = float(tw)
        cfg_ft = build_overrides(overrides_ft)
        print(f"[Shared-HP:Fine second] tw={tw}")
        res_ft = run_one(family_second, cfg_ft)
        fine_trials.append(
            {
                "text_weight": tw,
                "best_valid_result": res_ft.get("best_valid_result"),
                "test_result": res_ft.get("test_result"),
            }
        )
        print(f"[Fine Result] tw={tw} | valid={res_ft.get('best_valid_result')} | test={res_ft.get('test_result')}")

    # Report the best of second family (including shared and fine)
    best_second = {
        "text_weight": shared_overrides["text_weight"],
        "best_valid_result": res_second_shared.get("best_valid_result"),
        "test_result": res_second_shared.get("test_result"),
    }
    best_second_score = pick_metric(best_second["best_valid_result"])
    for t in fine_trials:
        sc = pick_metric(t["best_valid_result"])
        if sc > best_second_score:
            best_second = t
            best_second_score = sc
    print("\n================ Shared-HP Summary ================")
    print(f"First family best (shared): {shared_best}")
    print(f"Second family best: {best_second}")
    print("===================================================")


if __name__ == "__main__":
    main()


