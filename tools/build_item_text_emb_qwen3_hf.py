#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build item text embeddings using a HuggingFace Qwen3 model.

This script has been enhanced to act as a "Vector Amplifier" (LLM-FiBiNET style).
It can generate embeddings from multiple "views" (prompts) and combine them
to create richer representations (Long Vectors) or Multi-View Tensors.

Input: mapping CSV exported by tools/export_internal_item_mapping.py
Output: item_text_emb.qwen3.npy

Example:
  python tools/build_item_text_emb_qwen3_hf.py \
    --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
    --prompt_preset multiview \
    --output_mode concat

Output Modes:
- concat: Concatenate vectors from all prompts (Shape: [N, K * D]). Good for "Long Vector" input.
- mean: Average vectors from all prompts (Shape: [N, D]). Good for "Denoised" input.
- stack: Save as 3D tensor (Shape: [N, K, D]). Good for advanced models like FiBiNET.
"""

import argparse
import os
import sys
import json
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

# Make local project importable when running from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset, data_preparation


# --- Built-in Prompt Presets ---
PROMPT_PRESETS = {
    "base": [
        "[TITLE] {text}",
    ],
    "multiview": [
        # View 1: Identity/Title (Base)
        "Identify the item: [TITLE] {text}",
        # View 2: Function/Utility
        "What are the main functions and features of [TITLE] {text}?",
        # View 3: Target Audience
        "Who is the target audience or user group for [TITLE] {text}?",
        # View 4: Category/Context
        "Categorize the item [TITLE] {text} and describe its context.",
    ],
    "description": [
        "Describe [TITLE] {text} in detail.",
    ]
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Qwen3 HF-based item text embeddings (Vector Amplifier)")
    p.add_argument("--mapping", required=True, help="CSV from export_internal_item_mapping.py")
    p.add_argument("--model_name_or_path", required=True, help="HF model id or local path")
    p.add_argument("--output", required=True, help="Output .npy path for embeddings")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="Max sequence length; 0 means no truncation.",
    )
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default=None, help="cuda device or cpu")
    p.add_argument("--device_map", default=None, help="HF accelerate device_map (e.g. 'auto')")
    
    # --- Prompting Arguments ---
    p.add_argument(
        "--prompt_template",
        default=None,
        help="Single prompt template. Overrides --prompt_preset if set. Use {text} placeholder.",
    )
    p.add_argument(
        "--prompt_preset",
        default="base",
        choices=PROMPT_PRESETS.keys(),
        help="Use a built-in set of prompts (e.g., 'multiview' for amplification).",
    )
    p.add_argument(
        "--prompt_list",
        default=None,
        help="Path to a JSON file containing a list of prompt strings. Overrides preset/template.",
    )
    
    p.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Wrap prompt using tokenizer.apply_chat_template.",
    )
    p.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template even if available.",
    )
    p.add_argument(
        "--use_causal_lm",
        action="store_true",
        help="Load AutoModelForCausalLM instead of AutoModel.",
    )
    p.add_argument("--placeholder_text", default="N/A", help="Fallback text.")
    p.add_argument("--pad_placeholder_text", default="[PAD]", help="Placeholder for PAD row.")
    p.add_argument(
        "--split_output_dir",
        default=None,
        help="If set, dumps per-view embeddings (view_{i}.npy) and views.json metadata for multi-view downstream.",
    )
    
    # --- Output & Projection Arguments ---
    p.add_argument(
        "--output_mode",
        default="mean",
        choices=["mean", "concat", "stack"],
        help="How to combine vectors from multiple prompts. 'concat' = Long Vector, 'stack' = 3D Tensor.",
    )
    p.add_argument(
        "--project_dim",
        type=int,
        default=None, # Changed default to None to avoid accidental projection of concatenated vectors
        help="Reduce embedding dim via TruncatedSVD. Applied AFTER concatenation/mean. Default: None (keep original).",
    )
    p.add_argument(
        "--view_project_dim",
        type=int,
        default=None,
        help="Apply TruncatedSVD per prompt view before concat/stack (e.g., 64). Requires --split_output_dir.",
    )
    p.add_argument(
        "--svd_random_state",
        type=int,
        default=42,
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="RecBole dataset name for SVD train-split fitting.",
    )
    p.add_argument(
        "--config",
        nargs="+",
        default=[],
        help="YAML config files for dataset loading.",
    )
    return p.parse_args()


def _select_device(dev: str | None) -> torch.device:
    if dev is not None:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    # Determine truncation policy
    do_trunc = isinstance(max_length, int) and max_length > 0
    
    # Tokenize
    if getattr(tokenizer, "pad_token_id", None) is None:
        # Fallback: encode one by one if no pad token
        outs = []
        for txt in texts:
            enc = tokenizer(
                txt,
                padding=False,
                truncation=do_trunc,
                max_length=(max_length if do_trunc else None),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = outputs.hidden_states[-1]
            
            # Mean pooling
            attn_mask = enc.get("attention_mask", torch.ones_like(last_hidden[:, :, 0]))
            mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, dim=1)
            outs.append(emb.to(torch.float32).detach().cpu().numpy())
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
    else:
        # Batch encoding
        enc = tokenizer(
            texts,
            padding=True,
            truncation=do_trunc,
            max_length=(max_length if do_trunc else None),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        
        if enc["input_ids"].shape[1] == 0:
            # Handle empty edge case
            return np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            last_hidden = outputs.hidden_states[-1]
            
        attn_mask = enc.get("attention_mask", torch.ones_like(last_hidden[:, :, 0]))
        mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.to(torch.float32).detach().cpu().numpy()


def _load_train_item_ids(args, max_row: int):
    if args.dataset is None or len(args.dataset) == 0:
        return None
    try:
        cfg = Config(model="BPR", dataset=args.dataset, config_file_list=args.config)
        ds = create_dataset(cfg)
        train_data, _, _ = data_preparation(cfg, ds)
        iid_field = cfg["ITEM_ID_FIELD"]
        train_ids_raw = train_data.dataset.inter_feat[iid_field].numpy()
        train_ids = np.unique(train_ids_raw).astype(np.int64)
        train_ids = train_ids[(train_ids >= 1) & (train_ids <= max_row)]
        if len(train_ids) == 0:
            return None
        return train_ids
    except Exception as e:
        print(f"Warning: Failed to load dataset for SVD split ({e}). Using all items.")
        return None


def _apply_truncated_svd(mat: np.ndarray, target_dim: int, train_ids, random_state: int, label: str):
    if mat.ndim != 2:
        raise ValueError("SVD can only be applied to 2D matrices.")
    orig_dim = mat.shape[1]
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("--project_dim/--view_project_dim must be > 0")
    if target_dim >= orig_dim:
        if mat.shape[0] > 1:
            mat[1:, :] = l2_normalize(mat[1:, :], norm="l2", axis=1)
        return mat

    if train_ids is None or len(train_ids) == 0:
        subset = mat[1:, :].astype(np.float32, copy=False)
    else:
        subset = mat[train_ids, :].astype(np.float32, copy=False)
    svd_k = max(1, min(target_dim, subset.shape[1] - 1 if subset.shape[1] > 1 else 1))
    print(f"[SVD:{label}] fitting TruncatedSVD from {orig_dim} -> {target_dim} (k={svd_k})...")
    svd = TruncatedSVD(n_components=svd_k, random_state=random_state)
    svd.fit(subset)

    nonpad = mat[1:, :].astype(np.float32, copy=False)
    reduced = svd.transform(nonpad)
    if svd_k < target_dim:
        pad = np.zeros((reduced.shape[0], target_dim - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)
    reduced = l2_normalize(reduced, norm="l2", axis=1)
    projected = np.zeros((mat.shape[0], target_dim), dtype=reduced.dtype)
    projected[1:, :] = reduced
    return projected


def main():
    args = parse_args()

    # --- 1. Resolve Prompts ---
    if args.prompt_list:
        with open(args.prompt_list, "r") as f:
            prompts = json.load(f)
            if not isinstance(prompts, list):
                raise ValueError("prompt_list JSON must be a list of strings.")
    elif args.prompt_template:
        prompts = [args.prompt_template]
    else:
        prompts = PROMPT_PRESETS.get(args.prompt_preset, PROMPT_PRESETS["base"])

    print(f"Using {len(prompts)} prompt(s):")
    for i, p in enumerate(prompts):
        print(f"  [{i+1}] {p}")

    # --- 2. Load Data ---
    df = pd.read_csv(args.mapping)
    if "internal_item_id" not in df.columns or "item_token" not in df.columns:
        raise ValueError("mapping CSV must contain 'internal_item_id' and 'item_token'")
    has_title = "title" in df.columns

    # Sort by internal_item_id to align with row indices
    df = df.sort_values("internal_item_id")
    
    # Pre-compute raw text for each item to avoid re-looping
    # List of dicts or objects
    item_raw_texts = []
    for _, row in df.iterrows():
        if row["internal_item_id"] == 0:
            raw = args.pad_placeholder_text
        else:
            raw = str(row["title"]) if has_title else str(row["item_token"])
            if len(raw.strip()) == 0:
                raw = args.placeholder_text
        item_raw_texts.append(raw.strip())

    # --- 3. Load Model ---
    device = _select_device(args.device)
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        for cand in [getattr(tokenizer, "eos_token", None), getattr(tokenizer, "unk_token", None), getattr(tokenizer, "bos_token", None)]:
            if isinstance(cand, str) and tokenizer.convert_tokens_to_ids(cand) is not None:
                tokenizer.pad_token = cand
                break

    try:
        hf_cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model_type = getattr(hf_cfg, "model_type", "") or hf_cfg.__class__.__name__
        is_qwen_like = "qwen" in str(model_type).lower()
    except Exception:
        is_qwen_like = False

    use_causal = args.use_causal_lm or is_qwen_like
    if use_causal:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
        )
    
    if args.device_map is None:
        model = model.to(device)
    model.eval()

    use_chat = hasattr(tokenizer, "apply_chat_template") and (not args.no_chat_template)

    # --- 4. Encoding Loop ---
    # We will process items in batches. For each batch, we generate embeddings for ALL prompts.
    # Result structure: List of [Batch, K, D] or [Batch, K*D] depending on logic. 
    # To keep memory low, we'll accumulate processed arrays.

    needs_view_concat = args.output_mode == "concat" and args.view_project_dim is not None
    if needs_view_concat and not args.split_output_dir:
        raise ValueError("--view_project_dim requires --split_output_dir to store per-view tensors.")

    final_embs = [] if not needs_view_concat else None
    
    n_items = len(item_raw_texts)
    max_len_str = str(args.max_length) if isinstance(args.max_length, int) and args.max_length > 0 else "unlimited"
    print(f"Encoding {n_items} items with batch_size={args.batch_size}, {len(prompts)} prompts, mode={args.output_mode}...")
    
    split_chunks = None
    if args.split_output_dir:
        split_chunks = [[] for _ in range(len(prompts))]
    view_mats_for_concat = [] if needs_view_concat else None

    with tqdm(total=n_items, unit="items") as pbar:
        for i in range(0, n_items, args.batch_size):
            batch_raw = item_raw_texts[i : i + args.batch_size]
            
            # For this batch, calculate embeddings for each prompt
            batch_prompt_embs = [] # will hold [Batch_Size, Hidden] for each prompt
            
            for prompt_tmpl in prompts:
                # Prepare texts for this prompt
                batch_texts = []
                for raw in batch_raw:
                    base_prompt = prompt_tmpl.replace("{text}", raw)
                    if use_chat:
                        messages = [{"role": "user", "content": base_prompt}]
                        chat_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False
                        )
                        batch_texts.append(chat_text)
                    else:
                        batch_texts.append(base_prompt)

                # Encode
                emb = encode_batch(model, tokenizer, batch_texts, args.max_length, device, torch_dtype)
                batch_prompt_embs.append(emb) # [B, D]
            
            # Stack prompt embeddings: [B, K, D]
            # Note: batch_prompt_embs is List of [B, D]
            stacked = np.stack(batch_prompt_embs, axis=1) # [B, K, D]

            if split_chunks is not None:
                for view_idx in range(len(prompts)):
                    split_chunks[view_idx].append(stacked[:, view_idx, :].astype(np.float32, copy=False))
            
            # Process according to output mode
            if args.output_mode == "mean":
                # Average over K -> [B, D]
                final_batch = np.mean(stacked, axis=1)
                final_batch = l2_normalize(final_batch, axis=1) # Re-normalize after mean
            elif args.output_mode == "concat":
                # Concatenate over K -> [B, K*D]
                B, K, D = stacked.shape
                final_batch = stacked.reshape(B, K * D)
                # Note: Concat vectors are usually NOT re-normalized as a whole, 
                # but sub-vectors are already normalized.
            else: # stack
                final_batch = stacked
            
            if final_embs is not None:
                final_embs.append(final_batch)
            pbar.update(len(batch_raw))
            
    mat = None
    if final_embs is not None:
        if len(final_embs) == 0:
            mat = np.zeros((n_items, 0), dtype=np.float32)
        else:
            mat = np.concatenate(final_embs, axis=0)

    if mat is not None and mat.shape[0] > 0:
        if mat.ndim == 2:
            mat[0, :] = 0.0
        elif mat.ndim == 3:
            mat[0, :, :] = 0.0

    train_ids_cache = None
    if (args.view_project_dim is not None or args.project_dim is not None) and args.dataset:
        train_ids_cache = _load_train_item_ids(args, n_items - 1)

    if split_chunks is not None:
        os.makedirs(args.split_output_dir, exist_ok=True)
        split_meta = {
            "num_items": int(n_items),
            "num_prompts": len(prompts),
            "dtype": args.dtype,
            "prompts": [],
        }
        for view_idx, view_parts in enumerate(split_chunks):
            if len(view_parts) == 0:
                continue
            view_mat = np.concatenate(view_parts, axis=0).astype(np.float32, copy=False)
            if view_mat.shape[0] > 0:
                view_mat[0, :] = 0.0
            if args.view_project_dim is not None:
                view_mat = _apply_truncated_svd(
                    view_mat,
                    target_dim=args.view_project_dim,
                    train_ids=train_ids_cache,
                    random_state=args.svd_random_state,
                    label=f"view{view_idx}",
                )
            save_mat = view_mat
            if args.dtype == "float16":
                save_mat = save_mat.astype(np.float16)
            elif args.dtype == "bfloat16":
                save_mat = save_mat.astype(np.float32)
            view_path = os.path.join(args.split_output_dir, f"view_{view_idx}.npy")
            np.save(view_path, save_mat)
            split_meta["prompts"].append(
                {
                    "index": view_idx,
                    "prompt": prompts[view_idx],
                    "file": os.path.basename(view_path),
                    "vector_dim": int(view_mat.shape[1]),
                }
            )
            if view_mats_for_concat is not None:
                view_mats_for_concat.append(view_mat.astype(np.float32, copy=False))
        meta_path = os.path.join(args.split_output_dir, "views.json")
        with open(meta_path, "w") as mf:
            json.dump(split_meta, mf, ensure_ascii=False, indent=2)
        print(f"[Split] Saved per-view embeddings to {args.split_output_dir} (metadata: {meta_path})")

    if mat is None:
        if view_mats_for_concat is None or len(view_mats_for_concat) == 0:
            raise ValueError("No embeddings collected for final output; ensure output_mode supports view-based concat.")
        if args.output_mode == "concat":
            mat = np.concatenate(view_mats_for_concat, axis=1)
        else:
            raise ValueError("--view_project_dim currently only supports output_mode=concat.")
        if mat.shape[0] > 0:
            mat[0, :] = 0.0

    # --- 5. Optional Dimensionality Reduction (SVD) ---
    # NOTE: SVD only implemented for 2D matrices currently.
    if args.project_dim is not None:
        if mat.ndim != 2:
            print("Warning: SVD projection skipped because output is not 2D (mode=stack?).")
        else:
            mat = _apply_truncated_svd(
                mat,
                target_dim=args.project_dim,
                train_ids=train_ids_cache,
                random_state=args.svd_random_state,
                label="final",
            )

    # --- 6. Save ---
    if args.dtype == "float16":
        mat = mat.astype(np.float16)
    elif args.dtype == "bfloat16":
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, mat)
    print(
        f"Saved Qwen3 embeddings to: {os.path.abspath(args.output)}  "
        f"shape={mat.shape}  dtype={mat.dtype}  (prompts={len(prompts)}, mode={args.output_mode})"
    )


if __name__ == "__main__":
    main()
