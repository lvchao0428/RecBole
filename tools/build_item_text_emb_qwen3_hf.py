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
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import torch
import importlib
import transformers as _transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm


def _patch_transformers_namespace_for_stream_generator():
    """Qwen ``trust_remote_code`` may pull in ``transformers_stream_generator``, whose
    ``main.py`` does ``from transformers import BeamSearchScorer, ...``. Recent
    ``transformers`` releases moved or removed these from the top-level package;
    re-export from submodules or add minimal stubs so that import succeeds.

    This embedding script never uses beam search or streaming generation; only
    ``from_pretrained`` needs to load remote code without ``ImportError``.
    """
    tf = _transformers

    def _ensure(name: str, modules: tuple[str, ...]) -> None:
        if hasattr(tf, name):
            return
        for mod_name in modules:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, name):
                    setattr(tf, name, getattr(mod, name))
                    return
            except Exception:
                continue
        # Removed entirely in some versions (e.g. transformers v5+): placeholder class.
        setattr(tf, name, type(name, (), {}))

    # transformers_stream_generator/main.py (lines 1–11) — beam / constraint symbols.
    _ensure("BeamSearchScorer", ("transformers.generation.beam_search",))
    _ensure("ConstrainedBeamSearchScorer", ("transformers.generation.beam_search",))
    _ensure("DisjunctiveConstraint", ("transformers.generation.beam_constraints",))
    _ensure("PhrasalConstraint", ("transformers.generation.beam_constraints",))


_patch_transformers_namespace_for_stream_generator()
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize


def _tokenizer_has_usable_chat_template(tokenizer) -> bool:
    """Return True only if ``apply_chat_template`` can run (template string is set).

    Many tokenizers define ``apply_chat_template`` but raise
    ``ValueError: ... chat_template is not set`` when the model repo omits
    ``chat_template`` in ``tokenizer_config.json`` (e.g. incomplete local copy).
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    ct = getattr(tokenizer, "chat_template", None)
    if ct is None:
        return False
    if isinstance(ct, str) and not ct.strip():
        return False
    return True

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
    "multiview-opt": [
        "Core attributes of [TITLE] {text}: name, function, main category",
        "Key traits of [TITLE] {text}: applicable group, core material/ingredient, usage scenario",
        "User value of [TITLE] {text}: solved pain point/met demand",
        "Fine category of [TITLE] {text} (specific type only)"
    ],
    # Universal multi-view prompts based on 5W1H semantic orthogonality
    # Designed for cross-domain effectiveness (Beauty, Toys, etc.)
    # Key insight: What vs Who forms strong adversarial pair (-0.23 on Beauty)
    "multiview-universal": [
        # View 0: WHAT - functional capabilities (adversarial to View 1)
        "What are the main functions and features of [TITLE] {text}?",
        # View 1: WHO - user demographics (adversarial to View 0)
        "Who is the ideal user or target audience for [TITLE] {text}?",
        # View 2: WHEN/WHERE - usage context (adversarial to View 3)
        "When and where would someone typically use [TITLE] {text}?",
        # View 3: HOW - physical attributes (adversarial to View 2)
        "How is [TITLE] {text} designed? Describe its materials, size, and appearance.",
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
    p.add_argument(
        "--flash_attn",
        action="store_true",
        help="Enable Flash Attention 2 for faster inference (requires flash-attn package).",
    )
    p.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in INT8 quantization (requires bitsandbytes). Reduces VRAM ~50%%.",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in INT4 quantization (requires bitsandbytes). Reduces VRAM ~75%%.",
    )
    
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
    
    # --- Generative Mode Arguments ---
    p.add_argument(
        "--generative",
        action="store_true",
        help="Enable generative mode: generate answer first, then extract embedding from the answer.",
    )
    p.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate in generative mode (default: 128).",
    )
    p.add_argument(
        "--gen_temperature",
        type=float,
        default=0.0,
        help="Temperature for generation. 0 = deterministic greedy decoding (default: 0).",
    )
    p.add_argument(
        "--gen_top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0, no filtering).",
    )
    p.add_argument(
        "--gen_top_k",
        type=int,
        default=0,
        help="Top-k sampling parameter. 0 = disabled (default: 0).",
    )
    p.add_argument(
        "--gen_repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for generation (default: 1.0, no penalty).",
    )
    p.add_argument(
        "--save_generated_texts",
        default=None,
        help="Path to save generated texts as JSON (e.g., generated_texts.json). Only used in generative mode.",
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
        "--svd_report_evr",
        action="store_true",
        help="Print TruncatedSVD explained_variance_ratio_ (EVR) summary for each SVD fit. Default: off.",
    )
    p.add_argument(
        "--svd_report_evr_path",
        default=None,
        help="If set, save SVD EVR records as JSON to this path. Default: None (do not save).",
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
    p.add_argument(
        "--whiten",
        action="store_true",
        help="Enable whitening transformation (disabled by default).",
    )
    p.add_argument(
        "--center",
        action="store_true",
        help="Enable centering (mean subtraction). If not set, skip center+whiten and only L2 normalize.",
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


@torch.no_grad()
def encode_batch_generative(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    gen_max_new_tokens: int = 128,
    gen_temperature: float = 0.0,
    gen_top_p: float = 1.0,
    gen_top_k: int = 0,
    gen_repetition_penalty: float = 1.0,
    return_generated_texts: bool = False,
) -> Union[np.ndarray, tuple]:
    """Generate answers for prompts, then extract embeddings from the generated text.
    
    This is a generative approach: 
    1. Feed prompt to model
    2. Generate answer text (with temperature=0 for determinism)
    3. Extract embedding from the generated answer
    
    **Optimized**: Uses batch generation with left-padding for significant speedup.
    
    Args:
        model: HuggingFace CausalLM model
        tokenizer: HuggingFace tokenizer
        texts: List of prompt texts (with item info already filled in)
        max_length: Max input sequence length (0 = no truncation)
        device: torch device
        torch_dtype: torch dtype for computation
        gen_max_new_tokens: Max new tokens to generate
        gen_temperature: Temperature for sampling (0 = greedy)
        gen_top_p: Nucleus sampling parameter
        gen_top_k: Top-k sampling parameter
        gen_repetition_penalty: Repetition penalty
        return_generated_texts: If True, also return the generated texts
    
    Returns:
        If return_generated_texts is False:
            np.ndarray of shape [batch_size, hidden_dim]
        If return_generated_texts is True:
            tuple of (np.ndarray, List[str]) where the list contains generated texts
    """
    do_trunc = isinstance(max_length, int) and max_length > 0
    
    # Determine sampling strategy
    do_sample = gen_temperature > 0
    
    # =============================================
    # Find pad/eos token for Qwen 1.x (which has None for all special tokens)
    # =============================================
    vocab = tokenizer.get_vocab()
    pad_token_id_to_use = None
    eos_token_id_to_use = None
    
    for cand in ['<|endoftext|>', '<|extra_0|>', '<|im_end|>', '</s>', '<eos>']:
        if cand in vocab:
            if pad_token_id_to_use is None:
                pad_token_id_to_use = vocab[cand]
            if eos_token_id_to_use is None:
                eos_token_id_to_use = vocab[cand]
            break
    
    if pad_token_id_to_use is None:
        pad_token_id_to_use = 0
    if eos_token_id_to_use is None:
        eos_token_id_to_use = pad_token_id_to_use
    
    # Build generation config with valid token ids
    gen_kwargs = {
        "max_new_tokens": gen_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id_to_use,
        "eos_token_id": eos_token_id_to_use,
    }
    
    # Only add sampling parameters when do_sample=True
    if do_sample:
        gen_kwargs["temperature"] = gen_temperature
        gen_kwargs["top_p"] = gen_top_p
        gen_kwargs["repetition_penalty"] = gen_repetition_penalty
        if gen_top_k > 0:
            gen_kwargs["top_k"] = gen_top_k
    # When do_sample=False (greedy), don't set temperature/top_p/top_k to avoid warnings
    
    # Store generated texts for embedding extraction
    generated_texts = []
    
    # =============================================
    # Try batch generation first (Qwen 2.5 supports it)
    # Fall back to sequential if it fails (Qwen 1.x)
    # =============================================
    try:
        # Batch tokenize with left-padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        enc = tokenizer(
            texts,
            padding=True,
            truncation=do_trunc,
            max_length=(max_length if do_trunc else None),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        
        tokenizer.padding_side = original_padding_side
        
        # Batch generate
        output_ids = model.generate(
            **enc,
            **gen_kwargs,
        )
        
        # Decode each generated sequence
        seq_len = enc["input_ids"].shape[1]
        
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, seq_len:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            if len(generated_text.strip()) == 0:
                generated_text = texts[i]
            
            generated_texts.append(generated_text)
            
    except Exception as e:
        # Fallback to sequential generation (for Qwen 1.x or other issues)
        print(f"[Warning] Batch generation failed ({e}), falling back to sequential...")
        generated_texts = []
        
        for txt in texts:
            enc = tokenizer(
                txt,
                padding=False,
                truncation=do_trunc,
                max_length=(max_length if do_trunc else None),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            try:
                output_ids = model.generate(**enc, **gen_kwargs)
                input_len = enc["input_ids"].shape[1]
                new_tokens = output_ids[0, input_len:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                if len(generated_text.strip()) == 0:
                    generated_text = txt
                
                generated_texts.append(generated_text)
            except Exception:
                generated_texts.append(txt)
    
    # Now extract embeddings from the generated texts
    # We can reuse the non-generative encode_batch for this
    if getattr(tokenizer, "pad_token_id", None) is None:
        # Fallback: encode one by one if no pad token
        outs = []
        for gen_txt in generated_texts:
            enc = tokenizer(
                gen_txt,
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
        emb_result = np.concatenate(outs, axis=0) if outs else np.zeros((0, model.config.hidden_size), dtype=np.float32)
        if return_generated_texts:
            return emb_result, generated_texts
        return emb_result
    else:
        # Batch encoding of generated texts
        enc = tokenizer(
            generated_texts,
            padding=True,
            truncation=do_trunc,
            max_length=(max_length if do_trunc else None),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        
        if enc["input_ids"].shape[1] == 0:
            emb_result = np.zeros((len(generated_texts), model.config.hidden_size), dtype=np.float32)
            if return_generated_texts:
                return emb_result, generated_texts
            return emb_result

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
        emb_result = emb.to(torch.float32).detach().cpu().numpy()
        if return_generated_texts:
            return emb_result, generated_texts
        return emb_result


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


def _center_whiten_and_normalize(
    emb: np.ndarray,
    train_ids: np.ndarray,
    output_stats_path: Optional[str] = None,
    enable_whiten: bool = False,
    enable_center: bool = False,
) -> np.ndarray:
    """Center + Whiten + L2 normalize, using training set statistics only.
    
    Args:
        emb: Embedding matrix [n_items, d], row 0 is PAD (zeros)
        train_ids: Array of training item internal IDs (excluding PAD=0)
        output_stats_path: Optional path to save mean and whiten_matrix
        enable_whiten: Whether to apply whitening (if False, only center+L2)
        enable_center: Whether to apply centering (if False, skip center+whiten entirely and only L2 normalize)
    
    Returns:
        Processed embedding matrix with same shape as input
    """
    if emb is None or emb.size == 0:
        return emb
    
    # Handle 3D tensors (multi-view stacked)
    if emb.ndim == 3:
        print("[WARN] Skipping center/whiten for 3D tensor (mode=stack). Apply per-view instead.")
        return emb
    
    # If centering is disabled, skip center+whiten entirely and only L2 normalize
    if not enable_center:
        print("[Center] Centering disabled. Only applying L2 normalization.")
        emb_processed = emb.copy().astype(np.float32)
        norms = np.linalg.norm(emb_processed[1:], axis=1, keepdims=True)
        emb_processed[1:] = emb_processed[1:] / np.clip(norms, 1e-8, None)
        emb_processed[0, :] = 0.0  # Keep PAD as zeros
        return emb_processed
    
    # Extract training embeddings (exclude PAD=0)
    if train_ids is None or len(train_ids) == 0:
        train_mask = np.arange(1, len(emb))
    else:
        train_mask = train_ids[train_ids > 0]
    
    train_emb = emb[train_mask]
    
    if len(train_emb) == 0:
        print("[WARN] No training embeddings found; skipping center/whiten.")
        return emb
    
    # Step 1: Center (compute mean on training set only)
    mean = train_emb.mean(axis=0, keepdims=True).astype(np.float64)
    emb_centered = (emb - mean).astype(np.float64)
    emb_centered[0, :] = 0.0  # Keep PAD as zeros
    
    whiten_matrix = None
    if enable_whiten:
        # Step 2: Compute whitening matrix (on training set only)
        # Use float64 for numerical stability
        train_centered = (train_emb - mean).astype(np.float64)
        cov = (train_centered.T @ train_centered) / len(train_centered)
        
        # SVD decomposition: cov = U @ diag(S) @ U.T
        U, S, _ = np.linalg.svd(cov)
        
        # Debug: log eigenvalue statistics
        print(f"[Whiten] Eigenvalue stats: min={S.min():.6f}, max={S.max():.6f}, mean={S.mean():.6f}")
        print(f"[Whiten] Condition number: {S.max() / (S.min() + 1e-10):.2f}")
        
        # Whitening matrix: U @ diag(1/sqrt(S))
        whiten_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
        
        # Step 3: Apply whitening to all embeddings (use float64 for computation)
        emb_whitened = emb_centered @ whiten_matrix
        emb_whitened[0, :] = 0.0
        
        # NOTE: Do NOT L2 normalize after whitening!
        # Whitening already decorrelates features and sets Cov(X) = I
        # L2 normalization would destroy this property (variance becomes 1/D instead of 1)
        # If the model needs L2-normalized embeddings, do it at model load time
        
        emb_processed = emb_whitened.astype(np.float32)  # Cast back to float32
        
    else:
        emb_processed = emb_centered
        
        # Step 4: L2 normalize
        norms = np.linalg.norm(emb_processed[1:], axis=1, keepdims=True)
        emb_processed[1:] = emb_processed[1:] / np.clip(norms, 1e-8, None)
    
    # Step 5: Save statistics for inference reuse
    if output_stats_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_stats_path)), exist_ok=True)
        if enable_whiten and whiten_matrix is not None:
            np.savez(
                output_stats_path,
                mean=mean.astype(np.float32),
                whiten_matrix=whiten_matrix.astype(np.float32),
            )
            print(f"[Whiten] Saved center+whiten stats to: {output_stats_path}")
        else:
            np.savez(
                output_stats_path,
                mean=mean.astype(np.float32),
            )
            print(f"[Center] Saved center stats to: {output_stats_path}")
    
    return emb_processed.astype(np.float32)


def _apply_truncated_svd(
    mat: np.ndarray, 
    target_dim: int, 
    train_ids, 
    random_state: int, 
    label: str,
    normalize: bool = True,
    report_evr: bool = False,
    evr_records: Optional[list] = None,
):
    """Apply TruncatedSVD dimensionality reduction.
    
    Args:
        mat: Input matrix [n_items, d]
        target_dim: Target dimensionality
        train_ids: Training item IDs for fitting
        random_state: Random seed
        label: Label for logging
        normalize: Whether to L2 normalize after SVD (default: True)
                  Set to False if whitening will be applied afterwards.
    """
    if mat.ndim != 2:
        raise ValueError("SVD can only be applied to 2D matrices.")
    orig_dim = mat.shape[1]
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("--project_dim/--view_project_dim must be > 0")
    if target_dim >= orig_dim:
        if normalize and mat.shape[0] > 1:
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

    # Optional: report / record explained variance ratio (EVR) for analysis.
    if report_evr or evr_records is not None:
        evr = getattr(svd, "explained_variance_ratio_", None)
        if evr is not None and len(evr) > 0:
            evr_sum = float(np.sum(evr))
            top1 = float(evr[0])
            top5 = float(np.sum(evr[: min(5, len(evr))]))
            top10 = float(np.sum(evr[: min(10, len(evr))]))
            if report_evr:
                print(
                    f"[SVD:{label}] EVR(sum)={evr_sum:.6f}  "
                    f"top1={top1:.6f}  top5={top5:.6f}  top10={top10:.6f}"
                )
            if evr_records is not None:
                evr_records.append(
                    {
                        "label": label,
                        "orig_dim": int(orig_dim),
                        "target_dim": int(target_dim),
                        "svd_k": int(svd_k),
                        "fit_rows": int(subset.shape[0]),
                        "evr_sum": evr_sum,
                        "evr_top1": top1,
                        "evr_top5": top5,
                        "evr_top10": top10,
                        "explained_variance_ratio": evr.astype(float).tolist(),
                    }
                )

    nonpad = mat[1:, :].astype(np.float32, copy=False)
    reduced = svd.transform(nonpad)
    if svd_k < target_dim:
        pad = np.zeros((reduced.shape[0], target_dim - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)
    
    # Optionally normalize after SVD
    if normalize:
        reduced = l2_normalize(reduced, norm="l2", axis=1)
    
    projected = np.zeros((mat.shape[0], target_dim), dtype=reduced.dtype)
    projected[1:, :] = reduced
    return projected


def main():
    args = parse_args()

    svd_evr_records: Optional[list] = None
    if args.svd_report_evr or args.svd_report_evr_path is not None:
        svd_evr_records = []

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
    
    # Debug: print all special tokens
    print(f"[Tokenizer Debug] eos_token='{tokenizer.eos_token}', eos_token_id={tokenizer.eos_token_id}")
    print(f"[Tokenizer Debug] pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")
    
    # Ensure pad_token is set (Qwen 1.x doesn't support add_special_tokens)
    def _check_pad_valid(tok):
        try:
            pid = tok.pad_token_id
            return pid is not None and isinstance(pid, int) and pid >= 0
        except Exception:
            return False
    
    if not _check_pad_valid(tokenizer):
        vocab = tokenizer.get_vocab()
        # Qwen 1.x uses <|endoftext|> as eos, check vocab
        candidates = ['<|endoftext|>', '<|extra_0|>', '<|im_end|>', '<eos>', '</s>', '[PAD]', '<pad>']
        
        set_pad_token = None
        set_pad_id = None
        for cand in candidates:
            if cand in vocab:
                set_pad_token = cand
                set_pad_id = vocab[cand]
                break
        
        if set_pad_id is None:
            # Fallback: use token 0
            set_pad_id = 0
            set_pad_token = tokenizer.decode([0])
        
        # Directly set attributes (for tokenizers that don't support add_special_tokens)
        try:
            tokenizer.add_special_tokens({'pad_token': set_pad_token})
            print(f"[Tokenizer] Added pad_token = '{set_pad_token}'")
        except ValueError:
            # Qwen 1.x doesn't support adding tokens, set directly
            tokenizer.pad_token = set_pad_token
            tokenizer._pad_token = set_pad_token
            # Also need to set the internal special tokens map
            if hasattr(tokenizer, 'special_tokens'):
                tokenizer.special_tokens['<|padding|>'] = set_pad_id
            print(f"[Tokenizer] Directly set pad_token = '{set_pad_token}' (id={set_pad_id})")
    
    # For Qwen tokenizer, we need to handle padding manually if pad_token_id is still None
    print(f"[Tokenizer] Final: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")

    try:
        hf_cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model_type = getattr(hf_cfg, "model_type", "") or hf_cfg.__class__.__name__
        is_qwen_like = "qwen" in str(model_type).lower()
    except Exception:
        is_qwen_like = False

    # Generative mode requires CausalLM
    use_causal = args.use_causal_lm or is_qwen_like or args.generative
    if args.generative and not use_causal:
        print("[Generative] Forcing use_causal_lm=True for generative mode.")
        use_causal = True
    
    # Flash Attention 2 support (with fallback for models that don't support it)
    use_flash_attn = args.flash_attn
    if use_flash_attn:
        print("[Flash Attention 2] Attempting to enable - requires flash-attn package.")
    
    # Quantization support
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if args.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("[Quantization] INT4 (NF4) enabled - VRAM reduced ~75%")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                print("[Quantization] INT8 enabled - VRAM reduced ~50%")
        except ImportError:
            print("[Quantization] ⚠️  bitsandbytes not installed. Run: pip install bitsandbytes")
            quantization_config = None
    
    def _load_model_with_fallback(model_cls, use_flash: bool, quant_config):
        """Try to load model with flash attention, fallback if not supported."""
        base_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": args.device_map or ("auto" if quant_config else None),  # quantization requires device_map
            "low_cpu_mem_usage": True,
        }
        
        if quant_config:
            base_kwargs["quantization_config"] = quant_config
        
        if use_flash:
            try:
                model = model_cls.from_pretrained(
                    args.model_name_or_path,
                    attn_implementation="flash_attention_2",
                    **base_kwargs,
                )
                print("[Flash Attention 2] ✅ Enabled successfully.")
                return model
            except TypeError as e:
                if "attn_implementation" in str(e):
                    print(f"[Flash Attention 2] ⚠️  Model doesn't support attn_implementation, falling back to default.")
                else:
                    raise
            except ImportError:
                print("[Flash Attention 2] ⚠️  flash-attn not installed, falling back to default.")
        
        # Fallback: load without flash attention
        return model_cls.from_pretrained(args.model_name_or_path, **base_kwargs)
    
    if use_causal:
        model = _load_model_with_fallback(AutoModelForCausalLM, use_flash_attn, quantization_config)
    else:
        model = _load_model_with_fallback(AutoModel, use_flash_attn, quantization_config)
    
    # Move to device only if not using device_map or quantization
    if args.device_map is None and quantization_config is None:
        model = model.to(device)
    model.eval()

    use_chat = (
        _tokenizer_has_usable_chat_template(tokenizer) and (not args.no_chat_template)
    )
    if not use_chat and not args.no_chat_template:
        if args.use_chat_template:
            print(
                "[WARN] --use_chat_template set but tokenizer.chat_template is missing or empty; "
                "using raw prompts. Fix: add chat_template to tokenizer_config.json under "
                "--model_name_or_path, or pass --no_chat_template to silence this."
            )
        else:
            print(
                "[INFO] tokenizer.chat_template is unset; using raw prompts (no chat wrapping)."
            )
    elif use_chat:
        print("[INFO] Using tokenizer.apply_chat_template() for prompts.")

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
    
    # Print mode info
    mode_str = "generative" if args.generative else "embedding-only"
    if args.generative:
        print(f"[Generative Mode] temperature={args.gen_temperature}, max_new_tokens={args.gen_max_new_tokens}")
    print(f"Encoding {n_items} items with batch_size={args.batch_size}, {len(prompts)} prompts, mode={args.output_mode}, encode_mode={mode_str}...")
    
    split_chunks = None
    if args.split_output_dir:
        split_chunks = [[] for _ in range(len(prompts))]
    view_mats_for_concat = [] if needs_view_concat else None
    
    # Storage for generated texts (only used in generative mode with --save_generated_texts)
    save_gen_texts = args.generative and args.save_generated_texts is not None
    all_generated_texts = [] if save_gen_texts else None  # List of dicts per item

    with tqdm(total=n_items, unit="items") as pbar:
        for i in range(0, n_items, args.batch_size):
            batch_raw = item_raw_texts[i : i + args.batch_size]
            batch_start_idx = i
            
            # For this batch, calculate embeddings for each prompt
            batch_prompt_embs = [] # will hold [Batch_Size, Hidden] for each prompt
            batch_gen_texts_per_view = [] if save_gen_texts else None  # [K][B] texts
            
            for prompt_idx, prompt_tmpl in enumerate(prompts):
                # Prepare texts for this prompt
                batch_texts = []
                for raw in batch_raw:
                    base_prompt = prompt_tmpl.replace("{text}", raw)
                    if use_chat:
                        messages = [{"role": "user", "content": base_prompt}]
                        # For generative mode, add generation prompt to trigger response
                        chat_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=args.generative
                        )
                        batch_texts.append(chat_text)
                    else:
                        batch_texts.append(base_prompt)

                # Encode: use generative or embedding-only mode
                if args.generative:
                    if save_gen_texts:
                        emb, gen_texts = encode_batch_generative(
                            model, tokenizer, batch_texts, args.max_length, device, torch_dtype,
                            gen_max_new_tokens=args.gen_max_new_tokens,
                            gen_temperature=args.gen_temperature,
                            gen_top_p=args.gen_top_p,
                            gen_top_k=args.gen_top_k,
                            gen_repetition_penalty=args.gen_repetition_penalty,
                            return_generated_texts=True,
                        )
                        batch_gen_texts_per_view.append(gen_texts)
                    else:
                        emb = encode_batch_generative(
                            model, tokenizer, batch_texts, args.max_length, device, torch_dtype,
                            gen_max_new_tokens=args.gen_max_new_tokens,
                            gen_temperature=args.gen_temperature,
                            gen_top_p=args.gen_top_p,
                            gen_top_k=args.gen_top_k,
                            gen_repetition_penalty=args.gen_repetition_penalty,
                        )
                else:
                    emb = encode_batch(model, tokenizer, batch_texts, args.max_length, device, torch_dtype)
                batch_prompt_embs.append(emb) # [B, D]
            
            # Collect generated texts for this batch
            if save_gen_texts and batch_gen_texts_per_view:
                for batch_offset, raw_text in enumerate(batch_raw):
                    item_idx = batch_start_idx + batch_offset
                    item_record = {
                        "internal_item_id": item_idx,
                        "raw_text": raw_text,
                        "views": []
                    }
                    for view_idx, prompt_tmpl in enumerate(prompts):
                        item_record["views"].append({
                            "view_index": view_idx,
                            "prompt": prompt_tmpl,
                            "generated_text": batch_gen_texts_per_view[view_idx][batch_offset]
                        })
                    all_generated_texts.append(item_record)
            
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

    enable_whiten = args.whiten
    enable_center = args.center

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
                # Don't normalize here - let the whitening step handle it
                view_mat = _apply_truncated_svd(
                    view_mat,
                    target_dim=args.view_project_dim,
                    train_ids=train_ids_cache,
                    random_state=args.svd_random_state,
                    label=f"view{view_idx}",
                    normalize=False,  # Preserve variance for whitening
                    report_evr=args.svd_report_evr,
                    evr_records=svd_evr_records,
                )
            
            # Apply center + whiten to each view independently
            if enable_center or enable_whiten:
                view_stats_path = os.path.join(
                    args.split_output_dir, 
                    f"view_{view_idx}_whiten_stats.npz"
                ) if enable_center else None
                view_mat = _center_whiten_and_normalize(
                    view_mat,
                    train_ids_cache,
                    output_stats_path=view_stats_path,
                    enable_whiten=enable_whiten,
                    enable_center=enable_center,
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
    if mat is not None and mat.shape[0] > 0:
        if mat.ndim == 2:
            mat[0, :] = 0.0
        elif mat.ndim == 3:
            mat[0, :, :] = 0.0

    # --- 5. Optional Dimensionality Reduction (SVD) ---
    # NOTE: SVD only implemented for 2D matrices currently.
    if args.project_dim is not None:
        if mat.ndim != 2:
            print("Warning: SVD projection skipped because output is not 2D (mode=stack?).")
        else:
            # SVD projection BEFORE whitening - do NOT normalize to preserve variance
            mat = _apply_truncated_svd(
                mat,
                target_dim=args.project_dim,
                train_ids=train_ids_cache,
                random_state=args.svd_random_state,
                label="final",
                normalize=False,  # Preserve variance for subsequent whitening
                report_evr=args.svd_report_evr,
                evr_records=svd_evr_records,
            )

    # --- 6. Apply Center + Whiten normalization (after final projection) ---
    if (enable_center or enable_whiten) and mat is not None and mat.ndim == 2:
        stats_path = args.output.replace('.npy', '_whiten_stats.npz') if enable_center else None
        mat = _center_whiten_and_normalize(
            mat,
            train_ids_cache,
            output_stats_path=stats_path,
            enable_whiten=enable_whiten,
            enable_center=enable_center,
        )

    # --- 7. Save ---
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

    # --- 8. Save SVD EVR report (optional) ---
    if args.svd_report_evr_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.svd_report_evr_path)), exist_ok=True)
        evr_payload = {
            "meta": {
                "output": os.path.abspath(args.output),
                "output_mode": args.output_mode,
                "num_prompts": int(len(prompts)),
                "prompt_preset": args.prompt_preset,
                "model_name_or_path": args.model_name_or_path,
                "dataset": args.dataset,
                "project_dim": args.project_dim,
                "view_project_dim": args.view_project_dim,
                "svd_random_state": args.svd_random_state,
                "center": bool(args.center),
                "whiten": bool(args.whiten),
            },
            "records": svd_evr_records or [],
        }
        with open(args.svd_report_evr_path, "w", encoding="utf-8") as f:
            json.dump(evr_payload, f, ensure_ascii=False, indent=2)
        print(f"[SVD] Saved EVR report to: {os.path.abspath(args.svd_report_evr_path)}")
    
    # --- 9. Save Generated Texts (if enabled) ---
    if all_generated_texts is not None and len(all_generated_texts) > 0:
        gen_texts_output = {
            "num_items": len(all_generated_texts),
            "num_views": len(prompts),
            "prompts": prompts,
            "items": all_generated_texts,
        }
        gen_texts_path = args.save_generated_texts
        os.makedirs(os.path.dirname(os.path.abspath(gen_texts_path)), exist_ok=True)
        with open(gen_texts_path, "w", encoding="utf-8") as f:
            json.dump(gen_texts_output, f, ensure_ascii=False, indent=2)
        print(f"[Generative] Saved generated texts to: {os.path.abspath(gen_texts_path)} ({len(all_generated_texts)} items)")


if __name__ == "__main__":
    main()
