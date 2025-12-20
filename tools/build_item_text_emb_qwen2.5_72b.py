#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build item text embeddings using Qwen2.5-72B-Instruct-GPTQ-Int8 model.

该脚本专为 Qwen2.5-72B GPTQ-Int8 量化模型优化，支持多GPU推理。
基于 build_item_text_emb_qwen3_hf.py 改写，保持相同的 project 维度和接口。

主要特性：
- 支持 GPTQ-Int8 量化模型
- 支持多GPU tensor parallel (8x4090)
- 支持多视图 prompt 生成
- 支持 SVD 降维和白化

Input: mapping CSV exported by tools/export_internal_item_mapping.py
Output: item_text_emb.qwen2.5_72b.npy

Example:
  python tools/build_item_text_emb_qwen2.5_72b.py \
    --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
    --model_name_or_path /data/model/Qwen2.5-72B-Instruct-GPTQ-Int8 \
    --output dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.npy \
    --prompt_preset multiview \
    --output_mode concat \
    --batch_size 4 \
    --dtype float16

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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    "multiview-opt": [
        "Core attributes of [TITLE] {text}: name, function, main category",
        "Key traits of [TITLE] {text}: applicable group, core material/ingredient, usage scenario",
        "User value of [TITLE] {text}: solved pain point/met demand",
        "Fine category of [TITLE] {text} (specific type only)"
    ],
    # Universal multi-view prompts based on 5W1H semantic orthogonality
    "multiview-universal": [
        # View 0: WHAT - functional capabilities
        "What are the main functions and features of [TITLE] {text}?",
        # View 1: WHO - user demographics
        "Who is the ideal user or target audience for [TITLE] {text}?",
        # View 2: WHEN/WHERE - usage context
        "When and where would someone typically use [TITLE] {text}?",
        # View 3: HOW - physical attributes
        "How is [TITLE] {text} designed? Describe its materials, size, and appearance.",
    ],
    "description": [
        "Describe [TITLE] {text} in detail.",
    ]
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Qwen2.5-72B item text embeddings (GPTQ-Int8)")
    p.add_argument("--mapping", required=True, help="CSV from export_internal_item_mapping.py")
    p.add_argument("--model_name_or_path", required=True, help="HF model id or local path (e.g., /data/model/Qwen2.5-72B-Instruct-GPTQ-Int8)")
    p.add_argument("--output", required=True, help="Output .npy path for embeddings")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (smaller for large models)")
    p.add_argument(
        "--max_length",
        type=int,
        default=0,
        help="Max sequence length; 0 means no truncation.",
    )
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default=None, help="cuda device or cpu (ignored when device_map=auto)")
    p.add_argument("--device_map", default="auto", help="HF accelerate device_map (default: 'auto' for multi-GPU)")
    p.add_argument(
        "--flash_attn",
        action="store_true",
        help="Enable Flash Attention 2 for faster inference (requires flash-attn package).",
    )
    p.add_argument(
        "--max_memory",
        default=None,
        help="Max memory per GPU in GB (e.g., '20' for 20GB). Auto-detected if not set.",
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
        default=None,
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


def _get_max_memory_dict(max_memory_gb: Optional[str] = None) -> Optional[dict]:
    """获取多GPU显存分配配置"""
    if not torch.cuda.is_available():
        return None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None
    
    if max_memory_gb is not None:
        # 用户指定的显存限制
        max_mem = f"{max_memory_gb}GB"
    else:
        # 自动检测：使用90%的可用显存
        max_mem_dict = {}
        for i in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            max_mem_dict[i] = int(total_mem * 0.9)
        max_mem_dict["cpu"] = "32GB"
        return max_mem_dict
    
    return {i: max_mem for i in range(num_gpus)}


@torch.no_grad()
def encode_batch(
    model,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    """编码文本批次，提取隐藏状态作为embedding"""
    do_trunc = isinstance(max_length, int) and max_length > 0
    
    # 确保有pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 批量编码
    enc = tokenizer(
        texts,
        padding=True,
        truncation=do_trunc,
        max_length=(max_length if do_trunc else None),
        return_tensors="pt",
    )
    
    # 移动到正确的设备（对于multi-GPU，模型会自动处理）
    if hasattr(model, 'hf_device_map'):
        # Multi-GPU模式，输入会自动分配
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            first_device = torch.device(first_device)
        elif isinstance(first_device, int):
            first_device = torch.device(f"cuda:{first_device}")
        enc = {k: v.to(first_device) for k, v in enc.items()}
    else:
        enc = {k: v.to(device) for k, v in enc.items()}
    
    if enc["input_ids"].shape[1] == 0:
        hidden_size = model.config.hidden_size
        return np.zeros((len(texts), hidden_size), dtype=np.float32)

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
    model,
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
    """生成模式：先生成回答，再从回答中提取embedding"""
    do_trunc = isinstance(max_length, int) and max_length > 0
    do_sample = gen_temperature > 0
    
    # 确保有pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 生成配置
    gen_kwargs = {
        "max_new_tokens": gen_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if do_sample:
        gen_kwargs["temperature"] = gen_temperature
        gen_kwargs["top_p"] = gen_top_p
        gen_kwargs["repetition_penalty"] = gen_repetition_penalty
        if gen_top_k > 0:
            gen_kwargs["top_k"] = gen_top_k
    
    generated_texts = []
    
    # 批量生成
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    enc = tokenizer(
        texts,
        padding=True,
        truncation=do_trunc,
        max_length=(max_length if do_trunc else None),
        return_tensors="pt",
    )
    
    # 移动到正确的设备
    if hasattr(model, 'hf_device_map'):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            first_device = torch.device(first_device)
        elif isinstance(first_device, int):
            first_device = torch.device(f"cuda:{first_device}")
        enc = {k: v.to(first_device) for k, v in enc.items()}
    else:
        enc = {k: v.to(device) for k, v in enc.items()}
    
    tokenizer.padding_side = original_padding_side
    
    # 批量生成
    output_ids = model.generate(**enc, **gen_kwargs)
    seq_len = enc["input_ids"].shape[1]
    
    for i in range(output_ids.shape[0]):
        new_tokens = output_ids[i, seq_len:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if len(generated_text.strip()) == 0:
            generated_text = texts[i]
        
        generated_texts.append(generated_text)
    
    # 从生成的文本中提取embedding
    emb_result = encode_batch(model, tokenizer, generated_texts, max_length, device, torch_dtype)
    
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
    """Center + Whiten + L2 normalize，使用训练集统计量"""
    if emb is None or emb.size == 0:
        return emb
    
    if emb.ndim == 3:
        print("[WARN] Skipping center/whiten for 3D tensor (mode=stack). Apply per-view instead.")
        return emb
    
    if not enable_center:
        print("[Center] Centering disabled. Only applying L2 normalization.")
        emb_processed = emb.copy().astype(np.float32)
        norms = np.linalg.norm(emb_processed[1:], axis=1, keepdims=True)
        emb_processed[1:] = emb_processed[1:] / np.clip(norms, 1e-8, None)
        emb_processed[0, :] = 0.0
        return emb_processed
    
    if train_ids is None or len(train_ids) == 0:
        train_mask = np.arange(1, len(emb))
    else:
        train_mask = train_ids[train_ids > 0]
    
    train_emb = emb[train_mask]
    
    if len(train_emb) == 0:
        print("[WARN] No training embeddings found; skipping center/whiten.")
        return emb
    
    # Step 1: Center
    mean = train_emb.mean(axis=0, keepdims=True).astype(np.float64)
    emb_centered = (emb - mean).astype(np.float64)
    emb_centered[0, :] = 0.0
    
    whiten_matrix = None
    if enable_whiten:
        # Step 2: Whitening
        train_centered = (train_emb - mean).astype(np.float64)
        cov = (train_centered.T @ train_centered) / len(train_centered)
        U, S, _ = np.linalg.svd(cov)
        
        print(f"[Whiten] Eigenvalue stats: min={S.min():.6f}, max={S.max():.6f}, mean={S.mean():.6f}")
        print(f"[Whiten] Condition number: {S.max() / (S.min() + 1e-10):.2f}")
        
        whiten_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
        emb_whitened = emb_centered @ whiten_matrix
        emb_whitened[0, :] = 0.0
        emb_processed = emb_whitened.astype(np.float32)
        
    else:
        emb_processed = emb_centered
        norms = np.linalg.norm(emb_processed[1:], axis=1, keepdims=True)
        emb_processed[1:] = emb_processed[1:] / np.clip(norms, 1e-8, None)
    
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
    normalize: bool = True
):
    """应用 TruncatedSVD 降维"""
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

    nonpad = mat[1:, :].astype(np.float32, copy=False)
    reduced = svd.transform(nonpad)
    if svd_k < target_dim:
        pad = np.zeros((reduced.shape[0], target_dim - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)
    
    if normalize:
        reduced = l2_normalize(reduced, norm="l2", axis=1)
    
    projected = np.zeros((mat.shape[0], target_dim), dtype=reduced.dtype)
    projected[1:, :] = reduced
    return projected


def main():
    args = parse_args()

    # --- 1. 解析 Prompts ---
    if args.prompt_list:
        with open(args.prompt_list, "r") as f:
            prompts = json.load(f)
            if not isinstance(prompts, list):
                raise ValueError("prompt_list JSON must be a list of strings.")
    elif args.prompt_template:
        prompts = [args.prompt_template]
    else:
        prompts = PROMPT_PRESETS.get(args.prompt_preset, PROMPT_PRESETS["base"])

    print(f"使用 {len(prompts)} 个 prompt(s):")
    for i, p in enumerate(prompts):
        print(f"  [{i+1}] {p}")

    # --- 2. 加载数据 ---
    df = pd.read_csv(args.mapping)
    if "internal_item_id" not in df.columns or "item_token" not in df.columns:
        raise ValueError("mapping CSV must contain 'internal_item_id' and 'item_token'")
    has_title = "title" in df.columns

    df = df.sort_values("internal_item_id")
    
    item_raw_texts = []
    for _, row in df.iterrows():
        if row["internal_item_id"] == 0:
            raw = args.pad_placeholder_text
        else:
            raw = str(row["title"]) if has_title else str(row["item_token"])
            if len(raw.strip()) == 0:
                raw = args.placeholder_text
        item_raw_texts.append(raw.strip())

    # --- 3. 加载模型 ---
    device = _select_device(args.device)
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"\n===== 加载 Qwen2.5-72B 模型 =====")
    print(f"模型路径: {args.model_name_or_path}")
    print(f"device_map: {args.device_map}")
    print(f"dtype: {args.dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="left"
    )
    
    # 确保有 pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[Tokenizer] pad_token='{tokenizer.pad_token}', eos_token='{tokenizer.eos_token}'")
    
    # 加载模型配置
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    # 设置 device_map
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
        if args.max_memory:
            model_kwargs["max_memory"] = _get_max_memory_dict(args.max_memory)
    
    # Flash Attention 2 支持
    if args.flash_attn:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("[Flash Attention 2] 尝试启用...")
        except Exception as e:
            print(f"[Flash Attention 2] 警告: {e}")
    
    # 加载模型（自动检测 GPTQ 量化）
    print(f"正在加载模型... (这可能需要几分钟)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )
    model.eval()
    
    # 打印模型分布信息
    if hasattr(model, 'hf_device_map'):
        print(f"[Multi-GPU] 模型已分布到设备: {set(model.hf_device_map.values())}")
    
    use_chat = hasattr(tokenizer, "apply_chat_template") and (not args.no_chat_template)
    if use_chat:
        print("[Chat Template] 已启用")

    # --- 4. 编码循环 ---
    needs_view_concat = args.output_mode == "concat" and args.view_project_dim is not None
    if needs_view_concat and not args.split_output_dir:
        raise ValueError("--view_project_dim requires --split_output_dir to store per-view tensors.")

    final_embs = [] if not needs_view_concat else None
    
    n_items = len(item_raw_texts)
    
    mode_str = "generative" if args.generative else "embedding-only"
    if args.generative:
        print(f"[Generative Mode] temperature={args.gen_temperature}, max_new_tokens={args.gen_max_new_tokens}")
    print(f"\n编码 {n_items} 个 items, batch_size={args.batch_size}, {len(prompts)} prompts, mode={args.output_mode}, encode_mode={mode_str}...")
    
    split_chunks = None
    if args.split_output_dir:
        split_chunks = [[] for _ in range(len(prompts))]
    view_mats_for_concat = [] if needs_view_concat else None
    
    save_gen_texts = args.generative and args.save_generated_texts is not None
    all_generated_texts = [] if save_gen_texts else None

    with tqdm(total=n_items, unit="items") as pbar:
        for i in range(0, n_items, args.batch_size):
            batch_raw = item_raw_texts[i : i + args.batch_size]
            batch_start_idx = i
            
            batch_prompt_embs = []
            batch_gen_texts_per_view = [] if save_gen_texts else None
            
            for prompt_idx, prompt_tmpl in enumerate(prompts):
                batch_texts = []
                for raw in batch_raw:
                    base_prompt = prompt_tmpl.replace("{text}", raw)
                    if use_chat:
                        messages = [{"role": "user", "content": base_prompt}]
                        chat_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=args.generative
                        )
                        batch_texts.append(chat_text)
                    else:
                        batch_texts.append(base_prompt)

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
                batch_prompt_embs.append(emb)
            
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
            
            stacked = np.stack(batch_prompt_embs, axis=1)

            if split_chunks is not None:
                for view_idx in range(len(prompts)):
                    split_chunks[view_idx].append(stacked[:, view_idx, :].astype(np.float32, copy=False))
            
            if args.output_mode == "mean":
                final_batch = np.mean(stacked, axis=1)
                final_batch = l2_normalize(final_batch, axis=1)
            elif args.output_mode == "concat":
                B, K, D = stacked.shape
                final_batch = stacked.reshape(B, K * D)
            else:
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
            "model": args.model_name_or_path,
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
                    normalize=False,
                )
            
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
        print(f"[Split] 保存分视图 embeddings 到 {args.split_output_dir} (metadata: {meta_path})")

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

    # --- 5. SVD 降维 ---
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
                normalize=False,
            )

    # --- 6. Center + Whiten 归一化 ---
    if (enable_center or enable_whiten) and mat is not None and mat.ndim == 2:
        stats_path = args.output.replace('.npy', '_whiten_stats.npz') if enable_center else None
        mat = _center_whiten_and_normalize(
            mat,
            train_ids_cache,
            output_stats_path=stats_path,
            enable_whiten=enable_whiten,
            enable_center=enable_center,
        )

    # --- 7. 保存 ---
    if args.dtype == "float16":
        mat = mat.astype(np.float16)
    elif args.dtype == "bfloat16":
        mat = mat.astype(np.float32)
    else:
        mat = mat.astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, mat)
    print(
        f"\n✅ 保存 Qwen2.5-72B embeddings 到: {os.path.abspath(args.output)}  "
        f"shape={mat.shape}  dtype={mat.dtype}  (prompts={len(prompts)}, mode={args.output_mode})"
    )
    
    # --- 8. 保存生成的文本 ---
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
        print(f"[Generative] 保存生成文本到: {os.path.abspath(gen_texts_path)} ({len(all_generated_texts)} items)")


if __name__ == "__main__":
    main()

