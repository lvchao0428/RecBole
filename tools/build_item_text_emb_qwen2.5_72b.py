#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build item text embeddings using Qwen2.5-72B-Instruct-GPTQ-Int8 model with vLLM.

该脚本使用 vLLM 加载 Qwen2.5-72B GPTQ-Int8 量化模型，支持多GPU tensor parallel。
基于 build_item_text_emb_qwen3_hf.py 改写，保持相同的 project 维度和接口。

主要特性：
- 使用 vLLM 加载模型，支持 GPTQ-Int8 量化
- 支持多GPU tensor parallel (8x4090)
- 支持多视图 prompt 生成
- 支持 SVD 降维和白化
- 生成模式：先生成回答，再提取 embedding

Input: mapping CSV exported by tools/export_internal_item_mapping.py
Output: item_text_emb.qwen2.5_72b.npy

Example:
  python tools/build_item_text_emb_qwen2.5_72b.py \
    --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
    --model_name_or_path /data/model/Qwen2.5-72B-Instruct-GPTQ-Int8 \
    --output dataset/Amazon_Beauty/item_text_emb.qwen2.5_72b.npy \
    --tensor_parallel_size 8 \
    --prompt_preset multiview \
    --output_mode concat \
    --batch_size 32 \
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
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

# vLLM imports
from vllm import LLM, SamplingParams

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
    p = argparse.ArgumentParser(description="Build Qwen2.5-72B item text embeddings using vLLM (GPTQ-Int8)")
    p.add_argument("--mapping", required=True, help="CSV from export_internal_item_mapping.py")
    p.add_argument("--model_name_or_path", required=True, help="HF model id or local path (e.g., /data/model/Qwen2.5-72B-Instruct-GPTQ-Int8)")
    p.add_argument("--output", required=True, help="Output .npy path for embeddings")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for vLLM generation")
    p.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096).",
    )
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    
    # --- vLLM 配置参数 ---
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism (default: 8 for 8x4090).",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9).",
    )
    p.add_argument(
        "--swap_space",
        type=int,
        default=4,
        help="Swap space in GB (default: 4).",
    )
    p.add_argument(
        "--enforce_eager",
        action="store_true",
        default=True,
        help="Enforce eager mode (required for consumer GPUs like 4090).",
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
        help="Wrap prompt using chat template format.",
    )
    
    # --- 生成参数 ---
    p.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate (default: 128).",
    )
    p.add_argument(
        "--gen_temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7).",
    )
    p.add_argument(
        "--gen_top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter (default: 0.95).",
    )
    p.add_argument(
        "--save_generated_texts",
        default=None,
        help="Path to save generated texts as JSON.",
    )
    p.add_argument("--placeholder_text", default="N/A", help="Fallback text.")
    p.add_argument("--pad_placeholder_text", default="[PAD]", help="Placeholder for PAD row.")
    p.add_argument(
        "--split_output_dir",
        default=None,
        help="If set, dumps per-view embeddings (view_{i}.npy) and views.json metadata.",
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
        help="Reduce embedding dim via TruncatedSVD. Applied AFTER concatenation/mean. Default: None.",
    )
    p.add_argument(
        "--view_project_dim",
        type=int,
        default=None,
        help="Apply TruncatedSVD per prompt view before concat/stack (e.g., 64).",
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
        help="Enable whitening transformation.",
    )
    p.add_argument(
        "--center",
        action="store_true",
        help="Enable centering (mean subtraction).",
    )
    
    # --- Embedding 模型配置 ---
    p.add_argument(
        "--embed_model_name_or_path",
        default=None,
        help="Separate embedding model for extracting embeddings from generated text. If not set, uses sentence-transformers.",
    )
    p.add_argument(
        "--embed_model_dim",
        type=int,
        default=1024,
        help="Embedding dimension (default: 1024 for bge-large).",
    )
    
    return p.parse_args()


def build_chat_prompt(text: str, use_chat_template: bool = True) -> str:
    """构建 Qwen 格式的 chat prompt"""
    if use_chat_template:
        # Qwen chat template format
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return text


def extract_embeddings_from_texts(
    texts: List[str],
    embed_model_name_or_path: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    从文本中提取 embeddings。
    使用 sentence-transformers 或其他 embedding 模型。
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        if embed_model_name_or_path is None:
            # 默认使用 bge-large-zh-v1.5 或 bge-large-en-v1.5
            embed_model_name_or_path = "BAAI/bge-large-en-v1.5"
        
        print(f"[Embedding] 加载 embedding 模型: {embed_model_name_or_path}")
        embed_model = SentenceTransformer(embed_model_name_or_path)
        
        print(f"[Embedding] 编码 {len(texts)} 个文本...")
        embeddings = embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        return embeddings.astype(np.float32)
        
    except ImportError:
        print("[WARNING] sentence-transformers not installed. Using TF-IDF fallback.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=1024, ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(texts).toarray()
        embeddings = l2_normalize(embeddings, axis=1)
        
        return embeddings.astype(np.float32)


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
    """Center + Whiten + L2 normalize"""
    if emb is None or emb.size == 0:
        return emb
    
    if emb.ndim == 3:
        print("[WARN] Skipping center/whiten for 3D tensor (mode=stack).")
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
    
    mean = train_emb.mean(axis=0, keepdims=True).astype(np.float64)
    emb_centered = (emb - mean).astype(np.float64)
    emb_centered[0, :] = 0.0
    
    whiten_matrix = None
    if enable_whiten:
        train_centered = (train_emb - mean).astype(np.float64)
        cov = (train_centered.T @ train_centered) / len(train_centered)
        U, S, _ = np.linalg.svd(cov)
        
        print(f"[Whiten] Eigenvalue stats: min={S.min():.6f}, max={S.max():.6f}")
        
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
            np.savez(output_stats_path, mean=mean.astype(np.float32), whiten_matrix=whiten_matrix.astype(np.float32))
            print(f"[Whiten] Saved stats to: {output_stats_path}")
        else:
            np.savez(output_stats_path, mean=mean.astype(np.float32))
            print(f"[Center] Saved stats to: {output_stats_path}")
    
    return emb_processed.astype(np.float32)


def _apply_truncated_svd(
    mat: np.ndarray, 
    target_dim: int, 
    train_ids, 
    random_state: int, 
    label: str,
    normalize: bool = True
):
    """Apply TruncatedSVD dimensionality reduction"""
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

    # --- 3. 初始化 vLLM 模型 ---
    print("\n" + "=" * 60)
    print("初始化 vLLM 模型 (Qwen2.5-72B-Instruct-GPTQ-Int8)")
    print("=" * 60)
    print(f"模型路径: {args.model_name_or_path}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"量化类型: GPTQ (Int8)")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"dtype: {args.dtype}")
    print("")
    
    # vLLM 配置 - 与 test_qwen2.5_72b_gptq_int8_transformers.py 一致
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization="gptq",  # 关键：指定 GPTQ 量化
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        dtype=args.dtype,
        swap_space=args.swap_space,
    )
    
    print("✅ vLLM 模型加载成功！")
    
    # 生成参数
    sampling_params = SamplingParams(
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        max_tokens=args.gen_max_new_tokens,
        stop_token_ids=[151645],  # Qwen EOS token ID
        skip_special_tokens=True,
    )

    # --- 4. 生成循环 ---
    n_items = len(item_raw_texts)
    print(f"\n处理 {n_items} 个 items, {len(prompts)} prompts, batch_size={args.batch_size}")
    
    needs_view_concat = args.output_mode == "concat" and args.view_project_dim is not None
    if needs_view_concat and not args.split_output_dir:
        raise ValueError("--view_project_dim requires --split_output_dir")

    # 存储每个视图的生成文本
    all_view_generated_texts = [[] for _ in range(len(prompts))]
    save_gen_texts = args.save_generated_texts is not None
    all_generated_records = [] if save_gen_texts else None

    # 对每个 prompt 视图进行生成
    for prompt_idx, prompt_tmpl in enumerate(prompts):
        print(f"\n--- 处理视图 {prompt_idx + 1}/{len(prompts)}: {prompt_tmpl[:50]}... ---")
        
        # 准备所有文本
        all_prompts_for_view = []
        for raw in item_raw_texts:
            base_prompt = prompt_tmpl.replace("{text}", raw)
            if args.use_chat_template:
                chat_prompt = build_chat_prompt(base_prompt, use_chat_template=True)
            else:
                chat_prompt = base_prompt
            all_prompts_for_view.append(chat_prompt)
        
        # 批量生成
        view_generated_texts = []
        for i in tqdm(range(0, n_items, args.batch_size), desc=f"View {prompt_idx + 1}"):
            batch_prompts = all_prompts_for_view[i:i + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                if len(generated_text) == 0:
                    generated_text = item_raw_texts[i]  # fallback
                view_generated_texts.append(generated_text)
        
        all_view_generated_texts[prompt_idx] = view_generated_texts
        print(f"✅ 视图 {prompt_idx + 1} 生成完成: {len(view_generated_texts)} 条文本")
    
    # 保存生成的文本（如果需要）
    if save_gen_texts:
        for item_idx in range(n_items):
            record = {
                "internal_item_id": item_idx,
                "raw_text": item_raw_texts[item_idx],
                "views": []
            }
            for view_idx, prompt_tmpl in enumerate(prompts):
                record["views"].append({
                    "view_index": view_idx,
                    "prompt": prompt_tmpl,
                    "generated_text": all_view_generated_texts[view_idx][item_idx]
                })
            all_generated_records.append(record)
    
    # --- 5. 提取 Embeddings ---
    print("\n" + "=" * 60)
    print("提取 Embeddings")
    print("=" * 60)
    
    view_embeddings = []
    for view_idx in range(len(prompts)):
        print(f"\n提取视图 {view_idx + 1} embeddings...")
        view_texts = all_view_generated_texts[view_idx]
        
        # 使用 embedding 模型提取特征
        emb = extract_embeddings_from_texts(
            view_texts,
            embed_model_name_or_path=args.embed_model_name_or_path,
            batch_size=args.batch_size,
        )
        
        print(f"  视图 {view_idx + 1} embedding shape: {emb.shape}")
        view_embeddings.append(emb)
    
    # --- 6. 合并视图 ---
    # Stack: [N, K, D]
    stacked = np.stack(view_embeddings, axis=1)
    print(f"\n合并后 shape: {stacked.shape}")
    
    # 确保 PAD 为零
    stacked[0, :, :] = 0.0
    
    # 处理分视图输出
    split_chunks = None
    if args.split_output_dir:
        split_chunks = [view_embeddings[i] for i in range(len(prompts))]
    
    view_mats_for_concat = [] if needs_view_concat else None
    
    # 根据 output_mode 处理
    if args.output_mode == "mean":
        mat = np.mean(stacked, axis=1)
        mat = l2_normalize(mat, axis=1)
    elif args.output_mode == "concat":
        B, K, D = stacked.shape
        mat = stacked.reshape(B, K * D)
    else:  # stack
        mat = stacked
    
    mat[0] = 0.0  # PAD

    # --- 7. SVD 和后处理 ---
    train_ids_cache = None
    if (args.view_project_dim is not None or args.project_dim is not None) and args.dataset:
        train_ids_cache = _load_train_item_ids(args, n_items - 1)

    enable_whiten = args.whiten
    enable_center = args.center

    # 分视图保存
    if split_chunks is not None:
        os.makedirs(args.split_output_dir, exist_ok=True)
        split_meta = {
            "num_items": int(n_items),
            "num_prompts": len(prompts),
            "dtype": args.dtype,
            "model": args.model_name_or_path,
            "prompts": [],
        }
        for view_idx, view_mat in enumerate(split_chunks):
            view_mat = view_mat.copy()
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
                view_stats_path = os.path.join(args.split_output_dir, f"view_{view_idx}_whiten_stats.npz")
                view_mat = _center_whiten_and_normalize(
                    view_mat,
                    train_ids_cache,
                    output_stats_path=view_stats_path,
                    enable_whiten=enable_whiten,
                    enable_center=enable_center,
                )
            
            save_mat = view_mat.astype(np.float16 if args.dtype == "float16" else np.float32)
            view_path = os.path.join(args.split_output_dir, f"view_{view_idx}.npy")
            np.save(view_path, save_mat)
            split_meta["prompts"].append({
                "index": view_idx,
                "prompt": prompts[view_idx],
                "file": os.path.basename(view_path),
                "vector_dim": int(view_mat.shape[1]),
            })
            if view_mats_for_concat is not None:
                view_mats_for_concat.append(view_mat.astype(np.float32))
        
        meta_path = os.path.join(args.split_output_dir, "views.json")
        with open(meta_path, "w") as mf:
            json.dump(split_meta, mf, ensure_ascii=False, indent=2)
        print(f"[Split] 保存分视图到 {args.split_output_dir}")

    # 如果需要从分视图重新拼接
    if view_mats_for_concat is not None and len(view_mats_for_concat) > 0:
        if args.output_mode == "concat":
            mat = np.concatenate(view_mats_for_concat, axis=1)
            mat[0, :] = 0.0

    # 最终 SVD
    if args.project_dim is not None and mat.ndim == 2:
        mat = _apply_truncated_svd(
            mat,
            target_dim=args.project_dim,
            train_ids=train_ids_cache,
            random_state=args.svd_random_state,
            label="final",
            normalize=False,
        )

    # Center + Whiten
    if (enable_center or enable_whiten) and mat is not None and mat.ndim == 2:
        stats_path = args.output.replace('.npy', '_whiten_stats.npz')
        mat = _center_whiten_and_normalize(
            mat,
            train_ids_cache,
            output_stats_path=stats_path,
            enable_whiten=enable_whiten,
            enable_center=enable_center,
        )

    # --- 8. 保存 ---
    if args.dtype == "float16":
        mat = mat.astype(np.float16)
    else:
        mat = mat.astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, mat)
    print(f"\n✅ 保存 embeddings 到: {os.path.abspath(args.output)}")
    print(f"   shape={mat.shape}, dtype={mat.dtype}, prompts={len(prompts)}, mode={args.output_mode}")
    
    # 保存生成文本
    if all_generated_records is not None:
        gen_output = {
            "num_items": len(all_generated_records),
            "num_views": len(prompts),
            "prompts": prompts,
            "items": all_generated_records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_generated_texts)), exist_ok=True)
        with open(args.save_generated_texts, "w", encoding="utf-8") as f:
            json.dump(gen_output, f, ensure_ascii=False, indent=2)
        print(f"[Generative] 保存生成文本到: {args.save_generated_texts}")


if __name__ == "__main__":
    main()
