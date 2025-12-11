#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build base item text embeddings (TF-IDF + TruncatedSVD) aligned with RecBole internal item indices.

Outputs a matrix `item_text_emb.npy` with shape [n_items, d_text], where row 0 is [PAD] (all zeros),
and rows 1..n_items-1 align to RecBole internal item ids. This file can be consumed by RLMRec and
other models that accept `item_text_emb_path`.

Design goals:
- Fair base (no LLM): character-level TF-IDF over item titles, then SVD to a fixed dimension.
- Deterministic and reproducible: controlled random_state, frozen vectors.
- No information leakage: only uses static item features.

Usage example:
  python tools/build_item_text_emb_base.py \
    --dataset Amazon_Beauty \
    --config recbole/properties/model/GRU4RecCPR.yaml \
    --output data/Amazon_Beauty/item_text_emb.base.npy \
    --title_field title \
    --svd_dim 256

Notes:
- The script reads the raw `.item` file via RecBole dataset metadata to get titles as plain text.
- If the specified title field is missing, it will try common fallbacks and then fall back to empty strings.
- For Chinese and multilingual text, char-level n-grams work robustly without additional tokenizers.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

# Make local project importable when running from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset, data_preparation


def _detect_item_file(dataset) -> Optional[str]:
    """Return path to `<dataset_name>.item` if exists, else None."""
    dataset_dir = getattr(dataset, "dataset_path", None)
    dataset_name = getattr(dataset, "dataset_name", None)
    if not dataset_dir or not dataset_name:
        return None
    candidate = os.path.join(dataset_dir, f"{dataset_name}.item")
    return candidate if os.path.exists(candidate) else None


def _find_col_by_base(df: pd.DataFrame, base_names: List[str]) -> Optional[str]:
    """Find a column whose base name (before ':') matches one of base_names."""
    cols = list(df.columns)
    # exact match first
    for name in base_names:
        if name in cols:
            return name
    # typed header like `field:token`
    for name in base_names:
        for c in cols:
            if isinstance(c, str) and c.split(":")[0] == name:
                return c
    return None


def _choose_title_field(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    """Pick a reasonable title/textual field from `.item` DataFrame.

    Priority: preferred -> common candidates; supports typed headers like `title:token`.
    """
    if preferred:
        col = _find_col_by_base(df, [preferred])
        if col is not None:
            return col
    candidates = [
        "title",
        "item_title",
        "name",
        "item_name",
        "product_title",
        "product_name",
        "categories",
        "category",
    ]
    return _find_col_by_base(df, candidates)


def _build_token_to_title_map(
    item_df: pd.DataFrame, item_id_col: str, title_col: Optional[str]
) -> Dict[str, str]:
    """Map external item tokens -> raw title string (empty string if missing)."""
    token_to_title: Dict[str, str] = {}
    if title_col is None:
        # No textual column; map to empty string
        for tok in item_df[item_id_col].astype(str).tolist():
            token_to_title[tok] = ""
        return token_to_title

    # Ensure strings and NaNs handled
    titles = item_df[title_col].fillna("")
    # If title is not string (e.g., numbers), cast to string
    titles = titles.astype(str)
    for tok, title in zip(item_df[item_id_col].astype(str).tolist(), titles.tolist()):
        token_to_title[tok] = title.strip()
    return token_to_title


def _get_internal_item_tokens(dataset) -> List[str]:
    """Get internal id-ordered external tokens for items, including PAD at 0."""
    iid_field = dataset.iid_field
    n_items = dataset.num(iid_field)
    ids = np.arange(n_items, dtype=np.int64)
    tokens = dataset.id2token(iid_field, ids)
    # Ensure list of str
    return [str(t) for t in tokens.tolist()]


def _build_texts_in_internal_order(
    internal_tokens: List[str], token_to_title: Dict[str, str]
) -> List[str]:
    texts: List[str] = []
    for tok in internal_tokens:
        if tok == "[PAD]":
            texts.append("")
        else:
            texts.append(token_to_title.get(tok, ""))
    return texts


def _fit_tfidf_svd(
    texts: List[str],
    n_components: int,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: Optional[int] = None,
    random_state: int = 42,
    pre_svd_l2: bool = False,  # Changed: disable by default for better whitening
) -> np.ndarray:
    """Compute TF-IDF then reduce with TruncatedSVD.

    Returns dense array of shape [len(texts), n_components].
    
    IMPORTANT: 
    - pre_svd_l2 is now False by default to preserve natural covariance structure
    - This allows subsequent whitening to work correctly (Cov → I)
    - L2 normalization (if needed) should be done AFTER whitening, not before SVD
    
    NOTE: This function does NOT apply L2 normalization after SVD.
    The caller should apply center+whiten normalization afterwards.
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        norm=None,  # normalization will be done after whitening
        dtype=np.float32,
    )
    tfidf = vectorizer.fit_transform(texts)
    # Optional: row-wise L2 normalization before SVD (for numerical stability)
    if pre_svd_l2:
        tfidf = l2_normalize(tfidf, norm="l2", axis=1, copy=False)

    # Handle edge cases where vocabulary is tiny
    svd_k = max(1, min(n_components, tfidf.shape[1] - 1 if tfidf.shape[1] > 1 else 1))
    # Use randomized algorithm with parallel processing for speed
    svd = TruncatedSVD(
        n_components=svd_k, 
        random_state=random_state,
        algorithm='randomized',  # Faster for large matrices
        n_iter=5  # Default is 5, good balance between speed and accuracy
    )
    print(f"[SVD] Fitting TruncatedSVD: {tfidf.shape} -> {svd_k} components...")
    reduced = svd.fit_transform(tfidf)
    print(f"[SVD] Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

    # If reduced dim < requested, pad zeros to target dim
    if svd_k < n_components:
        pad = np.zeros((reduced.shape[0], n_components - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)

    # Do NOT L2 normalize here - let the whitening step handle normalization
    # This preserves the natural variance structure needed for whitening
    reduced[0, :] = 0.0  # Ensure PAD row is zeros
    return reduced.astype(np.float32)


def _center_whiten_and_normalize(
    emb: np.ndarray,
    train_ids: np.ndarray,
    output_stats_path: Optional[str] = None,
    enable_whiten: bool = True,
    enable_center: bool = True,
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
    
    # If centering is disabled, skip center+whiten entirely and only L2 normalize
    if not enable_center:
        print("[Center] Centering disabled. Only applying L2 normalization.")
        emb_processed = emb.copy().astype(np.float32)
        norms = np.linalg.norm(emb_processed[1:], axis=1, keepdims=True)
        emb_processed[1:] = emb_processed[1:] / np.clip(norms, 1e-8, None)
        emb_processed[0, :] = 0.0  # Keep PAD as zeros
        return emb_processed
    
    # Extract training embeddings (exclude PAD=0)
    train_mask = np.isin(np.arange(len(emb)), train_ids)
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
        train_centered = train_emb - mean
        cov = (train_centered.T @ train_centered) / len(train_centered)
        
        # SVD decomposition: cov = U @ diag(S) @ U.T
        U, S, _ = np.linalg.svd(cov)
        
        # Whitening matrix: U @ diag(1/sqrt(S))
        # Add small epsilon for numerical stability
        whiten_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5))
        
        # Step 3: Apply whitening to all embeddings
        emb_whitened = emb_centered @ whiten_matrix
        emb_whitened[0, :] = 0.0
        emb_processed = emb_whitened
        
        # NOTE: Do NOT L2 normalize after whitening!
        # Whitening already decorrelates features and sets Cov(X) = I
        # L2 normalization would destroy this property
        # If the model needs L2-normalized embeddings, do it at model load time
        
    else:
        emb_processed = emb_centered
        
        # Step 4: L2 normalize (only if whitening is disabled)
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


def _fit_on_train_transform_all(
    train_texts: List[str],
    all_texts: List[str],
    n_components: int,
    analyzer: str = "char",
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: Optional[int] = None,
    random_state: int = 42,
    pre_svd_l2: bool = False,  # Changed: disable by default for better whitening
) -> np.ndarray:
    """Fit TF-IDF and SVD on train_texts only, then transform all_texts.

    Returns dense array of shape [len(all_texts), n_components].
    
    IMPORTANT: 
    - pre_svd_l2 is now False by default to preserve natural covariance structure
    - This allows subsequent whitening to work correctly (Cov → I)
    - Fitting on train set only prevents data leakage
    
    NOTE: This function does NOT apply L2 normalization after SVD.
    The caller should apply center+whiten normalization afterwards.
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        norm=None,
        dtype=np.float32,
    )
    # Fit only on training texts
    tfidf_train = vectorizer.fit_transform(train_texts)
    tfidf_all = vectorizer.transform(all_texts)
    # Optional: row-wise L2 before SVD fit/transform (for numerical stability)
    if pre_svd_l2:
        tfidf_train = l2_normalize(tfidf_train, norm="l2", axis=1, copy=False)
        tfidf_all = l2_normalize(tfidf_all, norm="l2", axis=1, copy=False)

    # Handle edge cases where vocabulary is tiny
    svd_k = max(1, min(n_components, tfidf_train.shape[1] - 1 if tfidf_train.shape[1] > 1 else 1))
    # Use randomized algorithm for speed
    svd = TruncatedSVD(
        n_components=svd_k, 
        random_state=random_state,
        algorithm='randomized',
        n_iter=5
    )
    print(f"[SVD] Fitting TruncatedSVD on train: {tfidf_train.shape} -> {svd_k} components...")
    svd.fit(tfidf_train)
    print(f"[SVD] Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"[SVD] Transforming all items: {tfidf_all.shape[0]} items...")
    reduced = svd.transform(tfidf_all)

    # If reduced dim < requested, pad zeros to target dim
    if svd_k < n_components:
        pad = np.zeros((reduced.shape[0], n_components - svd_k), dtype=reduced.dtype)
        reduced = np.concatenate([reduced, pad], axis=1)

    # Do NOT L2 normalize here - let the whitening step handle normalization
    # This preserves the natural variance structure needed for whitening
    reduced[0, :] = 0.0  # Ensure PAD row is zeros
    return reduced.astype(np.float32)


def build_item_text_emb(
    dataset_name: str,
    config_files: List[str],
    output_path: str,
    title_field: Optional[str] = None,
    svd_dim: int = 256,
    analyzer: str = "char",
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    max_features: Optional[int] = None,
    dtype: str = "float16",
    svd_random_state: int = 42,
    pre_svd_l2: bool = False,  # Changed: disable by default for better whitening
    enable_whiten: bool = True,
    enable_center: bool = True,
) -> str:
    """Main pipeline to build base item text embeddings and save to output_path.

    Returns the absolute output path.
    """
    import time
    start_time = time.time()
    
    if not dataset_name:
        raise KeyError("--dataset is required (e.g., --dataset Amazon_Beauty)")
    
    print(f"[Step 1/6] Loading dataset: {dataset_name}...")
    cfg = Config(model="BPR", dataset=dataset_name, config_file_list=config_files)
    dataset = create_dataset(cfg)
    print(f"  → Dataset loaded: {dataset.num(dataset.iid_field)} items")

    print(f"[Step 2/6] Reading item metadata...")
    item_file = _detect_item_file(dataset)
    if item_file is None:
        print("[WARN] .item file not found; falling back to empty titles for all items.")
        item_df = pd.DataFrame({cfg["ITEM_ID_FIELD"]: []})
    else:
        item_df = pd.read_csv(item_file, sep="\t")
        print(f"  → Loaded {len(item_df)} item records")

    item_id_col = _find_col_by_base(item_df, [cfg["ITEM_ID_FIELD"], "item_id", "item", "iid"]) or cfg["ITEM_ID_FIELD"]

    chosen_title = _choose_title_field(item_df, title_field)
    if chosen_title:
        print(f"  → Using title field: '{chosen_title}'")
    token_to_title = _build_token_to_title_map(item_df, item_id_col, chosen_title)
    internal_tokens = _get_internal_item_tokens(dataset)
    texts_all = _build_texts_in_internal_order(internal_tokens, token_to_title)
    print(f"  → Prepared {len(texts_all)} text entries")

    # Build split and collect train-only item ids to avoid leakage
    print(f"[Step 3/6] Preparing data split...")
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    iid_field = cfg["ITEM_ID_FIELD"]
    try:
        train_iids = train_data.dataset.inter_feat[iid_field].numpy()
    except Exception:
        # Fallback: no split info available → treat all non-PAD as train (kept for robustness)
        train_iids = np.arange(1, len(internal_tokens), dtype=np.int64)
    train_iids_set = set([int(x) for x in np.unique(train_iids).tolist() if int(x) > 0])
    # Assemble train texts by internal id index (exclude PAD=0)
    train_texts = [texts_all[i] for i in range(1, len(texts_all)) if i in train_iids_set]
    # Guard: if train_texts ends up empty, fall back to all (rare/corrupt case)
    print(f"[Step 4/6] Building TF-IDF + SVD features...")
    print(f"  → Training set: {len(train_texts)} items")
    print(f"  → Total items: {len(texts_all)} items")
    
    if len(train_texts) == 0:
        print("[WARN] train_texts is empty; falling back to fitting on all_texts (may risk leakage).")
        emb = _fit_tfidf_svd(
            texts_all,
            n_components=svd_dim,
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
            random_state=svd_random_state,
            pre_svd_l2=pre_svd_l2,
        )
    else:
        emb = _fit_on_train_transform_all(
            train_texts=train_texts,
            all_texts=texts_all,
            n_components=svd_dim,
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_features=max_features,
            random_state=svd_random_state,
            pre_svd_l2=pre_svd_l2,
        )

    # Apply center + whiten normalization (using training set statistics)
    print(f"[Step 5/6] Applying center + whiten normalization...")
    if enable_center or enable_whiten:
        stats_path = output_path.replace('.npy', '_whiten_stats.npz') if enable_center else None
        emb = _center_whiten_and_normalize(
            emb,
            train_iids,
            output_stats_path=stats_path,
            enable_whiten=enable_whiten,
            enable_center=enable_center,
        )
    else:
        print("  → Center and whitening disabled (--no_center --no_whiten)")

    # Cast dtype if requested
    print(f"[Step 6/6] Saving embeddings...")
    if dtype == "float16":
        emb = emb.astype(np.float16)
    elif dtype == "float32":
        emb = emb.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, emb)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Saved item_text_emb to: {os.path.abspath(output_path)}")
    print(f"   - Shape: {emb.shape}")
    print(f"   - Dtype: {emb.dtype}")
    print(f"   - File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"   - Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return os.path.abspath(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build base item text embeddings (TF-IDF+SVD)")
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g., Amazon_Beauty",
    )
    p.add_argument(
        "--config",
        nargs="+",
        required=False,
        default=[],
        help="Optional YAML config files to customize dataset/model params",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for the generated .npy file (e.g., data/Amazon_Beauty/item_text_emb.base.npy)",
    )
    p.add_argument(
        "--title_field",
        default=None,
        help="Column name in .item for title; if omitted, tries common names like 'title'",
    )
    p.add_argument("--svd_dim", type=int, default=256, help="Output embedding dimension")
    p.add_argument(
        "--svd_random_state",
        type=int,
        default=42,
        help="Random state for TruncatedSVD (aligns with Qwen3 script).",
    )
    p.add_argument(
        "--ngram_min", type=int, default=1, help="Minimum n for character n-grams"
    )
    p.add_argument(
        "--ngram_max", type=int, default=2, help="Maximum n for character n-grams"
    )
    p.add_argument(
        "--min_df", type=int, default=2, help="Min document frequency for TF-IDF vocabulary"
    )
    p.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Limit TF-IDF vocabulary size (None means unlimited)",
    )
    p.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output dtype for the saved matrix",
    )
    p.add_argument(
        "--no_pre_svd_l2",
        action="store_true",
        help="Disable row-wise L2 normalization before SVD (enabled by default).",
    )
    p.add_argument(
        "--no_whiten",
        action="store_true",
        help="Disable whitening transformation (enabled by default). Only center + L2 normalize.",
    )
    p.add_argument(
        "--no_center",
        action="store_true",
        help="Disable centering (mean subtraction). If set, skip center+whiten entirely and only L2 normalize.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    build_item_text_emb(
        dataset_name=args.dataset,
        config_files=args.config,
        output_path=args.output,
        title_field=args.title_field,
        svd_dim=args.svd_dim,
        analyzer="char",
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
        dtype=args.dtype,
        svd_random_state=args.svd_random_state,
        pre_svd_l2=(not args.no_pre_svd_l2),
        enable_whiten=(not args.no_whiten),
        enable_center=(not args.no_center),
    )


if __name__ == "__main__":
    main()


