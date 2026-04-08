#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 book-crossing 的文本向量与 RecBole 数据集是否对齐（行数 = n_items，含 PAD 行 0）。

与 SASRecAlignV3 中 _load_text_embeddings 的约束一致：
  - 路径存在且为 2D
  - emb.shape[0] == dataset.num(iid_field)

用法（在项目根目录）:
  python tools/verify_book_crossing_embeddings.py
  python tools/verify_book_crossing_embeddings.py \\
    --config sasrec_align_book_crossing_qwen3_stratified_v3.yaml
  python tools/verify_book_crossing_embeddings.py --skip-multiview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset

# 与 gen_text_emb_book_crossing_full_fast.sh / 常见配置一致
DATASET_DIR = "dataset/book-crossing"
EXPECTED_EMB_DIM = 256
MULTIVIEW_DIR = os.path.join(DATASET_DIR, "qwen3_4views")
NUM_VIEWS = 4
VIEW_DIM = 64


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> str:
    print(f"  FAIL  {msg}")
    return msg


def _check_npy(
    path: str,
    n_items: int,
    expected_dim: Optional[int],
    label: str,
    check_pad_row: bool,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not os.path.isfile(path):
        errors.append(_fail(f"{label}: file missing -> {path}"))
        return False, errors
    try:
        emb = np.load(path, mmap_mode="r")
    except Exception as e:
        errors.append(_fail(f"{label}: load error {e}"))
        return False, errors
    if emb.ndim != 2:
        errors.append(_fail(f"{label}: expected 2D array, got shape {emb.shape}"))
        return False, errors
    if emb.shape[0] != n_items:
        errors.append(
            _fail(
                f"{label}: row mismatch emb.shape[0]={emb.shape[0]} vs n_items={n_items}"
            )
        )
    if expected_dim is not None and emb.shape[1] != expected_dim:
        errors.append(
            _fail(
                f"{label}: dim mismatch emb.shape[1]={emb.shape[1]} vs expected {expected_dim}"
            )
        )
    if check_pad_row and emb.shape[0] > 0:
        pad_norm = float(np.linalg.norm(np.asarray(emb[0], dtype=np.float64)))
        if pad_norm > 1e-3:
            print(
                f"  WARN  {label}: row-0 (PAD) L2 norm={pad_norm:.6f} (usually ~0)"
            )
    if not errors:
        extra = f" shape={tuple(emb.shape)} dtype={emb.dtype}"
        _ok(f"{label}:{extra}")
    return len(errors) == 0, errors


def _check_stats_npz(path: str, label: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not os.path.isfile(path):
        print(f"  SKIP  {label}: optional stats not found -> {path}")
        return True, errors
    try:
        z = np.load(path)
        if "center" not in z.files or "whiten" not in z.files:
            errors.append(_fail(f"{label}: stats missing 'center' or 'whiten' keys"))
            return False, errors
        c, w = z["center"], z["whiten"]
        if c.ndim != 1 or w.ndim != 2 or w.shape[0] != w.shape[1] or c.shape[0] != w.shape[0]:
            errors.append(
                _fail(f"{label}: bad stats shapes center={c.shape} whiten={w.shape}")
            )
        else:
            _ok(f"{label}: center={c.shape} whiten={w.shape}")
    except Exception as e:
        errors.append(_fail(f"{label}: stats load failed {e}"))
    return len(errors) == 0, errors


def _check_multiview_split_dir(n_items: int) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not os.path.isdir(MULTIVIEW_DIR):
        print(f"  SKIP  multiview split dir not found -> {MULTIVIEW_DIR}")
        return True, errors
    vj = os.path.join(MULTIVIEW_DIR, "views.json")
    if os.path.isfile(vj):
        try:
            with open(vj, "r", encoding="utf-8") as f:
                meta = json.load(f)
            n = len(meta.get("views", []))
            _ok(f"views.json: {n} view(s) in metadata")
        except Exception as e:
            errors.append(_fail(f"views.json: read error {e}"))
    for i in range(NUM_VIEWS):
        vp = os.path.join(MULTIVIEW_DIR, f"view_{i}.npy")
        ok, e = _check_npy(vp, n_items, VIEW_DIM, f"multiview view_{i}", check_pad_row=False)
        errors.extend(e)
        if not ok and not os.path.isfile(vp):
            break
    return len(errors) == 0, errors


def main() -> int:
    p = argparse.ArgumentParser(description="Verify book-crossing text embeddings vs RecBole n_items")
    p.add_argument(
        "--config",
        nargs="+",
        default=["sasrec_book_crossing_plain.yaml"],
        help="YAML list for Config (same load_col as training/embed gen)",
    )
    p.add_argument("--skip-multiview", action="store_true", help="Do not check qwen3_4views/*.npy")
    p.add_argument("--skip-stats", action="store_true", help="Do not check *_whiten_stats.npz")
    args = p.parse_args()

    os.chdir(ROOT_DIR)

    cfg = Config(model="BPR", dataset="book-crossing", config_file_list=list(args.config))
    dataset = create_dataset(cfg)
    n_items = dataset.num(dataset.iid_field)

    print("=" * 60)
    print("book-crossing embedding verification")
    print("=" * 60)
    print(f"  config_file_list: {args.config}")
    print(f"  n_items (incl. PAD row 0): {n_items}")
    print(f"  expected emb dim: {EXPECTED_EMB_DIM}")
    print()

    all_errors: List[str] = []

    checks = [
        ("TF-IDF base", os.path.join(DATASET_DIR, "item_text_emb.base.npy"), True),
        ("Qwen3 single-view", os.path.join(DATASET_DIR, "item_text_emb.qwen3.base.npy"), True),
        ("Qwen3 multiview concat", os.path.join(DATASET_DIR, "item_text_emb.qwen3.multiview.npy"), False),
    ]
    for label, rel, pad in checks:
        ok, err = _check_npy(rel, n_items, EXPECTED_EMB_DIM, label, check_pad_row=pad)
        all_errors.extend(err)

    if not args.skip_stats:
        for name, fname in [
            ("TF-IDF whiten stats", "item_text_emb.base_whiten_stats.npz"),
            ("Qwen3 base whiten stats", "item_text_emb.qwen3.base_whiten_stats.npz"),
            ("Qwen3 multiview whiten stats", "item_text_emb.qwen3.multiview_whiten_stats.npz"),
        ]:
            ok, err = _check_stats_npz(os.path.join(DATASET_DIR, fname), name)
            all_errors.extend(err)

    mapping_csv = os.path.join(DATASET_DIR, "item_index_mapping.csv")
    if os.path.isfile(mapping_csv):
        _ok(f"item_index_mapping.csv present ({mapping_csv})")
    else:
        print(f"  SKIP  item_index_mapping.csv not found (optional for LLM pipeline)")

    if not args.skip_multiview:
        ok, err = _check_multiview_split_dir(n_items)
        all_errors.extend(err)

    print()
    if all_errors:
        print("Result: FAILED")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    print("Result: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
