#!/usr/bin/env python3
"""
Rebuild Qwen single-view and multi-view embeddings with TS-aware center+whiten.

No need to re-run Qwen inference. This script:
  1. Loads existing Qwen embeddings (already SVD-reduced).
  2. Inverts old whitening (using saved whiten_stats.npz) if present.
  3. Re-applies center+whiten using ONLY items present before the TS train cutoff.
  4. Saves new files with .ts suffix.

Usage:
  python tools/rebuild_qwen_emb_ts.py \
    --dataset Amazon_Beauty \
    --config sasrec_baseline_50ep_stratified_ts.yaml

Output (for Beauty):
  dataset/Amazon_Beauty/item_text_emb.qwen2.5_7b.base.ts.npy
  dataset/Amazon_Beauty/qwen2.5_7b_4views_ts/view_{0..3}.npy  (+ views.json)
"""
import argparse
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_ts_train_ids(dataset_name, config_files):
    """Return internal item IDs (int array) that appear in the TS train split."""
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.data.utils import data_preparation

    cfg = Config(model="BPR", dataset=dataset_name, config_file_list=config_files)
    ds = create_dataset(cfg)
    train_data, _, _ = data_preparation(cfg, ds)
    iid_field = cfg["ITEM_ID_FIELD"]
    train_ids = np.unique(train_data._dataset.inter_feat[iid_field].numpy())
    train_ids = train_ids[train_ids > 0].astype(np.int64)
    print(f"  TS train items: {len(train_ids)} / {ds.num(iid_field)} total")
    return train_ids


def _invert_whiten(emb: np.ndarray, stats_path: str) -> np.ndarray:
    """Undo center+whiten using saved stats. Returns embeddings in SVD-output space."""
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float64)          # (1, D)
    W = stats["whiten_matrix"].astype(np.float64)    # (D, D)
    # emb = (orig - mean) @ W  =>  orig = emb @ W^{-1} + mean
    # W = U @ diag(1/sqrt(S)), so W^{-1} = diag(sqrt(S)) @ U.T = W.T @ diag(S)
    # But numerically simpler: just use lstsq / pinv
    W_inv = np.linalg.pinv(W)                        # (D, D)
    orig = emb.astype(np.float64) @ W_inv + mean     # (N, D)
    orig[0] = 0.0                                    # keep PAD zeros
    return orig.astype(np.float32)


def _center_whiten(emb: np.ndarray, train_ids: np.ndarray,
                   stats_out: str = None) -> np.ndarray:
    """Center + ZCA-whiten using train_ids statistics only. L2-norm NOT applied."""
    emb = emb.astype(np.float64)
    emb[0] = 0.0

    train_emb = emb[train_ids]
    mean = train_emb.mean(axis=0, keepdims=True)      # (1, D)
    emb_c = emb - mean
    emb_c[0] = 0.0

    train_c = train_emb - mean
    cov = (train_c.T @ train_c) / len(train_c)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + 1e-5))         # ZCA whitening matrix

    result = (emb_c @ W).astype(np.float32)
    result[0] = 0.0

    if stats_out:
        os.makedirs(os.path.dirname(os.path.abspath(stats_out)), exist_ok=True)
        np.savez(stats_out,
                 mean=mean.astype(np.float32),
                 whiten_matrix=W.astype(np.float32))
        print(f"  saved whiten stats → {stats_out}")

    return result


# ── main logic ────────────────────────────────────────────────────────────────

def rebuild_single_view(dataset_dir: str, train_ids: np.ndarray):
    """
    Re-process item_text_emb.qwen2.5_7b.base.npy → .ts.npy
    Original has no whitening → apply center+whiten with TS train_ids.
    """
    src = os.path.join(dataset_dir, "item_text_emb.qwen2.5_7b.base.npy")
    dst = os.path.join(dataset_dir, "item_text_emb.qwen2.5_7b.base.ts.npy")
    stats_dst = dst.replace(".npy", "_whiten_stats.npz")

    print(f"\n[Single-view] Loading {src} ...")
    emb = np.load(src).astype(np.float32)
    print(f"  shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"  norm before (sample): {np.linalg.norm(emb[1:5].astype(np.float64), axis=1)}")

    print("  applying center+whiten with TS train_ids ...")
    emb_ts = _center_whiten(emb, train_ids, stats_out=stats_dst)

    np.save(dst, emb_ts.astype(np.float16))
    print(f"  saved → {dst}  shape={emb_ts.shape}  ({os.path.getsize(dst)/1024/1024:.1f} MB)")


def rebuild_multi_view(dataset_dir: str, train_ids: np.ndarray):
    """
    Re-process qwen2.5_7b_4views/view_*.npy → qwen2.5_7b_4views_ts/view_*.npy
    Each view: invert old whitening, then re-apply with TS train_ids.
    """
    src_dir = os.path.join(dataset_dir, "qwen2.5_7b_4views")
    dst_dir = os.path.join(dataset_dir, "qwen2.5_7b_4views_ts")
    os.makedirs(dst_dir, exist_ok=True)

    with open(os.path.join(src_dir, "views.json")) as f:
        meta = json.load(f)

    new_meta = dict(meta)
    new_meta["prompts"] = []

    for view_info in meta["prompts"]:
        idx = view_info["index"]
        src_path = os.path.join(src_dir, view_info["file"])
        stats_path = os.path.join(src_dir, f"view_{idx}_whiten_stats.npz")
        dst_path = os.path.join(dst_dir, view_info["file"])
        dst_stats = os.path.join(dst_dir, f"view_{idx}_whiten_stats.npz")

        print(f"\n[Multi-view idx={idx}] Loading {src_path} ...")
        emb = np.load(src_path).astype(np.float32)
        print(f"  shape: {emb.shape}")

        if os.path.exists(stats_path):
            print(f"  inverting old whitening from {stats_path} ...")
            emb_svd = _invert_whiten(emb, stats_path)
        else:
            print(f"  no old whiten stats found — using raw embedding")
            emb_svd = emb

        print("  applying center+whiten with TS train_ids ...")
        emb_ts = _center_whiten(emb_svd, train_ids, stats_out=dst_stats)

        np.save(dst_path, emb_ts.astype(np.float16))
        print(f"  saved → {dst_path}  ({os.path.getsize(dst_path)/1024/1024:.1f} MB)")

        new_entry = dict(view_info)
        new_entry["vector_dim"] = int(emb_ts.shape[1])
        new_meta["prompts"].append(new_entry)

    with open(os.path.join(dst_dir, "views.json"), "w") as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)
    print(f"\n[Multi-view] saved views.json → {dst_dir}/views.json")


def main():
    p = argparse.ArgumentParser(description="Rebuild Qwen embeddings with TS-aware center+whiten")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", nargs="+", default=[])
    p.add_argument("--dataset_dir", default=None,
                   help="Override dataset root dir (default: dataset/<dataset>)")
    args = p.parse_args()

    dataset_dir = args.dataset_dir or os.path.join("dataset", args.dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    print("=" * 60)
    print(f"Rebuilding Qwen embeddings with TS train_ids")
    print(f"  dataset  : {args.dataset}")
    print(f"  data dir : {dataset_dir}")
    print("=" * 60)

    print("\n[Step 1/3] Getting TS train item IDs ...")
    train_ids = _get_ts_train_ids(args.dataset, args.config)

    print("\n[Step 2/3] Rebuilding single-view embedding ...")
    rebuild_single_view(dataset_dir, train_ids)

    print("\n[Step 3/3] Rebuilding multi-view embeddings ...")
    rebuild_multi_view(dataset_dir, train_ids)

    print("\n✅ Done. New files:")
    print(f"  {dataset_dir}/item_text_emb.qwen2.5_7b.base.ts.npy")
    print(f"  {dataset_dir}/qwen2.5_7b_4views_ts/view_{{0..3}}.npy")


if __name__ == "__main__":
    main()
