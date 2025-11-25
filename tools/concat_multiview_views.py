#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np


DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": np.float32,  # store as float32 if bf16 requested
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate per-view embeddings (view_*.npy) into a single long vector file."
    )
    parser.add_argument("--split_dir", required=True, help="Directory containing view_*.npy and views.json.")
    parser.add_argument("--output", required=True, help="Output path for concatenated .npy file.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=DTYPE_MAP.keys(),
        help="Output dtype. bfloat16 stores as float32 for compatibility.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Number of items to process per chunk when concatenating.",
    )
    return parser.parse_args()


def load_meta(split_dir: str) -> dict:
    meta_path = os.path.join(split_dir, "views.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"views.json not found in {split_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "prompts" not in meta or len(meta["prompts"]) == 0:
        raise ValueError(f"views.json in {split_dir} contains no prompts metadata.")
    return meta


def main():
    args = parse_args()
    split_dir = os.path.abspath(os.path.expanduser(args.split_dir))
    output_path = os.path.abspath(os.path.expanduser(args.output))

    meta = load_meta(split_dir)
    prompts = sorted(meta["prompts"], key=lambda p: p.get("index", 0))
    num_items = int(meta.get("num_items", 0))
    if num_items <= 0:
        raise ValueError("Invalid num_items in views.json.")

    view_dim = int(prompts[0]["vector_dim"])
    dtype = DTYPE_MAP[args.dtype]
    chunk = max(1, args.chunk_size)

    view_files = []
    for entry in prompts:
        file_name = entry.get("file")
        if not file_name:
            raise ValueError("views.json entry missing file field.")
        full_path = file_name if os.path.isabs(file_name) else os.path.join(split_dir, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"View file missing: {full_path}")
        view_files.append(full_path)

    total_dim = len(view_files) * view_dim
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(num_items, total_dim))

    for start in range(0, num_items, chunk):
        end = min(start + chunk, num_items)
        slices = []
        for vf in view_files:
            arr = np.load(vf, mmap_mode="r")[start:end]
            if arr.shape[1] != view_dim:
                raise ValueError(f"Unexpected dimension for {vf}: {arr.shape[1]} vs {view_dim}")
            slices.append(arr.astype(dtype, copy=False))
        out[start:end] = np.concatenate(slices, axis=1)

    out[0, :] = 0.0
    del out
    print(f"[Concat] Saved {output_path} with shape ({num_items}, {total_dim}) and dtype {dtype}.")


if __name__ == "__main__":
    main()

