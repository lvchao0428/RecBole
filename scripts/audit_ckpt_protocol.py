#!/usr/bin/env python3
"""Dump protocol-critical fields from RecBole checkpoints (no GPU)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

KEYS = [
    "model",
    "dataset",
    "seed",
    "learning_rate",
    "attn_dropout_prob",
    "hidden_dropout_prob",
    "use_cross",
    "use_multiview_text_cross",
    "use_seq_text_cross",
    "cold_text_boost",
    "infer_boost",
    "use_llm",
    "text_mode",
    "use_text_view_split",
    "mv_include_base",
    "num_text_views",
    "text_view_indices",
    "item_text_emb_path_base",
    "item_text_emb_path_llm",
    "item_text_emb_split_dir",
]


def pick_pth(path: Path) -> Path:
    if path.is_file() and path.suffix == ".pth":
        return path
    pths = list(path.glob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth in {path}")
    return max(pths, key=lambda p: p.stat().st_size)


def path_flags(p: str) -> str:
    s = (p or "").replace("\\", "/")
    flags = []
    if not s:
        return "empty"
    if s.endswith(".ts.npy") or "/qwen2.5_7b_4views_ts" in s or s.endswith("_ts") or "/qwen2.5_7b_4views_ts/" in s:
        flags.append("TS")
    if "ts_early60" in s:
        flags.append("early60")
    if s.endswith(".base.npy") or s.endswith("4views") or "/qwen2.5_7b_4views/" in s:
        if "TS" not in flags and "early60" not in flags:
            flags.append("NON_TS")
    if "4views_raw" in s:
        flags.append("raw")
    return ",".join(flags) if flags else "other"


def inspect(path: Path) -> dict:
    pth = pick_pth(path)
    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        return {"tag": path.name, "pth": str(pth), "error": "no config"}
    d = cfg.final_config_dict if hasattr(cfg, "final_config_dict") else dict(cfg)
    row = {"tag": path.name, "pth": pth.name, "pth_mb": f"{pth.stat().st_size / 1e6:.0f}"}
    for k in KEYS:
        val = d.get(k, "")
        if k in ("item_text_emb_path_base", "item_text_emb_path_llm", "item_text_emb_split_dir"):
            s = str(val) if val not in (None, "") else ""
            row[k] = s
            row[k + "_flag"] = path_flags(s)
        else:
            row[k] = val
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_list", required=True)
    ap.add_argument("--out", default="paper_recsys/checkpoint_protocol_audit.csv")
    args = ap.parse_args()

    paths = []
    for line in Path(args.ckpt_list).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(Path(line))

    rows = []
    for p in paths:
        try:
            row = inspect(p)
            print(f"OK {p} model={row.get('model')} ds={row.get('dataset')} "
                  f"base={row.get('item_text_emb_path_base_flag')} "
                  f"llm={row.get('item_text_emb_path_llm_flag')} "
                  f"views={row.get('item_text_emb_split_dir_flag')} "
                  f"cross={row.get('use_cross')} boost={row.get('cold_text_boost')}")
        except Exception as e:
            row = {"tag": p.name, "error": str(e)}
            print(f"FAIL {p}: {e}", file=sys.stderr)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
