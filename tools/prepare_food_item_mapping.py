#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Food item_index_mapping.csv with unified text = name + nutrition + tags.

All pipelines (TF-IDF / single-view LLM / 4-view MV) share the same `title` column.
MV uses Beauty/Toys multiview prompts; each view reads the full text and extracts
its own aspect — no manual tag splitting.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset

# Food.com 7-dim nutrition order (calories fat sugar sodium protein saturated_fat carbohydrate)
NUTRITION_LABELS = (
    "calories",
    "total_fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbohydrate",
)


def _find_col(df, base_names):
    for name in base_names:
        if name in df.columns:
            return name
    for name in base_names:
        for c in df.columns:
            if isinstance(c, str) and c.split(":")[0] == name:
                return c
    return None


def _format_nutrition(raw: str) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    vals = text.split()
    if len(vals) == len(NUTRITION_LABELS):
        pairs = [f"{label} {v}" for label, v in zip(NUTRITION_LABELS, vals)]
        return "nutrition " + " ".join(pairs)
    return "nutrition " + text


def _full_text(name: str, nutrition: str, tags: str) -> str:
    parts = []
    name = (name or "").strip()
    tags = (tags or "").strip()
    nut = _format_nutrition(nutrition)
    if name:
        parts.append(name)
    if nut:
        parts.append(nut)
    if tags:
        parts.append(tags)
    return " ".join(parts)


def prepare_food_mapping(dataset_name: str, config_files, output_csv: str) -> None:
    cfg = Config(model="BPR", dataset=dataset_name, config_file_list=config_files)
    dataset = create_dataset(cfg)

    iid_field = dataset.iid_field
    n_items = dataset.num(iid_field)
    ids = np.arange(n_items, dtype=np.int64)
    tokens = dataset.id2token(iid_field, ids).astype(str)

    df = pd.DataFrame({"internal_item_id": ids, "item_token": tokens})

    dataset_dir = dataset.dataset_path
    item_file = os.path.join(dataset_dir, f"{dataset_name}.item")
    if not os.path.isfile(item_file):
        raise FileNotFoundError(f"Missing item file: {item_file}")

    item_df = pd.read_csv(item_file, sep="\t")
    item_id_col = _find_col(item_df, [cfg["ITEM_ID_FIELD"], "item_id"])
    name_col = _find_col(item_df, ["name"])
    tags_col = _find_col(item_df, ["tags"])
    nutrition_col = _find_col(item_df, ["nutrition"])
    if not item_id_col or not name_col:
        raise ValueError(f"Cannot find item_id/name columns in {item_file}")

    rows = []
    for _, row in item_df.iterrows():
        tok = str(row[item_id_col])
        name = str(row[name_col]) if name_col else ""
        tags = str(row[tags_col]) if tags_col and pd.notna(row.get(tags_col)) else ""
        nutrition = row[nutrition_col] if nutrition_col and nutrition_col in row.index else ""
        rows.append(
            {
                "item_token": tok,
                "name": name.strip(),
                "nutrition": _format_nutrition(nutrition),
                "tags": tags.strip(),
                "title": _full_text(name, nutrition, tags),
            }
        )

    text_df = pd.DataFrame(rows)
    out = df.merge(text_df, on="item_token", how="left")
    out["title"] = out["title"].fillna("")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out.to_csv(output_csv, index=False)

    nonempty = (out["title"].str.len() > 0).sum()
    active = out.loc[out["internal_item_id"] > 0]
    lens = active["title"].str.split().str.len()
    has_nut = (active["nutrition"].str.len() > 0).sum()
    print(f"Saved: {output_csv}")
    print(f"  rows={len(out)}  nonempty_title={nonempty}  with_nutrition={has_nut}")
    print(
        f"  title tokens: mean={lens.mean():.1f} med={lens.median():.0f} "
        f"p90={lens.quantile(0.9):.0f} max={lens.max():.0f}"
    )
    if len(active) > 0:
        sample = active.loc[active["internal_item_id"] == 1, "title"]
        if len(sample):
            print(f"  sample[1]: {sample.iloc[0][:200]}...")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Food")
    p.add_argument("--config", nargs="+", default=["sasrec_food_plain.yaml"])
    p.add_argument("--output", default="dataset/Food/item_index_mapping.csv")
    args = p.parse_args()
    prepare_food_mapping(args.dataset, args.config, args.output)


if __name__ == "__main__":
    main()
