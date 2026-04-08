#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Book-Crossing 的 `book-crossing.inter` 补全 RecBole 所需的 `timestamp:float` 列。

## 时间戳含义（按「交互顺序」）

- 脚本**按当前文件从上到下的行序**（pandas 读入后的行序 = 磁盘文件行序）为每条交互赋值
  `timestamp = 0, 1, 2, …, n-1`。
- RecBole 里 `eval_args.order=TO` 会对 **全表** 按 `timestamp` 排序；由于这些值**全局唯一且递增**，
  排序后行序与**原文件行序一致**。
- 因此：**同一用户内**的交互顺序 = 该用户各条交互在**原文件中出现的先后顺序**。
  若你期望「用户内顺序 = 真实时间」而原始文件未按该顺序排列，请先自行排序/清洗后再运行本脚本。

## 如何执行（在仓库根目录）

  # 方式 A：指定目录（推荐，默认 `book-crossing.inter`）
  python tools/preprocess_book_crossing_inter_add_timestamp.py \\
    --dataset_dir dataset/book-crossing --in_place --backup

  # 方式 B：直接指定 .inter 文件
  python tools/preprocess_book_crossing_inter_add_timestamp.py \\
    --inter_path dataset/book-crossing/book-crossing.inter --in_place --backup

已含 `timestamp` 列时默认跳过（可用 `--force` 删掉后按行序重写）。

备份：`--in_place` 且 `--backup` 时，原文件复制为同目录下的 `.bak`。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(
        description="为 book-crossing.inter 按文件行序追加合成 timestamp:float（0..n-1）。"
    )
    p.add_argument(
        "--inter_path",
        default=None,
        help="book-crossing.inter 的绝对或相对路径（tab 分隔，RecBole 表头）",
    )
    p.add_argument(
        "--dataset_dir",
        default=None,
        help="数据目录，如 dataset/book-crossing；将读写 <dataset_dir>/<dataset_name>.inter",
    )
    p.add_argument(
        "--dataset_name",
        default="book-crossing",
        help="与 --dataset_dir 联用，默认文件名前缀 book-crossing",
    )
    p.add_argument(
        "--in_place",
        action="store_true",
        help="原地写回（建议与 --backup 同用）",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="原地写回前先复制为 .bak",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="即使已有 timestamp 也删除后按行序重写",
    )
    args = p.parse_args()

    if args.inter_path and args.dataset_dir:
        print("[ERROR] 请只指定 --inter_path 或 --dataset_dir 之一。", file=sys.stderr)
        sys.exit(2)
    if args.dataset_dir:
        path = os.path.abspath(
            os.path.join(args.dataset_dir, f"{args.dataset_name}.inter")
        )
    elif args.inter_path:
        path = os.path.abspath(args.inter_path)
    else:
        print(
            "[ERROR] 必须提供 --inter_path 或 --dataset_dir。",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path, sep="\t")
    base_cols = [c.split(":")[0] for c in df.columns]

    has_ts = "timestamp" in base_cols
    if has_ts and not args.force:
        print(f"[OK] {path} already has timestamp column; nothing to do.")
        sys.exit(0)

    if has_ts and args.force:
        ts_name = [c for c in df.columns if c.split(":")[0] == "timestamp"][0]
        df = df.drop(columns=[ts_name])
        has_ts = False

    # 合成时间戳：按当前文件行序 0..n-1（全局唯一，TO 排序后保持原行序）
    n = len(df)
    df["timestamp:float"] = pd.Series(range(n), dtype=float)

    out = path if args.in_place else path + ".with_timestamp"

    if args.in_place and args.backup:
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[INFO] Backup: {bak}")

    df.to_csv(out, sep="\t", index=False)
    print(f"[OK] Wrote {n} rows with synthetic timestamp:float (row order 0..{n-1}) -> {out}")

    if not args.in_place:
        print(f"[INFO] Review then: mv {out} {path}")


if __name__ == "__main__":
    main()
