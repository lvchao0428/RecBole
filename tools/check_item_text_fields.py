#!/usr/bin/env python3
"""Check .item files for natural-language text fields (title/name/description)."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

TEXT_KEYS = ("title", "name", "description", "text", "summary", "content", "genres", "categories", "brand", "tags")


def check_item_header(path: str):
    try:
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                items = [n for n in zf.namelist() if n.endswith(".item") and "__" not in n]
                if not items:
                    return None, "no .item"
                line = zf.open(items[0]).readline().decode("utf-8", errors="replace").strip()
                sample = zf.open(items[0]).readline()
                sample = zf.open(items[0]).read(500).decode("utf-8", errors="replace").split("\n")
                sample_row = sample[1][:80] if len(sample) > 1 else ""
        else:
            ip = Path(path)
            if ip.suffix == ".inter":
                ip = ip.parent / (ip.stem + ".item")
            if not ip.exists():
                return None, "no .item file"
            lines = ip.read_text(encoding="utf-8", errors="replace").split("\n")
            line = lines[0].strip()
            sample_row = lines[1][:80] if len(lines) > 1 else ""
        cols = [c.split(":")[0].lower() for c in line.split("\t")]
        text_cols = [c for c in cols if any(k in c for k in TEXT_KEYS)]
        has_title = any("title" in c or c == "name" for c in cols)
        return {
            "header": line,
            "cols": cols,
            "text_cols": text_cols,
            "has_title": has_title,
            "sample_row": sample_row,
        }, None
    except Exception as e:
        return None, str(e)


def main():
    root = Path(sys.argv[1])
    proj = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    found = {}
    for zp in root.rglob("*.zip"):
        if "example" in str(zp).lower():
            continue
        found[zp.stem.lower()] = str(zp)
    for ip in root.rglob("*.inter"):
        if "example" in str(ip).lower():
            continue
        found[ip.stem.lower()] = str(ip)
    if proj and proj.is_dir():
        for ip in proj.rglob("*.inter"):
            found[ip.stem.lower()] = str(ip)

    rows = []
    for lab in sorted(found):
        if "amazon" in lab:
            continue
        path = found[lab]
        info, err = check_item_header(path)
        rows.append({"label": Path(path).stem, "path": path, "info": info, "error": err})

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
