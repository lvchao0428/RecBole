#!/usr/bin/env python3
"""Summarize MRR/NDCG/HR for stopgate 3-seed + grid seed2025 refs."""
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def parse_test(path: Path):
    text = path.read_text(errors="ignore")
    blocks = re.findall(r"test result[:\s]*\{([^}]+)\}", text, flags=re.I)

    def pick(key, block):
        m = re.search(rf"[\"']?{re.escape(key)}[\"']?\s*[:=]\s*([0-9]+\.[0-9]+)", block, flags=re.I)
        return float(m.group(1)) if m else None

    if blocks:
        b = blocks[-1]
        hit = pick("hit@10", b) or pick("recall@10", b)
        return {"mrr": pick("mrr@10", b), "ndcg": pick("ndcg@10", b), "hit": hit}

    mrr = re.findall(r"[\"']mrr@10[\"']\s*:\s*([0-9.]+)", text)
    ndcg = re.findall(r"[\"']ndcg@10[\"']\s*:\s*([0-9.]+)", text)
    hit = re.findall(r"[\"'](?:hit|recall)@10[\"']\s*:\s*([0-9.]+)", text)
    if not mrr:
        return None
    return {
        "mrr": float(mrr[-1]),
        "ndcg": float(ndcg[-1]) if ndcg else None,
        "hit": float(hit[-1]) if hit else None,
    }


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "NA"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{st.mean(xs):.4f}±{st.stdev(xs):.4f}"


def main():
    root = Path(__file__).resolve().parents[1]
    logs = root / "logs"

    groups = {
        "LLM lr1e-4/do0.1": sorted(logs.glob("sg_beauty_llm_lr1e4_do0.1_seed*.log")),
        "TF  lr5e-4/do0.3": sorted(logs.glob("sg_beauty_tf_lr5e4_do0.3_seed*.log")),
        "MV  lr5e-4/do0.3": sorted(logs.glob("sg_beauty_mv_lr5e4_do0.3_seed*.log")),
        "MV* lr5e-4/do0.5": sorted(logs.glob("sg_beauty_mv_lr5e4_do0.5_seed*.log")),
    }

    print("=" * 90)
    print("3-SEED (42/2024/2026) — per seed")
    print(f"{'config':<22} {'seed':>6} {'MRR@10':>8} {'NDCG@10':>8} {'HR@10':>8}")
    print("-" * 90)
    agg = defaultdict(lambda: {"mrr": [], "ndcg": [], "hit": []})
    for name, paths in groups.items():
        for p in paths:
            seed = p.stem.split("seed")[-1]
            m = parse_test(p)
            if not m:
                continue
            print(f"{name:<22} {seed:>6} {m['mrr']:8.4f} {m['ndcg']:8.4f} {m['hit']:8.4f}")
            agg[name]["mrr"].append(m["mrr"])
            agg[name]["ndcg"].append(m["ndcg"])
            agg[name]["hit"].append(m["hit"])

    print()
    print("=" * 90)
    print("3-SEED mean ± std")
    print(f"{'config':<22} {'MRR@10':>16} {'NDCG@10':>16} {'HR@10':>16}")
    print("-" * 90)
    for name in groups:
        a = agg[name]
        print(f"{name:<22} {fmt(a['mrr']):>16} {fmt(a['ndcg']):>16} {fmt(a['hit']):>16}")

    print()
    print("=" * 90)
    print("GRID seed=2025 reference")
    refs = [
        ("LLM test=valid", logs / "grid_beauty_SASRecAlignV3_lr0.0001_do0.1_seed2025.log"),
        ("TF  test=valid", logs / "grid_beauty_tfidf_lr0.0005_do0.3_seed2025.log"),
        ("MV  valid do0.3", logs / "grid_beauty_mv_fix_lr0.0005_do0.3_seed2025.log"),
        ("MV  test  do0.5", logs / "grid_beauty_mv_fix_lr0.0005_do0.5_seed2025.log"),
    ]
    print(f"{'config':<22} {'MRR@10':>8} {'NDCG@10':>8} {'HR@10':>8}")
    print("-" * 90)
    for name, path in refs:
        if not path.exists():
            print(f"{name:<22} MISSING")
            continue
        m = parse_test(path)
        print(f"{name:<22} {m['mrr']:8.4f} {m['ndcg']:8.4f} {m['hit']:8.4f}")


if __name__ == "__main__":
    main()
