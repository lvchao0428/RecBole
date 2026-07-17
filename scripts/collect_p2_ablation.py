#!/usr/bin/env python3
"""
Advisor P(2) Ablation 结果收集
Cross on/off × 4 text configs，V2 TS-aware leakage-free, no-boost, seed=2025

运行方式：
  python3 scripts/collect_p2_ablation.py
"""
import re, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")

KEY = ["mrr@10", "ndcg@10", "recall@10", "Recall_new@10", "Recall_few@10", "Recall_frequent@10"]
KEY_DISPLAY = ["MRR@10", "NDCG@10", "R@10", "R_new@10", "R_few@10", "R_freq@10"]


def extract(path):
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    lines = [l for l in txt.split("\n") if "test result: OrderedDict" in l]
    # fallback: Run Summary JSON
    if not lines:
        import json
        m = re.search(r"Run Summary JSON\] ({.+})", txt)
        if m:
            try:
                d = json.loads(m.group(1))
                return d["results"]["test_result"]
            except Exception:
                return None
        return None
    last = lines[-1]
    vals = {}
    for k in KEY:
        m = re.search(
            re.escape("'" + k + "'") + r"['\s]*:[\s]*(?:np\.float64\()?([\d.]+)\)?", last
        )
        if m:
            vals[k] = float(m.group(1))
    return vals if vals else None


MATRIX = [
    # (label, cross, log_filename)
    ("ID-only",          "—",   "ts_beauty_id_only_min5_seed2025.log"),
    ("TF-IDF + Cross",   "on",  "ts_beauty_tfidf_v2_seed2025.log"),
    ("TF-IDF no-Cross",  "off", "ts_beauty_tfidf_nocross_v2_seed2025.log"),
    ("LLM + Cross",      "on",  "ts_beauty_tfidf_llm_v2_seed2025.log"),
    ("LLM no-Cross",     "off", "ts_beauty_llm_nocross_v2_seed2025.log"),
    ("MV + Cross",       "on",  "ts_beauty_mv_v2_seed2025.log"),
    ("MV no-Cross",      "off", "ts_beauty_mv_nocross_v2_seed2025.log"),
]


def main():
    print("=" * 100)
    print("Advisor P(2) Ablation: Cross effect under V2 TS-aware leakage-free protocol")
    print("Config: no-SE, no-boost, seed=2025, min_train>=5")
    print("=" * 100)
    print()

    # Header
    col_w = 20
    hdr = f"{'Config':<22} {'Cross':^5}  " + "  ".join(f"{k:>9}" for k in KEY_DISPLAY)
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for label, cross, fn in MATRIX:
        path = os.path.join(LOG_DIR, fn)
        d = extract(path)
        results[label] = d
        if d:
            vals = "  ".join(f"{d.get(k, 0):>9.4f}" for k in KEY)
            status = ""
        else:
            vals = "  ".join(f"{'N/A':>9}" for _ in KEY)
            status = "  ← NOT FOUND"
        print(f"{label:<22} {cross:^5}  {vals}{status}")

    print()
    print("=" * 100)
    print("Cross effect summary (with − without Cross, per text family):")
    print("-" * 60)

    pairs = [
        ("TF-IDF",      "TF-IDF + Cross",  "TF-IDF no-Cross"),
        ("TF-IDF+LLM",  "LLM + Cross",     "LLM no-Cross"),
        ("MV-Align",    "MV + Cross",       "MV no-Cross"),
    ]
    for family, with_lbl, without_lbl in pairs:
        dw = results.get(with_lbl)
        dwo = results.get(without_lbl)
        if dw and dwo:
            delta_mrr  = dw["mrr@10"]  - dwo["mrr@10"]
            delta_ndcg = dw["ndcg@10"] - dwo["ndcg@10"]
            delta_r    = dw["recall@10"] - dwo["recall@10"]
            delta_rnew = dw.get("Recall_new@10", 0) - dwo.get("Recall_new@10", 0)
            sign = lambda x: "+" if x >= 0 else ""
            print(
                f"  {family:<14}  Δ MRR={sign(delta_mrr)}{delta_mrr:+.4f}  "
                f"Δ NDCG={sign(delta_ndcg)}{delta_ndcg:+.4f}  "
                f"Δ R@10={sign(delta_r)}{delta_r:+.4f}  "
                f"Δ R_new={sign(delta_rnew)}{delta_rnew:+.4f}"
            )
        else:
            missing = []
            if not dw:  missing.append(with_lbl)
            if not dwo: missing.append(without_lbl)
            print(f"  {family:<14}  MISSING: {', '.join(missing)}")

    print()
    print("Interpretation guide:")
    print("  Cross ON  vs OFF:")
    print("    + MRR/NDCG  → Cross improves ranking quality")
    print("    - Recall    → Cross reduces HR (coverage trade-off)")
    print("    + R_new     → Cross helps new/cold items")
    print()
    print("  If Cross Δ MRR < 0 consistently → remove from main model (advisor advice)")
    print("  If Cross Δ MRR > 0 but Δ Recall < 0 → present as MRR-HR trade-off mechanism")


if __name__ == "__main__":
    main()
