#!/usr/bin/env python3
import json
runs = [
    ("baseline", "20260620-232134_SASRecAlign_Phase-A.txt"),
    ("tfidf_0618", "20260618-180132_SASRecAlignV3_Phase-B.txt"),
    ("tfidf_aligned", "20260620-064619_SASRecAlignV3_Phase-B.txt"),
    ("llm_0618", "20260618-210359_SASRecAlignV3_Phase-B.txt"),
    ("llm_aligned", "20260620-094844_SASRecAlignV3_Phase-B.txt"),
    ("mv_0621", "20260621-093732_SASRecAlignMultiViewV3_Phase-B.txt"),
    ("mv_g0", "20260620-173233_SASRecAlignMultiViewV3_Phase-B.txt"),
]
base = "/home/charlie/project/RecBole/run_metrics"
for name, fn in runs:
    j = json.load(open(f"{base}/{fn}"))
    pb = j["results"].get("phase_b_result", {})
    te = pb.get("test_result") or j["results"].get("best_valid_result", {})
    print(
        f"{name:14} mrr@10={te.get('mrr@10')} rec@10={te.get('recall@10')} "
        f"new={te.get('MRR_new@10')} few={te.get('MRR_few@10')} fr={te.get('MRR_frequent@10')}"
    )
