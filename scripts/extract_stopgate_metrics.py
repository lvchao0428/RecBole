#!/usr/bin/env python3
"""Extract valid/test metrics from stopgate 3-seed logs."""
import re
from pathlib import Path


def pick_from_block(block: str, key: str):
    m = re.search(rf"[\"']?{re.escape(key)}[\"']?\s*[:=]\s*([0-9]+\.[0-9]+)", block, flags=re.I)
    return m.group(1) if m else None


def main():
    root = Path(__file__).resolve().parents[1]
    logs = sorted((root / "logs").glob("sg_beauty_*.log"))
    out_csv = root / "logs" / "stopgate_3seed_beauty_metrics.csv"
    rows = ["tag,model,lr,dropout,seed,valid_mrr,test_mrr,ndcg@10,hit@10,mrr_new"]

    print(rows[0])
    for path in logs:
        text = path.read_text(errors="ignore")
        tag = path.stem
        # parse tag: sg_beauty_{model}_lr..._do..._seed...
        m = re.match(r"sg_beauty_(llm|tf|mv)_lr([0-9e.]+)_do([0-9.]+)_seed(\d+)", tag)
        if m:
            model, lr, do, seed = m.groups()
        else:
            model, lr, do, seed = "?", "?", "?", "?"

        best_valid = re.findall(r"best valid.*?mrr@10.*?([0-9]+\.[0-9]+)", text, flags=re.I | re.S)
        test_blocks = re.findall(r"test result[:\s]*\{([^}]+)\}", text, flags=re.I)
        valid_blocks = re.findall(r"valid result[:\s]*\{([^}]+)\}", text, flags=re.I)

        valid = best_valid[-1] if best_valid else None
        if valid is None and valid_blocks:
            valid = pick_from_block(valid_blocks[-1], "mrr@10")
        test = pick_from_block(test_blocks[-1], "mrr@10") if test_blocks else None
        ndcg = pick_from_block(test_blocks[-1], "ndcg@10") if test_blocks else None
        hit = None
        if test_blocks:
            hit = pick_from_block(test_blocks[-1], "hit@10") or pick_from_block(test_blocks[-1], "recall@10")
        mrr_new = pick_from_block(test_blocks[-1], "mrr_new@10") if test_blocks else None

        all_mrr = re.findall(r"[\"']mrr@10[\"']\s*:\s*([0-9]+\.[0-9]+)", text)
        if test is None and all_mrr:
            test = all_mrr[-1]
        if valid is None and len(all_mrr) >= 2:
            valid = all_mrr[-2]
        if ndcg is None:
            nd = re.findall(r"[\"']ndcg@10[\"']\s*:\s*([0-9]+\.[0-9]+)", text)
            ndcg = nd[-1] if nd else "NA"
        if hit is None:
            ht = re.findall(r"[\"'](?:hit|recall)@10[\"']\s*:\s*([0-9]+\.[0-9]+)", text)
            hit = ht[-1] if ht else "NA"
        if mrr_new is None:
            mn = re.findall(r"[\"']mrr_new@10[\"']\s*:\s*([0-9]+\.[0-9]+)", text)
            mrr_new = mn[-1] if mn else "NA"

        line = f"{tag},{model},{lr},{do},{seed},{valid or 'NA'},{test or 'NA'},{ndcg},{hit},{mrr_new}"
        rows.append(line)
        print(line)

    out_csv.write_text("\n".join(rows) + "\n")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
