# Experiment Status — 2026-08-14

## Summary

Based on advisor guidance (`0813zhidao.txt`), running three experiment blocks + citation fix.

## Completed Today

### 1. LOOV under GTS-Last (P2) — DONE
- 6 runs completed: Beauty/Toys × 3 seeds (42, 2024, 2026)
- Zero-mask on MV GTS-Last warm checkpoints, no retraining needed
- Results saved: `paper_recsys/loov_gts_last.csv`
- Key finding: LOOV deltas remain very small (< 0.0005) under GTS-Last, consistent with prior GTS-All findings

### 2. refs.bib Citation Fix — DONE
Fixed 4 hallucinated/incorrect citations:
| Entry | Problem | Fix |
|-------|---------|-----|
| `liang2024timetosplit` | Wrong authors/year/venue entirely | Gusak et al., RecSys 2025, pp 874-883 |
| `meng2020temporal` | Fabricated title/venue (no such paper) | "Exploring Data Splitting Strategies...", RecSys 2020, pp 681-686 |
| `ji2023evaluation` | Wrong title/venue | "A Critical Study on Data Leakage...", ACM TOIS 41(3), 2023 |
| `sun2023reevaluation` | Wrong title/authors | "Take a Fresh Look...", Sun (sole), SIGIR 2023, pp 2629-2638 |

Removed 2 duplicate entries: `hidasi2015gru4rec`, `zhao2021recbolefull`

### 3. Desktop Display Manager Disabled on 5090
- Stopped `gdm3` to free GPU from Xorg/gnome-shell
- Resolved `cudaErrorLaunchTimeout` that was killing training at ~2h mark
- GPU memory freed from 286MiB to 12MiB

## Currently Running on 5090 (PID 403003)

### P4: Early60 Equal-Horizon [0.6,0.1,0.1] — IN PROGRESS
- Beauty TF/LLM/MV × 3 seeds
- seed42: **all 3 done** (TF=0.0237/0.006, LLM=0.0228/0.0058, MV=0.0237/0.006)
- seed2024: TF done (0.024/0.0064), **LLM running**, MV pending
- seed2026: all 3 pending
- ETA: ~3h from 15:26

### P5: Pure LLM-only (no TF-IDF base) — QUEUED
- Beauty/Toys × 3 seeds (42, 2024, 2026) = 6 runs
- ETA: ~3.5h after P4 completes

### 4. Raw 256-d CKA/ER Audit (P3a) — DONE
- Script: `scripts/cka_raw_256d_audit.py`
- Uses 256-d `no_whiten` embeddings (post-SVD, pre-whitening)
- Results:
  - **Beauty**: CKA(base vs mv) = 0.970, ER(base) = 167/256 (65%), ER(mv) = 99/256 (39%)
  - **Toys**: CKA(base vs mv) = 0.973, ER(base) = 172/256 (67%), ER(mv) = 104/256 (41%)
- **Conclusion**: Redundancy is already extreme (CKA > 0.97) at 256-d pre-whitening stage

### 5. Raw 4096-d CKA/ER Audit (P3b) — QUEUED (after GPU training)
- Script: `scripts/generate_raw_4096d_for_cka.sh`
- Re-runs Qwen2.5-7B inference with 4 prompts, saves raw hidden states (no SVD)
- Datasets: Beauty + Toys + Grocery
- Computes pairwise CKA among 4 views in the original 4096-d space
- **This provides the strongest possible evidence** that redundancy originates from encoder
- ETA: ~30 min after P4+P5 training completes (~23:30)

## Remaining (text-only, no experiments)

1. **[P1] Wording fixes** in `main_wsdm2027_v3.tex`:
   - RQ1: split effect vs target effect distinction
   - "inflates" → "yields higher"
   - LOO explanation → "a plausible explanation is..."
   - "genuine complementarity" → "potential complementarity"/"representational dissimilarity"
   - Remove engineering log statements (REDUNDANT/collapse=True)
   - Clarify warm candidate catalog = train catalog

2. **Update paper tables** with new LOOV GTS-Last results, Early60-EQ results, Pure LLM results

## Overall Progress (vs 0813 advisor requirements)

| Requirement | Status |
|------------|--------|
| RQ1 split/target effect wording | Pending (text edit) |
| RQ2 Early equal-horizon [0.6,0.1,0.1] | Running |
| RQ3 wording: potential complementarity | Pending (text edit) |
| TF+LLM hybrid → Pure LLM-only | Queued (after P4) |
| LOOV GTS-Last 3 seeds | **DONE** |
| Raw CKA/ER pre-whitening audit (256-d) | **DONE** (CKA=0.97) |
| Raw 4096-d CKA audit (pre-SVD, strongest) | Queued (after training) |
| Citation fix (Time to Split etc.) | **DONE** |
| Engineering log cleanup | Pending (text edit) |
| Candidate catalog clarification | Pending (text edit) |
