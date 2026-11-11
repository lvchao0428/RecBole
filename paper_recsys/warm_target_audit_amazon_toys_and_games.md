# Warm-target Audit — Amazon_Toys_and_Games

> Generated: 2026-08-02 17:00
> Script: `scripts/audit_warm_targets.py`

## Cutoffs

| | |
|---|---|
| Train cutoff | 2014-01-17 (ts=1389916800.0) |
| Valid cutoff | 2014-04-08 (ts=1396915200.0) |
| Train catalog size | 283233 |
| Eligible users (hist≥5) | 44024 |

## Unseen target share

| Split | Unseen share | Unseen / Total |
|-------|:------------:|:--------------:|
| Valid | 21.1% | 1543 / 7307 |
| Test | **34.5%** | 2300 / 6665 |

Test catalog turnover (unseen items among test targets): **31.1%**

## Target popularity (train count)

| Split | mean | p50 |
|-------|:----:|:---:|
| Valid | 69.3 | 9.0 |
| Test | 47.2 | 3.0 |

## Protocol recommendation (0723)

- Main table: **warm targets only** (item appeared in train before cutoff).
- Unseen targets: separate table; do **not** compare ID-only vs text models unless content-only scoring is enabled.
- Candidate catalog: train warm items (full ranking).
- Paper Methods must state: GTS cutoff, target selection, candidate catalog, cold filtering, valid/test construction.
