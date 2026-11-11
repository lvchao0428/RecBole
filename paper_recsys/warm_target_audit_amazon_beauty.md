# Warm-target Audit — Amazon_Beauty

> Generated: 2026-08-02 17:00
> Script: `scripts/audit_warm_targets.py`

## Cutoffs

| | |
|---|---|
| Train cutoff | 2014-03-09 (ts=1394323200.0) |
| Valid cutoff | 2014-05-17 (ts=1400284800.0) |
| Train catalog size | 217470 |
| Eligible users (hist≥5) | 39530 |

## Unseen target share

| Split | Unseen share | Unseen / Total |
|-------|:------------:|:--------------:|
| Valid | 15.2% | 1193 / 7845 |
| Test | **27.8%** | 1762 / 6332 |

Test catalog turnover (unseen items among test targets): **23.3%**

## Target popularity (train count)

| Split | mean | p50 |
|-------|:----:|:---:|
| Valid | 65.7 | 11.0 |
| Test | 53.7 | 6.0 |

## Protocol recommendation (0723)

- Main table: **warm targets only** (item appeared in train before cutoff).
- Unseen targets: separate table; do **not** compare ID-only vs text models unless content-only scoring is enabled.
- Candidate catalog: train warm items (full ranking).
- Paper Methods must state: GTS cutoff, target selection, candidate catalog, cold filtering, valid/test construction.
