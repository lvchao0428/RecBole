# Main Tables — GTS-Last / GTS-Random (MRR@10)

> Protocol lock (0803): primary = **Last / Random**, warm preferred for main claims;
> GTS-All and Clean LOO are contrast / appendix only.
>
> Last source: `paper_recsys/gts_last_eval.csv`
> Random source: `paper_recsys/gts_random_eval.csv`

min_train_interactions=5; 3 seeds {42,2024,2026}; mean±sample-std.

## GTS-Last

### All-user (one target / user)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0022±0.0004 | 0.0036±0.0006 | 0.0041±0.0003 | **0.0042±0.0004** | 0.0036±0.0001 |
| Toys | 0.0019±0.0001 | 0.0025±0.0003 | 0.0022±0.0001 | **0.0026±0.0004** | 0.0026±0.0000 |

### Warm targets only (target ∈ train catalog)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0032±0.0006 | 0.0051±0.0008 | 0.0059±0.0005 | **0.0061±0.0005** | 0.0052±0.0002 |
| Toys | 0.0031±0.0002 | 0.0040±0.0005 | 0.0035±0.0002 | **0.0042±0.0006** | 0.0041±0.0001 |

### Ranking summary (GTS-Last, warm)

- **Beauty**: MV(0.0061) > LLM(0.0059) > UniSRec(0.0052) > TF(0.0051) > ID(0.0032)
- **Toys**: MV(0.0042) > UniSRec(0.0041) > TF(0.0040) > LLM(0.0035) > ID(0.0031)

## GTS-Random

### All-user (one target / user)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0055±0.0002 | **0.0095±0.0009** | 0.0089±0.0010 | 0.0092±0.0008 | 0.0067±0.0001 |
| Toys | 0.0058±0.0004 | **0.0088±0.0004** | 0.0085±0.0004 | 0.0085±0.0005 | 0.0062±0.0003 |

### Warm targets only (target ∈ train catalog)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0081±0.0003 | **0.0139±0.0013** | 0.0130±0.0015 | 0.0134±0.0011 | 0.0098±0.0001 |
| Toys | 0.0096±0.0006 | **0.0146±0.0006** | 0.0140±0.0007 | 0.0141±0.0008 | 0.0103±0.0006 |

### Ranking summary (GTS-Random, warm)

- **Beauty**: TF(0.0139) > MV(0.0134) > LLM(0.0130) > UniSRec(0.0098) > ID(0.0081)
- **Toys**: TF(0.0146) > MV(0.0141) > LLM(0.0140) > UniSRec(0.0103) > ID(0.0096)

## Contrast: GTS-All (appendix reference, not main)

| Dataset | ID | TF | LLM | MV | UniSRec |
|---|---:|---:|---:|---:|---:|
| Beauty | 0.0106±0.0005 | **0.0165±0.0008** | 0.0161±0.0005 | 0.0160±0.0008 | 0.0123±0.0001 |
| Toys | 0.0070±0.0001 | **0.0113±0.0003** | 0.0107±0.0004 | 0.0109±0.0006 | 0.0082±0.0001 |

Clean LOO (contrast): Beauty TF 0.0251 / LLM 0.0254 / MV 0.0253; Toys TF 0.0247 / LLM≈MV 0.0258.

