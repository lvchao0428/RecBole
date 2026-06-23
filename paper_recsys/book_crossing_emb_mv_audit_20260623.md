# 文本 embedding 去相关 / MV 潜力分析

| Dataset | TF-IDF sim | LLM sim | view diversity | whiten offdiag | MV 优势 |
|---------|------------|---------|----------------|----------------|---------|
| book-crossing | 0.000 | 0.836 | 0.945 | 1.191 | moderate |
| Amazon_Beauty | 0.000 | -0.000 | 1.023 | 1.032 | moderate |

## Pairwise view cosine (lower = more diverse)

**book-crossing**:
- view_0_vs_view_1: 0.048
- view_0_vs_view_2: 0.113
- view_0_vs_view_3: -0.003
- view_1_vs_view_2: 0.065
- view_1_vs_view_3: 0.008
- view_2_vs_view_3: 0.102

**Amazon_Beauty**:
- view_0_vs_view_1: -0.094
- view_0_vs_view_2: -0.017
- view_0_vs_view_3: 0.103
- view_1_vs_view_2: -0.192
- view_1_vs_view_3: -0.028
- view_2_vs_view_3: 0.088