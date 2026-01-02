# Budgeted Scaling Ablation Plan (LLM size × SVD compression budget)

This guide turns the proposed 2D ablation into **actionable steps** using the existing feature-generation scripts and the newly added SVD EVR logging in `tools/build_item_text_emb_qwen3_hf.py`.

## Goal

We want to verify whether the observed non-monotonic “LLM scaling” behavior is actually caused by a **fixed low-dimensional bottleneck** (e.g., multi-view per-view `--view_project_dim 64`) rather than the LLM itself.

In the paper, this is framed as **budgeted scaling**: scaling under a given **embedding budget / compression strength**.

## What stays fixed (downstream)

To make the ablation interpretable, keep all downstream settings identical across runs:

- **Two-phase training schedule** and hyper-parameters (same YAMLs / same run scripts)
- **Seed**, **split**, **evaluation protocol** (full ranking + stratified)
- **Model architecture** (SASRecAlignMultiViewV2, cross, alignment, etc.)

Only change the **text embedding generation / compression budget**.

## What varies (the 2D grid)

- **LLM size**: `{7B, 14B, 32B}`
- **Compression budget** (per-view SVD dim): `view_project_dim ∈ {32, 64, 128, 256}`
  - Current default is typically **64** in scripts such as:
    - `tools/gen_text_emb_beauty_qwen2.5_7b.sh`
    - `tools/gen_text_emb_beauty_qwen2.5_14b.sh`
    - `tools/gen_text_emb_beauty_qwen2.5_32b.sh`
    - `tools/gen_text_emb_toys_qwen2.5_*.sh`

## Outputs to collect (per run)

For each (dataset, llm_size, view_project_dim), collect:

1. **Downstream metrics**: overall + stratified (new/few/frequent) + coverage
2. **EVR (explained variance ratio)** from TruncatedSVD:
   - Captures how much variance is retained under the chosen budget.

This repository now supports:

- `--svd_report_evr` (print summary)
- `--svd_report_evr_path <path.json>` (save full EVR vector + summary)

## Step-by-step procedure

### Step 1: Decide naming / directories

For multi-view split outputs, use a directory that encodes the budget, e.g.:

- Beauty:
  - `dataset/Amazon_Beauty/qwen2.5_7b_4views_vdim64/`
  - `dataset/Amazon_Beauty/qwen2.5_7b_4views_vdim128/`
  - `dataset/Amazon_Beauty/qwen2.5_32b_4views_vdim256/`
- Toys:
  - `dataset/Amazon_Toys_and_Games/qwen2.5_14b_4views_vdim64/`
  - `dataset/Amazon_Toys_and_Games/qwen2.5_32b_4views_vdim128/`

Also store EVR JSON next to the split outputs:

- `.../svd_evr.json`

### Step 2: Generate embeddings for each budget

Use the existing generator scripts as the base, and for each desired `view_project_dim` do:

1. Set `--view_project_dim <DIM>`
2. Set `--split_output_dir <DIR_WITH_DIM>`
3. Add EVR logging flags:
   - `--svd_report_evr_path "<DIR_WITH_DIM>/svd_evr.json"`
   - (optional) `--svd_report_evr` to also print EVR summary

#### Example: Beauty + Qwen2.5-32B + view_project_dim=128

Take `tools/gen_text_emb_beauty_qwen2.5_32b.sh` (section “4. multi-view”), and edit:

- `--split_output_dir "dataset/Amazon_Beauty/qwen2.5_32b_4views_vdim128" \`
- `--view_project_dim 128 \`
- add:
  - `--svd_report_evr_path "dataset/Amazon_Beauty/qwen2.5_32b_4views_vdim128/svd_evr.json" \`

#### Example: Toys + Qwen2.5-7B + view_project_dim=256

Take `tools/gen_text_emb_toys_qwen2.5_7b.sh` (section “4. multi-view”), and edit similarly:

- `--split_output_dir "dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_vdim256" \`
- `--view_project_dim 256 \`
- add:
  - `--svd_report_evr_path "dataset/Amazon_Toys_and_Games/qwen2.5_7b_4views_vdim256/svd_evr.json" \`

### Step 3: Point YAML configs to the correct split dir

For each (dataset, llm_size, view_project_dim), create or edit a config that differs **only** in:

```yaml
item_text_emb_split_dir: dataset/<DATASET>/<MODEL>_4views_vdim<DIM>
```

Keep all other training hyper-parameters identical so the comparison isolates the embedding budget.

Examples (you may create new YAMLs to avoid overwriting existing ones):

- Beauty:
  - `sasrec_align_multi_view_v2_stratified_32b_vdim128.yaml`
  - `sasrec_align_multi_view_v2_stratified_7b_vdim256.yaml`
- Toys:
  - `sasrec_align_multi_view_v2_toys_stratified_14b_vdim128.yaml`

### Step 4: Run training (same scripts, only swap YAML)

Run the same two-phase training entrypoint, but change `--config_files` to the YAML that points to your chosen embedding directory.

Keep:

- seed
- two-phase hyperparams
- dataset/model

fixed across the sweep.

### Step 5: Analyze results (what to plot)

To make the “budgeted scaling” claim clean and dataset-consistent, plot:

1. **Metric vs view_project_dim** (one curve per LLM size)
   - overall: Recall@10/20, NDCG@10, MRR@10
   - stratified: Recall\_new@10, Recall\_few@10, Recall\_frequent@10
   - coverage: Coverage\_new@10 (and/or @20)
2. **EVR\_sum vs view_project_dim** (one curve per LLM size)
   - Use `svd_evr.json` and take `evr_sum`.

### Step 6: Interpreting outcomes (decision rules)

- If **32B improves as view_project_dim increases** and/or when EVR retained increases, then:
  - Non-monotonic scaling at `vdim=64` is likely a **compression bottleneck artifact**.
- If **32B remains worse** even at large `view_project_dim` and high EVR retained, then:
  - The issue is more likely in **fusion/alignment/training dynamics** rather than compression.

## Notes / Gotchas

- `--view_project_dim` only applies to **multi-view split** generation (`--split_output_dir` + `--output_mode concat`).
- EVR values should be compared **within the same pipeline** (same dataset, same prompt preset, same whitening/centering flags).
- This plan deliberately avoids changing downstream hyper-parameters; if you later tune per-budget hyperparams, treat it as a separate study.


