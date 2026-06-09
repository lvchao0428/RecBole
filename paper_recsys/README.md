### SIGIR LaTeX Draft (ACM `acmart` / `sigconf`)

This folder contains a **SIGIR-format LaTeX draft** aligned with the current codebase:

- **Multi-view model**: `two_phase_run_multiview_v2_stratified.sh` (`SASRecAlignMultiViewV2`)
- **Strong text baseline**: `two_phase_run_tfidf_stratified.sh` (`SASRec_Align` / TF-IDF base)
- **ID-only baseline**: `run50epBase_stratified.sh` (`SASRecAlign` with text disabled)

### Files

- `main.tex`: paper draft (review mode: anonymous)
- `refs.bib`: bibliography
- `figures/`: put your figures here (e.g., `framework.pdf`)

### Compile (local)

From repo root:

```bash
cd paper_sigir
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Compile (Overleaf)

- Upload the whole `paper_sigir/` directory to Overleaf.
- Set the main file to `main.tex`.

### Switch to camera-ready

In `main.tex`, change:

- `\documentclass[sigconf,review,anonymous]{acmart}` → `\documentclass[sigconf]{acmart}`
- Add author blocks (currently omitted for double-blind review).


