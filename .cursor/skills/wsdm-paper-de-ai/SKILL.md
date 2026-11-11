---
name: wsdm-paper-de-ai
description: >-
  De-AI-tone and claim-strength rules for the WSDM 2027 leakage-aware text-for-SR
  paper. Use when editing paper_recsys/main_wsdm2027*.tex, the Chinese counterpart,
  Overleaf copies, abstracts, figure captions, or when the user mentions 去AI味道,
  de-AI, 润色, overclaim, hedge, 结论限定再降调, or advisor writing comments.
---

# WSDM paper: de-AI writing

This paper is an empirical study, not a model-proposal paper. Reviewers who have seen many LLM-polished drafts will flag stacked hedges, citation slips, and silent number edits before they flag missing experiments.

**Default files**

| Role | File | Touch? |
|------|------|--------|
| Stable mainline | `paper_recsys/main_wsdm2027_v3_0822.tex` (+ `_zh_0822`) | Only if the user names 0822 |
| Working de-AI | `paper_recsys/main_wsdm2027_v3_0823.tex` (+ `_zh_0823`) | Default for style edits |
| Overleaf | `paper_recsys/overleaf_deploy/main_wsdm2027_v3_0822.tex` | Sync only after the user asks |

Do not start a full-paper rewrite. Propose a short sentence list first; edit only after the user agrees.

Current leftover locations: [inventory.md](inventory.md). Advisor wording to keep: [advisor-quotes.md](advisor-quotes.md).

## Hard stops (advisor)

Do **not** let the model auto-polish the whole paper. The failure mode is silently changing approved numbers, protocols, or wording.

- Last-round edits: sentence-level diffs only. No “while I’m here” rewrites of nearby paragraphs.
- Never change a table number, caption number, or in-text citation of a table number unless the user asked to fix a known inconsistency.
- Never invent a mechanism to fill a gap. If the data do not show *why*, state the association and stop.
- Never cite a paper for a claim it does not make (TALLRec is not “LLM descriptions beat TF-IDF”; UniSRec is not that comparison either).
- Copy BibTeX from Crossref / the venue PDF / Google Scholar. Do not reconstruct titles, years, pages, or author given names from memory.
- Do not leave lab-process language in the paper (`paused to finish`, `initially evaluated without`, `earlier catalog-fit run`, `min_train_inter=5`). Write the final protocol in prose.

## Consistency first (do not optimize away the paper)

De-AI and hedge trims must stay consistent with what the draft already committed to. Do not overturn prior **data**, **hypotheses**, or **conclusions** just to make a paragraph shorter or cleaner.

- Numbers, table bolding, seed lists, protocol cutoffs, and in-text citations of those numbers stay as already approved unless the user asked to fix a known inconsistency.
- A locked hypothesis stays a hypothesis (`may`, `plausible account`, `empirical observation`). Do not upgrade it to a finding, and do not delete it and replace it with a different story.
- A locked conclusion (Last vs Random changes the winner on the same checkpoints; Early ranking is not preserved on Beauty; redundancy already in raw Qwen outputs under our prompt design) may be restated once, not reversed or narrowed into a different claim.
- If abstract / contribution bullets / RQ lead / results / discussion / conclusion disagree after an edit, that is a bug. Align wording; do not pick a new conclusion because it sounds better.
- If a proposed trim would drop a fact the introduction still asserts, stop and flag. Do not silently drop Beauty-only, three-dataset, or GTS-Last/Random scope.
- Style edits are not license to reopen 0822, change a table, or invent a mechanism that “fits the narrative better.”

## Two kinds of hedge (do not confuse)

**Keep** a single, evidence-matched hedge. The advisor asked for these:

- Mechanism without a direct measurement → one of `one plausible hypothesis is…` / `this pattern is consistent with…` / `a plausible explanation is…`. Then stop.
- RQ3 locus → `redundancy is already present in the raw Qwen outputs before SVD/whitening under our prompt design`. Not `originates in the Qwen encoder itself`.
- RQ3 geometry → `the observed geometry is consistent with insufficient view complementarity`. Not `views fail the complementarity condition` or `exactly predicted by`.
- Last vs Random solid claim → same checkpoints, change target selection, change winner. Do **not** claim Last prefers niche/trending items unless there is a statistic.
- Figure 1 → point-estimate ordering; gaps inside 1 std use `\approx`. Do not draw a deterministic champion.
- Early window → first 60% train, next 10% valid, next 10% test; final 20% unused. The evaluation window shifts earlier; valid/test horizon lengths stay fixed.
- Train/test boundary → `interactions` are separated by a global cutoff. Not `items are temporally disjoint` (warm targets sit in the train catalog).
- Variant names → TF, TF+LLM, TF+MV. Not three independent families. UniSRec is an adapted backbone.

**Remove** the stacked template the advisor named:

> 结论 → 限定 → 再降调

That is: a finding, then a hedge, then a second hedge that restates “we do not claim / only an association / the solid claim is”. LLM polish produces this stack several times in a row. One hedge is scientific; two in the same paragraph is the AI tell.

Also remove self-contradictory mechanism sentences (hypothesis in one direction, number in the other). Prefer the advisor’s RQ2 hedge (0823 dropped the *Despite catalog shift* prefix; see §RQ2):

> TF-IDF exhibits the smallest Main-to-Early drop among the three text variants; we treat this as an empirical robustness observation rather than evidence for a causal mechanism.

## How to trim a stacked paragraph

1. Keep the measured fact and at most one hedge.
2. Delete the second meta-sentence (`we treat that only as an association`, `the solid claim is`, `we read this as`, `we emphasize association rather than causation`) if the first sentence already limited the claim.
3. If the paragraph title already says `A plausible account`, do not open the body with `A plausible explanation is`.
4. Do not add new caveats while trimming. Do not promote a hypothesis into a finding.

**Bad (three layers)**

> One plausible hypothesis is that Last over-represents recently popular or niche items while Random draws a broader mix, but we treat that only as an association: the solid claim is that changing target selection on the same checkpoints changes the winner.

**Good (one hedge + the claim)**

> Last may over-represent recently popular or niche items while Random draws a broader mix; regardless of the cause, our claim is only that changing target selection on the same checkpoints changes the winner.

**Bad (title + body repeat)**

> **A plausible account of the LOO gap.** … A plausible explanation is that this overlap disproportionately benefits…

**Good**

> **A plausible account of the LOO gap.** … This overlap may disproportionately benefit…

**Bad (read-as sermon)**

> …the margins are within roughly one seed-level std, so we read this as *no loss from dropping three views* plus a consistent small edge, not as a large additive gain.

**Good**

> …with margins within one seed-level std---no loss from dropping three views, not a large additive gain.

## Other tells to strip (only if they appear)

- Unsourced crowd claims: `This hypothesis is widely held`, `a consideration that is seldom examined`.
- Sermon close: `Our evidence suggests the community should…`.
- Template openers: `Three findings emerge` is locked in the abstract (advisor draft). Do not clone it again in the conclusion if a shorter lead works.
- Mid-sentence `\textbf{chronologically last}` / `uniformly random` / `warm-target` — keep bold on paragraph leads and table/figure callouts, not mid-prose.
- Code tokens in prose (`min_train_inter=5`) → `users with at least five training interactions`.
- Winner language in figures/captions when the gap is inside 1 std.

Do **not** strip ordinary academic words (`however`, `consistent with`, `we cannot isolate`) when they carry a real limitation. The problem is repetition of the same overclaim-defense, not caution itself.

## RQ1 results prose (do not restack)

The Last-vs-Random claim is already locked in the Why paragraph. Do not say it again in Implications, the figure caption, or a third “protocol artifact / superiority is unreliable” closer.

- Table caption, figure caption, and body bullets must not each reprint the same per-protocol ranking. Figure 1: *point estimates* + $\approx$ for gaps inside 1 std + pointer to Table 1. No Toys walkthrough and no *not a deterministic winner* sermon.
- A paragraph titled *A plausible account* may keep one *may*. Do not convert it into a mechanism (`this advantage vanishes`, `the model memorizes`). Under GTS, write that the LOO ranking is no longer visible.
- Do not write *all text margins often within one std* unless every named pair is actually inside 1 std. Beauty/Grocery Last TF vs TF+MV are not. Point to the per-seed table for seed-dependence only.
- Do not restack *same checkpoints change the winner* in Implications after the Why paragraph already has the advisor sentence. Leave Implications for a new fact (e.g., +13–100% vs ID).
- No *Claims of … superiority are therefore unreliable* sermon. One Amazon-family study does not support that.
- Prose protocol filters: `users with at least five training interactions`, not `min-5` / `min_train_inter=5`.
- `leads` / `highest`, not `dominates`, when the ratio is $\sim$1.8$\times$.
- Random averaging: *3 target-sampling seeds*, not *sample-seed* (that collides with training seeds and with sample-std).

## RQ2 results prose (do not restack)

Keep the advisor hedge on the TF-IDF drop:

> we treat this as an empirical robustness observation rather than evidence for a causal mechanism.

Do not add a second *do not explain / not causal* sentence after it. Table 5’s caption already limits those numbers.

- Table~\ref{tab:early} is Beauty only. Say *on Beauty* in the body; do not write as if Toys/Grocery Early were run.
- Tables~\ref{tab:shift}--\ref{tab:bucket} are Main GTS **valid→test** catalog context, not Main→Early. Do not write *Despite catalog shift* or *co-occur with the Main→Early drop* as if those tables measured the Early experiment. Unseen 31.9% (Table 5) and 17.8% (Table 6) are different definitions; do not treat them as one “catalog shift” number.
- Do not claim Early *isolates* training-window size from the test period. Horizon **lengths** stay 10%/10%; the test **slice** still moves earlier. Use the locked Early wording: 60/10/10, final 20% unused, window shifted earlier.
- No standalone *No single text variant dominates once the window is varied* after the ranking sentence already said the Main order is not preserved.
- `dominates` in RQ2 means “wins both windows”; prefer *leads both windows* if the sentence is kept at all.

## RQ3 results prose (do not restack)

Keep the locked locus and geometry hedges. Do not upgrade them, and do not hang extra caveats on the wrong citation or table.

- Do not write *all LLM view pairs $\ge$ 0.82 / exceed 0.82*. Toys Function--Style is $.818$. The range $0.82$--$0.94$ is fine if that cell is named; Abstract rounding may stay.
- Cite Kornblith only for Gram-matrix / CKA-as-similarity. Do not cite it for *cannot prove task-irrelevance* (that is why we add LOVO, not a Kornblith result).
- Do not cite Table~\ref{tab:cost} for projection width or $4\times256$. Cost is recommendation-stage params. Source-redundancy points at Table~\ref{tab:cka-prepost} / raw 3584-d. We did not run a $4\times256$ ablation.
- Concat capacity: *quadruples nominal width but only about doubles effective rank ($64\to136$)*, not *adds $\sim$50\% new capacity*.
- Published audit stages: raw 3584-d, 64-d pre-whiten, 64-d post-whiten. Not *256-d SVD-projected* as a main-text stage.
- Keep locked: *already present in the raw Qwen outputs before SVD/whitening under our prompt design*; *consistent with insufficient view complementarity*. Do not write *originates in the encoder* or *fail the complementarity condition*.

## Discussion prose (do not restack)

Section 7 explains RQ3; it must not reprint RQ1’s Last-vs-Random sermon or upgrade CKA into usefulness.

- TF-IDF paragraph: title already asks *why*. Do not open with a second *Why can TF-IDF…*. State the Random fact once, then First / Second. Do not repeat *geometric diversity per dimension* in the closer. Do not write *each dimension contributes distinct information* (ER/dim ≈0.98 already says high utilization).
- Hybrid: keep *no loss from dropping three views, not a large additive gain* and *CKA licenses only geometric dissimilarity*. Do not add *the lexical–semantic pair carries the useful signal* after that. Write *3 target-sampling seeds*, not *sampling seeds*.
- Implications: Table~\ref{tab:early} is Beauty only. Do not add a fifth *do not overclaim a single best / rankings reverse* bullet; item 1 already says report GTS with target mode.
- Warm-last exclusion is 30–38% of eligible last targets (Beauty 30.6, Grocery 29.6, Toys 38.3), not 28–35%. Unseen among all post-cutoff targets stays 30–41% (Table 5).
- Cost table: *limited multi-view gains are not a recommendation-stage capacity issue*. Not *multi-view failure*.

## Conclusion prose (do not restack)

Keep the positioning sentence and the one closer hedge. Do not reprint RQ1–RQ3 sermons or upgrade hybrid into a budget prescription.

- Positioning: `empirical investigation rather than a model-proposal paper`. Leave it.
- Item 1 may recap Table 1 rankings once, including UniSRec on Grocery until the cutoff-fit CSVs replace that cell. Keep `driven entirely by target selection, not by model changes` here; do not say it again in Future or Ethical.
- Item 2: Table~\ref{tab:early} is Beauty only. Use locked Early wording (60/10/10, final 20\% unused). Do not add *Text helps, but which text…* after the ranking sentence already said the Main order is not preserved.
- Item 3: hybrid *matches or slightly exceeds* \tfmv. Stop there. Do not add *the prompt-view budget is better spent…*.
- Closer: keep `not a free upgrade`. Drop the unsourced *LLM upgrade problem / assumed to monotonically improve* lead; Implications already lists the practical default.
- Future (1): do not restate the hybrid scores. Future (2) keeps *under our prompt design*; do not upgrade to encoder-origin or a general transformer property.
- Ethical: one PII sentence + one bias concern + warm-target diagnostics + no human subjects. Do not restack *object of study rather than a recommended default*.

## Math and numbers

- Effective rank of a sample matrix is bounded by `min(n, d)`. With 3000 items and 14336-d raw features, do not report `1567/14336 ≈ 10.9%` or `~90% collapse`. Keep raw CKA ≈ 0.96; 256-d ER 53–55% is valid because `n > d`.
- If two cells round to the same 4-decimal display, add one digit on that row. Do not invent a story that MV features were dropped.
- After any table edit, grep the prose for the old number. Abstract, RQ bullets, figure captions, conclusion, and appendix must match the table.

## Edit workflow

1. Work on `0823` unless the user names `0822`.
2. List the sentences you will change (old → new). Wait if the user asked for a plan first.
3. Apply only those sentences. Mirror the Chinese file the same turn.
4. Do not “improve” adjacent advisor-locked protocol/claim sentences. Do not overturn earlier numbers, hypotheses, or conclusions for style.
5. Compile if the user will look at the PDF; report overfull boxes, do not reflow the whole page to hide one line.
6. **Log every 0823 change** the same turn in `paper_recsys/experiment_status_20260822.md` §变更流水: location, old → new, reason. Do not skip “small” wording edits.

7. **Sync overleaf_deploy every time you edit the paper.** After editing `main_wsdm2027_v3_0823.tex`, `main_wsdm2027_v3_zh_0823.tex`, or `refs.bib`, always copy them to `paper_recsys/overleaf_deploy/`:
   - `main_wsdm2027_v3_0823.tex` → `overleaf_deploy/main_wsdm2027_v3_0823.tex` AND `overleaf_deploy/main.tex`
   - `main_wsdm2027_v3_zh_0823.tex` → `overleaf_deploy/main_wsdm2027_v3_zh_0823.tex`
   - `refs.bib` → `overleaf_deploy/refs.bib`
   Do not forget this step. Overleaf compiles `main.tex` by default.
