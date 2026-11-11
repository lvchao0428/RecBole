# AI-tone inventory

Line numbers drift; find sentences by quote. Work on `0823`, not `0822`.

## Done on 0823 (pass 1 + pass 2)

| Location | Change |
|----------|--------|
| RQ1 LOO title + body | Body is `This overlap may…` (title already hedges) |
| RQ1 Last vs Random | `Last may…; regardless of the cause, our claim is only that…` |
| Hybrid | `margins within one seed-level std---no loss…, not a large additive gain` |
| Intro | Dropped `This hypothesis is widely held` and `seldom examined` |
| RQ1 LOO lead | Dropped `not merely an additive shift` |
| RQ1 Implications | Dropped `narrow claim … not a broad one`; 0823 later dropped the third Last-vs-Random restack + *superiority unreliable* |
| RQ1 Fig.1 caption | Point estimates + $\approx$ + see Table 1; no Toys ranking walkthrough |
| RQ1 LOO closer | *advantage vanishes* → LOO ranking no longer visible |
| RQ1 Last bullet | *often within one std* → seed-dependent winners (Table 3) |
| RQ1 UniSRec | `min-5` → five training interactions; `dominates` → `leads` |
| Practitioner 4 | Dropped `Such claims should be verified… rather than assumed` |
| Conclusion | Dropped `Three conclusions follow` and `community should` closer |
| Ethical | Shortened; no second hedge stack |
| UniSRec filter | `inflates` → `raises` / 升到 |
| Future work (2) | No longer `confirms the redundancy locus in Qwen` |
| RQ2 isolate / Beauty | Dropped *isolate training-window size*; body says Table 4 is Beauty |
| RQ2 dominates one-liner | Deleted *No single text variant dominates…* |
| RQ2 TF-IDF stack | Dropped *Despite catalog shift* and the *co-occur… do not explain* third hedge |
| RQ3 tab:cka caption + (1) | *all pairs exceed / $\ge$ 0.82* → range $0.82$--$0.94$, name Toys Function--Style $.818$ |
| RQ3 lead | Kornblith only on Gram-matrix similarity; dropped *cannot prove task-irrelevance* as a [11] claim |
| RQ3 4×256 | No *not justified* + tab:cost; *unlikely to remove overlap*, cite tab:cka-prepost / 3584-d |
| Discussion concat | *adds ~50% new capacity* → *doubles ER ($64\to136$)* |
| Limitations locus stages | *256-d SVD-projected* → *64-d SVD-projected pre-whitening* |
| §7 TF-IDF para | Dropped second Why + *each dimension distinct* + repeated geometric-diversity closer |
| §7 hybrid | Dropped *pair carries the useful signal*; *sampling seeds* → target-sampling |
| §7 Implications | Early *on Beauty*; dropped *do not overclaim a single best* bullet |
| §7 Limitations / cost | Warm-last 28–35% → 30–38%; *multi-view failure* → limited gains / not capacity |
| §8 item 2 | Early *on Beauty* + 60/10/10; dropped *Text helps, but which text…* |
| §8 item 3 / closer | Dropped *budget is better spent* and *LLM upgrade / monotonically* lead |
| §8 Future (1) | Dropped hybrid-score restatement |
| Ethical | Dropped *object of study rather than a recommended default* |

## Keep (advisor-locked, one hedge each)

- Abstract `Three findings emerge` (0816 draft)
- RQ2: `we treat this as an empirical robustness observation rather than evidence for a causal mechanism`
- RQ3: `already present in the raw Qwen outputs … under our prompt design` + `We cannot separate encoder…`
- RQ3: `consistent with insufficient view complementarity` / `consistent with the redundancy analysis`
- Conclusion positioning: `empirical investigation rather than a model-proposal paper`
- Table 5 caption: `associations with catalog shift, not causal claims`
- Limitations “Redundancy locus”

## Remaining mild tells (do not auto-trim)

| Location | Why it can stay |
|----------|-----------------|
| Intro `This hypothesis rests on fragile evaluation foundations` | Needed bridge into LOO/GTS; only one leftover `hypothesis` after dropping `widely held` |
| Intro `specifically decisive` | Emphasis, not a hedge stack |
| Conclusion `not a free upgrade` | One hedge; sits alone after dropping the community-should sentence |
| `Five questions remain open` | Functional list lead |

## Already fixed earlier (do not reopen)

Early 60/10/10; items→interactions; RQ2 contradiction; raw 10.9%/90% collapse; mid-prose bold; `min_train_inter=5`; process talk; TALLRec/UniSRec mis-cite; TF/TF+LLM/TF+MV names; Last vs Random vs LOO split; `genuine complementarity`; Beauty GTS-Rand 5 decimals.
