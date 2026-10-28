# Ablation results: anchors, granularity, claim tiers

Date: 2026-10-12. Model: gpt-6-sol. Raw table: `eval/ablation_table.txt`; rows: `eval/ablation_results.json`. Reproduce: `python3 ablation_runner.py score <variants>`. Rerunning `run` reuses checkpoints.

## Setup

**Cases.** Five cases, 245 gold units in total (all gold was built by sub-agents, not humans):
- vershynin, perelman, string_topology, nesterov (the colleague's gold);
- hdp_ex: HDP pp.99–106, 45 exercises, 21 of them with sub-parts. Its gold has two granularities, whole (69 units) and subpart (97 units).

**Metrics.**
- R / P: one-to-one recall and precision.
- iou: word-span IoU between a matched prediction and its gold unit, measured on the PDF text layer. This is the boundary-quality signal.
- kind: kind-family accuracy.

**Noise.** A_rep repeats A with identical settings and a fresh cache. Run-to-run spread is large: R differs by up to 0.11 and F1 by up to 0.13 (nesterov). Differences below roughly ±0.05 are not meaningful at this sample size.

## B: anchor-based source spans

| case | R (A / A_rep / B) | P (A / A_rep / B) | iou (A / A_rep / B) | kind (A / A_rep / B) |
|---|---|---|---|---|
| vershynin | 0.88 / 0.86 / 0.90 | 0.84 / 0.89 / 0.88 | 0.81 / 0.83 / 0.84 | 0.96 / 0.96 / 0.96 |
| perelman | 0.83 / 0.75 / 0.70 | 0.84 / 0.81 / 0.85 | 0.83 / 0.81 / 0.84 | 0.81 / 0.85 / 0.87 |
| string_topology | 0.98 / 0.91 / 1.00 | 0.71 / 0.65 / 0.70 | 0.90 / 0.88 / 0.92 | 0.76 / 0.76 / 0.80 |
| nesterov (scan, no text layer) | 0.94 / 0.83 / 0.89 | 0.52 / 0.40 / 0.41 | – | 0.88 / 0.87 / 0.94 |

Means across the four cases, B minus the A average: R −0.02, P +0.03, iou +0.02, kind +0.04, F1 ±0.00.

**Conclusion: B helps only weakly.**
- Boundaries and kinds improve slightly, but only inside the noise band.
- Model anchors were located on 96 of 98 text-layer claims, so the mechanism itself works.
- Most of the deterministic value comes from span-based dedup, and B0 gets that at zero cost.
- The Perelman recall drop is mostly the miner returning fewer candidates this run (32 vs 39). It is not span logic: the lost units were present in the raw responses but did not survive.
- B uses 30–60% more tokens because of the anchor fields and the extra verify batches.

**Kept:** span grounding (`span_grounding`) as the default for PDFs with a text layer. Model anchors stay optional.

## C: granularity

### Exercise sub-parts (hdp_ex, scored at subpart granularity)

| variant | R | R labeled | P | F1 | iou | iou ≥ 0.8 |
|---|---|---|---|---|---|---|
| baseline (one record per exercise) | 0.46 | 0.53 | 1.00 | 0.63 | 0.57 | 0.20 |
| C_record (normalize, no atomizing) | 0.48 | 0.55 | 0.96 | 0.64 | 0.59 | 0.23 |
| A (free atomizing) | 0.78 | 0.86 | 0.89 | 0.83 | 0.73 | 0.42 |
| B | 0.78 | 0.88 | 0.93 | 0.85 | 0.76 | 0.47 |
| **C_subparts (explicit sub-part rule)** | **0.79** | **0.93** | **0.96** | **0.87** | **0.79** | **0.55** |

C_subparts is the clear winner for exercises.
- Labeled sub-part recall goes from 0.53 to 0.93.
- 55% of matched units have near-exact boundaries, against 20% for the baseline.
- Each sub-part statement carries the shared stem and the earlier-part hypotheses it uses. 79 of 80 parts were verified faithful.
- Sub-parts now get labels such as `3.13(b)`, taken deterministically from the printed marker.

### Prose atomization (C_record vs B on the other cases)

Turning atomization off (C_record) lowers recall on every case:
- vershynin −0.07
- perelman −0.12
- string_topology −0.04
- nesterov −0.06

It raises P by 0.03–0.06. F1 gets worse or stays flat, except string_topology, where it rises by 0.03.

**Kept:** atomize prose records, plus the explicit sub-part rule (`subpart_policy: split`). On non-exercise books C_subparts behaves like B: F1 within ±0.01 and lower token cost.

## D: claim tiers (one tier-labelled mining run, three acceptance filters)

| case | D1 explicit R / P | D2 +implicit R / P | D3 all R / P |
|---|---|---|---|
| vershynin | 0.76 / 0.90 | 0.88 / 0.90 | 0.90 / 0.90 |
| perelman | 0.45 / 0.88 | 0.64 / 0.80 | 0.66 / 0.76 |
| string_topology | 0.70 / 0.78 | 0.85 / 0.66 | 0.96 / 0.69 |
| nesterov | 0.94 / 0.53 | 0.94 / 0.45 | 0.94 / 0.42 |
| hdp_ex (subpart) | 0.73 / 0.93 | 0.74 / 0.91 | 0.74 / 0.91 |

**Conclusions:**
- D1 alone loses too much: R falls by 12–25 points on the prose-heavy cases. In these books most prose content is implicit.
- D3 has the best F1 on 3 of 5 cases. D2 never wins clearly.
- Adding the tier field changes the miner's behaviour: D3 recall on Perelman (0.66) is below A (0.83). With the field the model is more conservative.

**Recommendation:** keep the tier as a label on every record, not as an extraction filter. Downstream users can filter `claim_tier` as they need. The miner prompt should therefore not ask for tiers until the tier field is decoupled from the find step (for example, label tiers in the verify pass instead).

## Recommended default configuration

```json
"prose_claims": {
  "enabled": true,
  "span_grounding": true,
  "anchors": false,
  "subpart_policy": "split",
  "granularity": "atomize",
  "claim_tiers": false,
  "sweep": true
}
```

## Code changes in this round

- `python -m parse_ocr.text_spans`: anchor locator over the native text layer. It tolerates gaps where math tokens were stripped, joins hyphenated words, and computes word-interval IoU and containment.
- `python -m parse_ocr.prose_claims`:
  - per-variant `cache_dir` and `output_suffix`;
  - `anchors`, `span_grounding`, `granularity`, `subpart_policy`, `claim_tiers` / `allowed_tiers`;
  - span-based dedup;
  - sub-part labels;
  - a stable sweep cache key;
  - pruning of orphaned failed checkpoints.
- `python -m parse_ocr.common`: backoff that honours 429 rate limits.
- `evaluate.py`: whole/subpart granularity views and boundary IoU.
- `ablation_runner.py`: run/score harness. Merge-only variants reuse model outputs.
- Tests: 90 pass, 6 of them new for spans and sub-part labels.

## Open issues

- Run-to-run variance is about as large as most effects. Telling apart effects of ±0.03 needs either 2–3 repeats per variant or larger gold sets.
- hdp_ex has 11 missed prose units: remarks, notation and recalled facts between exercises. The miner under-reads exercise-dense pages.
- Gold is model-built. The next priority is a human pass on `gold/hdp_ex.json` and `gold/perelman.json`.
