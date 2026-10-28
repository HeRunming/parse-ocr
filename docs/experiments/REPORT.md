# v3 bench: prose-claim mining and normalized statements

Date: 2026-10-12. Model: `gpt-6-sol` (OpenAI-compatible gateway). Code: `src/parse_ocr/prose_claims.py`.

## What changed

There is a new stage, `prose_claims.py` (mine → sweep → normalize → verify → repair → merge). It can be called as `direct_pipeline.py prose-claims`, or switched on inside `run` with `"prose_claims": {"enabled": true}` in `run.json`. It does not modify `environments.json`, which stays as the verbatim evidence layer. It writes:

- `dataset/structured_statements.json` is the new primary view. It has one entry per atomic mathematical unit. Each entry carries `source_statement` (verbatim), `normalized_statement` (standalone, model-written, verified), `origin` (labeled_environment / unlabeled_environment / *_atomized / prose_derived), `kind`, pages, and dependencies.
- `dataset/prose_claims.json`, `prose_claims_rejected.json`, `prose_claims_informal.json` and `prose_claims_summary.json`.

How it works:

1. **mine:** page-image windows (2 pages, overlap 1) receive the already-extracted objects. The model returns content stated in prose (inline definitions, asserted facts, recalled results, claims, conjectures). Each item has a verbatim `source_excerpt`, a `normalized_statement` and an `assertion_status`.
2. **sweep:** a second pass is shown everything found so far and targets the typical misses: one-sentence consequences, asymptotic conclusions, named results mentioned in prose, and results attributed to others.
3. **normalize:** every existing record is rewritten into standalone form and atomized into parts. One record often holds a definition plus a property, or a theorem plus a remark.
4. **verify:** an independent page-image auditor checks only faithfulness (added or dropped hypotheses, quantifiers, formulas, invented content). Unfaithful items are excluded, not patched silently.
5. **repair:** one bounded round re-writes the unfaithful items using the auditor's issues. The result gets a new uid and is verified again from scratch.
6. **merge:** grounding against the PDF text layer, deduplication against the atomized parts, and a separate tier for heuristic or physical-intuition claims.

## Results (gold = assistant-built annotation of every labeled and prose unit; nesterov uses the colleague's reviewed annotation)

R = one-to-one recall, P = one-to-one precision.

| case | system | gold | pred | R all | R labeled | R unlabeled | P |
|---|---|---|---|---|---|---|---|
| Vershynin 8p | baseline direct | 58 | 35 | 0.59 | 0.87 | 0.49 | 0.97 |
| | **v3.3** | 58 | 61 | **0.88** | 1.00 | **0.84** | 0.84 |
| | mathdoc-parser | 58 | 22 | 0.38 | 1.00 | 0.16 | 1.00 |
| Perelman 7p | baseline | 64 | 19 | 0.30 | 0.60 | 0.24 | 1.00 |
| | **v3.3** | 64 | 63 | **0.83** | 0.70 | **0.85** | 0.84 |
| | mathdoc-parser | 64 | 33 | 0.09 | 0.20 | 0.07 | 0.18 |
| String topology 7p | baseline | 46 | 29 | 0.59 | n/a | 0.59 | 0.93 |
| | **v3.3** | 46 | 63 | **0.98** | n/a | **0.98** | 0.71 |
| | mathdoc-parser | 46 | 280 | 0.67 | n/a | 0.67 | 0.11 |
| Nesterov 8p (colleague gold) | baseline | 18 | 18 | 0.83 | 1.00 | 0.79 | 0.83 |
| | **v3.3** | 18 | 33 | **0.94** | 1.00 | 0.93 | 0.52 |
| | mathdoc-parser | 18 | 18 | 0.50 | 0.25 | 0.57 | 0.50 |

Normalization fidelity after one repair round: 132 of 137 parts are faithful or have only a minor issue.

Cost: the prose layer uses 145k–250k tokens per 8-page case, about 0.3–1× the baseline.

## Caveats

- The gold sets were built by model sub-agents reading the page images, not by human experts. Granularity is subjective, so treat the numbers as relative. A human pass is still needed.
- The precision drop is mostly a granularity disagreement, not hallucination. Nesterov's colleague gold is coarse: 18 units, with prose mostly excluded. Most "unmatched" predictions there are real prose claims ("These bounds are valid for all reasonable iterative schemes").
- Perelman misses on labeled items: the gold counts the equation pairs (1.1)–(1.4) as labeled units. The baseline merges them into the surrounding Proposition 1.2 / Remark 1.3 text.
- Remaining recall misses are informal roadmap claims ("In §8 it is applied to obtain …").
- `must_have` (key-fragment containment in the normalized statement) fell from about 0.92 to about 0.86. Normalization rewrites the notation (⌈ ⌉ vs \lceil), and the string metric under-counts that. The auditor's verdicts are the better fidelity signal.

## Ablation status (2026-10-12, interrupted)

Harness: `ablation_runner.py run|score VARIANT...`. Each variant has its own checkpoint dir (`output/<cache_dir>/`) and output suffix (`structured_statements__<variant>.json`). Boundary quality is scored as word-span IoU against gold, computed on the PDF text layer (`text_spans.py`). A new case, `hdp_ex` (HDP pp.99–106, 45 exercises, 21 of them with sub-parts), has gold at two granularities (whole / subpart).

- **B0 (deterministic text-layer spans, same model outputs as A):** results are neutral. F1 is within ±0.01 on all cases. Its only effect is span-based dedup, which is fine.
- **B (model anchors + spans), incomplete:** the endpoint returned 429 and then "no available channel" partway through. 8–16 sweep/verify checkpoints per case failed, so B is not yet comparable to A. In what did finish:
  - Model anchors located 96/98 claims on cases that have a text layer. Nesterov has none and falls back as designed.
  - Vershynin: F1 went from 0.86 to 0.89 and mean span IoU from 0.81 to 0.85.
  - Perelman: recall went from 0.83 to 0.66, but 5 of its sweep windows failed, so this drop is confounded.
- **Exercise granularity (hdp_ex, A vs baseline):** the baseline has 1 record per exercise. Scored against subpart gold, its recall is 0.46. A atomizes sub-parts on its own initiative and reaches recall 0.78 / P 0.89 at subpart granularity. Explicit sub-part splitting (C_subparts) has not run yet.
- **Not run:** C_record, C_subparts, D1–D3. All are wired in `ablation_runner.py`.
