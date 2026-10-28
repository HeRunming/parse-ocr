# Direct visual route

## Contents

1. Configuration
2. Inventory
3. Extraction and reconciliation
4. Independent coverage audit
5. Gap and target retries
6. Operational recovery
7. Relationship stage

## 1. Configuration

`run.json` uses `route: direct` and a `direct` block:

```json
{
  "inventory_window_pages": 10,
  "inventory_overlap_pages": 2,
  "inventory_zoom": 1.25,
  "inventory_workers": 32,
  "inventory_split_window_ids": [],
  "inventory_window_page_overrides": {},
  "extraction_window_pages": 5,
  "extraction_overlap_pages": 1,
  "extraction_zoom": 2.4,
  "extraction_workers": 64,
  "source_grounding_enabled": true,
  "source_verification_enabled": true,
  "source_verification_zoom": 4.0,
  "source_verification_workers": 8,
  "source_verification_max_tokens": 8000,
  "extraction_split_window_ids": [],
  "coverage_audit_enabled": true,
  "coverage_audit_window_pages": 4,
  "coverage_audit_overlap_pages": 1,
  "coverage_audit_zoom": 2.0,
  "coverage_audit_workers": 32,
  "coverage_audit_split_window_ids": [],
  "coverage_audit_page_ranges": [],
  "extraction_window_page_overrides": {},
  "extraction_extra_windows": [],
  "retry_page_padding": 1,
  "retry_detected_span_gaps": true,
  "retry_max_pages": 8,
  "retry_zoom": 2.8,
  "retry_workers": 32,
  "retry_group_targets_by_pages": false,
  "retry_targets_per_request": 12,
  "retry_round_tag": "",
  "manual_retry_ids": [],
  "retry_focus": {}
},
"relationships": {
  "enabled": true,
  "implicit_dependencies": true,
  "workers": 32,
  "request_attempts": 4,
  "error_retry_rounds": 2,
  "error_retry_min_workers": 4,
  "source_batch_size": 10,
  "same_section_lookback": 30,
  "same_section_lookahead": 30,
  "cross_section_candidate_limit": 12,
  "fail_on_request_errors": true
}
```

Use lower concurrency when the endpoint throttles or returns transport errors. Split only the failing
large window IDs before rerunning the entire stage. Increase extraction
zoom for small old type, dense subscripts, or degraded scans. Do not increase zoom merely to fix
object recall; recall is controlled by inventory and window coverage.

## 2. Inventory

Inventory uses low-resolution overlapping page windows and returns only kinds, labels, short cues,
page spans, and boundary status. It must not reproduce formulas. Unnumbered explicit objects receive
stable page-derived labels such as `p40.d2` or `p41.x1`.

Inventory has two levels. `record_type=section` is a structural container used for navigation and
page scope; `record_type=object` is an atomic mathematical unit eligible for output. A survey/problem
article containing multiple results must be a section, not one giant exercise. Configure its stable
heading namespace with `structural_label_regex`.

Check `direct/inventory/summary.json` for window errors, kind counts, and continuity candidates.
Overlapping inventory votes improve robustness but do not prove completeness.

## 3. Extraction and reconciliation

High-resolution windows receive inventory targets and may discover extras. Reconciliation groups by
`(record_type, kind, label)`, prefers complete/longer candidates, and records materially different
overlap transcriptions as conflicts instead of silently averaging them.

The extractor also returns blocks in strict visual order plus a short opening anchor and per-page ordinal.
These are fallbacks, not object identity. During `build`, `source_grounding.py` aligns the VLM statement to
native PDF text blocks and records page, bbox, block interval, and block-token interval. The native text
layer controls geometry only; it never replaces VLM mathematics.

Canonicalization is provenance-aware:

- formatting-equivalent text is merged only on overlapping source spans;
- the canonical object retains all `candidate_kinds` and `merged_source_ids`;
- identical wording at distinct spans is retained;
- a lowercase/connective fragment is merged only when it is the adjacent part of the same source sentence
  and has the same kind; otherwise it receives a context link or stays in the audit queue;
- a displayed formula that grammatically completes “is”, “given by”, “specimen is”, or a similarly unfinished
  statement is moved from misclassified support back into `statement` without changing its content.

Final order is the source-position key. Structural membership is assigned to the latest preceding heading
anchor, including when two headings share a physical page. Never sort by kind or synthetic label.

Inspect:

- `direct/extraction/windows/`: raw checkpointed responses;
- `direct/reconciliation/merged.json`: selected candidates;
- `direct/reconciliation/report.json`: missing, incomplete, conflicts, request errors, and usage.
- `dataset/source_grounding_report.json`: alignment confidence, section coverage, merge events, context links,
  ID remaps, and low-confidence visual-audit targets.
- `dataset/source_ledger.json`: state of every detected source object/container and later discovery.
- `dataset/formula_conflicts.json`: same-span semantic-critical conflicts and verification evidence.
- `dataset/source_span_collisions.json`: exact/high-overlap canonical-role decisions.
- `dataset/context_only_entries.json` and `dataset/superseded_entries.json`: preserved noncanonical layers.
- `dataset/id_aliases.json`: compatibility ID, graph ID, canonical source ID, physical page, and span mapping.

After the first build, `run` verifies unresolved hard formula conflicts and high-precision source-span
canonical-role collisions, then rebuilds before relationship extraction. A standalone
`direct_pipeline.py verify` does the same. Use a padded native-resolution crop when alignment is sound; fall
back to the full physical page when it is weak. For cross-page candidates send every page in `source_spans[]`.
Successful checkpoints are reused; low-confidence canonical-role results are never auto-applied.
The same stage verifies every `transcription=low` entry. It corrects only inside the candidate boundary and
rejects outputs that grow disproportionately or newly contain a neighboring same-page object. A low tight-crop
reading retries once on complete physical pages; persistent clipping remains in
`dataset/transcription_review_queue.json`.

For formula conflicts, first diff candidates and send the verifier an explicit critical-token checklist. For
tables, matrices, factor lists, and other structured long-number blocks, use a complete row/cell checklist even
when most cells agree. Require two readings of long integers, a blind observer, and a deterministic
statement-to-observation postcheck. A disagreement triggers at most one tie-break observer; no majority means
unresolved. To retry one case without rerunning the queue:

```bash
python -m parse_ocr.verify_source_conflicts --config PROJECT_DIR/run.json --candidate-id OBJECT_ID --force-stage
```

If bounded observers still fail, record a narrowly scoped human PDF-visual decision in
`source_verified_conflict_overrides`; do not silently patch the canonical JSON. Preserve the rejected candidate
in `dataset/superseded_entries.json`. Single-token formula conflicts already resolved by an independent
canonical-span crop audit may share that decision only under the exact-pair/delta postcheck; whole-table
conflicts cannot use this shortcut.

## 4. Independent coverage audit

After initial extraction, a separate high-resolution pass receives the page images plus a compact list of
already covered objects. It scans continuation pages for omitted definitions, named/unnamed results,
examples, constructions, conjectures, formulas with a mathematical assertion, and explicit exercises.
It returns only short visible cues; confirmed candidates are added to inventory and transcribed by the
ordinary targeted retry. This pass must remain independent of the first inventory response.

Inspect `direct/coverage_audit/summary.json`. “All inventory targets matched” measures internal
consistency only; it is not a source-completeness claim. A clean run requires the independent audit to run
without request errors, but even then visually inspect dense continuation pages and section-level outliers.
Use `coverage_audit_page_ranges` only for a focused diagnostic or an explicitly scoped rerun; leave it empty
for the completion run so every included page is audited.

`dataset/page_coverage_ledger.json` must contain one row for every included physical page and explicitly list
pages with audit errors or no detected mathematical object. `dataset/benchmark_metrics.json` reports measured
conflict/duplicate audit quantities, while leaving gold recall and residual hard-transcription error unset
unless a human gold manifest actually exists.

## 5. Gap and target retries

The retry stage scans short numerical gaps only when neighboring labels share the same hierarchy and
occur within a small page distance. The gap prompt asks for visible source evidence and must not infer
an object from numbering alone. Newly confirmed inventory objects are extracted at higher zoom.

The same targeted extraction mechanism handles inventory misses, cross-window incomplete objects,
conflicting duplicates, and short inline fragments whose pronouns or connective words require a
visible antecedent sentence/formula. Remaining failures stay visible in `direct/retry/summary.json`.
Put visually audited high-risk IDs in `manual_retry_ids` when a specific formula needs a higher-zoom
adjudication even though deterministic checks did not flag it.

After a build materializes the independent candidate-span ledger, a subsequent retry automatically adds every
unresolved detector span as a high-resolution target when `retry_detected_span_gaps=true`. Rebuild and repeat
until each candidate is consumed or explicitly ignored. This closes recall over the detector universe without
claiming that the detector universe is complete source truth.

For books with hundreds of omissions, set `retry_group_targets_by_pages=true` and cap each request with
`retry_targets_per_request`. A retry checkpoint name must hash the sorted target IDs as well as its page
range. Set a new `retry_round_tag` only when the prompt/focus changes materially; page-only cache names
can silently substitute an earlier target set and are forbidden.

After model retries are exhausted, use profile overrides only for page-verified facts:

- `record_overrides`: correct a specific transcription or restore the visible antecedent;
- `section_id_overrides`: correct a same-page structural assignment;
- `source_verified_additions`: add a visibly present object missed by inventory and audit;
- `drop_object_ids`: remove a visually confirmed false positive or redundant alias.

Every manual addition must retain source pages and identify visual PDF inspection as its source method.
Do not use these fields to write a mathematically plausible replacement from memory.

## 6. Operational recovery

All remote window results are individual JSON checkpoints. Re-run a failed stage without
`--force-stage` to execute only missing files. Use `--force-stage` after changing a prompt/profile or
when deliberately replacing successful checkpoints. Never delete the entire output directory to
retry one stage.

If only a few dense windows repeatedly stall, list their IDs in `inventory_split_window_ids`,
`extraction_split_window_ids`, or `coverage_audit_split_window_ids` as appropriate; narrow
their core pages with `inventory_window_page_overrides` or `extraction_window_page_overrides`, and add
any uncovered single-page calls through `extraction_extra_windows`. Use `retry_focus` only for
visually audited targets that need an explicit source boundary; changing one target's focus changes
only that target's checkpoint hash.

Validation may downgrade historical request errors to warnings only when successful overlapping windows
or retries cover the same source pages and the current reconciliation has zero missing/incomplete targets.
Keep the failed checkpoint in the report. Apply the same exception to a coverage-audit error only after a
visual page-hole review; never use a relaxed policy merely to obtain `ok: true`.

## 7. Relationship stage

After `build`, the text-only `relationships` stage preserves explicit source references and then
extracts conservative direct prerequisites. Read [relationships.md](relationships.md) before changing
candidate scope or relation types. Run it independently after relationship prompt/config changes:

```bash
python -m parse_ocr.direct_pipeline relationships --config PROJECT_DIR/run.json --force-stage
```

Ordinary resume omits `--force-stage` and reuses prompt-hashed successful checkpoints. When only the
deterministic relationship validator changes, it may also revalidate a compatible cached raw response with
the same source/candidate IDs. Relationship output is transactional: request failure must not replace an
existing valid graph with a partial or empty one. A clean completion
requires `dataset/dependency_graph.json`, zero dependency cycles/dangling targets, and no unhandled
relationship request errors.
Cached request errors are retried automatically. Error-only retry rounds progressively lower concurrency so
gateway failures do not force successful relationship tasks to run again.
