---
name: adaptive-math-pdf-qa
description: Inspect mathematics textbook PDFs visually, adapt a per-book extraction profile, and extract definitions, notation, theorems, lemmas, propositions, corollaries, facts, conjectures, examples, exercises, remarks, algorithms, proofs, hints, solutions, and answers into source-faithful structured JSON. Use when building or auditing textbook-to-JSON/Q&A datasets. Default to the direct VLM route with structural and atomic inventory, high-resolution transcription, independent coverage audit, and targeted retries; use the retained MinerU route only when the user explicitly requests MinerU.
---

# Adaptive Math PDF Q&A

Keep one extraction engine and a small evidence-based profile per book. Default to direct visual
parsing. Do not invoke MinerU merely because a PDF lacks a text layer.

## Required workflow

1. Locate the PDF and choose a project/output directory outside this Skill.
2. Run `python -m parse_ocr.init_project PDF PROJECT_DIR`. This writes `route: direct` unless the user
   explicitly requests `--route mineru`.
3. Run `python -m parse_ocr.book_inspector`. Open every image listed in `inspection/manifest.json` and inspect
   title/front matter, representative body pages, theorem-heavy pages, examples/exercises,
   appendices, answer/hint sections, and front/back-matter glossaries of terminology,
   abbreviations, notation, or symbols. Native PDF text is secondary evidence.
4. Read [profile-schema.md](../docs/reference/profile-schema.md), then edit only `book_profile.json` to
   describe observed labels, unheaded objects, proof boundaries, distant support, figures, and
   high-risk glyphs. Apply the glossary gate below before excluding front/back matter. Preserve the
   all-environment taxonomy unless the user narrows it.
5. Run `python -m parse_ocr.preflight --config PROJECT_DIR/run.json --network`. A direct project requires
   only the configured vision-model credential. Never print or store secret values.
6. Read [direct-route.md](../docs/reference/direct-route.md), then run:

   ```bash
   python -m parse_ocr.direct_pipeline run --config PROJECT_DIR/run.json
   ```

   The pipeline separately extracts configured glossary pages, checkpoints inventory and extraction
   windows, audits numbering, retries only missing/incomplete/conflicting targets, builds the
   dataset, detects and crop-verifies semantic-critical formula conflicts, extracts source references and
   conservative document-local dependencies, and validates it.
   During `build`, ground every VLM transcription in a deterministic PDF source span. Use page/block/bbox/
   block-token position for source order, structural-section assignment, and duplicate identity; never sort
   same-page objects by model-assigned kind or label.
   The retry stage includes an independent visual coverage audit that compares
   continuation pages against already extracted objects. Failed large windows may be split, while targeted
   retries may be grouped by page; cache keys must include the exact target-ID set so a later retry round
   cannot reuse an unrelated response. Resume normally;
   use `--force-stage` only for the narrow stage that changed.
7. Inspect `dataset/quality_report.json`, `dataset/dependency_graph.json`,
   `dataset/source_grounding_report.json`, `dataset/visual_audit_queue.json`,
   `dataset/source_ledger.json`, `dataset/formula_conflicts.json`,
   `dataset/source_span_collisions.json`, `dataset/context_only_entries.json`,
   `dataset/superseded_entries.json`, `dataset/id_aliases.json`,
   `dataset/structured_verification_audit_queue.json`,
   `dataset/page_coverage_ledger.json`, `dataset/benchmark_metrics.json`,
   `direct/reconciliation/report.json`,
   `direct/coverage_audit/summary.json`, and `direct/retry/summary.json`. Visually sample every kind,
   all uncertain/conflicting items,
   formula-heavy records, page-window boundaries, distant support, and statement-length outliers.
8. When an independent TeX/XML/HTML export is available and the user wants maximum recall or better
   atomic boundaries, read [hybrid-route.md](../docs/reference/hybrid-route.md). Keep the direct dataset as
   PDF-grounded candidate stream G and treat the external export only as independent candidate stream T.
   Do not enable hybrid merely because the direct route has ordinary low-confidence items.

## Front/back-matter glossary gate

Before excluding a terminology, abbreviation, notation, or symbol table, decide whether every
extracted object remains understandable when read alone without that table. Treat the table as
required context if it defines book-specific terminology, non-universal conventions, overloaded
symbols, named sequences/functions, conjecture markers, or abbreviations needed to interpret a
statement. Do not assume a convention is universal merely because it is common in the subject.

If the table is unnecessary, record that conclusion in `inspection_notes`. If it is necessary:

- keep its physical pages in `glossary_page_ranges` even when they lie inside an excluded
  front/back-matter range;
- transcribe only printed meanings—never expand abbreviations or infer definitions from memory;
- save the complete table in `dataset/symbol_glossary.json` and attach only relevant entries as
  `glossary_context`; never rewrite or expand the source `statement` in place;
- prefer the table's printed problem/section applicability labels, then allow conservative literal
  statement matching only for distinctive terms or symbols;
- list overloaded or ambiguous forms in `glossary_content_match_disabled_symbols`, so a symbol such
  as `P(n)` is not assigned one article's meaning in another article;
- visually adjudicate conflicting glossary transcriptions with narrow `glossary_overrides`, keeping
  source page traceability and the match reason.

The `symbol` field may contain a printed term or abbreviation as well as mathematical notation; the
`meaning` field contains its source-faithful explanation.

## Direct-route rules

- Treat page images as authoritative. The inventory identifies objects and boundaries but never
  transcribes mathematics.
- Separate structural containers from atomic objects. A long problem/survey article containing several
  definitions, known results, examples, and questions is `record_type=section`, not one giant exercise.
  Do not emit both the section wrapper and atomic children by default.
- Use high-resolution overlapping windows for source-faithful text. Include inventory targets and
  allow the extractor to report additional visible objects.
- Compare inventory IDs with extracted IDs. Retry only missing source objects, incomplete
  cross-window records, materially conflicting duplicates, and visible labels confirmed by a
  continuity gap scan.
- After the first extraction, run the independent visual coverage audit over all included pages. It receives
  the already covered objects and returns only visibly supported omissions; add those candidates to inventory
  and transcribe them through targeted retry. Treat inventory reconciliation as internal consistency, not as
  proof that the source was fully covered.
- After reconciliation, ground objects against the native PDF text geometry without replacing any VLM
  mathematics from native OCR. Auto-merge presentation-equivalent transcriptions only when their source
  spans overlap; preserve `candidate_kinds` and `merged_source_ids`. Identical wording at a different span
  remains a separate object. Merge an adjacent split sentence only under narrow same-kind/same-span rules;
  otherwise retain it with explicit `context_before_id`/`context_after_id` or send it to visual retry.
- Treat the text layer as a positional scaffold, not a transcription authority. For a poor/scanned text
  layer, use the extractor's per-page `source_order` only as a fallback and add low-confidence records to
  the visual audit queue. Never force a geometric match.
- A shared native-text block is only a coarse location. Adjacent non-overlapping block-token intervals are
  distinct objects. Same-span candidates that otherwise match but disagree on a number, variable, subscript,
  exponent, sign, relation, or semantic command are hard conflicts; presentation-only LaTeX differences are not.
- Build a high-precision source-span conflict queue from exact-span collisions, very high semantic overlap,
  and same-kind containment. Ordinary shared setup is context evidence, not automatically a duplicate.
  Three independent crop audits must choose `retain_distinct`, `merge`, or `context_only`; canonical-role
  changes require an exact 2-of-3 decision. Render every page in `source_spans[]` for a cross-page candidate—never
  adjudicate it from only its primary page. Keep every removed candidate and the resolution evidence.
- Send every `transcription=low` object through independent source-span verification. A single-record verifier
  may confirm an existing candidate, but a model-proposed novel glyph never auto-corrects the statement.
  It must not absorb neighboring mathematical units. Reject large length expansion or newly contained same-page neighbors deterministically. If a tight
  crop returns low confidence, retry once with full physical pages; source scans that remain clipped stay low.
- Before asking a verifier to reread a conflict, deterministically diff same-region candidates. For an
  ordinary formula conflict, verify only the differing critical numbers/variables/subscripts/exponents/signs;
  for a table or long structured object, enumerate every row/cell as a checklist even when candidates agree.
  Require literal observations for each item and double-read long integers. Free-form “looks correct” prose cannot upgrade
  transcription confidence.
- Use three candidate-blind visual reads for structured checks and require a unique 2-of-3 value majority for
  every atom. Do not expose candidate statements, candidate values, prior conclusions, or candidate-derived
  mathematical templates. Match the resulting atom map to existing candidates deterministically and write
  back only the uniquely selected candidate's unchanged statement. A novel value remains unresolved even if
  all three observers agree. A verifier failure leaves the candidate low; it never licenses a guess. After
  bounded model attempts, a human may supply a narrow PDF-visual override,
  but it must record the exact page, checked values, method, and retain every losing candidate in
  `superseded_entries.json`.
- Treat operator, negation, inequality, variable, subscript, exponent, bound, and matrix-cell changes as
  glyph-critical. Inspect the complete page and a 4x/5x crop, report the literal glyphs before interpreting
  them, and reject any correction justified only by mathematical plausibility. Preserve visibly awkward or
  false source text with a `source_anomaly` note. An empty or wrong bbox triggers a full-page retry; it never
  authorizes deleting the object. Include awkward literal negative controls in recurring audit strata.
- Keep canonical-span and formula-conflict ledgers consistent. A high-confidence canonical crop decision may
  close the matching single-delta formula conflict only when the candidate-pair digest matches, the action is
  `merge`, the audit used the current evidence protocol, and the selected candidate statement is reused unchanged.
  This bridge never bypasses exhaustive verification for whole-table conflicts.
- Maintain `source_ledger.json`: materialize both primary-inventory and independent-coverage-audit candidate
  regions, then mark each `consumed`, `intentionally_ignored`, or `unresolved`. Also account for each source
  object as `extracted`, `merged_into`,
  `intentionally_ignored`, or `unresolved`; structural containers and later discoveries remain explicit.
  This accounts for detector output and must not be advertised as proof of every mathematical span in the PDF.
- Maintain `page_coverage_ledger.json` for every included physical page. Distinguish accounted pages,
  glossary-route pages, audit errors, and pages where neither detector found a mathematical object. List every
  non-accounted page for review. This is a completeness ledger over detector passes, not human-gold recall.
- Feed unresolved detector spans into the next targeted retry automatically (`retry_detected_span_gaps=true`).
  Rebuild the ledger after retry; a clean completion has zero unresolved detector spans.
- Keep backward-compatible `id`/`legacy_id`, use `entry_id` for graph edges, and use
  `canonical_source_id`/`source_spans[]` for complete provenance. Keep `source_span_id` only as the compatible
  primary span. Hash all component spans into the canonical identity. Never parse a synthetic `pNN` label as
  the true page.
- Make narrow post-extraction corrections only after viewing the cited PDF pages. Use
  `record_overrides` for a source-faithful transcription repair, `section_id_overrides` for a verified
  boundary assignment, and `source_verified_additions` only for a visibly present object missed by all
  model passes. Preserve page provenance and mark such additions as manual PDF visual evidence.
- A numbering gap is an audit signal, never permission to invent an object. A gap retry must return
  nothing unless the relevant printed heading/cue is visible.
- Preserve formulas, hypotheses, quantifiers, subparts, equation tags, primes, stars, inequality
  directions, dimensions, printed order, and spelling. Undo only typographic line-break
  hyphenation unless the user explicitly requests conservative spelling repair.
- Treat mathematical role changes as atomic boundaries even inside one printed sentence: separate a proved
  theorem from an attributed conjecture, and a definition from the open problem that follows it. Copy only
  visibly shared local subject/notation needed to make each statement self-contained; keep common source
  context and adjacent source order. For visually reviewed same-page reordering, store an idempotent
  `reviewed_source_order_rank` rather than overwriting a previously stabilized physical-page key.
- Do not classify a leading-display record as context-only by length alone. Strip the complete first display
  and inspect its tail: prose such as “is a 5-cycle” or “is a square if …” is the mathematical predicate and
  makes the record a canonical example. An empty display tail is context-only by default; when a blind PDF
  audit confirms that a captionless table is the source's standalone example or counterexample, restore it only
  through the hash-guarded `promote_context_only` override channel and remove its old provenance event.
- Keep source proofs/answers/hints only when printed. Never solve or generate missing support.
- After statements are stable, run the text-only relationship stage. Keep explicit source
  `cross_references` separate from logical `dependencies`: a `see`/`cf.` pointer may legally point
  forward, while a formalization prerequisite must resolve to an existing atomic `entry_id`. Book order
  is a sorting hint rather than a hard constraint for encyclopedia-style books. Require
  exact source and target evidence for every inferred edge and never infer edges from topical similarity
  alone. Keep `illustrates/specializes/comments_on/continues` as `semantic_relations`; they must not
  constrain the Lean formalization order.
- Do not use relationship edges or hidden `source_context` as a substitute for an independently readable
  canonical statement. Before QA projection, reject or repair records whose visible text still depends on
  local deixis such as “again”, “the above”, “this number”, or an unexplained equation label. Restore only
  the minimum same-source antecedent, widen the literal source span, or retain the complete parent; never
  invent a premise. The graph remains provenance and scheduling metadata, not missing question text.
- Use deterministic cleanup only for book-confirmed layout artifacts, such as a proof-ending glyph
  at the very end of proof text, a full CDATA envelope, or a lone response-closing tag. Audit every removal.
  Never globally delete vertical bars, arbitrary markup-like text, or normalize ambiguous operators.
- Put book-specific glyph distinctions in `visual_prompt_overlay`. Describe local semantics when
  similar glyphs mean different operations; do not impose one LaTeX macro book-wide from shape
  alone.

Read [relationships.md](../docs/reference/relationships.md) before changing relationship extraction or graph
fields. Read [source-grounding.md](../docs/reference/source-grounding.md) and
[auditing.md](../docs/reference/auditing.md) before changing ordering, deduplication, audit, verification, or writeback,
section assignment, or context grouping. Read [prompting.md](../docs/reference/prompting.md) for unusual boxes, multi-column layouts, distant
answers, interleaved examples, or inline definitions.

## Hybrid route: independent secondary source

Use hybrid only when a direct-route dataset and an independently produced structured text export both exist,
or when the user explicitly requests the comparison. Run `python -m parse_ocr.hybrid_pipeline` in the order
`segment`, `align`, `reconcile`, `build`, `refine`; then rerun relationships and validation on the final
hybrid objects. The optional `refine` stage is boundary-only: it constructs exact source substrings from
addressable units and preserves composite parents as context, never allowing model-written mathematical text.

Hybrid is candidate fusion, not majority voting. Prefer direct PDF transcription for one-to-one objects,
use the secondary stream to find omissions and repair object boundaries, and preserve both candidates when a
semantic-critical formula disagreement lacks source evidence. Never union every matched direct span into a
secondary child. Independently locate it across the whole structural-section page range; otherwise inherit
only the best direct counterpart or mark it section-only/unlocated. Preserve exact source order from the
secondary stream with direct-candidate interpolation, while keeping physical page/bbox as separate provenance.

After every material fusion change, run a fresh context-free audit against only the PDF and candidate output.
Separate discovery from writeback, reproduce claimed issues visually, retain false alarms, then rerun the same
fixed audit strata after repair. See [hybrid-route.md](../docs/reference/hybrid-route.md) for the full contract.

## MinerU route: explicit opt-in only

Use MinerU only when the user explicitly asks for it. Initialize with `--route mineru`, configure a
cached/local/API backend, run `python -m parse_ocr.preflight`, then run:

```bash
python -m parse_ocr.mineru_pipeline run --config PROJECT_DIR/run.json
```

The retained `pipeline.py` is the legacy MinerU + segmentation + per-object visual transcription
engine. Prefer cached MinerU, then an explicitly configured local command, then the API. MinerU API
uses `model_version=vlm`. Never silently fall back from direct to MinerU.

## Authentication

- Configure environment-variable names, not values, in `run.json`.
- The VLM endpoint must be OpenAI-compatible and accept image input. Use the strongest available
  vision model by default.
- Direct mode requires only `vlm.api_key_env`; MinerU credentials are irrelevant.
- MinerU mode additionally requires its selected cached/local/API backend.
- If authentication is absent, stop before the affected remote stage. Do not search files for keys.

## Completion gate

The canonical output is `dataset/environments.json`; `dataset/qa_pairs.json` is the flattened view.
Read [output-schema.md](../docs/reference/output-schema.md) before changing fields.

Require:

- no empty statements or duplicate IDs;
- no same-page formatting-equivalent duplicates, source-order inversions, or non-contiguous section runs;
- no reversed structural heading anchors and no unresolved canonical same-source text-overlap candidates;
- every object has a source position or is explicitly listed as a low-confidence visual-audit target;
- a fragment with an unresolved pronoun/connective is either merged with its source sentence or carries
  a narrow context link; a display completing a statement sentence is not mislabeled as an answer/proof;
- no failed windows left unaudited; a historical failed checkpoint may remain only when overlapping
  successful windows or target retries cover its pages and reconciliation has no missing objects;
- independent visual coverage audit completed without request errors, or the failed audit window is
  explicitly covered by successful overlaps plus a visual page-hole review and reported as a warning;
- every inventory target extracted or explicitly listed as unresolved;
- no unresolved incomplete/conflicting record accepted silently;
- no unresolved semantic-critical formula conflict or source-ledger detection;
- every source-span collision explicitly resolved, with no hard conflict left open; canonical outputs contain
  only `canonical`, while `context_only` and `superseded` records remain in their audit artifacts;
- every cross-page statement source represented in `source_spans[]` and included in `canonical_source_id`;
- every transcription-low object has a saved verifier result; any remaining low item stays in
  `transcription_review_queue.json` with source evidence rather than being guessed;
- every multi-row table/array and long-integer structure has an explicit atom checklist; it remains low until
  exhaustive candidate-blind 2-of-3 verification selects one existing candidate, or a documented human PDF-visual gold
  override supplies the same evidence;
- every long comma-separated sequence and grouped integer is itemized; printed invariants such as “first 100”
  and “58 digits” agree with deterministic counts;
- every included physical page appears in `page_coverage_ledger.json`, and all pages requiring review are
  reported rather than folded into an aggregate coverage percentage;
- no absolute image paths in final prose;
- every proof/answer/hint traceable to source pages;
- saved inventory, continuity, retry, run-summary, and quality reports;
- a post-build visual-audit queue saved for low-confidence alignment, unresolved overlap transcription
  conflicts, uncertain spans, and containment context lacking exact span overlap;
- relationship fields present on every object, all dependency targets resolved, no dependency cycles,
  and every unresolved explicit cross-reference retained for audit rather than guessed;
- no structural-section wrappers duplicated with their atomic children;
- visual sampling across every object kind and high-risk typography.

`uncertain_spans = 0` is not proof of perfect transcription. Report coverage as detected source
objects unless a separate audit establishes completeness.

## Script map

- `python -m parse_ocr.init_project`: create direct-default config and all-object profile.
- `python -m parse_ocr.book_inspector`: render representative pages and optionally draft a profile.
- `python -m parse_ocr.preflight`: route-aware dependency, credential, endpoint, and backend check.
- `python -m parse_ocr.direct_pipeline`: direct `inventory`, `extract`, `coverage-audit`, `reconcile`, `retry`,
  `build`, `verify`, `relationships`, `validate`, or `run` stages.
- `python -m parse_ocr.source_grounding`: deterministic source-span alignment, reading order, section assignment,
  provenance-aware canonicalization, critical-formula conflict detection, statement-completion repair,
  and context links.
- `python -m parse_ocr.verify_source_conflicts`: independent source-crop adjudication for hard formula conflicts and
  canonical-role source-span collisions, including multi-page evidence, structured checklists, blind observers,
  deterministic postchecks, and targeted `--candidate-id` reruns.
- `python -m parse_ocr.extract_relationships`: text-only explicit-reference, implicit-dependency, and graph stage.
- `python -m parse_ocr.hybrid_pipeline`: optional direct-PDF plus independent-TeX candidate fusion with bounded
  segmentation, alignment, reconciliation, grounding, provenance ledgers, and audit queues.
- `python -m parse_ocr.mineru_pipeline`: retained MinerU route; use only by explicit request.
- `python -m parse_ocr.validate_dataset`: deterministic schema and anomaly audit.

All scripts support `--help`; paths in `run.json` may be relative to that config.
