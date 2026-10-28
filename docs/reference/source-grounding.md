# Source-grounded ordering and canonicalization

Use this design after VLM transcription and before relationship extraction. It addresses four failures that
cannot be solved reliably by a stronger transcription prompt alone: same-page order loss, duplicate aliases,
same-page section misassignment, and atomic statements detached from their source context.

## Invariants

1. The page image/VLM output is authoritative for mathematical text.
2. The PDF text layer or a layout parser supplies geometry only. Never replace a formula from native OCR.
3. Object identity is `(document, source span)`, not `(model kind, model label)` and not text similarity alone.
4. Final reading order is `(physical page, bbox y, block index, token offset)`.
5. A structural section begins at its heading anchor, not at the top of its first physical page.
6. Atomicity and readability are separate layers: keep the atomic statement; carry surrounding source context
   in a context group rather than cloning or concatenating every neighboring object.
7. A native text block is only a coarse location. When block-token intervals are available, they must overlap
   before candidates are treated as occupying the same source unit.
8. Heading localization precedes object cleanup. Search exact section labels in all page blocks, including the
   page-top band; when both a running header and an in-body heading match, prefer the in-body heading.

## Grounding cascade

Prefer, in order:

1. native PDF text blocks when the text layer has a usable match;
2. an external parser's content-list/model blocks with page and bbox provenance;
3. the VLM's strict per-page source ordinal and opening anchor;
4. page-only fallback, explicitly marked `unlocated` and queued for visual audit.

Alignment must be conservative. Record several candidate locations and a score. A weak match is evidence for
manual/visual retry, not permission to move the object to a semantically plausible paragraph.

## Canonicalization layers

- Merge presentation-equivalent strings only when their grounded block intervals overlap. Ignore bold/HTML,
  harmless math wrappers, spacing, and Unicode case-fold differences. Preserve all source IDs and candidate
  kinds on the winner.
- Keep equal strings at different source spans. Repeated definitions and repeated problem formulations can be
  intentional.
- Merge a split sentence only when items are adjacent in source order, same kind, same source block, and the
  second starts as a grammatical continuation. Otherwise keep both and link context.
- If a compact display completes an unfinished statement sentence, move the unchanged display from
  `supporting_text` back into `statement`; do not relabel a genuine proof, hint, or solution.
- Never auto-delete fuzzy or semantic containment. Store same-span objects in `source_context_groups.json`.
  When a larger VLM statement contains an atomic child, attach the larger source text as `source_context` to the
  child. This preserves a clean Lean target and a readable/checkable prompt context.
- After canonicalization, emit residual same-source containment/long-overlap candidates to
  `canonical_text_overlaps.json`. Similarity only opens an audit case. A source-backed decision must partition
  shared setup into context, retain distinct atomic objects, or supersede a true duplicate.
- Normalize presentation-only formula wrappers before comparison. A disagreement in a number, variable,
  subscript, exponent, sign, relation, or semantic command is a hard conflict when candidates otherwise
  transcribe the same overlapping unit. Semantic containment is context/atomicity evidence, not automatically
  a transcription conflict.
- Save hard conflicts in `formula_conflicts.json`. Verify only those conflicts against rendered source evidence;
  use the full physical page when alignment is too weak for a safe crop. Only a candidate-blind 2-of-3 atom
  consensus that uniquely matches one existing candidate may choose it automatically. Other readings remain unresolved.
- Run a separate high-precision canonical-role audit for exact source spans, very high overlap/similarity, and
  same-kind containment. Its valid outcomes are: keep both atomic units, merge duplicate candidates, or move a
  redundant host/setup candidate to context-only storage. Preserve source text and provenance for every outcome.
  Only high-confidence role decisions apply automatically. A cross-page decision must render all component
  source spans; a primary-page crop alone is insufficient evidence.
- Verify every low-transcription record independently, but treat the existing candidate boundary as fixed.
  The machine verifier can confirm existing text but cannot write a novel character correction. It does not prepend/append neighboring objects. A deterministic boundary
  gate rejects disproportionate length growth and any newly absorbed same-page statement, unless an already
  source-verified canonical/context resolution explicitly consolidates that pair. A low crop may retry on full
  pages once; persistent source clipping remains unresolved.

## Structured transcription verification

Resolve candidate identity before correcting text. First compute a deterministic candidate diff, then build
the smallest adequate verification unit:

- a formula-only conflict gets a checklist of the differing critical tokens;
- a multi-row table, matrix, factor list, or long-integer block gets an exhaustive row/cell checklist, including
  values on which candidates agree;
- each long integer is read twice, preferably in digit groups, and the two readings must agree;
- each verifier returns structured observations, not a prose assurance;
- automatic writeback reuses an existing candidate statement exactly; model-generated full text is evidence only.

Run three independent observers that see only the source crop and value-free checklist roles. Hide candidate
statements, candidate values, prior decisions, and candidate-derived math templates. Require a unique 2-of-3
majority for every atom and match that consensus to exactly one existing candidate. Save every attempt.
Non-convergence, an ambiguous candidate match, or any novel value is a valid unresolved result and must keep
the item low. Read [auditing.md](auditing.md) for the full trust and writeback protocol.

After bounded automatic attempts, a human PDF-visual override is permitted for a narrowly identified conflict.
It must list candidate IDs, selected ID, physical page, exact checked atoms, and the fact that the decision was
human visual evidence. The selected candidate becomes canonical; losing candidates remain in the superseded
layer with their original text and provenance.

Formula and canonical-span ledgers may independently flag the same candidate pair. Use their shared digest to
propagate a high-confidence `merge` decision only for a formula-token delta whose selected values are found in
the canonical verifier's complete statement. Never use this shortcut for a cross-candidate table conflict.

## Identity and source accounting

Keep historical `id=kind:label` for compatibility. Add `source_spans[]` from every physical
page/block/token component, retain `source_span_id` as the primary compatibility anchor, and hash all spans into
`canonical_source_id`; use `entry_id` for graph edges. A synthetic label such as `p185.x1`
can disagree with the physical page, so consumers must never parse it as provenance. `id_aliases.json` records
the mapping.

`source_ledger.json` materializes the independent page-level candidate-region universe from both inventory and
coverage-audit passes. Candidate spans are `consumed`, `intentionally_ignored`, or `unresolved`. Each detected object is `extracted`,
`merged_into`, `intentionally_ignored`, or `unresolved`; structural records are `structural_container`, and
objects found only during transcription/retry are `extracted_discovered`. Its coverage claim must state that
detector accounting is not proof that every mathematical source span was detected. Unresolved candidate spans
become targeted retry inputs on the next retry pass; after rebuild they must either be consumed or explicitly ignored.

## Validation

Require zero source-order inversions, zero non-contiguous section runs, and zero same-page
presentation-equivalent duplicate groups. Report:

- high/medium/low/unlocated grounding counts;
- section headings found and section IDs used;
- exact, contained-fragment, and sentence-fragment merge events;
- candidate-kind disagreements;
- context groups and membership;
- unresolved anaphoric statements;
- unexplained containment pairs;
- low-confidence IDs for visual audit.
- decomposed localization/transcription/segmentation/deduplication confidence;
- hard formula conflicts and their resolution evidence;
- source-ledger states and legacy-label/source-page mismatches.
- physical-page accounting states and pages requiring review;
- structured items checked/corrected, unresolved hard-conflict rate, and whether human gold exists for recall
  or residual transcription-error estimation.
- section-heading anchor reversals and unresolved canonical text overlaps.

`dataset/visual_audit_queue.json` combines those positional risks with unresolved overlapping-window
transcription conflicts and `uncertain_spans`. Treat it as the input to a later targeted visual-adjudication
pass; do not rerun the whole book. An item can be structurally well placed yet still need formula review.

Do not claim full coverage from a clean source-grounding report. Coverage still requires structural inventory,
independent visual audit, numbering/section continuity checks, and targeted gap retries.
`benchmark_metrics.json` must leave gold recall and hard transcription error rate as `null` when no human unit/
token gold manifest exists; model confidence counts are not substitutes for those metrics.

## Design lineage

This scheme combines ideas that should remain distinct:

- `mathdoc-parser` contributes stable source-position ordering and declaration identity; its conservative rule
  against merging uncertain cross-declaration similarity is retained.
- Docling and MinerU demonstrate that reading-order output should carry page/bbox provenance rather than only
  flattened text.
- LayoutReader shows why reading order is a layout problem, especially outside single-column books.
- arXiTeX demonstrates the strongest case: when TeX source exists, exact character/source spans and deterministic
  references should replace PDF-text alignment.

For multi-column, side-note, or heavily boxed layouts, a top-to-bottom native-block order is insufficient. Use a
layout reading-order model or VLM adjudication while keeping the same source-position schema.
