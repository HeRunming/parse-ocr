# Hybrid direct-PDF plus structured-text route

Use this route only after the direct pipeline is stable and an independently produced TeX/XML/HTML export is
available. Its purpose is to improve recall and atomic boundaries without allowing a second OCR system to
overwrite PDF-grounded mathematics.

## Evidence roles

- **G (direct PDF/VLM)** is the preferred transcription source and supplies PDF page/bbox provenance.
- **T (secondary structured text)** supplies an independent recall signal, exact document order, prose
  continuity, and alternative object boundaries. It is not PDF truth.
- **The rendered PDF** is authoritative when G and T disagree on a number, variable, subscript, exponent,
  sign, relation, matrix/table cell, or semantic LaTeX command.

Do not ask a reconciliation model to synthesize a third statement. It may select only existing candidate IDs.
Store its reason and whether visual review is required.

## Configuration and stages

Add a `hybrid` block to `run.json`:

```json
{
  "route": "hybrid",
  "atomicity_refinement": {
    "enabled": true,
    "workers": 32,
    "max_tokens": 3000
  },
  "hybrid": {
    "direct_dataset": "../direct/output/dataset/environments.json",
    "direct_sections": "../direct/output/dataset/sections.json",
    "secondary_articles": "../secondary/article_segments.json",
    "secondary_source_root": "../secondary/source",
    "audit_overrides": "reviewed_overrides.json",
    "secondary_grounding_min_score": 0.36,
    "segment_workers": 32,
    "reconcile_workers": 32
  }
}
```

Run:

```bash
python -m parse_ocr.hybrid_pipeline segment --config PROJECT_DIR/run.json
python -m parse_ocr.hybrid_pipeline align --config PROJECT_DIR/run.json
python -m parse_ocr.hybrid_pipeline reconcile --config PROJECT_DIR/run.json
python -m parse_ocr.hybrid_pipeline build --config PROJECT_DIR/run.json
python -m parse_ocr.hybrid_pipeline refine --config PROJECT_DIR/run.json
python -m parse_ocr.extract_relationships --config PROJECT_DIR/run.json --force-stage
python -m parse_ocr.validate_dataset --config PROJECT_DIR/run.json
```

The segment, reconcile, and final atomicity-refinement stages are checkpointed. Use `--force-stage` only for
the changed stage. A build is local and deterministic once candidate, reconciliation, and refinement
checkpoints exist.

## Fusion rules

1. Segment T on source block boundaries. Long articles use overlapping block windows; deduplicate overlap by
   stable candidate digest.
2. Align within structural sections using prose and math-aware similarity. One G object may align to several T
   children and vice versa; this is boundary evidence, not a duplicate decision.
3. Prefer G text for a one-to-one equivalent object. Retain T when it is a genuine new object or a needed child
   of an overmerged G host.
4. If G and T disagree on semantic-critical formula tokens, keep G and enqueue the pair for candidate-blind
   PDF verification. For tables, matrices, long integer lists, and other structured layouts, do not promote T
   without exhaustive source checking.
5. A T child must be grounded independently over every physical page between its section heading and the next
   heading, augmented by pages occupied by direct objects. Accept a unique native-text match; otherwise inherit
   only one best G counterpart. Never union spans from multiple matches. If neither is defensible, retain a
   section-scope span with `source_pdf_page=null` and `grounding_confidence=unlocated`.
6. Keep atomic statements atomic. If a retained child begins with an anaphor such as “They”, “This”, “An
   example”, “these bounds”, or “the corresponding maximum”, attach exact neighboring source text through
   `context_before`/`source_context`; do not silently prepend it to the statement.
7. A lexical repair requires both an aligned G host and independent native-PDF text evidence for the replacement.
   Accent-only normalization may use the aligned G candidate. Do not use unconstrained spell checking, and do
   not replace a proper name merely because two OCR systems differ.
8. Preserve T article order and interpolate unmatched G candidates between aligned neighbors. Store this as an
   initial abstract source-order key. After source grounding and reviewed overrides, order final objects by
   confirmed physical page and use article order only within a page. Interpolate truly unlocated rows between
   their located article neighbors. A PDF-reviewed insertion or reorder must carry a separate
   `reviewed_source_order_rank`; do not encode it by mutating an already stabilized page-bearing key, because
   the next rebuild would reinterpret that page component as an abstract rank and move the object to the end.
   Test that two consecutive stabilization passes are idempotent. Never let noisy candidate order create a
   backward physical-page run.
9. Clear inherited relationship edges after fusion. Object IDs and boundaries changed, so rerun relationship
   extraction and graph validation rather than rebinding stale edges by resemblance. Set a relationship
   `max_prompt_chars` budget for long encyclopedia sections: shard the candidate target set while repeating the
   same source batch, so every target is still considered. A completed raw response may be reused for a shard
   only when source entry IDs match exactly and the old candidate set is a superset; always reparse and filter
   the raw XML against the current shard. Recursively HTML-unescape evidence before storing it so evidence is
   the exact source substring accepted by graph validation.
10. After fusion, enqueue only objects with strong compound-object signals for boundary refinement. The model
    receives exact addressable sentence units and may return only `keep` or increasing, non-overlapping unit
    ranges. Construct children by byte-for-byte slicing, reject malformed math delimiters, retain the complete
    parent in `context_only_entries.json`, and attach it as `source_context` to every child. Never accept model-
    written replacement text. Treat `source_context` as provenance and reader support, not as permission to
    emit fragments: if a proposed child has an unresolved pronoun, deictic phrase, undefined error term, or
    says only that something “is a solution”, keep the parent intact. Run relationships only after this final
    object layer is frozen.
11. Apply candidate-blind PDF-confirmed corrections through a checked override artifact, not ad hoc code. Every
    update or context-only action must name the exact pre-edit statement SHA-256; reject stale/missing targets,
    retain `statement_before_reviewed_override`, and emit `reviewed_override_audit.json`. A visual auditor's
    suspected finding never qualifies for this layer. Record source-disproved findings as
    `rejected_no_writeback`; this prevents a later reviewer from repeatedly proposing a plausible but visibly
    false normalization. If conservative factoring placed a real standalone object in the provenance layer,
    restore it through `promote_context_only`, guarded by its exact statement hash and rendered-page evidence;
    remove the corresponding context-only event rather than duplicating the row. This channel is especially
    useful for captionless data tables, but must not be driven by an extraction model alone. If a coverage
    audit finds an object missed by both streams, use the distinct
    `source_verified_additions` channel with an explicit stable ID, physical page, source order, literal
    transcription, and audit evidence; reject duplicate IDs and never use this channel to generate an answer.
    For a long support field where one visually confirmed atom is wrong, use an exact-count, hash-guarded
    `replace_fields` edit instead of copying the whole field into an override. Reject the entire edit when the
    old atom occurs a different number of times. After reviewed overrides, strip only unambiguous API envelope
    artifacts (a full outer CDATA wrapper or a lone XML closing tag), and emit `transport_cleanup_audit.json`;
    never generalize this into mathematical-symbol or arbitrary HTML cleanup.
12. Keep the canonical inventory disjoint at source-clause level. Do not retain both a complete paragraph and
    standalone children that repeat its clauses. Choose one of three evidence-preserving outcomes: keep the
    complete object and move duplicate children to `context_only`; retain disjoint children and trim the host to
    its unique source text; or keep a reusable definition/result separately and remove only its repeated wording
    from the dependent object while attaching an explicit context target. Never delete the losing candidate;
    preserve it in `context_only_entries.json` with its canonical target and audit evidence.
    Conversely, do not keep distinct mathematical roles fused merely because the source prints them in one
    sentence. In particular, split a proved result from a following conjecture and split a definition from a
    following open question; restore the subject/notation from the same visible sentence so each canonical
    statement remains independently understandable. Preserve their shared source context and adjacent order.
    Relationship metadata cannot rescue a fragment at QA time: before projection, each canonical statement
    must expose enough source text to resolve local pronouns, deictic phrases, and equation labels on its own.
    Either copy the minimum antecedent visible in the same source passage, widen its literal source span, or
    keep the unsplit parent. Preserve a graph/context link as additional provenance, not as replacement text.
    A record that begins with a display is not automatically support-only: inspect the prose after the closing
    display. If it asserts that the displayed object is a cycle, solution, square, counterexample, or otherwise
    supplies a mathematical predicate, retain it canonically. A genuinely bare display with an empty tail is
    context-only by default, but a candidate-blind page audit may promote a captionless table or array when the
    surrounding prose visibly identifies it as the promised example or counterexample. Make that exception via
    the checked promotion channel, not a global table heuristic.
13. Resolve an explicit figure/table pointer to a unique canonical caption object when one exists, including a
    parenthetical bare pointer such as `(Figure 2)`, and attach
    the exported asset directly to the consumer when the visual shape is material to the statement. A bare
    non-parenthetical mention is not a caption alias. Leave ambiguous or absent anchors explicitly unresolved, and distinguish an
    interpretation-critical figure from a merely illustrative one in release policy. Build a figure-label asset
    index from article-level TeX before candidate selection, because a direct-PDF winner can legitimately need an
    image carried only by an unselected secondary candidate.

## Accounting and audit

The build must emit candidate accounting for both streams in `source_ledger.json`, section-page accounting in
`page_coverage_ledger.json`, source context groups, the full conflict queue, reconciliation decisions, dropped
exact duplicates, and a visual-audit queue. A clean ledger means every detector candidate has a recorded
selection/rejection state; it does not prove human-gold recall.

The final boundary pass additionally emits `atomicity_review_queue.json`, `atomicity_refinements.json`,
`atomicity_refinement_summary.json`, and `hybrid_atomicity_audit.json`. Preserve rejected/kept decisions so a
later auditor can distinguish “not reviewed” from “reviewed and deliberately kept.”

Evaluate hybrid against the direct baseline with a fixed, source-blind sample that includes ordinary objects,
structured formulas, cross-page objects, boundary conflicts, and predicted direct misses. Score recall,
transcription, atomicity, context, order, and grounding separately. Record confirmed regressions, suspected
issues, and false alarms. Repair deterministic construction bugs first; change prompts or models only when the
same failure survives source-grounded reproduction.

An auditor must judge the complete consumer record (`statement`, context fields, source-context memberships,
assets, and graph edges), not the statement string alone. Context-only provenance is not a canonical duplicate.
Require direct rendered-page evidence for every confirmed transcription finding and keep a negative-control
sample; a model's mathematical preference or OCR-text reading is not source evidence.

Do not promote a hybrid output while it has wrong physical pages, span unions from unrelated objects, dangling
anaphora without context links, stale relationship edges, or a semantic-critical disagreement silently resolved
from the secondary text.
