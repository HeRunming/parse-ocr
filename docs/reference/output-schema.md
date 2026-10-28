# Output schema

Canonical `dataset/environments.json` is an array of:

```json
{
  "id": "theorem:2.5",
  "legacy_id": "theorem:2.5",
  "entry_id": "3.2.004",
  "source_span_id": "p17.b4-5.t12-49",
  "source_spans": [
    {"source_span_id":"p17.b4-5.t12-49","pdf_page":17,"bbox":[18.7,121.1,343.3,196.0]},
    {"source_span_id":"p18.b1-2.t0-31","pdf_page":18,"bbox":[20.1,44.0,344.2,91.0]}
  ],
  "canonical_source_id": "theorem:3.2:p17.b4-5.t12-49:9a1b2c3d4e",
  "canonical_role": "canonical",
  "section_id": "3.2",
  "kind": "theorem",
  "label": "2.5",
  "title": "Optional printed title",
  "statement": "Source-faithful Markdown/LaTeX",
  "supporting_text": "Source proof, derivation, answer, hint, or solution",
  "supporting_text_role": "proof",
  "supporting_text_origin": "book_source",
  "statement_sources": [{"part":"0","block_id":42,"pdf_page":17,"bbox":[0,0,0,0]}],
  "supporting_sources": [],
  "source_pdf_page": 17,
  "source_pages": [17, 18],
  "source_position": {
    "page": 17,
    "block_start": 4,
    "block_end": 5,
    "bbox": [18.7, 121.1, 343.3, 196.0],
    "token_start": 12,
    "token_end": 49,
    "score": 0.91,
    "method": "native_text_geometry_alignment"
  },
  "source_order_key": [17, 121.1, 4, 12],
  "grounding_confidence": "high",
  "confidence": {
    "localization": "high",
    "localization_score": 0.91,
    "transcription": "high",
    "segmentation_atomicity": "medium",
    "deduplication": "high",
    "method": "deterministic_risk_signals_not_model_probability"
  },
  "verification_status": "unverified_risk_screened",
  "candidate_locations": [],
  "candidate_kinds": ["theorem", "fact"],
  "merged_source_ids": ["theorem:p17.t1", "fact:p17.f1"],
  "context_before": "Narrow preceding source context when required",
  "context_before_id": "3.2.003",
  "source_context_group_id": "src:p17.b4-5",
  "source_context": "A larger same-span source passage containing this atomic statement",
  "source_context_host_id": "3.2.005",
  "assets": [{"image":"assets/theorem_2.5_1.png","caption":"...","pdf_page":17}],
  "glossary_context": [{"id":"glossary:abc123", "symbol":"A(x)", "meaning":"...", "source_pdf_page":13}],
  "cross_references": [{
    "edge_id": "xref:...",
    "target_type": "section",
    "target_id": "section:2.1",
    "target_entry_id": "",
    "target_section_id": "2.1",
    "target_label": "2.1",
    "relation": "see_also",
    "explicit": true,
    "evidence": "see Section 2.1",
    "source_field": "statement",
    "resolved": true,
    "method": "deterministic"
  }],
  "dependencies": [{
    "edge_id": "dep:...",
    "target_id": "definition:2.1",
    "target_entry_id": "2.1.001",
    "target_section_id": "2.1",
    "relation": "uses_definition",
    "explicit": false,
    "evidence": "compact convex set",
    "target_evidence": "A set is compact when ...",
    "source_field": "statement_or_supporting_text",
    "confidence": "high",
    "method": "llm_conservative"
  }],
  "semantic_relations": [{
    "edge_id": "sem:...",
    "target_id": "definition:2.1",
    "target_entry_id": "2.1.001",
    "target_section_id": "2.1",
    "relation": "illustrates",
    "evidence": "For example",
    "target_evidence": "A set is compact when ...",
    "confidence": "high",
    "method": "llm_conservative"
  }],
  "uncertain_spans": []
}
```

Rules:

- `id = kind:label`; different kinds may share a label.
- `id`/`legacy_id` remain backward-compatible extraction identifiers. `entry_id` is the graph identifier;
  `source_span_id` is the primary compatibility anchor; `source_spans[]` is the complete statement provenance;
  `canonical_source_id` combines kind, structural scope, every component span, and a statement hash. Never
  infer a physical page by parsing a synthetic `pNN` label.
- Canonical object views contain only `canonical_role=canonical`. Redundant context passages live in
  `context_only_entries.json`; duplicate or losing conflict candidates live in `superseded_entries.json`.
  Both audit layers retain full source text, source positions, and the canonical target/winner.
- `section_id` is the printed structural container; `entry_id` is a stable source-order identifier within
  that container. Structural containers themselves live in `dataset/sections.json` and are not duplicated
  as exercises unless the profile explicitly enables wrappers.
- `statement` is always nonempty and contains the printed object heading when present.
- `supporting_text` is empty unless present in the source.
- `supporting_text_origin` is `book_source` for printed support and empty otherwise. Generated material must
  use a separate field/dataset and must never be presented as source support.
- `supporting_text_role` is one of `proof`, `derivation`, `explanation`, `answer`, `hint`,
  `solution`, `hint_or_solution`, or empty.
- Sources preserve traceability; final prose contains no local file path.
- `source_position` is the canonical reading-order/provenance anchor. Native PDF text is used only to locate
  the VLM transcription; it is not trusted to repair mathematical content. `source_order_key` determines
  final ordering. Low-confidence or page-only fallbacks remain explicit audit targets.
- `confidence` separates localization, transcription, segmentation/atomicity, and deduplication signals.
  These are deterministic risk categories, not calibrated model probabilities and not interchangeable with
  the legacy `grounding_confidence` alignment score.
- `verification_status` is one of `human_pdf_verified`, `machine_source_verified`,
  `unverified_risk_screened`, or `verification_required`. Risk-screened content is not described as having
  been checked against the PDF merely because no detector fired.
- `candidate_kinds` and `merged_source_ids` preserve extraction disagreements after two records are proven to
  occupy the same source span. Text similarity without span overlap never authorizes a merge.
- `context_before`/`context_after` and their ID fields provide the minimum source context for an otherwise
  non-self-contained atomic statement. They do not become part of the source statement or a dependency edge,
  and therefore cannot by themselves make the projected QA `question` independently readable.
- `source_context_group_id` groups atomic objects grounded in an overlapping paragraph/block span.
  `source_context` is populated only when a larger VLM-transcribed same-span passage contains the atomic
  statement. It is prompt/display context, not a second theorem and not a logical dependency.
- `glossary_context` is optional source material from a printed global notation table. It supplements
  rather than rewrites `statement`; the complete table is saved as `dataset/symbol_glossary.json`.
  Despite the field name, `symbol` may contain a printed term or abbreviation when the source table
  defines prose terminology rather than mathematical notation. `matched_by` records whether the
  entry came from the table's applicability labels, a conservative statement-symbol match, or both.
- `cross_references` preserves visible `see`, `cf.`, named-item, table, figure, or section pointers.
  It is source-faithful navigation metadata and may point earlier or later. An unresolved pointer is
  retained with `resolved=false`; never guess its target.
- `dependencies` contains direct document-local prerequisites only. The source object depends on the
  target. Targets must be existing `entry_id` values, and every model-inferred edge must carry
  a short exact evidence excerpt from the source object. Do not store transitive closure or generic
  mathematical background. A target may occur later in source order when the book introduces a question
  before the definition/result needed to interpret it; the acyclic graph determines formalization order.
- `semantic_relations` stores `illustrates`, `specializes`, `comments_on`, and `continues` edges. These
  help navigation and context assembly but do not constrain Lean formalization order. Model-produced
  dependency and semantic edges carry exact excerpts from both source and target records.
- `qa_pairs.json` maps `statement → question` and `supporting_text → answer` without synthesizing
  either side, and retains all three relationship arrays. Canonical records containing unresolved local
  deixis or equation labels must be repaired from visible source context or withheld before this projection;
  consumers are not required to reconstruct missing question text from the graph.

`dataset/source_context_groups.json` is the deterministic geometry/context view. It records each group's page,
bbox, source IDs, and entry IDs without treating the group as a mathematical object.

`dataset/visual_audit_queue.json` is a risk-ranked post-build queue. Reasons include low source alignment,
unadjudicated overlap transcription conflicts, explicit uncertain spans, and context containment without exact
span overlap. It is an audit artifact, not another object dataset.

`dataset/transcription_review_queue.json` contains the remaining `transcription=low` canonical records after
source verification. Applied repairs retain `statement_before_verification`, `uncertain_spans_original`, and
the complete `transcription_verification` record. Boundary-expanding verifier outputs are rejected, not applied.

`dataset/structured_verification_audit_queue.json` lists canonical records containing multi-row tables,
matrices, or long critical integers whose atom checklist has not been exhaustively source-verified. Each item
records the required checklist IDs. Such a record must not be labeled transcription-high merely because a
free-form verifier returned high confidence.

`dataset/source_ledger.json` accounts for each inventory/coverage-detected source object as `extracted`,
`merged_into`, `intentionally_ignored`, or `unresolved`, plus structural containers and later discoveries.
Its declared coverage is detector accounting, not proof that every source span in the PDF was found.
Hybrid builds use the analogous states `selected`, `split_to_atomic_children`, `context_only`,
`replaced_by_direct`, and `not_selected`. A split ledger row records every canonical `target_ids` child plus
the preserved `context_only_target_ids` parent; it must not collapse a one-to-many split into one arbitrary
target.

`dataset/page_coverage_ledger.json` accounts for every included physical page as `accounted`,
`accounted_glossary_route`, `coverage_audit_error`, or `no_detected_mathematical_object`, and separately lists
pages requiring review. It does not imply human-gold unit recall.

`dataset/benchmark_metrics.json` reports structured atoms checked/corrected, semantic-duplicate audit counts,
and unresolved hard-conflict rate. `gold_unit_recall` and `hard_transcription_error_rate` remain `null` with an
explicit status until corresponding human gold manifests exist.

`dataset/formula_conflicts.json` records overlapping near-duplicate candidates with semantic-critical formula
differences, per-token/per-cell verification checklists, and any crop-verification resolution. An unresolved
hard conflict fails validation by default. Resolved conflicts keep all losing candidates in the superseded
artifact rather than deleting provenance.

`dataset/source_span_collisions.json` records exact-span/high-overlap canonical-role conflicts and the independent
multi-page crop decision. The resolution is `retain_distinct`, `merge`, or `context_only`; unresolved hard
collisions fail validation. A resolved collision documents an explicit audit decision, not an assertion that
the candidates have no shared wording.

`dataset/canonical_text_overlaps.json` records residual canonical statements that repeat a long contiguous
source passage while occupying overlapping grounded spans. Unresolved items block strict release; they are not
automatically deleted on text similarity.

`dataset/id_aliases.json` maps compatibility IDs and `entry_id` values to physical page/span provenance and
`canonical_source_id`, including legacy-label/page mismatches.

`dataset/dependency_graph.json` is the normalized graph view. It contains entry, section, and source-context nodes,
dependency edges, semantic-relation edges, cross-reference edges, unresolved references, target-before-source
`formalization_order`, source-context memberships, cycles, request errors, and aggregate counts. Context membership
never constrains formalization order. Edge objects use
`source_entry_id` for the dependent/referring item and target fields for the prerequisite/referenced item.

The dataset directory also contains JSONL, Markdown, per-kind JSON, transcription audit,
segmentation summary, run summary, and deterministic `quality_report.json`.
