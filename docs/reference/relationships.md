# Relationship extraction

## Contents

1. Semantic boundary
2. Extraction stages
3. Validation
4. Lean-facing use

## 1. Semantic boundary

Keep three edge families separate:

- `cross_references` are printed navigation links. They may point backward, forward, to a whole
  section, or to a source anchor that is not an extracted object.
- `dependencies` are direct document-local prerequisites needed to understand or formalize the
  source object. They resolve to existing atomic entries.
- `semantic_relations` capture `illustrates`, `specializes`, `comments_on`, and `continues`. They are
  useful context links but do not determine Lean declaration order.

Do not turn every `see` or `cf.` into a prerequisite. Do not erase forward references merely because
they point later in the book. Store them as cross-references. A logical dependency may also point to a
later source entry in an encyclopedia-style book when the definition/background follows the question;
enforce acyclicity on the resulting logical graph and compute a separate formalization order.

The graph is not a claim about all foundations of the mathematics. Common background concepts that
the book never defines are outside the document-local dependency graph.

## 2. Extraction stages

The relationship stage runs after final statement construction and does not need page images.

1. Deterministically extract explicit section and typed-object references from `statement` and
   source `supporting_text`. Retain the exact cue, source field, span, and resolution status.
2. For implicit dependencies, give the model only existing source and candidate `entry_id` values.
   Candidate scope consists of nearby objects on both sides in the same section plus conservatively
   ranked objects from explicitly cited sections.
3. Accept an inferred edge only when its target exists and both source evidence and target evidence are
   exact contiguous excerpts of their respective statements/support. For logical dependencies, require
   a shared distinctive term, symbol, or formula between the two evidence spans. Topical similarity and
   adjacency alone are insufficient. Reject the edge when the target evidence is already repeated in the
   source or when the source evidence locally defines the same term/notation; those records are
   self-contained rather than dependent.
4. Cache each request by the full prompt hash. A changed source item or candidate set must create a
   new checkpoint.

Allowed dependency relations are `uses_definition`, `uses_notation`, `uses_result`, and
`uses_construction`. Semantic relations are `illustrates`, `specializes`, `comments_on`, and `continues`.

## 3. Validation

Require:

- relationship arrays on every environment;
- unique edge IDs;
- resolved dependency target `id` and `entry_id` pairs;
- no self edges or cycles;
- graph and per-record edge counts identical;
- unresolved explicit references reported, never silently dropped;
- relationship request failures visible and fatal by default.

If a visually/textually audited model edge is wrong or creates a false cycle, list its stable `edge_id`
in profile `drop_relationship_edge_ids`. The graph retains that audit list. Do not break cycles by silently
dropping every later-target edge or by changing source order.

The graph direction stored in edge records is dependent/referrer → prerequisite/referenced target.
`formalization_order` is separately computed target-before-dependent.

## 4. Lean-facing use

Use `formalization_order` as a scheduling hint, not as Lean's compiled declaration dependency graph.
For an individual object, provide its direct dependency statements and glossary context to the
formalizer. Let Lean compilation determine imports and elaborated constant dependencies later.

Preserve relation types: a definition/notation dependency is usually stronger context than
`comments_on` or `illustrates`. If a downstream service supports only a flat list, pass
`target_entry_id` values but retain the full graph as the auditable canonical form.
