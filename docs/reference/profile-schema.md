# Book profile schema

The profile is the only book-specific control surface. Keep it evidence-based and short.

```json
{
  "schema_version": 1,
  "book_id": "stable-short-id",
  "title": "Printed title",
  "languages": ["en"],
  "include_unnumbered_named": false,
  "kinds": [
    "definition", "theorem", "lemma", "proposition", "corollary",
    "conjecture", "claim", "fact", "example", "exercise", "algorithm", "notation", "remark"
  ],
  "kind_aliases": {"problem": "exercise"},
  "label_regex": "(?:[0-9]+|[A-Z])(?:\\.[0-9]+)+",
  "structural_label_regex": "",
  "emit_structural_wrappers": false,
  "numbering_scope": "book",
  "question_section_headings": ["Exercises", "Problems"],
  "support_section_headings": ["Hints", "Answers", "Solutions"],
  "distant_support_alignment": true,
  "preserve_markers": ["K", "KK", "KKK"],
  "worked_example_policy": "split_when_source_has_setup_and_resolution",
  "figure_policy": "only_explicit_dependencies",
  "segmentation_prompt_overlay": "Book-specific ID-selection rules only.",
  "visual_prompt_overlay": "Book-specific transcription rules only.",
  "direct_conflict_similarity": 0.94,
  "strip_terminal_proof_marks": false,
  "exclude_pdf_page_ranges": [],
  "glossary_page_ranges": [],
  "glossary_overrides": {},
  "glossary_content_match_disabled_symbols": [],
  "source_verified_conflict_overrides": [],
  "source_verified_text_replacements": {},
  "resolved_conflict_ids": [],
  "merge_object_groups": [],
  "drop_object_ids": [],
  "inspection_notes": ["Facts observed from rendered pages."],
  "expected_counts": {}
}
```

## Decisions

- `include_unnumbered_named`: enable only when headings such as “Theorem” reliably delimit complete
  objects without labels. The engine assigns stable source labels such as `p17.b42`; synthetic
  labels cannot safely align distant support unless the support section prints another identifier.
- `numbering_scope`: use `book` when `(kind,label)` is globally unique; use `chapter` or `section`
  when labels restart. A scoped book needs a profile overlay telling segmentation to include the
  printed chapter/section prefix. Do not rely on source order to disambiguate repeated labels.
- `structural_label_regex`: identifies headings that are containers rather than atomic mathematical
  objects. Use it for problem-compendium articles such as `A1`–`F31` when each article contains many
  definitions, results, examples, and open questions. The inventory records these as `record_type=section`;
  they provide `section_id` and page scope but do not enter the object dataset.
- `emit_structural_wrappers`: normally false. Enable only when the source genuinely presents each matching
  heading as one inseparable exercise rather than a container. A heading containing “Problem” is not enough
  evidence. Never emit both a full-section wrapper and its atomic children merely to improve apparent recall.
- `distant_support_alignment`: enable only if answer/proof labels use the same full label namespace.
- `worked_example_policy`: normally split visible setup/question from visible derivation/answer.
  Keep compact, inseparable explanatory examples entirely in `statement`.
- `expected_counts`: optional human/audited counts by kind. Use them as coverage alerts, not as a
  reason for the model to invent missing records.
- `direct_conflict_similarity`: normalized overlap similarity below which two transcriptions of the
  same ID require a targeted retry. Keep conservative; surface real disagreements.
- `strip_terminal_proof_marks`: enable only after visual inspection confirms a distinctive terminal
  proof glyph. The cleanup applies only at the end of supporting proof text.
- `exclude_pdf_page_ranges`: physical PDF page ranges known from visual inspection to contain only
  front matter, indexes, or other out-of-scope material. Do not exclude pages merely because native
  text extraction is empty.
- `glossary_page_ranges`: physical page ranges containing a printed terminology, abbreviation,
  notation, or symbol table. Direct mode extracts these pages separately even if they overlap an
  excluded front/back-matter range, and attaches source-faithful relevant entries as
  `glossary_context` without modifying the original statement. Leave this empty only after checking
  that standalone extracted objects remain understandable without the table.
- `glossary_overrides`: optional, visually verified corrections keyed by the parser's normalized
  symbol string. Use only to resolve an ambiguous printed glyph or canonicalize an operator
  (for example, `\\Sigma` to `\\sum`); never add a meaning not present in the printed table.
- `glossary_content_match_disabled_symbols`: ambiguous or overloaded glossary symbols that must be
  attached only through the table's printed problem/section labels, never by raw text matching.
  Use this for symbols such as `P(n)` when different articles define it differently.
- `resolved_conflict_ids`: source-visually adjudicated IDs whose overlap variants are known to be a
  faithful shorter/longer rendering of the same text. This suppresses only the conflict alert; the
  normal complete/longer candidate selection still applies.
- `source_verified_conflict_overrides`: last-resort, narrowly scoped human PDF-visual adjudications after
  bounded structured observers fail to converge. Each record must include `candidate_ids`, `selected_id`,
  `status=resolved`, `confidence=high`, `human_pdf_visual_verification=true`, physical page, method, and exact
  checked evidence. This may select a faithful candidate but must never erase the losing provenance.
- `source_verified_text_replacements`: narrow exact `old`/`new` substitutions keyed by stable ID. Each old
  string must occur exactly once or the build fails. Use this for a visually verified digit, sign, subscript,
  or name correction without cloning a long statement into the profile.
- `merge_object_groups`: visually confirmed inventory fragments that form one inseparable source
  definition/example. Each entry has `target_id` and ordered `source_ids`; targeted retry transcribes
  one combined target and reconciliation removes the subordinate IDs.
- `drop_object_ids`: visually confirmed false-positive inventory objects, such as an introductory
  sentence mistaken for a standalone fact. These IDs are excluded from coverage, retry, and output.
  Use this only after checking the rendered source page.
- Source grounding is a run-level direct-route setting (`direct.source_grounding_enabled`, default true),
  not a book-specific profile heuristic. Use `section_id_overrides` only for visually verified cases that
  remain ambiguous after heading-bbox assignment. A per-book override must never replace the global
  source-order key with kind/label sorting.

## Visual inspection checklist

Inspect title/front matter, chapter starts, theorem-heavy pages, exercise sections, worked examples,
appendices, answer/hint sections, multi-column pages, algorithm boxes, pages with footnotes, and
cross-page environments. Record what is actually visible; do not infer a convention from one page.
