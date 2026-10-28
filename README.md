# parse-ocr

[English](README.md) | [中文](README.zh-CN.md)

`parse-ocr` turns mathematics PDFs (born-digital or OCR'd scans) into structured JSON. It extracts
definitions, theorems, lemmas, propositions, corollaries, examples, exercises, remarks, algorithms and
the printed proofs, hints and answers attached to them, and exports QA/VQA records.

The design rule is that **the page image is the authority**. A vision-language model (VLM) reads
rendered pages. The PDF text layer, when present, is used only to locate, order and deduplicate what the
model read. It never replaces the model's mathematics.

## Contents

1. [Outputs](#outputs)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Pipeline](#pipeline)
5. [Command reference](#command-reference)
6. [Configuration](#configuration)
7. [Guarantees and non-guarantees](#guarantees-and-non-guarantees)
8. [Evaluation](#evaluation)
9. [Repository layout](#repository-layout)
10. [Development](#development)
11. [Known limitations](#known-limitations)

## Outputs

A run writes two layers under `<output>/dataset/`. They have different contracts.

| Layer | File | Written by | Contract |
|---|---|---|---|
| Evidence | `environments.json` (also `.jsonl`, `.md`) | `direct build` | One record per printed environment. `statement` and `supporting_text` are VLM transcriptions of the page, located on the PDF text layer when one exists. No paraphrase. |
| Evidence (QA view) | `qa_pairs.json` | `direct build` | Flattened `environments.json`: `question` = `statement`, `answer` = `supporting_text`, `answer_role` = `proof`, `hint`, `solution`, ... |
| Structured | `structured_statements.json` | `prose-claims merge` | Evidence records plus claims mined from prose, each with a model-written `normalized_statement`, a `source_statement`, an `origin` and a verifier verdict. Never edits `environments.json`. |

Every record keeps its provenance: `source_pages`, `source_pdf_page`, and, when a text layer exists,
`source_spans` (`source_span_id`, `pdf_page`, `bbox`).

**Evidence record** (`environments.json`, abridged):

```json
{
  "id": "theorem:2.5", "entry_id": "3.2.004", "section_id": "3.2",
  "kind": "theorem", "label": "2.5", "title": "Optional printed title",
  "statement": "Source-faithful Markdown/LaTeX",
  "supporting_text": "Source proof, derivation, answer, hint, or solution",
  "supporting_text_role": "proof", "supporting_text_origin": "book_source",
  "source_pages": [17, 18], "source_pdf_page": 17,
  "source_spans": [{"source_span_id": "p17.b4-5.t12-49", "pdf_page": 17, "bbox": [18.7, 121.1, 343.3, 196.0]}],
  "uncertain_spans": [], "glossary_context": []
}
```

**Structured record** (`structured_statements.json`) adds:

| Field | Values |
|---|---|
| `origin` | `labeled_environment`, `unlabeled_environment`, `labeled_environment_atomized`, `unlabeled_environment_atomized`, `prose_derived` |
| `kind` | `definition`, `notation`, `theorem`, `lemma`, `proposition`, `corollary`, `claim`, `fact`, `conjecture`, `assumption`, `example`, `exercise` (plus `remark`, `algorithm` for environments) |
| `assertion_status` (prose claims) | `proved_in_text`, `stated_without_proof`, `cited_result`, `recalled_standard`, `left_to_reader`, `conjectured`, `heuristic`, `definitional` |
| verification `verdict` | `faithful`, `minor_issue`, `unfaithful`, `deterministic_verbatim` (no model rewrite), `unverified` |

`*_atomized` means one printed environment was split into several atomic statements; `uid` is then
`<id>#<n>`. Prose claims judged `unfaithful` go to `prose_claims_rejected.json`, not to the structured
layer.

Field-level schema: [`docs/reference/output-schema.md`](docs/reference/output-schema.md).

## Installation

Requires Python 3.9–3.13 and an OpenAI-compatible VLM endpoint (local or remote).

```bash
pip install -e .
# Development:
pip install -e '.[dev]'
pytest
```

The `parse-ocr` command-line script is installed in your environment.

## Quick start

```bash
export VLM_BASE_URL=http://127.0.0.1:3000/v1
export VLM_API_KEY=your-api-key

# 1. Create my_project/run.json and my_project/book_profile.json.
#    The quoted '${VLM_BASE_URL}' is stored literally and expanded when the config is loaded.
parse-ocr init book.pdf my_project --vlm-base-url '${VLM_BASE_URL}' --vlm-model gpt-5.5

# 2. Check the PDF, dependencies and credentials. --network also probes the endpoint.
parse-ocr preflight --config my_project/run.json --network

# 3. Render sample pages and ask the VLM for a draft profile.
#    Review output/inspection/book_profile.suggested.json, then copy what is correct into book_profile.json.
parse-ocr inspect --config my_project/run.json --auto

# 4. Run every stage, including validation.
parse-ocr direct run --config my_project/run.json
```

Results are in `my_project/output/dataset/`. Start with `environments.json`, `qa_pairs.json` and
`quality_report.json`.

Relative paths in `run.json` resolve against the directory that contains `run.json`, not the current
directory. The API key is read only from the environment variable named by `vlm.api_key_env` (default
`VLM_API_KEY`). It is never written to `run.json` or to any output file.

## Pipeline

`parse-ocr direct run` runs these stages in order. Each stage reads the previous stage's checkpoints
under `<output>/direct/` and skips work that is already cached, unless `--force-stage` is given.

| # | Stage | What it does | Main output |
|---|---|---|---|
| 1 | `glossary` | Reads notation and symbol pages | `direct/glossary/`, later `dataset/symbol_glossary.json` |
| 2 | `inventory` | Lists every printed object (kind, label, page range) in overlapping page windows | `direct/inventory/index.json` |
| 3 | `extract` | Transcribes each inventoried object from page images at extraction zoom | `direct/extraction/windows/` |
| 4 | `retry` | Runs `reconcile` and `coverage-audit`, then re-extracts missing objects, missing proofs/answers and detected gaps from padded page ranges | `direct/retry/`, `direct/coverage_audit/` |
| 5 | `build` | Merges, deduplicates and grounds rows on the text layer; writes the evidence layer and audit ledgers | `dataset/environments.json`, `qa_pairs.json`, ... |
| 6 | `verify` | Re-reads high-zoom source crops where the passes in `dataset/formula_conflicts.json` disagree, then rebuilds | `direct/source_verification/` |
| 7 | `relationships` | Explicit cross-references plus model-proposed implicit dependencies | `dataset/dependency_graph.json` |
| 8 | `prose-claims` | Runs only if `prose_claims.enabled` is true: mine → normalize → verify → repair → merge | `dataset/structured_statements.json` |
| 9 | `validate` | Deterministic checks on the dataset | `dataset/quality_report.json` |

`reconcile` and `coverage-audit` can also be run on their own. In `run` they are executed inside
`retry`.

```
 PDF ──render──► page images ──VLM──► inventory ──► extraction ──► retry (reconcile + coverage audit)
  │                                                                     │
  └──text layer (if any)──► locate / order / dedup ───────────────► build ──► verify ──► relationships
                                                                                            │
                                         validate ◄── prose-claims (optional) ◄─────────────┘
```

### Other routes

| Route | Command | Use when |
|---|---|---|
| MinerU | `parse-ocr mineru {mineru,segment,transcribe,build,validate,run}` | A MinerU export (local, cached or API) is available. Crops are re-read by the VLM. `--ids` limits `transcribe` to given objects. |
| Hybrid | `parse-ocr hybrid {segment,align,reconcile,build,refine,run}` | The direct route is stable and an independent TeX/XML/HTML export exists. The secondary text adds recall and boundaries; it never overwrites PDF-grounded mathematics. See [`docs/reference/hybrid-route.md`](docs/reference/hybrid-route.md). |

## Command reference

`parse-ocr <command> [args]` calls `parse_ocr.<module>.main()`. `python -m parse_ocr.<module>` does the
same. Every command except `init` takes `--config path/to/run.json`.

| Command | Module | Arguments | Writes | Exit code |
|---|---|---|---|---|
| `init` | `init_project` | `pdf project_dir [--book-id] [--output] [--vlm-base-url] [--vlm-model] [--vlm-key-env] [--route direct\|mineru] [--mineru-backend auto\|cache\|local\|api]` | `run.json`, `book_profile.json` | 0 |
| `preflight` | `preflight` | `[--network]` | `output/preflight.json` | 0 if ok, 2 if any check fails |
| `inspect` | `book_inspector` | `[--pages 12] [--zoom 1.6] [--auto]` | `output/inspection/manifest.json`; with `--auto` also `profile_suggestion.json`, `book_profile.suggested.json` | 0 |
| `direct` | `direct_pipeline` | `<stage> [--force-stage]` | see [Pipeline](#pipeline) | non-zero if a stage fails |
| `verify` | `verify_source_conflicts` | `[--force-stage] [--candidate-id ID ...]` | `direct/source_verification/{resolutions,canonical_resolutions,transcription_resolutions}.json` | non-zero if `dataset/formula_conflicts.json` is missing |
| `relationships` | `extract_relationships` | `[--force-stage] [--reuse-existing-graph]` | `dataset/dependency_graph.json` | non-zero on request errors when `relationships.fail_on_request_errors` is true (default). The previous graph is kept. |
| `prose-claims` | `prose_claims` | `{mine,normalize,verify,repair,merge,run} [--force-stage]` | `dataset/prose_claims*.json`, `structured_statements*.json` | non-zero if a stage fails |
| `validate` | `validate_dataset` | — | `dataset/quality_report.json` | 0 if ok, 2 if any error check fails |
| `mineru` | `mineru_pipeline` | `<stage> [--force-stage] [--ids ID ...]` | `output/mineru/`, `dataset/` | non-zero if a stage fails |
| `hybrid` | `hybrid_pipeline` | `<stage> [--force-stage]` | `output/hybrid/`, `dataset/` | non-zero if a stage fails |

`--reuse-existing-graph` rebinds saved implicit edges by stable source ID without calling the model.
`--candidate-id` limits source verification to conflicts that contain the given candidate.

### Dataset files

Besides the two layers above, `build` and later stages write audit files to `dataset/`:

| File | Content |
|---|---|
| `quality_report.json` | Result of `validate`: `ok`, failed checks with the offending items, warnings |
| `source_ledger.json`, `page_coverage_ledger.json` | Which inventoried objects were extracted, and on which pages |
| `source_grounding_report.json` | How each record was located on the text layer |
| `formula_conflicts.json` | Disagreements between transcription passes, with the verification result |
| `transcription_review_queue.json`, `visual_audit_queue.json` | Records that need a human look |
| `source_span_collisions.json`, `canonical_text_overlaps.json` | Records that claim the same text |
| `superseded_entries.json`, `context_only_entries.json`, `id_aliases.json` | Removed duplicates, context-only rows, ID renames |
| `dependency_graph.json` | Explicit and implicit dependency edges |
| `symbol_glossary.json` | Notation table attached to records as `glossary_context` |
| `run_summary.json`, `benchmark_metrics.json` | Counts and run metadata |

## Configuration

`run.json` holds every pipeline setting. `init` creates a working baseline. All relative paths resolve
against the directory that contains `run.json`. Environment variables in `vlm.base_url` (e.g.,
`"${VLM_BASE_URL}"`) are expanded when the config is loaded; unset variables raise an error.

Key sections:

| Section | What it controls |
|---|---|
| `vlm` | `base_url`, `model`, `api_key_env`, `timeout_seconds`, `request_attempts`, `segment_workers`, `vision_workers`, `double_pass` |
| `direct` | Inventory, extraction, coverage audit, source verification, retry: window sizes, overlaps, zoom, workers, page ranges |
| `relationships` | `enabled`, `implicit_dependencies`, `workers`, `request_attempts`, batch sizes, candidate limits, `fail_on_request_errors` |
| `prose_claims` | `enabled`, `span_grounding`, `anchors`, `subpart_policy` (`split`, or `auto` to leave sub-parts to the model), `granularity` (`atomize`, `record`, `atomize_merge_dependent`), `claim_tiers`, `window_pages`, `overlap_pages`, `workers`, `output_suffix` |
| `mineru` | `backend`, `api_base`, `api_key_env` (default `MINERU_API_KEY`), `model_version`, `cache_dir` |
| `pipeline` | MinerU route only: `segment_chunk_chars`, `segment_overlap_pages`, `crop_padding`, `crop_zoom` |
| `hybrid` | Hybrid route only; see [`docs/reference/hybrid-route.md`](docs/reference/hybrid-route.md) |

[`configs/run.example.json`](configs/run.example.json) lists every key with its default value.

`book_profile.json` holds every book-specific setting. Its main keys are:
- `kinds` and `kind_aliases`;
- `label_regex`;
- `question_section_headings` and `support_section_headings`;
- `visual_prompt_overlay`;
- `exclude_pdf_page_ranges` and `glossary_page_ranges`;
- `drop_object_ids` and `merge_object_groups`;
- `source_verified_*` overrides.

Full schema: [`docs/reference/profile-schema.md`](docs/reference/profile-schema.md).

**Recommended `prose_claims` settings** (from the ablation in
[`docs/experiments/ABLATION_RESULTS.md`](docs/experiments/ABLATION_RESULTS.md)):

```json
"prose_claims": {
  "enabled": true,
  "span_grounding": true,
  "anchors": false,
  "subpart_policy": "split",
  "granularity": "atomize",
  "claim_tiers": false
}
```

These defaults are in `configs/run.example.json`.

## Guarantees and non-guarantees

What the pipeline guarantees:

- `statement` and `supporting_text` in `environments.json` come from VLM reads of the rendered pages.
  Text-layer, MinerU or secondary-export text is used to locate, order and deduplicate records, never as
  a replacement for the transcription.
- When two passes disagree on a formula, `verify` re-reads a high-zoom crop and selects one of the
  existing candidates. It does not synthesize a new one.
- Proofs, hints and answers are attached only when they are printed in the book
  (`supporting_text_origin = "book_source"`). Missing support is left empty, not generated.
- `relationships` is transactional: a run with request errors fails and leaves the previous
  `dependency_graph.json` in place.
- `validate` and `preflight` exit with code 2 when a check fails, so they can gate scripts and CI.
- API keys are read only from environment variables.

What it does not guarantee:

- Complete recall. Inventory, coverage audit and retry reduce misses but cannot rule them out. Coverage
  numbers are measured against detector passes, not against a human reading of the book.
- Error-free transcription. Records the pipeline is unsure about are listed in
  `transcription_review_queue.json` and `visual_audit_queue.json`. Records outside these queues can still
  contain errors.
- A faithful `structured_statements.json`. `normalized_statement` is model-written, and its verdict is
  another model's judgment. Normalization can add hypotheses that are not printed. Use
  `source_statement` or `environments.json` when you need only the printed text.
- Text-layer grounding for scans with no text layer. Such records have no `source_spans`, and
  span-based deduplication does not apply.

## Evaluation

The harness lives in `bench/` (see [`bench/README.md`](bench/README.md)). It has seven cases. Each case
directory commits only `run.json` and `book_profile.json`. PDFs, gold files and outputs stay local,
because they contain copyrighted text.

| Tool | Measures |
|---|---|
| `bench/evaluate.py` | Against gold units: one-to-one recall and precision, word-span IoU on the text layer (boundary quality), kind accuracy |
| `bench/ablation_runner.py` | Runs the `prose_claims` variants in `VARIANTS`, each with its own cache and output suffix, and scores them |
| `bench/goldfree_eval.py report` | Without gold: hygiene (control characters, CJK garbage, leaked references/URLs, hallucinated prefixes, empty statements, proofs not found on the source pages) and printed-label coverage |
| `bench/goldfree_eval.py judge` | Blind pairwise comparison of two systems page by page against the page image, in random order |

The current gold units were built by models, not people. Repeat runs with the same settings differ by up
to about 0.1 in recall and F1, so effects smaller than about ±0.05 need repeated runs. Results and
analysis: [`docs/experiments/`](docs/experiments/).

## Repository layout

```
src/parse_ocr/
  cli.py                      parse-ocr entry point (command -> module map)
  common.py                   config loading, VLM client, JSON I/O
  init_project.py             init
  preflight.py                preflight
  book_inspector.py           inspect
  direct_pipeline.py          direct route (main route)
  source_grounding.py         text-layer grounding used by build
  text_spans.py               word-span location and IoU helpers
  verify_source_conflicts.py  verify
  extract_relationships.py    relationships
  prose_claims.py             prose-claims (structured layer)
  validate_dataset.py         validate
  mineru_pipeline.py          MinerU route
  hybrid_pipeline.py          hybrid route
configs/        run.example.json, book_profile.example.json
docs/reference/ schemas and per-stage design notes
docs/experiments/ ablation plan, results and reports
bench/          evaluation harness and per-case configs (PDFs, gold, outputs not committed)
skill/          agent skill wrapper describing the operating workflow
legacy/         earlier MinerU-based HDP QA/VQA scripts, kept for reference, not tested
tests/          unit tests (no network, no VLM calls)
```

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

The tests need no network access and no VLM. CI (`.github/workflows/tests.yml`) runs them on Python
3.9, 3.10, 3.12 and 3.13.

Conventions:
- A new command needs a module with a `main()` that parses `sys.argv`, and an entry in `cli.COMMANDS`.
  `tests/test_cli_config.py` checks that every registered module imports and has `main`.
- Model-calling stages write checkpoints under `<output>/` and reuse them unless `--force-stage` is
  given. Keep new stages restartable in the same way.
- Never commit PDFs, gold files, outputs, endpoint addresses or keys.

## Known limitations

- Scanned books with simplex tableaux (the Dantzig case) lose the tableaux.
- Normalization can add hypotheses that are not printed, and the verifier does not always catch them.
- The prose-claim miner under-reads exercise-dense pages: remarks, notation and recalled facts between
  exercises are often missed (11 missed units on hdp_ex).
- All gold sets are model-built. A human pass over `hdp_ex` and `perelman` gold is the next evaluation
  priority.
