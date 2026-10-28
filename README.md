# parse-ocr

Source-faithful extraction of mathematical objects from PDF books and OCR'd scans: definitions,
theorems, lemmas, propositions, corollaries, examples, exercises, remarks, algorithms, proofs, hints and
answers, exported as structured JSON and flattened QA/VQA records.

Two layers are produced and never mixed:

| Layer | File | Written by | Contract |
|---|---|---|---|
| Evidence | `dataset/environments.json` | VLM transcription, grounded on the PDF | verbatim source text, page/bbox provenance |
| Structured | `dataset/structured_statements.json` | `prose-claims` stage | one atomic unit per entry, self-contained `normalized_statement`, verified against the page |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Requires Python ≥ 3.9, PyMuPDF and requests. Any OpenAI-compatible endpoint that accepts image input works.

## Quick start

```bash
export VLM_BASE_URL=http://HOST:PORT/v1   # referenced as ${VLM_BASE_URL} in run.json
export VLM_API_KEY=...                    # read from the environment only, never written to files

parse-ocr init book.pdf projects/book --vlm-base-url '${VLM_BASE_URL}' --vlm-model gpt-5.5
parse-ocr inspect   --config projects/book/run.json      # render pages, then edit book_profile.json
parse-ocr preflight --config projects/book/run.json --network
parse-ocr direct run --config projects/book/run.json     # full direct pipeline
parse-ocr prose-claims run --config projects/book/run.json
```

`parse-ocr COMMAND` is equivalent to `python -m parse_ocr.<module>`; every command supports `--help`.
Stages are checkpointed and resumable. Use `--force-stage` to rerun only the stage you changed.

## Pipeline

```
PDF ──► inventory ──► extract ──► coverage-audit ──► reconcile/retry ──► build ──► verify ──► relationships ──► validate
          (objects,     (high-res     (independent       (targeted          (ground on     (crop        (refs, deps,
           no text)      windows)      omission scan)     re-reads)          text layer)    adjudication) graph)
                                                                                    │
                                                    environments.json ◄─────────────┘
                                                            │
             prose-claims: mine ──► sweep ──► normalize/atomize ──► verify ──► repair ──► merge
                                                            │
                                                structured_statements.json
```

- Page images are authoritative. The native text layer, when present, is only a positional scaffold for
  ordering, deduplication and span identity; OCR text never replaces VLM mathematics.
- Scans without a text layer fall back to the extractor's per-page order and are queued for visual audit.
- Every removed, merged or unresolved candidate stays in an audit artifact (`source_ledger.json`,
  `superseded_entries.json`, `visual_audit_queue.json`, ...).

## Commands and modules

| Command | Module | Purpose |
|---|---|---|
| `init` | `init_project` | create `run.json` + `book_profile.json` |
| `inspect` | `book_inspector` | render representative pages, optional draft profile |
| `preflight` | `preflight` | dependency, credential, endpoint check |
| `direct` | `direct_pipeline` | stages `glossary inventory extract coverage-audit reconcile retry build verify relationships prose-claims validate run` |
| `prose-claims` | `prose_claims` | stages `mine normalize verify repair merge run` |
| `verify` | `verify_source_conflicts` | blind 2-of-3 crop adjudication of formula and span conflicts |
| `relationships` | `extract_relationships` | cross-references, dependencies, cycle check |
| `validate` | `validate_dataset` | deterministic schema/anomaly audit (exit 2 on failure) |
| `hybrid` | `hybrid_pipeline` | fuse direct output with an independent TeX export |
| `mineru` | `mineru_pipeline` | legacy MinerU route, explicit opt-in only |

Library modules: `common` (config, HTTP with 429 backoff, chat call), `source_grounding` (span alignment,
order, canonicalization, conflict detection), `text_spans` (text-layer anchors, word-span IoU).

## Configuration

- `configs/run.example.json`: all run settings. Relative `pdf`/`output`/`profile` paths resolve against
  the config file; `vlm.base_url` may reference environment variables.
- `configs/book_profile.example.json`: per-book evidence (labels, kinds, proof boundaries, glossary pages,
  glyph overlays). See `docs/reference/profile-schema.md`.
- Recommended `prose_claims` defaults (from the ablation in `docs/experiments/ABLATION_RESULTS.md`):
  `span_grounding=true`, `anchors=false`, `subpart_policy=split`, `granularity=atomize`, `claim_tiers=false`.

## Repository layout

```
src/parse_ocr/     package: pipeline stages + shared libraries, unified CLI
tests/             offline unit tests (no network, synthetic PDFs)
configs/           example run.json and book_profile.json
docs/reference/    contracts: output schema, profile schema, grounding, auditing, routes, prompting
docs/experiments/  design review, ablation plan/results, bench report, next steps
bench/             v3 evaluation harness and per-case configs (PDFs, gold, outputs not committed)
skill/             agent skill wrapper (SKILL.md) describing the operating workflow
legacy/            earlier MinerU-based HDP QA/VQA scripts, kept for reference
```

## Evaluation

See `bench/README.md`. Two modes: scoring against model-built gold units (`evaluate.py`,
`ablation_runner.py score`), and gold-free checks (hygiene, printed-label coverage, blind pairwise page
judge) in `goldfree_eval.py`. Run-to-run noise is about ±0.05–0.1, so smaller effects need repeated runs.

## Known limitations

- Scanned books with simplex tableaux (Dantzig case) lose the tableaux.
- Normalization can add hypotheses that are not printed; the verifier does not always catch them.
- Coverage numbers are relative to detector passes, not to human gold.
