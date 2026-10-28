# v3 bench

Seven cases: `vershynin`, `perelman`, `string_topology`, `nesterov`, `hdp_ex` (with gold) and
`dantzig`, `kochenderfer` (scans, gold-free only). Each case directory holds `run.json` and
`book_profile.json`; the PDF slices, gold files and outputs are not committed (copyrighted source text).

Expected local layout before running:

```
bench/pdfs/<case>.pdf
bench/gold/<case>.json            # gold units: {gid, page, kind, labeled, label, source_excerpt, ...}
refdata/mathdoc_parser_problem_cases_20260909/...   # baseline outputs for goldfree_eval (or set PARSE_OCR_REFDATA)
```

```bash
export VLM_BASE_URL=... VLM_API_KEY=...
parse-ocr direct run --config bench/vershynin/run.json        # evidence layer
python bench/ablation_runner.py run FINAL --cases vershynin   # prose-claims variant
python bench/ablation_runner.py score                         # -> bench/eval/ablation_results.json
python bench/evaluate.py bench/gold/vershynin.json bench/vershynin/output/dataset/structured_statements.json
python bench/goldfree_eval.py report                          # hygiene + label coverage
JUDGE_BASE_URL=... python bench/goldfree_eval.py judge --pair ours,mathdoc
```

Variants are defined in `ablation_runner.py:VARIANTS`; each uses its own checkpoint directory and output
suffix, so variants never overwrite each other. Results and analysis: `docs/experiments/`.
