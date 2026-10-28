#!/usr/bin/env python3
"""Initialize a portable project config and all-environment book profile."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from parse_ocr.common import write_json


KINDS = [
                "definition", "theorem", "lemma", "proposition", "corollary",
                "conjecture", "claim", "fact", "example", "exercise", "algorithm", "notation", "remark",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--book-id")
    parser.add_argument("--output", default="output", help="output path written into run.json")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:3000/v1")
    parser.add_argument("--vlm-model", default="gpt-5.5")
    parser.add_argument("--vlm-key-env", default="VLM_API_KEY")
    parser.add_argument("--route", choices=["direct", "mineru"], default="direct")
    parser.add_argument("--mineru-backend", choices=["auto", "cache", "local", "api"], default="auto")
    args = parser.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")
    project = args.project_dir.resolve()
    project.mkdir(parents=True, exist_ok=True)
    book_id = args.book_id or re.sub(r"[^a-z0-9]+", "-", pdf.stem.lower()).strip("-")
    profile_path = project / "book_profile.json"
    config_path = project / "run.json"

    if not profile_path.exists():
        write_json(profile_path, {
            "schema_version": 1,
            "book_id": book_id,
            "title": pdf.stem,
            "languages": ["en"],
            "include_unnumbered_named": True,
            "kinds": KINDS,
            "kind_aliases": {"problem": "exercise"},
            "label_regex": r"(?:[0-9]+|[A-Z])(?:\.[0-9]+)+",
            "structural_label_regex": "",
            "emit_structural_wrappers": False,
            "numbering_scope": "book",
            "question_section_headings": ["Exercises", "Problems"],
            "support_section_headings": ["Hints", "Answers", "Solutions"],
            "distant_support_alignment": True,
            "preserve_markers": [],
            "worked_example_policy": "split_when_source_has_setup_and_resolution",
            "figure_policy": "only_explicit_dependencies",
            "segmentation_prompt_overlay": "",
            "visual_prompt_overlay": "",
            "direct_conflict_similarity": 0.94,
            "strip_terminal_proof_marks": False,
            "exclude_pdf_page_ranges": [],
            "glossary_page_ranges": [],
            "source_verified_conflict_overrides": [],
            "merge_object_groups": [],
            "drop_object_ids": [],
            "drop_relationship_edge_ids": [],
            "inspection_notes": [],
            "expected_counts": {},
        })
    if not config_path.exists():
        write_json(config_path, {
            "schema_version": 1,
            "route": args.route,
            "pdf": str(pdf),
            "output": args.output,
            "profile": "book_profile.json",
            "vlm": {
                "base_url": args.vlm_base_url,
                "model": args.vlm_model,
                "api_key_env": args.vlm_key_env,
                "timeout_seconds": 1200,
                "request_attempts": 6,
                "segment_workers": 32,
                "vision_workers": 64,
                "double_pass": True,
            },
            "direct": {
                "inventory_window_pages": 10,
                "inventory_overlap_pages": 2,
                "inventory_zoom": 1.25,
                "inventory_workers": 32,
                "extraction_window_pages": 5,
                "extraction_overlap_pages": 1,
                "extraction_zoom": 2.4,
                "extraction_workers": 64,
                "source_grounding_enabled": True,
                "source_verification_enabled": True,
                "source_verification_zoom": 4.0,
                "source_verification_workers": 8,
                "source_verification_max_tokens": 8000,
                "coverage_audit_enabled": True,
                "coverage_audit_window_pages": 4,
                "coverage_audit_overlap_pages": 1,
                "coverage_audit_zoom": 2.0,
                "coverage_audit_workers": 32,
                "coverage_audit_page_ranges": [],
                "retry_page_padding": 1,
                "retry_max_pages": 8,
                "retry_zoom": 2.8,
                "retry_workers": 32,
                "glossary_zoom": 3.0,
                "glossary_workers": 4,
                "manual_retry_ids": []
            },
            "mineru": {
                "backend": args.mineru_backend,
                "api_base": "https://mineru.net/api/v4",
                "api_key_env": "MINERU_API_KEY",
                "model_version": "vlm",
                "max_pages_per_part": 200,
                "local_command": [],
                "cache_dir": "output",
            },
            "pipeline": {
                "segment_chunk_chars": 80000,
                "segment_overlap_pages": 2,
                "crop_padding": 35,
                "crop_zoom": 3.0,
            },
            "relationships": {
                "enabled": True,
                "implicit_dependencies": True,
                "workers": 32,
                "request_attempts": 4,
                "error_retry_rounds": 2,
                "error_retry_min_workers": 4,
                "source_batch_size": 10,
                "same_section_lookback": 30,
                "same_section_lookahead": 30,
                "cross_section_candidate_limit": 12,
                "fail_on_request_errors": True,
            },
        })
    print(json.dumps({"ok": True, "config": str(config_path), "profile": str(profile_path)}))


if __name__ == "__main__":
    main()
