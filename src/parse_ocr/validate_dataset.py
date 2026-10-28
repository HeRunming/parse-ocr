#!/usr/bin/env python3
"""Deterministic schema, traceability, and anomaly audit for final datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from parse_ocr.common import load_json, resolve_config, write_json


ABSOLUTE_PATH = re.compile(
    r"(?:file://\S+|/(?:Users|home|volume)/[^\s)]+|"
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:Users|Documents|Downloads|Windows|Program Files)/[^\s)]*)"
)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return sorted(values)[round((len(values) - 1) * fraction)]


def normalized_statement(value: str) -> str:
    value = re.sub(r"[`*_#$\\{}\[\]()]", "", value.casefold())
    return re.sub(r"\s+", "", value)


def anaphoric_fragment(value: str) -> bool:
    text = re.sub(r"^[#*\s]+", "", value).strip()
    if re.match(r"it is (?:easy|hard|difficult|possible|impossible|clear|known)\s+to\b", text, flags=re.I):
        return False
    if len(text) >= 120:
        return False
    if re.match(
        r"It (?:was|is) (?:noted|shown|proved|defined|observed) in [A-Z]\d+ that\b",
        text, flags=re.I,
    ):
        return False
    return bool(re.search(
        r"^(?:are there others|are any of them|are there infinitely many of them|"
        r"it is conjectured that|he also calls|this|these|those|it\b|its\b|otherwise|"
        r"similarly|and\b|the above|the latter|the former|是否还有|上述|这些|它们)",
        text, flags=re.I,
    ))


def has_resolving_context(row: dict[str, Any]) -> bool:
    """Return true when a short anaphoric statement has a usable context chain."""
    return bool(
        row.get("context_before")
        or row.get("source_context")
        or row.get("source_context_host_id")
        or row.get("source_context_entry_ids")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(); config = resolve_config(args.config)
    output, profile = Path(config["output"]), load_json(Path(config["profile"]))
    validation = config.get("validation", {})
    dataset_path = output / "dataset" / "environments.json"
    if not dataset_path.exists():
        raise SystemExit(f"dataset missing: {dataset_path}")
    rows: list[dict[str, Any]] = load_json(dataset_path)
    ids = [row.get("id", "") for row in rows]
    kind_label = [(row.get("kind", ""), row.get("label", "")) for row in rows]
    lengths = [len(row.get("statement", "")) for row in rows]
    errors, warnings = [], []
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    duplicate_kind_labels = sorted([list(key) for key, count in Counter(kind_label).items() if count > 1])
    empty = [row.get("id") for row in rows if not str(row.get("statement", "")).strip()]
    malformed_ids = [row.get("id") for row in rows if row.get("id") != f"{row.get('kind')}:{row.get('label')}"]
    unexpected_kinds = sorted(set(row.get("kind", "") for row in rows) - set(profile["kinds"]))
    structural_pattern_text = str(profile.get("structural_label_regex", "")).strip()
    structural_pattern = re.compile(structural_pattern_text) if structural_pattern_text else None
    leaked_structural_wrappers = [
        row.get("id") for row in rows
        if structural_pattern and structural_pattern.fullmatch(row.get("label", ""))
        and row.get("record_type", "object") == "section"
        and not bool(profile.get("emit_structural_wrappers", False))
    ]
    bad_roles = [row.get("id") for row in rows if row.get("supporting_text_role", "") not in {
        "", "proof", "derivation", "explanation", "answer", "hint", "solution", "hint_or_solution"
    }]
    missing_sources = [row.get("id") for row in rows if not row.get("statement_sources")]
    path_leaks = [row.get("id") for row in rows if ABSOLUTE_PATH.search(
        row.get("statement", "") + "\n" + row.get("supporting_text", ""))]
    nonportable_sources = [row.get("id") for row in rows if any(
        source.get("image_source") or ABSOLUTE_PATH.search(json.dumps(source, ensure_ascii=False))
        for source in row.get("statement_sources", []) + row.get("supporting_sources", [])
    )]
    duplicate_entry_ids = sorted(key for key, count in Counter(
        row.get("entry_id", "") for row in rows if row.get("entry_id")
    ).items() if count > 1)
    missing_entry_ids = [row.get("id") for row in rows if not row.get("entry_id")]
    fragment_candidates = [
        row.get("id") for row in rows
        if anaphoric_fragment(row.get("statement", "")) and not has_resolving_context(row)
    ]
    resolved_fragment_source_ids = set()
    for row in rows:
        source_ids = set(row.get("merged_source_ids", [])) | {row.get("id")}
        if (
            not anaphoric_fragment(row.get("statement", ""))
            or has_resolving_context(row)
        ):
            resolved_fragment_source_ids.update(source_id for source_id in source_ids if source_id)
    source_position_missing = [row.get("id") for row in rows if not row.get("source_position")]
    missing_source_identity = [row.get("id") for row in rows if not all(
        row.get(field) for field in ("source_span_id", "canonical_source_id", "legacy_id")
    )]
    malformed_source_spans = []
    incomplete_cross_page_provenance = []
    malformed_canonical_roles = []
    for row in rows:
        spans = row.get("source_spans")
        span_ids = {
            span.get("source_span_id") for span in spans or []
            if isinstance(span, dict) and span.get("source_span_id")
        }
        if not isinstance(spans, list) or not spans or row.get("source_span_id") not in span_ids:
            malformed_source_spans.append(row.get("id"))
        source_pages = {
            int(source.get("pdf_page")) for source in row.get("statement_sources", [])
            if source.get("pdf_page")
        }
        span_pages = {
            int(span.get("pdf_page")) for span in spans or []
            if isinstance(span, dict) and span.get("pdf_page")
        }
        if not source_pages <= span_pages:
            incomplete_cross_page_provenance.append({
                "id": row.get("id"), "missing_pages": sorted(source_pages - span_pages),
            })
        if row.get("canonical_role") != "canonical":
            malformed_canonical_roles.append({
                "id": row.get("id"), "canonical_role": row.get("canonical_role"),
            })
    duplicate_canonical_source_ids = sorted(key for key, count in Counter(
        row.get("canonical_source_id", "") for row in rows if row.get("canonical_source_id")
    ).items() if count > 1)
    confidence_dimensions = (
        "localization", "transcription", "segmentation_atomicity", "deduplication"
    )
    malformed_confidence = [row.get("id") for row in rows if (
        not isinstance(row.get("confidence"), dict)
        or any(row.get("confidence", {}).get(dimension) not in {
            "high", "medium", "low", "unlocated"
        } for dimension in confidence_dimensions)
    )]
    allowed_verification_status = {
        "human_pdf_verified", "machine_source_verified",
        "unverified_risk_screened", "verification_required",
    }
    malformed_verification_status = [
        row.get("id") for row in rows
        if row.get("verification_status") not in allowed_verification_status
    ]
    if malformed_verification_status:
        errors.append({
            "check": "missing or malformed content verification status",
            "items": malformed_verification_status,
        })
    legacy_label_page_mismatches = [
        row.get("id") for row in rows if row.get("legacy_label_page_mismatch")
    ]
    low_source_positions = [
        row.get("id") for row in rows
        if row.get("grounding_confidence") in {"low", "unlocated"}
    ]
    source_order_keys = [tuple(row.get("source_order_key", [])) for row in rows]
    source_order_inversions = [
        {"left": rows[index - 1].get("id"), "right": rows[index].get("id")}
        for index in range(1, len(rows))
        if source_order_keys[index - 1] and source_order_keys[index]
        and source_order_keys[index - 1] > source_order_keys[index]
    ]
    semantic_exact_groups = []
    exact_index: dict[tuple[int, str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for row in rows:
        statement = row.get("statement", "")
        # TeX control sequences are case-sensitive: \omega and \Omega are
        # different mathematical objects even though prose-level formatting
        # normalization intentionally case-folds the surrounding sentence.
        math_commands = tuple(re.findall(r"\\[A-Za-z]+", statement))
        key = (int(row.get("source_pdf_page") or 0), normalized_statement(statement), math_commands)
        exact_index.setdefault(key, []).append(row)
    for (page, fingerprint, _math_commands), values in exact_index.items():
        if fingerprint and len(values) > 1:
            semantic_exact_groups.append({"pdf_page": page, "ids": [row.get("id") for row in values]})
    odd_math_delimiters = [row.get("id") for row in rows if len(re.findall(
        r"(?<!\\)\$", row.get("statement", "") + "\n" + row.get("supporting_text", "")
    )) % 2]
    nonstandard_conjecture_markers = [
        row.get("id") for row in rows if "¿" in row.get("statement", "")
    ]
    containment_pairs = []
    explained_containment_pairs = []
    by_section: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_section.setdefault(row.get("section_id", ""), []).append(row)
    for section_rows in by_section.values():
        normalized = [(row, normalized_statement(row.get("statement", ""))) for row in section_rows]
        for index, (left, left_text) in enumerate(normalized):
            if len(left_text) < 40:
                continue
            for right, right_text in normalized[index + 1:]:
                if len(right_text) < 40:
                    continue
                pair = None
                contained = host = None
                if left_text in right_text:
                    contained, host = left, right
                elif right_text in left_text:
                    contained, host = right, left
                if contained is not None:
                    pair = {"contained": contained.get("id"), "host": host.get("id")}
                    same_source_group = (
                        contained.get("source_context_group_id")
                        and contained.get("source_context_group_id") == host.get("source_context_group_id")
                    )
                    if same_source_group or contained.get("source_context"):
                        explained_containment_pairs.append(pair)
                    else:
                        containment_pairs.append(pair)
    raw_path = output / "segmentation" / "environments_raw.json"
    missing_final, unexpected_final = [], []
    if raw_path.exists():
        raw_ids = {row.get("id", "") for row in load_json(raw_path)}
        final_ids = set(ids)
        missing_final = sorted(raw_ids - final_ids)
        unexpected_final = sorted(final_ids - raw_ids)
    missing_assets = []
    for row in rows:
        for asset in row.get("assets", []):
            if not (output / "dataset" / asset.get("image", "")).is_file():
                missing_assets.append({"id": row.get("id"), "image": asset.get("image")})
    for label, values in (
        ("empty dataset", ["no rows"] if not rows else []),
        ("duplicate ids", duplicate_ids), ("duplicate kind+label", duplicate_kind_labels),
        ("malformed ids", malformed_ids), ("unexpected kinds", unexpected_kinds),
        ("empty statements", empty), ("bad roles", bad_roles), ("absolute path leaks", path_leaks),
        ("nonportable source metadata", nonportable_sources), ("missing assets", missing_assets),
        ("structural wrappers leaked into object dataset", leaked_structural_wrappers),
        ("duplicate entry ids", duplicate_entry_ids), ("missing entry ids", missing_entry_ids),
        ("missing source identity fields", missing_source_identity),
        ("malformed multi-span source provenance", malformed_source_spans),
        ("incomplete cross-page source provenance", incomplete_cross_page_provenance),
        ("non-canonical records leaked into canonical environment dataset", malformed_canonical_roles),
        ("duplicate canonical source ids", duplicate_canonical_source_ids),
        ("malformed decomposed confidence", malformed_confidence),
        ("raw records missing from final dataset", missing_final),
        ("unexpected final records", unexpected_final),
    ):
        if values: errors.append({"check": label, "items": values})
    if missing_sources: warnings.append({"check": "missing statement source trace", "items": missing_sources})
    if source_position_missing:
        warnings.append({"check": "missing deterministic source position", "items": source_position_missing})
    if low_source_positions:
        warnings.append({"check": "low-confidence native-text source alignment", "items": low_source_positions})
    if source_order_inversions:
        errors.append({"check": "source-order inversions", "items": source_order_inversions})
    if semantic_exact_groups:
        errors.append({"check": "same-page formatting-equivalent duplicates", "items": semantic_exact_groups})
    if fragment_candidates:
        warnings.append({"check": "possibly non-self-contained fragments", "items": fragment_candidates})
    if containment_pairs:
        warnings.append({"check": "unexplained same-section contained statements", "items": containment_pairs})
    if odd_math_delimiters:
        warnings.append({"check": "odd unescaped dollar delimiters", "items": odd_math_delimiters})
    if nonstandard_conjecture_markers:
        warnings.append({
            "check": "source conjecture markers require render-safe view",
            "items": nonstandard_conjecture_markers,
        })
    if legacy_label_page_mismatches:
        warnings.append({
            "check": "legacy synthetic label page differs from physical source page; use source_pdf_page/source_span_id",
            "items": legacy_label_page_mismatches,
        })
    if any(row.get("uncertain_spans") for row in rows):
        warnings.append({"check": "uncertain spans", "items": [row["id"] for row in rows if row.get("uncertain_spans")]})
    counts = dict(sorted(Counter(row.get("kind", "") for row in rows).items()))
    missing_kinds = [kind for kind in profile["kinds"] if counts.get(kind, 0) == 0]
    if missing_kinds:
        warnings.append({"check": "configured kinds with zero detections", "items": missing_kinds})
    expected = profile.get("expected_counts", {})
    count_mismatches = {kind: {"expected": count, "actual": counts.get(kind, 0)}
                        for kind, count in expected.items() if counts.get(kind, 0) != count}
    if count_mismatches:
        warnings.append({"check": "expected count mismatch", "items": count_mismatches})
    direct_retry = output / "direct" / "retry" / "summary.json"
    if direct_retry.is_file():
        retry = load_json(direct_retry)
        unresolved = {
            "missing_objects": retry.get("remaining_missing_objects", []),
            "missing_support": retry.get("remaining_missing_support", []),
            "incomplete": retry.get("remaining_incomplete", []),
            "context_fragments": retry.get("remaining_context_fragments", []),
        }
        unresolved["context_fragments"] = [
            row for row in unresolved["context_fragments"]
            if row.get("id") not in resolved_fragment_source_ids
        ]
        retry_conflicts = retry.get("remaining_conflicts", [])
        if validation.get("fail_on_conflicts", True):
            unresolved["conflicts"] = retry_conflicts
        elif retry_conflicts:
            warnings.append({"check": "direct route candidate conflicts", "items": retry_conflicts})
        unresolved = {key: value for key, value in unresolved.items() if value}
        if unresolved:
            errors.append({"check": "direct route unresolved after retry", "items": unresolved})
    direct_reconciliation = output / "direct" / "reconciliation" / "report.json"
    if direct_reconciliation.is_file():
        reconciliation = load_json(direct_reconciliation)
        if reconciliation.get("request_errors"):
            target = errors if validation.get("fail_on_request_errors", True) else warnings
            target.append({"check": "direct route request errors", "items": reconciliation["request_errors"]})
        unresolved_reconciliation = {
            key: reconciliation.get(key, [])
            for key in ("missing_objects", "missing_support", "incomplete", "context_fragments")
            if reconciliation.get(key)
        }
        if "context_fragments" in unresolved_reconciliation:
            unresolved_reconciliation["context_fragments"] = [
                row for row in unresolved_reconciliation["context_fragments"]
                if row.get("id") not in resolved_fragment_source_ids
            ]
            if not unresolved_reconciliation["context_fragments"]:
                del unresolved_reconciliation["context_fragments"]
        reconciliation_conflicts = reconciliation.get("conflicts", [])
        if validation.get("fail_on_conflicts", True) and reconciliation_conflicts:
            unresolved_reconciliation["conflicts"] = reconciliation_conflicts
        if unresolved_reconciliation:
            errors.append({
                "check": "direct reconciliation still has unresolved targets",
                "items": unresolved_reconciliation,
            })
    coverage_audit_path = output / "direct" / "coverage_audit" / "summary.json"
    coverage_audit: dict[str, Any] = {"enabled": False, "windows": 0, "added": [], "errors": []}
    if coverage_audit_path.is_file():
        coverage_audit = load_json(coverage_audit_path)
    if not coverage_audit.get("enabled"):
        warnings.append({"check": "independent visual coverage audit disabled", "items": []})
    if coverage_audit.get("errors"):
        target = errors if validation.get("fail_on_coverage_audit_errors", True) else warnings
        target.append({"check": "coverage audit request errors", "items": coverage_audit["errors"]})
    low, high = percentile(lengths, 0.02), percentile(lengths, 0.98)
    anomalies = [{"id": row["id"], "statement_chars": len(row["statement"])} for row in rows
                 if len(row["statement"]) <= low or len(row["statement"]) >= high]
    distant = []
    for row in rows:
        statement_pages = [x.get("pdf_page") for x in row.get("statement_sources", []) if x.get("pdf_page")]
        support_pages = [x.get("pdf_page") for x in row.get("supporting_sources", []) if x.get("pdf_page")]
        if statement_pages and support_pages:
            distance = min(abs(a-b) for a in statement_pages for b in support_pages)
            if distance >= 5: distant.append({"id": row["id"], "page_distance": distance})
    sections_path = output / "dataset" / "sections.json"
    sections = load_json(sections_path) if sections_path.is_file() else []
    section_sequence = []
    for row in rows:
        if row.get("section_id") and (not section_sequence or section_sequence[-1] != row.get("section_id")):
            section_sequence.append(row.get("section_id"))
    repeated_section_runs = sorted(section for section, count in Counter(section_sequence).items() if count > 1)
    if repeated_section_runs:
        errors.append({"check": "non-contiguous section runs in source order", "items": repeated_section_runs})
    relationships_cfg = config.get("relationships", {})
    relationships_enabled = bool(relationships_cfg.get("enabled", True))
    graph_path = output / "dataset" / "dependency_graph.json"
    graph = load_json(graph_path) if graph_path.is_file() else {}
    entry_index = {row.get("entry_id", ""): row for row in rows}
    section_ids = {section.get("section_id", "") for section in sections}
    missing_relationship_fields = [
        row.get("id") for row in rows
        if (
            not isinstance(row.get("cross_references"), list)
            or not isinstance(row.get("dependencies"), list)
            or not isinstance(row.get("semantic_relations"), list)
        )
    ]
    malformed_dependency_edges = []
    dangling_dependency_edges = []
    unsupported_dependency_evidence = []
    malformed_semantic_edges = []
    dangling_semantic_edges = []
    unsupported_semantic_evidence = []
    malformed_cross_reference_edges = []
    dangling_resolved_cross_references = []
    unsupported_cross_reference_evidence = []
    dependency_edge_ids, semantic_edge_ids, cross_reference_edge_ids = [], [], []
    for row in rows:
        source_entry = row.get("entry_id", "")
        for edge in row.get("dependencies", []):
            dependency_edge_ids.append(edge.get("edge_id", ""))
            if not edge.get("edge_id") or not edge.get("target_entry_id") or not edge.get("relation"):
                malformed_dependency_edges.append({"source": source_entry, "edge": edge})
                continue
            target_entry = edge.get("target_entry_id", "")
            target = entry_index.get(target_entry)
            if target is None or target.get("id") != edge.get("target_id"):
                dangling_dependency_edges.append({"source": source_entry, "edge": edge})
            elif target.get("section_id", "") != edge.get("target_section_id", ""):
                dangling_dependency_edges.append({"source": source_entry, "edge": edge})
            evidence = re.sub(r"\s+", " ", str(edge.get("evidence", ""))).strip().casefold()
            source_text = re.sub(
                r"\s+", " ", row.get("statement", "") + "\n" + row.get("supporting_text", "")
            ).strip().casefold()
            if not evidence or evidence not in source_text:
                unsupported_dependency_evidence.append({"source": source_entry, "edge": edge})
            target_evidence = re.sub(
                r"\s+", " ", str(edge.get("target_evidence", ""))
            ).strip().casefold()
            target_text = "" if target is None else re.sub(
                r"\s+", " ", target.get("statement", "") + "\n" + target.get("supporting_text", "")
            ).strip().casefold()
            if not target_evidence or target_evidence not in target_text:
                unsupported_dependency_evidence.append({"source": source_entry, "edge": edge})
        for edge in row.get("semantic_relations", []):
            semantic_edge_ids.append(edge.get("edge_id", ""))
            if not edge.get("edge_id") or not edge.get("target_entry_id") or not edge.get("relation"):
                malformed_semantic_edges.append({"source": source_entry, "edge": edge})
                continue
            target = entry_index.get(edge.get("target_entry_id", ""))
            if (
                target is None or target.get("id") != edge.get("target_id")
                or target.get("section_id", "") != edge.get("target_section_id", "")
            ):
                dangling_semantic_edges.append({"source": source_entry, "edge": edge})
            source_text = re.sub(
                r"\s+", " ", row.get("statement", "") + "\n" + row.get("supporting_text", "")
            ).strip().casefold()
            target_text = "" if target is None else re.sub(
                r"\s+", " ", target.get("statement", "") + "\n" + target.get("supporting_text", "")
            ).strip().casefold()
            evidence = re.sub(r"\s+", " ", str(edge.get("evidence", ""))).strip().casefold()
            target_evidence = re.sub(
                r"\s+", " ", str(edge.get("target_evidence", ""))
            ).strip().casefold()
            if (
                not evidence or evidence not in source_text
                or not target_evidence or target_evidence not in target_text
            ):
                unsupported_semantic_evidence.append({"source": source_entry, "edge": edge})
        for edge in row.get("cross_references", []):
            cross_reference_edge_ids.append(edge.get("edge_id", ""))
            if not edge.get("edge_id") or not edge.get("target_type") or not edge.get("evidence"):
                malformed_cross_reference_edges.append({"source": source_entry, "edge": edge})
                continue
            if edge.get("resolved"):
                if edge.get("target_type") == "section" and edge.get("target_section_id") not in section_ids:
                    dangling_resolved_cross_references.append({"source": source_entry, "edge": edge})
                if edge.get("target_type") == "entry" and edge.get("target_entry_id") not in entry_index:
                    dangling_resolved_cross_references.append({"source": source_entry, "edge": edge})
            source_field = edge.get("source_field", "")
            field_text = str(row.get(source_field, ""))
            evidence = re.sub(r"\s+", " ", str(edge.get("evidence", ""))).strip().casefold()
            field_text = re.sub(r"\s+", " ", field_text).strip().casefold()
            if source_field not in {"statement", "supporting_text"} or not evidence or evidence not in field_text:
                unsupported_cross_reference_evidence.append({"source": source_entry, "edge": edge})
    duplicate_dependency_edge_ids = sorted(
        edge_id for edge_id, count in Counter(dependency_edge_ids).items() if edge_id and count > 1
    )
    duplicate_cross_reference_edge_ids = sorted(
        edge_id for edge_id, count in Counter(cross_reference_edge_ids).items() if edge_id and count > 1
    )
    duplicate_semantic_edge_ids = sorted(
        edge_id for edge_id, count in Counter(semantic_edge_ids).items() if edge_id and count > 1
    )
    relationship_request_errors = graph.get("request_errors", []) if graph else []
    graph_cycles = graph.get("cycles", []) if graph else []
    graph_dependency_count = len(graph.get("dependency_edges", [])) if graph else 0
    graph_cross_reference_count = len(graph.get("cross_reference_edges", [])) if graph else 0
    graph_semantic_count = len(graph.get("semantic_relation_edges", [])) if graph else 0
    graph_context_group_ids = {
        node.get("group_id", "") for node in graph.get("nodes", [])
        if node.get("node_type") == "source_context"
    }
    graph_context_memberships = {
        (item.get("group_id", ""), item.get("entry_id", ""))
        for item in graph.get("source_context_memberships", [])
    }
    dangling_context_memberships = sorted(
        [group_id, entry_id] for group_id, entry_id in graph_context_memberships
        if group_id not in graph_context_group_ids or entry_id not in entry_index
    )
    expected_context_memberships = {
        (row.get("source_context_group_id", ""), row.get("entry_id", ""))
        for row in rows if row.get("source_context_group_id")
    }
    missing_context_memberships = sorted(
        [group_id, entry_id]
        for group_id, entry_id in expected_context_memberships - graph_context_memberships
    )
    row_dependency_count = sum(len(row.get("dependencies", [])) for row in rows)
    row_cross_reference_count = sum(len(row.get("cross_references", [])) for row in rows)
    row_semantic_count = sum(len(row.get("semantic_relations", [])) for row in rows)
    formalization_order = graph.get("formalization_order", []) if graph else []
    bad_formalization_order = bool(graph) and (
        len(formalization_order) != len(rows)
        or len(set(formalization_order)) != len(formalization_order)
        or set(formalization_order) != set(entry_index)
    ) and not graph_cycles
    relationship_errors = []
    if relationships_enabled:
        if not graph:
            relationship_errors.append("dependency_graph.json missing")
        if missing_relationship_fields:
            relationship_errors.append({"missing relationship fields": missing_relationship_fields})
        for label, values in (
            ("malformed dependency edges", malformed_dependency_edges),
            ("dangling dependency edges", dangling_dependency_edges),
            ("dependency evidence absent from source", unsupported_dependency_evidence),
            ("malformed semantic-relation edges", malformed_semantic_edges),
            ("dangling semantic-relation edges", dangling_semantic_edges),
            ("semantic-relation evidence absent from source/target", unsupported_semantic_evidence),
            ("malformed cross-reference edges", malformed_cross_reference_edges),
            ("dangling resolved cross-references", dangling_resolved_cross_references),
            ("cross-reference evidence absent from source field", unsupported_cross_reference_evidence),
            ("duplicate dependency edge ids", duplicate_dependency_edge_ids),
            ("duplicate cross-reference edge ids", duplicate_cross_reference_edge_ids),
            ("duplicate semantic-relation edge ids", duplicate_semantic_edge_ids),
            ("dangling source-context memberships", dangling_context_memberships),
            ("missing source-context memberships", missing_context_memberships),
            ("dependency graph cycles", graph_cycles),
        ):
            if values:
                relationship_errors.append({label: values})
        if graph and graph_dependency_count != row_dependency_count:
            relationship_errors.append({"dependency graph count mismatch": {
                "graph": graph_dependency_count, "rows": row_dependency_count,
            }})
        if graph and graph_cross_reference_count != row_cross_reference_count:
            relationship_errors.append({"cross-reference graph count mismatch": {
                "graph": graph_cross_reference_count, "rows": row_cross_reference_count,
            }})
        if graph and graph_semantic_count != row_semantic_count:
            relationship_errors.append({"semantic-relation graph count mismatch": {
                "graph": graph_semantic_count, "rows": row_semantic_count,
            }})
        if bad_formalization_order:
            relationship_errors.append({"invalid formalization order": {
                "expected_entries": len(rows), "actual_entries": len(formalization_order),
            }})
        if relationship_request_errors and relationships_cfg.get("fail_on_request_errors", True):
            relationship_errors.append({"relationship request errors": relationship_request_errors})
    if relationship_errors:
        errors.append({"check": "relationship graph integrity", "items": relationship_errors})
    elif relationship_request_errors:
        warnings.append({"check": "relationship request errors", "items": relationship_request_errors})
    unresolved_cross_refs = [
        {"source": row.get("entry_id"), "edge": edge}
        for row in rows for edge in row.get("cross_references", []) if not edge.get("resolved")
    ]
    if unresolved_cross_refs:
        warnings.append({"check": "unresolved explicit cross-references", "items": unresolved_cross_refs})
    formula_conflicts_path = output / "dataset" / "formula_conflicts.json"
    formula_conflicts = load_json(formula_conflicts_path) if formula_conflicts_path.is_file() else []
    unresolved_formula_conflicts = [
        conflict for conflict in formula_conflicts
        if conflict.get("hard_conflict") and conflict.get("status") != "resolved"
    ]
    if unresolved_formula_conflicts:
        target = errors if validation.get("fail_on_formula_conflicts", True) else warnings
        target.append({
            "check": "unresolved semantic-critical same-span formula conflicts",
            "items": unresolved_formula_conflicts,
        })
    span_collision_path = output / "dataset" / "source_span_collisions.json"
    span_collisions = load_json(span_collision_path) if span_collision_path.is_file() else []
    unresolved_span_collisions = [
        conflict for conflict in span_collisions if conflict.get("status") != "resolved"
    ]
    unresolved_hard_span_collisions = [
        conflict for conflict in unresolved_span_collisions if conflict.get("hard_conflict")
    ]
    row_by_id = {row.get("id"): row for row in rows}
    unflagged_exact_span_collisions = []
    for conflict in unresolved_span_collisions:
        if not conflict.get("exact_source_span"):
            continue
        high = [
            candidate_id for candidate_id in conflict.get("candidate_ids", [])
            if row_by_id.get(candidate_id, {}).get("confidence", {}).get("deduplication") == "high"
        ]
        if high:
            unflagged_exact_span_collisions.append({
                "collision_set_id": conflict.get("collision_set_id"), "high_ids": high,
            })
    if unresolved_hard_span_collisions:
        target = errors if validation.get("fail_on_source_span_conflicts", True) else warnings
        target.append({
            "check": "unresolved source-span semantic-critical conflicts",
            "items": unresolved_hard_span_collisions,
        })
    nonhard_span_collisions = [
        conflict for conflict in unresolved_span_collisions if not conflict.get("hard_conflict")
    ]
    if nonhard_span_collisions:
        warnings.append({
            "check": "source-span canonical-role decisions awaiting verification",
            "items": nonhard_span_collisions,
        })
    if unflagged_exact_span_collisions:
        errors.append({
            "check": "unresolved exact source-span collision incorrectly marked deduplication high",
            "items": unflagged_exact_span_collisions,
        })
    canonical_overlap_path = output / "dataset" / "canonical_text_overlaps.json"
    canonical_overlaps = load_json(canonical_overlap_path) if canonical_overlap_path.is_file() else []
    unresolved_canonical_overlaps = [
        item for item in canonical_overlaps if item.get("status") != "resolved"
    ]
    if unresolved_canonical_overlaps:
        target = errors if validation.get("fail_on_canonical_text_overlaps", True) else warnings
        target.append({
            "check": "canonical records retain overlapping source text",
            "items": unresolved_canonical_overlaps,
        })
    grounding_report_path = output / "dataset" / "source_grounding_report.json"
    grounding_report = load_json(grounding_report_path) if grounding_report_path.is_file() else {}
    section_anchor_order_issues = grounding_report.get("section_anchor_order_issues", [])
    if section_anchor_order_issues:
        errors.append({
            "check": "localized section headings reverse structural book order",
            "items": section_anchor_order_issues,
        })
    ledger_path = output / "dataset" / "source_ledger.json"
    source_ledger = load_json(ledger_path) if ledger_path.is_file() else {}
    ledger_unresolved = source_ledger.get("unresolved", []) if source_ledger else []
    if not source_ledger:
        errors.append({"check": "source content ledger missing", "items": [str(ledger_path)]})
    elif ledger_unresolved:
        target = errors if validation.get("fail_on_ledger_unresolved", True) else warnings
        target.append({"check": "source ledger unresolved detections", "items": ledger_unresolved})
    detected_span_unresolved = source_ledger.get("detected_span_unresolved", []) if source_ledger else []
    if detected_span_unresolved:
        target = errors if validation.get("fail_on_detected_span_unresolved", False) else warnings
        target.append({
            "check": "independent detector spans not mapped to extraction or ignore decision",
            "items": detected_span_unresolved,
        })
    page_ledger_path = output / "dataset" / "page_coverage_ledger.json"
    page_ledger = load_json(page_ledger_path) if page_ledger_path.is_file() else {}
    if not page_ledger:
        errors.append({"check": "physical-page coverage ledger missing", "items": [str(page_ledger_path)]})
    elif page_ledger.get("pages_requiring_review"):
        warnings.append({
            "check": "physical pages require completeness review",
            "items": page_ledger.get("pages_requiring_review", []),
        })
    transcription_queue_path = output / "dataset" / "transcription_review_queue.json"
    transcription_resolution_path = output / "direct" / "source_verification" / "transcription_resolutions.json"
    transcription_queue = load_json(transcription_queue_path) if transcription_queue_path.is_file() else []
    transcription_resolutions = (
        load_json(transcription_resolution_path) if transcription_resolution_path.is_file() else []
    )
    resolved_or_attempted_transcription_ids = {
        item.get("selected_id") for item in transcription_resolutions if item.get("selected_id")
    }
    unattempted_transcription_ids = [
        item.get("id") for item in transcription_queue
        if item.get("id") not in resolved_or_attempted_transcription_ids
    ]
    if unattempted_transcription_ids:
        errors.append({
            "check": "transcription review item has no saved verifier attempt",
            "items": unattempted_transcription_ids,
        })
    false_high_structured = []
    for row in rows:
        if not row.get("structured_verification_risk_reasons"):
            continue
        complete = any(
            verification.get("human_pdf_visual_verification") is True
            or (
                verification.get("structured_verification", {}).get("complete")
                and verification.get("independent_observer", {}).get("complete")
                and {
                    str(item.get("id")) for item in verification.get("checks", []) if item.get("id")
                } == set(row.get("structured_verification_checklist_ids", []))
            )
            for verification in (
                row.get("source_verification") or {}, row.get("transcription_verification") or {},
            )
        ) or row.get("verification_status") == "human_pdf_verified"
        if row.get("confidence", {}).get("transcription") == "high" and not complete:
            false_high_structured.append(row.get("id"))
    if false_high_structured:
        errors.append({
            "check": "structured high-risk entry marked transcription high without exhaustive postcheck",
            "items": false_high_structured,
        })
    covered_pages = {page for row in rows for page in row.get("source_pages", []) if isinstance(page, int)}
    semantic_page_holes = []
    section_page_total = 0
    for section in sections:
        pages = [int(page) for page in section.get("source_pages", [])]
        section_page_total += len(pages)
        missing = [page for page in pages if page not in covered_pages]
        if missing:
            semantic_page_holes.append({"section_id": section.get("section_id", ""), "pages": missing})
    if semantic_page_holes:
        warnings.append({
            "check": "structural section pages without any extracted atomic object",
            "items": semantic_page_holes,
        })
    release_blockers = []
    if page_ledger.get("pages_requiring_review"):
        release_blockers.append("physical_pages_require_completeness_review")
    verification_required = [
        row.get("id") for row in rows if row.get("verification_status") == "verification_required"
    ]
    if verification_required:
        release_blockers.append("structured_or_atomic_content_requires_source_verification")
    if unresolved_canonical_overlaps:
        release_blockers.append("canonical_text_overlaps_unresolved")
    report = {
        "ok": not errors, "dataset": str(dataset_path), "environments": len(rows),
        "release_ready": not errors and not release_blockers,
        "release_blockers": release_blockers,
        "counts_by_kind": counts, "with_supporting_text": sum(bool(x.get("supporting_text")) for x in rows),
        "errors": errors, "warnings": warnings,
        "sampling": {
            "length_anomalies": anomalies,
            "distant_support": sorted(distant, key=lambda x: -x["page_distance"]),
            "one_per_kind": [next(row["id"] for row in rows if row["kind"] == kind) for kind in counts],
        },
        "statement_length": {"p02": low, "p50": percentile(lengths, .5), "p98": high},
        "semantic_coverage": {
            "claim": "inventory-plus-independent-audit object recall; not proof of full source-text coverage",
            "structural_sections": len(sections),
            "section_pages": section_page_total,
            "section_pages_with_objects": len(covered_pages & {
                page for section in sections for page in section.get("source_pages", [])
            }),
            "section_page_holes": semantic_page_holes,
            "coverage_audit": {
                "enabled": bool(coverage_audit.get("enabled")),
                "windows": int(coverage_audit.get("windows", 0)),
                "candidates": int(coverage_audit.get("candidates", 0)),
                "added": len(coverage_audit.get("added", [])),
                "errors": len(coverage_audit.get("errors", [])),
            },
        },
        "source_grounding": {
            "positions_present": len(rows) - len(source_position_missing),
            "low_confidence_positions": len(low_source_positions),
            "source_order_inversions": len(source_order_inversions),
            "format_equivalent_duplicate_groups": len(semantic_exact_groups),
            "non_contiguous_section_runs": repeated_section_runs,
            "entries_with_context_links": sum(bool(row.get("context_before_id")) for row in rows),
            "explained_containment_pairs": len(explained_containment_pairs),
            "unexplained_containment_pairs": len(containment_pairs),
            "formula_conflicts": len(formula_conflicts),
            "unresolved_formula_conflicts": len(unresolved_formula_conflicts),
            "source_span_collisions": len(span_collisions),
            "unresolved_source_span_collisions": len(unresolved_span_collisions),
            "unresolved_hard_source_span_collisions": len(unresolved_hard_span_collisions),
            "canonical_text_overlaps": len(canonical_overlaps),
            "unresolved_canonical_text_overlaps": len(unresolved_canonical_overlaps),
            "section_anchor_order_issues": len(section_anchor_order_issues),
            "cross_page_entries": sum(len(row.get("source_pages", [])) > 1 for row in rows),
            "entries_with_multi_span_provenance": sum(len(row.get("source_spans", [])) > 1 for row in rows),
            "legacy_label_page_mismatches": len(legacy_label_page_mismatches),
            "source_ledger_state_counts": source_ledger.get("state_counts", {}),
            "source_ledger_unresolved": len(ledger_unresolved),
            "detected_span_ledger_state_counts": source_ledger.get("detected_span_state_counts", {}),
            "detected_span_ledger_unresolved": len(detected_span_unresolved),
            "confidence_dimensions": {
                dimension: dict(sorted(Counter(
                    row.get("confidence", {}).get(dimension, "missing") for row in rows
                ).items()))
                for dimension in confidence_dimensions
            },
            "verification_status": dict(sorted(Counter(
                row.get("verification_status", "missing") for row in rows
            ).items())),
        },
        "relationships": {
            "enabled": relationships_enabled,
            "entries_with_dependencies": sum(bool(row.get("dependencies")) for row in rows),
            "dependency_edges": row_dependency_count,
            "entries_with_semantic_relations": sum(bool(row.get("semantic_relations")) for row in rows),
            "semantic_relation_edges": row_semantic_count,
            "entries_with_cross_references": sum(bool(row.get("cross_references")) for row in rows),
            "cross_reference_edges": row_cross_reference_count,
            "unresolved_cross_references": len(unresolved_cross_refs),
            "source_context_nodes": len(graph_context_group_ids),
            "source_context_memberships": len(graph_context_memberships),
            "cycles": len(graph_cycles),
            "request_errors": len(relationship_request_errors),
        },
    }
    structured_verifications = []
    for row in rows:
        for field in ("source_verification", "transcription_verification"):
            verification = row.get(field) or {}
            structured = verification.get("structured_verification", {})
            if structured.get("complete") and verification.get("independent_observer", {}).get("complete"):
                structured_verifications.append(structured)
                break
    hard_conflicts = [
        conflict for conflict in formula_conflicts + span_collisions if conflict.get("hard_conflict")
    ]
    unresolved_hard_conflicts = [
        conflict for conflict in hard_conflicts if conflict.get("status") != "resolved"
    ]
    resolved_formula_disagreements = [
        conflict for conflict in formula_conflicts if conflict.get("status") == "resolved"
    ]
    benchmark_metrics = {
        "schema_version": 1,
        "gold_unit_recall": page_ledger.get("gold_unit_recall"),
        "gold_unit_recall_status": page_ledger.get(
            "gold_unit_recall_status", "not_measured_no_human_gold_manifest"
        ),
        "hard_transcription_error_rate": None,
        "hard_transcription_error_rate_status": (
            "not_estimable_without_token_or_cell_level_human_gold"
        ),
        "structured_verified_items": sum(
            int(item.get("checked_items", 0)) for item in structured_verifications
        ),
        "structured_corrected_items": sum(
            int(item.get("corrected_items", 0)) for item in structured_verifications
        ),
        "observed_candidate_atom_correction_rate": (
            sum(int(item.get("corrected_items", 0)) for item in structured_verifications)
            / max(1, sum(int(item.get("checked_items", 0)) for item in structured_verifications))
        ),
        "resolved_candidate_atom_disagreements": sum(
            len(conflict.get("critical_differences", []))
            for conflict in resolved_formula_disagreements
        ),
        "human_verified_candidate_atom_disagreements": sum(
            len(conflict.get("critical_differences", []))
            for conflict in resolved_formula_disagreements
            if conflict.get("resolution", {}).get("human_pdf_visual_verification") is True
        ),
        "model_verified_candidate_atom_disagreements": sum(
            len(conflict.get("critical_differences", []))
            for conflict in resolved_formula_disagreements
            if conflict.get("resolution", {}).get("human_pdf_visual_verification") is not True
        ),
        "semantic_duplicate_audit": {
            "unexplained_containment_pairs": len(containment_pairs),
            "explained_containment_pairs": len(explained_containment_pairs),
            "resolved_source_span_collisions": sum(
                conflict.get("status") == "resolved" for conflict in span_collisions
            ),
            "total_source_span_collisions": len(span_collisions),
        },
        "unresolved_conflict_rate": (
            len(unresolved_hard_conflicts) / max(1, len(hard_conflicts))
        ),
        "unresolved_hard_conflicts": len(unresolved_hard_conflicts),
        "hard_conflicts": len(hard_conflicts),
        "note": (
            "Correction rate measures atoms rewritten inside selected candidates by exhaustive verification. "
            "Resolved candidate disagreements count atoms fixed by canonical selection, separately split by "
            "human versus model source evidence. Neither quantity estimates the residual error rate."
        ),
    }
    report["benchmark_metrics"] = benchmark_metrics
    write_json(output / "dataset" / "benchmark_metrics.json", benchmark_metrics)
    write_json(output / "dataset" / "quality_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
