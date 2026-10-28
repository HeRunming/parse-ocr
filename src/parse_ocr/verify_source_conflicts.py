#!/usr/bin/env python3
"""Resolve only hard same-span transcription conflicts from native PDF crops.

This stage is intentionally narrow.  It does not re-extract pages and it never
generates mathematical content from memory: the rendered crop is the authority.
Low-confidence adjudications remain unresolved for human review.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz

from parse_ocr.common import call_chat, image_data_url, load_json, resolve_config, write_json
from parse_ocr.source_grounding import (
    normalize_structured_observation, structured_item_map,
)


def verification_input_sha256(conflict: dict[str, Any]) -> str:
    payload = {
        "structured_verifier_version": 6,
        "candidate_ids": conflict.get("candidate_ids", []),
        "candidates": [{
            "id": item.get("id", ""), "statement": item.get("statement", ""),
            "source_spans": item.get("source_spans", []),
            "source_position": item.get("source_position", {}),
        } for item in conflict.get("candidates", [])],
        "verification_checklist": conflict.get("verification_checklist", []),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def xml_field(block: str, name: str) -> str:
    match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", block, flags=re.S | re.I)
    return html.unescape(match.group(1).strip()) if match else ""


def parse_resolution(raw: str, conflict: dict[str, Any]) -> dict[str, Any]:
    block_match = re.search(r"<resolution>\s*(.*?)\s*</resolution>", raw, flags=re.S | re.I)
    block = block_match.group(1) if block_match else raw
    result = {
        "conflict_set_id": xml_field(block, "conflict_set_id") or conflict["conflict_set_id"],
        "selected_id": xml_field(block, "selected_id"),
        "verified_statement": xml_field(block, "verified_statement"),
        "evidence": xml_field(block, "evidence"),
        "confidence": xml_field(block, "confidence").lower(),
        "method": "targeted_native_pdf_crop_vlm",
    }
    checks_raw = xml_field(block, "checks_json")
    try:
        result["checks"] = json.loads(checks_raw) if checks_raw else []
    except json.JSONDecodeError:
        result["checks"] = []
        result["checks_parse_error"] = "invalid_checks_json"
    compatible_single_id = (
        len(conflict.get("candidate_ids", [])) == 1
        and result["conflict_set_id"] == f"transcription:{conflict['candidate_ids'][0]}"
    )
    valid = (
        (result["conflict_set_id"] == conflict["conflict_set_id"] or compatible_single_id)
        and result["selected_id"] in conflict.get("candidate_ids", [])
        and bool(result["verified_statement"])
        and result["confidence"] in {"high", "medium", "low"}
    )
    result["status"] = "resolved" if valid and result["confidence"] in {"high", "medium"} else "unresolved"
    if compatible_single_id:
        result["conflict_set_id"] = conflict["conflict_set_id"]
    if not valid:
        result["parse_error"] = "missing_or_invalid_required_field"
    return result


def enforce_structured_checklist(
    result: dict[str, Any], conflict: dict[str, Any], *, require_all: bool = True,
) -> dict[str, Any]:
    """Reject model confidence unless every generated atom is independently evidenced."""
    expected = conflict.get("verification_checklist", [])
    if not expected:
        result["structured_verification"] = {
            "required": False, "complete": False, "checked_items": 0,
            "note": "no_deterministic_checklist_available",
        }
        return result
    checks = result.get("checks", [])
    check_by_id = {
        str(item.get("id", "")): item for item in checks if isinstance(item, dict) and item.get("id")
    }
    expected_ids = [str(item["id"]) for item in expected]
    issues = []
    verified_items = structured_item_map(result.get("verified_statement", ""))
    novel_observations = []
    for item in expected:
        item_id = str(item["id"])
        check = check_by_id.get(item_id)
        if check is None:
            issues.append({"id": item_id, "reason": "missing_check"})
            continue
        first = str(check.get("observed_first", "")).strip()
        second = str(check.get("observed_second", "")).strip()
        if check.get("legible") is not True or not first:
            issues.append({"id": item_id, "reason": "not_legible_or_empty"})
            continue
        role = str(item.get("role", ""))
        if item.get("requires_double_read") and (
            not second
            or normalize_structured_observation(first, role)
            != normalize_structured_observation(second, role)
        ):
            issues.append({"id": item_id, "reason": "independent_readings_disagree"})
            continue
        rendered = verified_items.get(item_id)
        if (
            rendered is None
            or normalize_structured_observation(first, role) != rendered.get("normalized_candidate")
        ):
            issues.append({"id": item_id, "reason": "verified_statement_postcheck_failed"})
            continue
        candidate_values = item.get("candidate_values", {})
        if not candidate_values and item.get("candidate") is not None:
            candidate_values = {"candidate": item.get("candidate", "")}
        observed = normalize_structured_observation(first, role)
        if candidate_values and observed not in {
            normalize_structured_observation(value, role) for value in candidate_values.values()
        }:
            novel_observations.append({
                "id": item_id, "observed": first,
                "reason": "observation_not_present_in_any_existing_candidate",
            })
    extra = sorted(set(check_by_id) - set(expected_ids))
    if require_all and extra:
        issues.append({"reason": "unexpected_check_ids", "ids": extra})
    complete = not issues and len(check_by_id) == len(expected_ids)
    result["structured_verification"] = {
        "required": True, "complete": complete,
        "expected_items": len(expected_ids), "checked_items": len(check_by_id),
        "corrected_items": 0, "novel_observations": novel_observations, "issues": issues,
        "method": "candidate_diff_critical_atom_checklist_deterministic_postcheck",
    }
    if not complete:
        result["status"] = "unresolved"
    return result


def render_conflict_crop(pdf_path: Path, conflict: dict[str, Any], path: Path, zoom: float) -> Path:
    page_number = int(conflict["pdf_page"])
    bbox = [float(value) for value in conflict["bbox"]]
    document = fitz.open(pdf_path)
    try:
        page = document[page_number - 1]
        alignment_scores = [
            float(candidate.get("source_position", {}).get("score", 0))
            for candidate in conflict.get("candidates", [])
        ]
        structured_bbox_trustworthy = bool(conflict.get("verification_checklist")) and (
            bbox[2] - bbox[0] >= float(page.rect.width) * 0.45
            and bbox[3] - bbox[1] >= float(page.rect.height) * 0.08
        )
        if (not alignment_scores or max(alignment_scores) < 0.45) and not structured_bbox_trustworthy:
            # A bad native text layer can place a correct transcription at the
            # wrong geometric block. In that case a tight crop would hide the
            # evidence; retain the full physical page as the safe fallback.
            clip = page.rect
        else:
            if conflict.get("verification_checklist"):
                padding_x, padding_y = 14.0, 16.0
            else:
                padding_x, padding_y = 32.0, 56.0
            clip = fitz.Rect(
                bbox[0] - padding_x, bbox[1] - padding_y,
                bbox[2] + padding_x, bbox[3] + padding_y,
            ) & page.rect
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(path)
    finally:
        document.close()
    return path


def independent_observer_prompt(conflict: dict[str, Any]) -> str:
    checklist = [{
        "id": item.get("id", ""), "role": item.get("role", ""),
        "requires_double_read": bool(item.get("requires_double_read")),
    } for item in conflict.get("verification_checklist", [])]
    return f"""You are a candidate-blind independent observer of a mathematical PDF crop.

You are intentionally not shown any candidate transcription or candidate numeric values. Read only the
attached image. Locate the single mathematical object/table covered by this checklist, transcribe it
source-faithfully, and fill every checklist id exactly once. For table ids, row=... is the printed row key.
For critical:NNNN or context-critical:NNNN ids, enumerate critical mathematical tokens in visual reading order. For every long
integer, read it twice independently, with the second pass grouped from the opposite direction. If a glyph
or boundary is not fully legible, set legible=false and confidence low. Do not infer or solve. The printed
source may itself contain a typo, inconsistent index, or mathematically false expression. Preserve it
literally. Never repair a bound-variable mismatch or replace a visible token with a mathematically more
natural one.

Blind checklist:
{json.dumps(checklist, ensure_ascii=False, indent=2)}

Return exactly:
<resolution>
<conflict_set_id>{conflict['conflict_set_id']}</conflict_set_id>
<selected_id>{conflict['candidate_ids'][0]}</selected_id>
<verified_statement>complete source-faithful statement or table</verified_statement>
<checks_json>[{{"id":"...","observed_first":"...","observed_second":"...","legible":true}}]</checks_json>
<evidence>visible boundary only; do not quote candidate values because none were supplied</evidence>
<confidence>high|medium|low</confidence>
</resolution>"""


def _candidate_values(item: dict[str, Any], conflict: dict[str, Any]) -> dict[str, str]:
    values = {
        str(candidate_id): str(value)
        for candidate_id, value in item.get("candidate_values", {}).items()
    }
    if not values and item.get("candidate") is not None and len(conflict.get("candidate_ids", [])) == 1:
        values[str(conflict["candidate_ids"][0])] = str(item.get("candidate", ""))
    return values


def resolve_from_blind_observers(
    conflict: dict[str, Any], observers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select an existing candidate from blind atom reads; never synthesize source text."""
    checklist = conflict.get("verification_checklist", [])
    base = {
        "conflict_set_id": conflict["conflict_set_id"],
        "status": "unresolved", "confidence": "low", "selected_id": "",
        "verified_statement": "", "observer_attempts": observers,
        "method": "candidate_blind_atom_consensus_existing_candidate_only_v6",
        "audit_safety": {
            "evidence_protocol": "v6_existing_candidate_only",
            "model_generated_text_applied": False,
            "novel_value_auto_apply": False,
            "printed_inconsistency_policy": "preserve_literal_source",
        },
    }
    if not checklist:
        base["reason"] = "no_deterministic_checklist_available"
        return base
    valid_observers = [
        observer for observer in observers
        if observer.get("status") == "resolved"
        and observer.get("confidence") == "high"
        and observer.get("structured_verification", {}).get("complete")
    ]
    consensus: dict[str, str] = {}
    consensus_raw: dict[str, str] = {}
    disagreements = []
    for item in checklist:
        item_id = str(item["id"])
        role = str(item.get("role", ""))
        counts: dict[str, int] = {}
        raw_for_normalized: dict[str, str] = {}
        readings = []
        for observer in valid_observers:
            check = next((
                value for value in observer.get("checks", [])
                if str(value.get("id", "")) == item_id
            ), {})
            raw_value = str(check.get("observed_first", "")).strip()
            normalized = normalize_structured_observation(raw_value, role)
            readings.append(raw_value)
            if normalized:
                counts[normalized] = counts.get(normalized, 0) + 1
                raw_for_normalized.setdefault(normalized, raw_value)
        winners = [value for value, count in counts.items() if count >= 2]
        if len(winners) != 1:
            disagreements.append({"id": item_id, "readings": readings, "reason": "no_unique_2_of_3_read"})
            continue
        consensus[item_id] = winners[0]
        consensus_raw[item_id] = raw_for_normalized[winners[0]]
    if disagreements or len(consensus) != len(checklist):
        base["reason"] = "blind_observer_consensus_incomplete"
        base["observer_consensus"] = {"complete": False, "disagreements": disagreements}
        return base

    novel = []
    for item in checklist:
        item_id = str(item["id"])
        role = str(item.get("role", ""))
        allowed = {
            normalize_structured_observation(value, role)
            for value in _candidate_values(item, conflict).values()
        }
        if consensus[item_id] not in allowed:
            novel.append({"id": item_id, "observed": consensus_raw[item_id]})
    if novel:
        base["reason"] = "novel_observation_requires_human_verification"
        base["novel_observations"] = novel
        base["observer_consensus"] = {"complete": True, "readings": consensus_raw}
        return base

    matching_ids = []
    for candidate_id in conflict.get("candidate_ids", []):
        matches = True
        for item in checklist:
            item_id = str(item["id"])
            role = str(item.get("role", ""))
            values = _candidate_values(item, conflict)
            if candidate_id not in values or normalize_structured_observation(
                values[candidate_id], role
            ) != consensus[item_id]:
                matches = False
                break
        if matches:
            matching_ids.append(candidate_id)
    if len(matching_ids) != 1:
        base["reason"] = "consensus_does_not_uniquely_select_existing_candidate"
        base["matching_candidate_ids"] = matching_ids
        base["observer_consensus"] = {"complete": True, "readings": consensus_raw}
        return base
    selected_id = matching_ids[0]
    candidate = next((
        item for item in conflict.get("candidates", []) if item.get("id") == selected_id
    ), None)
    if not candidate:
        base["reason"] = "selected_candidate_payload_missing"
        return base
    statement = str(candidate.get("statement", "")).strip()
    if not statement:
        base["reason"] = "selected_candidate_statement_empty"
        return base
    base.update({
        "status": "resolved", "confidence": "high", "selected_id": selected_id,
        "verified_statement": statement,
        "selected_candidate_statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
        "evidence": "Two-of-three candidate-blind visual reads matched one existing candidate.",
        "checks": [
            {"id": item_id, "observed_first": raw, "observed_second": raw, "legible": True}
            for item_id, raw in consensus_raw.items()
        ],
        "structured_verification": {
            "required": True, "complete": True, "expected_items": len(checklist),
            "checked_items": len(checklist), "corrected_items": 0, "issues": [],
            "method": "blind_2_of_3_existing_candidate_match",
        },
        "independent_observer": {
            "complete": True, "observer_count": len(observers),
            "complete_observers": len(valid_observers), "confidence": "high",
            "method": "three_candidate_blind_visual_reads",
        },
        "observer_consensus": {"complete": True, "readings": consensus_raw},
    })
    return base


def parse_canonical_resolution(raw: str, conflict: dict[str, Any]) -> dict[str, Any]:
    block_match = re.search(r"<resolution>\s*(.*?)\s*</resolution>", raw, flags=re.S | re.I)
    block = block_match.group(1) if block_match else raw
    context_ids = [
        value.strip() for value in xml_field(block, "context_only_ids").split(",") if value.strip()
    ]
    result = {
        "conflict_set_id": xml_field(block, "conflict_set_id") or conflict["collision_set_id"],
        "action": xml_field(block, "action").lower(),
        "selected_id": xml_field(block, "selected_id"),
        "context_only_ids": context_ids,
        "verified_statement": xml_field(block, "verified_statement"),
        "evidence": xml_field(block, "evidence"),
        "confidence": xml_field(block, "confidence").lower(),
        "method": "independent_native_pdf_crop_canonical_verifier",
    }
    ids = set(conflict.get("candidate_ids", []))
    valid = (
        result["conflict_set_id"] == conflict["collision_set_id"]
        and result["action"] in {"retain_distinct", "merge", "context_only"}
        and result["selected_id"] in ids
        and set(context_ids) <= ids
        and result["confidence"] in {"high", "medium", "low"}
        and (result["action"] != "merge" or bool(result["verified_statement"]))
        and (result["action"] != "context_only" or bool(context_ids))
    )
    # Canonical-role changes are more consequential than token correction: only a
    # fully legible, high-confidence decision is automatically applied.
    result["status"] = "resolved" if valid and result["confidence"] == "high" else "unresolved"
    if not valid:
        result["parse_error"] = "missing_or_invalid_required_field"
    return result


def _canonical_decision_key(result: dict[str, Any]) -> tuple[Any, ...] | None:
    if result.get("status") != "resolved" or result.get("confidence") != "high":
        return None
    action = result.get("action")
    if action == "retain_distinct":
        return (action,)
    if action == "merge":
        return (action, result.get("selected_id", ""))
    if action == "context_only":
        return (
            action, result.get("selected_id", ""),
            tuple(sorted(result.get("context_only_ids", []))),
        )
    return None


def resolve_canonical_consensus(
    conflict: dict[str, Any], attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require two matching independent role decisions and reuse candidate text verbatim."""
    counts: dict[tuple[Any, ...], int] = {}
    for attempt in attempts:
        key = _canonical_decision_key(attempt)
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    winners = [key for key, count in counts.items() if count >= 2]
    base = {
        "conflict_set_id": conflict["collision_set_id"],
        "status": "unresolved", "confidence": "low", "canonical_attempts": attempts,
        "method": "independent_canonical_role_consensus_existing_candidate_only_v6",
        "audit_safety": {
            "evidence_protocol": "v6_existing_candidate_only",
            "model_generated_text_applied": False,
            "novel_value_auto_apply": False,
            "printed_inconsistency_policy": "preserve_literal_source",
        },
        "canonical_consensus": {"complete": False, "attempt_count": len(attempts)},
    }
    if len(winners) != 1:
        base["reason"] = "no_unique_2_of_3_canonical_decision"
        base["decision_counts"] = {json.dumps(key): count for key, count in counts.items()}
        return base
    key = winners[0]
    action = str(key[0])
    selected_id = (
        str(key[1]) if len(key) > 1 else str(conflict.get("candidate_ids", [""])[0])
    )
    context_ids = list(key[2]) if action == "context_only" else []
    verified_statement = ""
    statement_hash = ""
    if action == "merge":
        candidate = next((
            item for item in conflict.get("candidates", []) if item.get("id") == selected_id
        ), None)
        if not candidate or not str(candidate.get("statement", "")).strip():
            base["reason"] = "selected_candidate_payload_missing"
            return base
        verified_statement = str(candidate["statement"]).strip()
        statement_hash = hashlib.sha256(verified_statement.encode()).hexdigest()
    agreeing = [attempt for attempt in attempts if _canonical_decision_key(attempt) == key]
    base.update({
        "status": "resolved", "confidence": "high", "action": action,
        "selected_id": selected_id, "context_only_ids": context_ids,
        "verified_statement": verified_statement,
        "selected_candidate_statement_sha256": statement_hash,
        "evidence": "Two-of-three independent visual role decisions agreed; no generated text was used.",
        "canonical_consensus": {
            "complete": True, "attempt_count": len(attempts),
            "support": len(agreeing), "decision": list(key),
        },
    })
    return base


def canonical_prompt(conflict: dict[str, Any]) -> str:
    candidates = json.dumps(conflict.get("candidates", []), ensure_ascii=False, indent=2)
    pages = sorted({
        int(span.get("pdf_page"))
        for candidate in conflict.get("candidates", [])
        for span in candidate.get("source_spans", []) if span.get("pdf_page")
    } or {int(conflict["pdf_page"])})
    return f"""You are an independent source-faithful canonicalization auditor.

The images are native-resolution crops from physical PDF pages {pages}, in page order.
The extraction system found overlapping candidates. Decide their canonical roles only from the
visible source. Do not solve, paraphrase, modernize, or add mathematical content.
The printed source can contain mistakes or awkward notation. Preserve its literal content and do not
use mathematical plausibility as evidence for a boundary or role decision.

Candidates:
{candidates}

Choose exactly one action:
- retain_distinct: the crop visibly contains multiple independently useful atomic mathematical units.
- merge: the candidates are duplicate/transcription variants of one mathematical unit. Select the
  best candidate and provide a complete source-faithful verified_statement.
- context_only: one candidate contributes no independent mathematical unit beyond another canonical
  candidate (for example, a repeated setup-only child or a broader prose host). Select the complete
  canonical unit and list only redundant/context ids in context_only_ids.

An exercise/problem/theorem/conjecture that contains the setup required for its mathematical request
is normally the complete canonical unit; do not demote it merely because a notation-only candidate
repeats its setup. Do not use context_only merely because two distinct propositions share a paragraph. If boundaries,
subscripts, numbers, or roles are not fully visible, use confidence low. Only high confidence applies.

Return exactly:
<resolution>
<conflict_set_id>{conflict['collision_set_id']}</conflict_set_id>
<action>retain_distinct|merge|context_only</action>
<selected_id>one candidate id</selected_id>
<context_only_ids>comma-separated ids, empty unless action=context_only</context_only_ids>
<verified_statement>complete source statement if action=merge, otherwise empty</verified_statement>
<evidence>brief visible evidence for boundaries and critical mathematical tokens</evidence>
<confidence>high|medium|low</confidence>
</resolution>"""


def render_canonical_crops(
    pdf_path: Path, conflict: dict[str, Any], directory: Path, safe_id: str, zoom: float,
) -> list[Path]:
    spans_by_page: dict[int, list[dict[str, Any]]] = {}
    for candidate in conflict.get("candidates", []):
        for span in candidate.get("source_spans", []):
            if span.get("pdf_page") and span.get("bbox"):
                spans_by_page.setdefault(int(span["pdf_page"]), []).append(span)
    if not spans_by_page:
        path = directory / f"{safe_id}_p{conflict['pdf_page']}.png"
        return [render_conflict_crop(pdf_path, conflict, path, zoom)]
    paths = []
    for page, spans in sorted(spans_by_page.items()):
        bboxes = [span["bbox"] for span in spans]
        page_conflict = {
            "pdf_page": page,
            "bbox": [
                min(float(bbox[0]) for bbox in bboxes), min(float(bbox[1]) for bbox in bboxes),
                max(float(bbox[2]) for bbox in bboxes), max(float(bbox[3]) for bbox in bboxes),
            ],
            "candidates": [{"source_position": {"score": max(
                float(span.get("score", 0)) for span in spans
            )}}],
        }
        path = directory / f"{safe_id}_p{page}.png"
        paths.append(render_conflict_crop(pdf_path, page_conflict, path, zoom))
    return paths


def run_independent_observer(
    config: dict[str, Any], conflict: dict[str, Any], crops: list[Path], observer_round: int = 1,
) -> dict[str, Any]:
    settings = config.get("direct", {})
    round_protocols = {
        1: (
            "Read line by line in normal visual order. Copy glyphs before interpreting their role."
        ),
        2: (
            "Inspect the requested atoms in reverse checklist and right-to-left glyph order, then report "
            "them in normal source order. Read subscripts and superscripts before their bases."
        ),
        3: (
            "First classify each visible glyph only by shape and position (base/subscript/superscript); "
            "only afterward assemble the literal token. Do not test whether the formula makes sense."
        ),
    }
    content = [{
        "type": "text",
        "text": independent_observer_prompt(conflict) + (
            f"\nIndependent observer round {observer_round}. Start a fresh visual read; do not reuse "
            f"any earlier reading. {round_protocols.get(observer_round, round_protocols[1])}"
        ),
    }]
    content.extend({
        "type": "image_url", "image_url": {"url": image_data_url(crop), "detail": "high"}
    } for crop in crops)
    raw, usage = call_chat(
        config, content, max_tokens=int(settings.get("source_verification_max_tokens", 8000))
    )
    result = enforce_structured_checklist(parse_resolution(raw, conflict), conflict)
    result["usage"] = usage
    if result.get("confidence") != "high":
        result["status"] = "unresolved"
    if result.get("status") != "resolved":
        result["raw_response"] = raw
    return result


def run_canonical_observer(
    config: dict[str, Any], conflict: dict[str, Any], crops: list[Path], observer_round: int,
) -> dict[str, Any]:
    settings = config.get("direct", {})
    round_protocols = {
        1: "Trace printed headings, paragraph starts, and terminal punctuation before choosing a role.",
        2: (
            "Try each allowed action as a competing boundary hypothesis and reject it using visible source "
            "evidence; do not prefer the extraction system's apparent intent."
        ),
        3: (
            "Count independently useful mathematical assertions in physical reading order, then decide "
            "whether shared setup is context or a duplicate."
        ),
    }
    content = [{
        "type": "text",
        "text": canonical_prompt(conflict) + (
            f"\nIndependent canonical audit round {observer_round}. Start from the pixels; "
            f"do not rely on another audit. {round_protocols.get(observer_round, round_protocols[1])}"
        ),
    }]
    content.extend({
        "type": "image_url", "image_url": {"url": image_data_url(crop), "detail": "high"}
    } for crop in crops)
    raw, usage = call_chat(
        config, content, max_tokens=int(settings.get("source_verification_max_tokens", 8000))
    )
    result = parse_canonical_resolution(raw, conflict)
    result["usage"] = usage
    if result.get("status") != "resolved":
        result["raw_response"] = raw
    return result


def verify_one(config: dict[str, Any], conflict: dict[str, Any], force: bool) -> dict[str, Any]:
    output = Path(config["output"])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", conflict["conflict_set_id"])
    checkpoint = output / "direct/source_verification/checkpoints" / f"{safe_id}.json"
    crop = output / "direct/source_verification/crops" / f"{safe_id}.png"
    input_sha256 = verification_input_sha256(conflict)
    if checkpoint.is_file() and not force:
        cached = load_json(checkpoint)
        requires_structured = bool(conflict.get("verification_checklist"))
        if (
            cached.get("verification_input_sha256") == input_sha256
            and (
                not requires_structured
                or (
                    cached.get("structured_verification", {}).get("complete")
                    and cached.get("independent_observer", {}).get("complete")
                )
            )
        ):
            return cached
    settings = config.get("direct", {})
    render_conflict_crop(Path(config["pdf"]), conflict, crop, float(settings.get("source_verification_zoom", 4.0)))
    try:
        if not conflict.get("verification_checklist"):
            result = {
                "conflict_set_id": conflict["conflict_set_id"], "status": "unresolved",
                "confidence": "low", "reason": "no_deterministic_checklist_available",
                "method": "v6_no_free_form_auto_adjudication",
            }
        else:
            observers = [
                run_independent_observer(config, conflict, [crop], round_number)
                for round_number in range(1, 4)
            ]
            result = resolve_from_blind_observers(conflict, observers)
        result.update({
            "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
            "verification_input_sha256": input_sha256,
        })
    except Exception as exc:
        result = {
            "conflict_set_id": conflict["conflict_set_id"], "status": "unresolved",
            "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(checkpoint, result)
    return result


def verify_canonical_one(config: dict[str, Any], conflict: dict[str, Any], force: bool) -> dict[str, Any]:
    output = Path(config["output"])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", conflict["collision_set_id"])
    checkpoint = output / "direct/source_verification/canonical_checkpoints" / f"{safe_id}.json"
    input_sha256 = verification_input_sha256(conflict)
    if checkpoint.is_file() and not force:
        cached = load_json(checkpoint)
        if (
            cached.get("verification_input_sha256") == input_sha256
            and cached.get("canonical_consensus", {}).get("complete")
        ):
            return cached
    settings = config.get("direct", {})
    crops = render_canonical_crops(
        Path(config["pdf"]), conflict,
        output / "direct/source_verification/canonical_crops", safe_id,
        float(settings.get("source_verification_zoom", 4.0)),
    )
    try:
        attempts = [
            run_canonical_observer(config, conflict, crops, round_number)
            for round_number in range(1, 4)
        ]
        result = resolve_canonical_consensus(conflict, attempts)
        result.update({
            "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
            "verification_input_sha256": input_sha256,
        })
    except Exception as exc:
        result = {
            "conflict_set_id": conflict["collision_set_id"], "status": "unresolved",
            "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(checkpoint, result)
    return result


def verify_transcription_one(config: dict[str, Any], conflict: dict[str, Any], force: bool) -> dict[str, Any]:
    output = Path(config["output"])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", conflict["conflict_set_id"])
    checkpoint = output / "direct/source_verification/transcription_checkpoints" / f"{safe_id}.json"
    input_sha256 = verification_input_sha256(conflict)
    if checkpoint.is_file() and not force:
        cached = load_json(checkpoint)
        requires_structured = bool(conflict.get("verification_checklist"))
        if cached.get("verification_input_sha256") != input_sha256:
            cached = {}
        elif requires_structured and not (
            cached.get("structured_verification", {}).get("complete")
            and cached.get("independent_observer", {}).get("complete")
        ):
            cached = {}
        if cached:
            return cached
    settings = config.get("direct", {})
    crops = render_canonical_crops(
        Path(config["pdf"]), conflict,
        output / "direct/source_verification/transcription_crops", safe_id,
        float(settings.get("source_verification_zoom", 4.0)),
    )
    try:
        if not conflict.get("verification_checklist"):
            result = {
                "conflict_set_id": conflict["conflict_set_id"], "status": "unresolved",
                "confidence": "low", "reason": "no_deterministic_checklist_available",
                "method": "v6_no_free_form_auto_adjudication",
            }
        else:
            observers = [
                run_independent_observer(config, conflict, crops, round_number)
                for round_number in range(1, 4)
            ]
            result = resolve_from_blind_observers(conflict, observers)
        result.update({
            "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
            "verification_input_sha256": input_sha256,
        })
        if (
            conflict.get("verification_checklist")
            and result.get("status") != "resolved"
            and result.get("reason") != "novel_observation_requires_human_verification"
        ):
            # A tight crop can hide the first/last glyph even when its geometry is
            # otherwise correct. Retry once with complete physical pages; never
            # widen the text from memory.
            full_conflict = json.loads(json.dumps(conflict, ensure_ascii=False))
            for candidate in full_conflict.get("candidates", []):
                for span in candidate.get("source_spans", []):
                    span["score"] = 0
            full_crops = render_canonical_crops(
                Path(config["pdf"]), full_conflict,
                output / "direct/source_verification/transcription_full_pages", safe_id,
                float(settings.get("source_verification_zoom", 4.0)),
            )
            full_observers = [
                run_independent_observer(config, conflict, full_crops, round_number)
                for round_number in range(1, 4)
            ]
            full_result = resolve_from_blind_observers(conflict, full_observers)
            full_result.update({
                "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
                "full_page_fallback_attempted": True,
                "initial_crop_result": result,
                "verification_input_sha256": input_sha256,
            })
            result = full_result
    except Exception as exc:
        result = {
            "conflict_set_id": conflict["conflict_set_id"], "status": "unresolved",
            "pdf_page": conflict["pdf_page"], "bbox": conflict["bbox"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(checkpoint, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument(
        "--candidate-id", action="append", default=[],
        help="verify only conflicts/queue items containing this candidate id; may repeat",
    )
    args = parser.parse_args()
    config = resolve_config(args.config)
    output = Path(config["output"])
    conflict_path = output / "dataset/formula_conflicts.json"
    if not conflict_path.is_file():
        raise SystemExit(f"build the dataset before source verification: {conflict_path}")
    resolution_path = output / "direct/source_verification/resolutions.json"
    previous = load_json(resolution_path) if resolution_path.is_file() else []
    conflicts = [
        conflict for conflict in load_json(conflict_path)
        if conflict.get("hard_conflict") and conflict.get("status") != "resolved"
        and (
            not args.candidate_id
            or set(args.candidate_id) & set(conflict.get("candidate_ids", []))
        )
    ]
    settings = config.get("direct", {})
    if not bool(settings.get("source_verification_enabled", True)):
        write_json(resolution_path, previous)
        print("source verification disabled")
        return
    workers = max(1, int(settings.get("source_verification_workers", 8)))
    new_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(verify_one, config, conflict, args.force_stage): conflict for conflict in conflicts}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            new_results.append(result)
            print(f"[{index}/{len(conflicts)}] {result['conflict_set_id']}: {result['status']}", flush=True)
    by_id = {result.get("conflict_set_id"): result for result in previous if result.get("conflict_set_id")}
    by_id.update({result["conflict_set_id"]: result for result in new_results})
    results = list(by_id.values())
    results.sort(key=lambda item: (int(item.get("pdf_page") or 0), item["conflict_set_id"]))
    write_json(resolution_path, results)
    active_formula_ids = {conflict["conflict_set_id"] for conflict in conflicts}
    active_formula_results = [
        result for result in results if result.get("conflict_set_id") in active_formula_ids
    ]
    write_json(output / "direct/source_verification/summary.json", {
        "schema_version": 1, "hard_conflicts": len(conflicts),
        "resolved": sum(result.get("status") == "resolved" for result in active_formula_results),
        "unresolved": sum(result.get("status") != "resolved" for result in active_formula_results),
        "novel_observation_cases": sum(
            result.get("reason") == "novel_observation_requires_human_verification"
            for result in active_formula_results
        ),
        "historical_results": len(results),
        "evidence_protocol": "v6_existing_candidate_only",
        "errors": [result for result in active_formula_results if result.get("error")],
    })

    span_path = output / "dataset/source_span_collisions.json"
    canonical_path = output / "direct/source_verification/canonical_resolutions.json"
    previous_canonical = load_json(canonical_path) if canonical_path.is_file() else []
    span_conflicts = [
        conflict for conflict in (load_json(span_path) if span_path.is_file() else [])
        if conflict.get("status") != "resolved"
        and (
            not args.candidate_id
            or set(args.candidate_id) & set(conflict.get("candidate_ids", []))
        )
    ]
    canonical_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(verify_canonical_one, config, conflict, args.force_stage): conflict
            for conflict in span_conflicts
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); canonical_results.append(result)
            print(
                f"[canonical {index}/{len(span_conflicts)}] "
                f"{result['conflict_set_id']}: {result['status']}", flush=True,
            )
    canonical_by_id = {
        result.get("conflict_set_id"): result for result in previous_canonical
        if result.get("conflict_set_id")
    }
    canonical_by_id.update({result["conflict_set_id"]: result for result in canonical_results})
    canonical_values = list(canonical_by_id.values())
    canonical_values.sort(key=lambda item: (int(item.get("pdf_page") or 0), item["conflict_set_id"]))
    write_json(canonical_path, canonical_values)
    active_canonical_ids = {conflict["collision_set_id"] for conflict in span_conflicts}
    active_canonical_values = [
        result for result in canonical_values if result.get("conflict_set_id") in active_canonical_ids
    ]
    write_json(output / "direct/source_verification/canonical_summary.json", {
        "schema_version": 1, "conflicts": len(span_conflicts),
        "resolved": sum(result.get("status") == "resolved" for result in active_canonical_values),
        "unresolved": sum(result.get("status") != "resolved" for result in active_canonical_values),
        "historical_results": len(canonical_values),
        "evidence_protocol": "v6_existing_candidate_only",
        "actions": {
            action: sum(result.get("action") == action and result.get("status") == "resolved"
                        for result in active_canonical_values)
            for action in ("retain_distinct", "merge", "context_only")
        },
        "errors": [result for result in active_canonical_values if result.get("error")],
    })

    transcription_queue_path = output / "dataset/transcription_review_queue.json"
    transcription_path = output / "direct/source_verification/transcription_resolutions.json"
    previous_transcription = load_json(transcription_path) if transcription_path.is_file() else []
    queue = load_json(transcription_queue_path) if transcription_queue_path.is_file() else []
    if args.candidate_id:
        queue = [item for item in queue if item.get("id") in set(args.candidate_id)]
    transcription_conflicts = []
    for item in queue:
        spans = item.get("source_spans", [])
        primary = spans[0] if spans else {}
        digest = re.sub(r"[^A-Za-z0-9_.-]+", "_", item["id"])
        transcription_conflicts.append({
            "conflict_set_id": f"transcription:{digest}",
            # Some inventory/coverage candidates can reach the review queue
            # before a native source span has been assigned.  Keep those
            # candidates unresolved rather than crashing on min([]); page 0
            # is an explicit "unknown" sentinel and is handled by the
            # verifier as a non-grounded record.
            "candidate_ids": [item["id"]], "pdf_page": min(item.get("source_pages") or [0]),
            "bbox": primary.get("bbox", [0, 0, 0, 0]),
            "verification_checklist": item.get("verification_checklist", []),
            "verification_reasons": item.get("verification_reasons", []),
            "candidates": [{
                "id": item["id"], "kind": item.get("kind", ""),
                "statement": item.get("statement", ""),
                "uncertain_spans": item.get("uncertain_spans", []),
                "source_position": primary, "source_spans": spans,
            }],
        })
    transcription_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(verify_transcription_one, config, conflict, args.force_stage): conflict
            for conflict in transcription_conflicts
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result(); transcription_results.append(result)
            print(
                f"[transcription {index}/{len(transcription_conflicts)}] "
                f"{result['conflict_set_id']}: {result['status']}", flush=True,
            )
    # Key by the canonical candidate rather than the model-echoed conflict id so
    # a harmless separator repair cannot leave a stale duplicate resolution.
    transcription_by_id = {
        result.get("selected_id"): result for result in previous_transcription
        if result.get("selected_id")
    }
    transcription_by_id.update({
        result.get("selected_id"): result for result in transcription_results
        if result.get("selected_id")
    })
    transcription_values = list(transcription_by_id.values())
    transcription_values.sort(key=lambda item: (int(item.get("pdf_page") or 0), item["conflict_set_id"]))
    write_json(transcription_path, transcription_values)
    active_ids = {conflict["candidate_ids"][0] for conflict in transcription_conflicts}
    active_values = [item for item in transcription_values if item.get("selected_id") in active_ids]
    write_json(output / "direct/source_verification/transcription_summary.json", {
        "schema_version": 1, "queued": len(transcription_conflicts),
        "resolved": sum(result.get("status") == "resolved" for result in active_values),
        "unresolved": sum(result.get("status") != "resolved" for result in active_values),
        "historical_results": len(transcription_values),
        "novel_observation_cases": sum(
            result.get("reason") == "novel_observation_requires_human_verification"
            for result in active_values
        ),
        "evidence_protocol": "v6_existing_candidate_only",
        "errors": [result for result in active_values if result.get("error")],
    })

    # A direct pairwise conflict pass can still be anchored by the bad candidate.
    # If one candidate later passes a genuinely blind, independent, exhaustive
    # single-record verification, use that stronger evidence to resolve the pair.
    formula_all = load_json(conflict_path)
    formula_by_id = {item.get("conflict_set_id"): item for item in results}
    transcription_by_candidate = {
        item.get("selected_id"): item for item in transcription_values
        if item.get("status") == "resolved" and item.get("confidence") == "high"
        and item.get("structured_verification", {}).get("complete")
        and item.get("independent_observer", {}).get("complete")
        and item.get("audit_safety", {}).get("evidence_protocol")
            == "v6_existing_candidate_only"
        and item.get("audit_safety", {}).get("model_generated_text_applied") is False
    }
    promotions = []
    for conflict in formula_all:
        if not conflict.get("verification_checklist"):
            continue
        existing = formula_by_id.get(conflict["conflict_set_id"], {})
        if (
            existing.get("status") == "resolved"
            and existing.get("independent_observer", {}).get("complete")
            and existing.get("audit_safety", {}).get("evidence_protocol")
                == "v6_existing_candidate_only"
        ):
            continue
        candidates = [
            transcription_by_candidate[candidate_id]
            for candidate_id in conflict.get("candidate_ids", [])
            if candidate_id in transcription_by_candidate
        ]
        if not candidates:
            continue
        # Prefer the verified candidate covering the most checklist/source text.
        selected = max(candidates, key=lambda item: len(item.get("verified_statement", "")))
        promoted = {
            **selected,
            "conflict_set_id": conflict["conflict_set_id"],
            "selected_id": selected["selected_id"],
            "method": "promoted_from_blind_independent_candidate_transcription_verification",
            "promotion_source_conflict_id": selected.get("conflict_set_id", ""),
            "prior_pairwise_attempt": existing,
            "status": "resolved", "confidence": "high",
        }
        formula_by_id[conflict["conflict_set_id"]] = promoted
        promotions.append({
            "conflict_set_id": conflict["conflict_set_id"],
            "selected_id": selected["selected_id"], "method": promoted["method"],
        })
    results = sorted(formula_by_id.values(), key=lambda item: (
        int(item.get("pdf_page") or 0), item.get("conflict_set_id", "")
    ))
    write_json(resolution_path, results)
    write_json(output / "direct/source_verification/promotions.json", promotions)
    relevant_ids = {
        item["conflict_set_id"] for item in formula_all if item.get("hard_conflict")
    }
    relevant_results = [
        item for item in results if item.get("conflict_set_id") in relevant_ids
    ]
    write_json(output / "direct/source_verification/summary.json", {
        "schema_version": 2, "hard_conflicts": len(relevant_ids),
        "resolved": sum(item.get("status") == "resolved" for item in relevant_results),
        "unresolved": sum(item.get("status") != "resolved" for item in relevant_results),
        "promoted_from_independent_candidate_verification": len(promotions),
        "errors": [item for item in relevant_results if item.get("error")],
    })


if __name__ == "__main__":
    main()
