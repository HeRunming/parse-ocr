#!/usr/bin/env python3
"""Ground extracted math objects in deterministic PDF source positions.

The VLM remains authoritative for mathematical transcription.  This module uses
the native PDF text layer only as a geometric scaffold for:

* reading order;
* section membership on pages containing more than one section heading;
* duplicate identity (same source span, not merely similar wording);
* conservative repair of sentence fragments split into multiple objects; and
* narrow context links for entries that still cannot stand alone.

No mathematical token is replaced from native PDF text.  A weak or missing text
layer therefore lowers grounding confidence instead of corrupting the VLM text.
"""

from __future__ import annotations

import html
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import fitz


@dataclass(frozen=True)
class TextBlock:
    page: int
    index: int
    bbox: tuple[float, float, float, float]
    text: str
    page_width: float
    page_height: float


def _plain(value: str) -> str:
    """Remove presentation-only markup while retaining readable math tokens."""
    value = html.unescape(value or "")
    # Strip only known presentation tags.  A generic ``<[^>]+>`` pattern can
    # swallow whole mathematical passages between an inequality ``<`` and a
    # later ``>``.
    value = re.sub(r"</?(?:b|strong|em|i|span|sup|sub)>", " ", value, flags=re.I)
    value = value.replace("**", " ").replace("__", " ")
    value = re.sub(r"\\(?:textbf|mathbf|mathrm|mathit|operatorname)\s*\{([^{}]*)\}", r" \1 ", value)
    value = value.replace("\\(", " ").replace("\\)", " ")
    value = value.replace("\\[", " ").replace("\\]", " ").replace("$$", " ").replace("$", " ")
    value = value.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    value = re.sub(r"[{}\\^_]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def semantic_tokens(value: str) -> list[str]:
    value = re.sub(r"\\(?:mid|vert)\b", " | ", value)
    value = re.sub(r"\\lvert\b|\\rvert\b", " | ", value)
    value = _plain(value).casefold()
    value = value.replace("≥", " ge ").replace("≤", " le ").replace("≠", " ne ")
    return re.findall(r"[a-z]+|\d+|[α-ωΑ-Ω]+|\|", value)


def semantic_fingerprint(value: str) -> str:
    """Fingerprint insensitive to bold/HTML and harmless LaTeX presentation."""
    aliases = {"geq": "ge", "leq": "le", "dfrac": "frac", "tfrac": "frac"}
    return " ".join(aliases.get(token, token) for token in semantic_tokens(value))


def semantic_compact(value: str) -> str:
    return re.sub(r"[^a-z0-9α-ω]+", "", _plain(value).casefold())


def prose_skeleton(value: str) -> str:
    value = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\$[^$]*\$|\\\([^)]*\\\)", " <math> ", value, flags=re.S)
    return re.sub(r"\s+", " ", _plain(value).casefold()).strip()


def formula_tokens(value: str) -> list[str]:
    """Tokenize formula content while ignoring presentation-only environments."""
    segments = re.findall(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\$([^$]*)\$|\\\((.*?)\\\)", value, flags=re.S)
    text = " ".join(next((part for part in parts if part), "") for parts in segments)
    text = re.sub(r"\\(?:left|right)\s*[()[\]{}.|]", " ", text)
    ignored_commands = {
        "begin", "end", "left", "right", "big", "bigg", "Big", "Bigg",
        "bmatrix", "pmatrix", "matrix", "array", "smallmatrix", "aligned", "align", "cases", "displaystyle",
        "text", "mathrm", "mathbf", "mathit", "mathbin", "quad", "qquad",
        "!", ",", ";", "colon",
    }
    tokens = re.findall(r"\\[A-Za-z]+|\d+|[A-Za-zα-ωΑ-Ω]+|\^|_|[+\-=<>|]|[()[\],]", text)
    result = []
    for token in tokens:
        key = token.lstrip("\\")
        if key in ignored_commands or re.fullmatch(r"[clr]+", key):
            continue
        normalized = {r"\mid": "|", r"\vert": "|", r"\lvert": "|", r"\rvert": "|"}.get(token, token)
        result.append(normalized.casefold())
    return result


def critical_formula_tokens(value: str) -> list[str]:
    """Return tokens whose disagreement can change mathematical meaning."""
    return [token for token in formula_tokens(value) if (
        token.isdigit() or token in {"^", "_", "+", "-", "=", "<", ">", "|"}
        or token.startswith("\\") or re.fullmatch(r"[a-zα-ω]+", token)
    )]


def normalize_verified_atom(value: str) -> str:
    """Normalize one checklist atom without changing mathematical content."""
    value = html.unescape(str(value or "")).strip()
    value = value.replace("\\(", "").replace("\\)", "").replace("$", "")
    value = re.sub(r"\\text\s*\{\s*([^{}]*)\s*\}", r"\1", value)
    value = re.sub(r"\\(?:quad|qquad)(?![A-Za-z])|\\[,;!]", "", value)
    value = re.sub(r"\\\s+", "", value)
    value = value.replace("\u00b7", "*").replace("\u00d7", "*")
    value = re.sub(r"\\(?:cdot|times)(?![A-Za-z])", "*", value)
    return re.sub(r"\s+", "", value).casefold()


def normalize_structured_observation(value: str, role: str = "") -> str:
    if role == "context_formula":
        value = value.replace("−", "-").replace("⋯", r"\cdots").replace("…", r"\cdots")
        wrapped = value if re.search(r"\$|\\\(|\\\[", value) else f"${value}$"
        return "|".join(critical_formula_tokens(wrapped))
    return normalize_verified_atom(value)


STRUCTURED_LATEX_ENVIRONMENTS = (
    "array", "matrix", "bmatrix", "pmatrix", "vmatrix", "Vmatrix",
    "smallmatrix", "aligned", "cases",
)


def _latex_array_rows(value: str) -> list[list[str]]:
    rows: list[list[str]] = []
    environments = "|".join(map(re.escape, STRUCTURED_LATEX_ENVIRONMENTS))
    for match in re.finditer(
        rf"\\begin\{{({environments})\}}(?:\{{[^{{}}]*\}})?(.*?)\\end\{{\1\}}",
        value, flags=re.S,
    ):
        for raw_row in re.split(r"\\\\(?:\[[^\]]*\])?", match.group(2)):
            cells = [cell.strip() for cell in raw_row.strip().split("&")]
            if len(cells) >= 2:
                rows.append(cells)
    return rows


def _dense_numeric_sequence_items(value: str) -> list[dict[str, Any]]:
    """Inventory long printed lists and grouped integers item-by-item."""
    candidates: list[tuple[str, list[str]]] = []
    for match in re.finditer(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", value, flags=re.S):
        body = re.sub(r"\\\\|&", " ", match.group(1))
        values = re.findall(r"\d+", body)
        if len(values) >= 10:
            candidates.append(("comma_list", values))
    for match in re.finditer(r"(?:\d+\s*,\s*\\?\s*){9,}\d+", value):
        candidates.append(("comma_list", re.findall(r"\d+", match.group(0))))
    for match in re.finditer(r"(?:\d+\s*(?:\\,|\\;|\s)\s*){5,}\d+", value):
        values = re.findall(r"\d+", match.group(0))
        if len(values) >= 6:
            candidates.append(("grouped_integer", values))
    if not candidates:
        return []
    role, values = max(candidates, key=lambda item: len(item[1]))
    return [{
        "id": f"{role}:item={index:04d}", "role": role,
        "candidate": item, "normalized_candidate": item,
        "requires_double_read": len(item) >= 8,
    } for index, item in enumerate(values, 1)]


def _markdown_table_rows(value: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def structured_math_items(value: str) -> list[dict[str, Any]]:
    """Create stable table-cell or critical-token checklist items.

    Table rows are keyed by their printed row label so two differently formatted
    transcriptions can be compared cell-by-cell.  Non-tabular mathematics falls
    back to occurrence-indexed critical tokens.
    """
    table_rows = _latex_array_rows(value) + _markdown_table_rows(value)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cells in table_rows:
        pairs: list[tuple[str, str]] = []
        if len(cells) >= 4:
            pairs.extend(((cells[0], cells[1]), (cells[2], cells[3])))
        elif len(cells) >= 2:
            pairs.append((cells[0], cells[1]))
        for key, cell_value in pairs:
            key_norm = normalize_verified_atom(key)
            if not re.fullmatch(r"[a-z]?\d+(?:\.\d+)?", key_norm):
                continue
            item_id = f"table:row={key_norm}:value"
            if item_id in seen:
                continue
            seen.add(item_id)
            normalized = normalize_verified_atom(cell_value)
            items.append({
                "id": item_id, "role": "table_value", "row_key": key_norm,
                "candidate": cell_value.strip(), "normalized_candidate": normalized,
                "requires_double_read": bool(re.search(r"\d{8,}", normalized)),
            })
    if items:
        non_table = re.sub(
            r"\\begin\{array\}(?:\{[^{}]*\})?.*?\\end\{array\}", " ", value, flags=re.S,
        )
        non_table = "\n".join(
            line for line in non_table.splitlines() if not line.strip().startswith("|")
        )
        formula_segments = re.findall(
            r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\$([^$]*)\$|\\\((.*?)\\\)",
            non_table, flags=re.S,
        )
        for index, parts in enumerate(formula_segments, 1):
            formula = next((part for part in parts if part), "").strip()
            if not formula:
                continue
            items.append({
                "id": f"context-formula:{index:04d}", "role": "context_formula",
                "candidate": formula,
                "normalized_candidate": normalize_structured_observation(formula, "context_formula"),
                "requires_double_read": bool(re.search(r"\d{8,}", formula)),
            })
        return items
    dense_items = _dense_numeric_sequence_items(value)
    if dense_items:
        return dense_items
    for index, token in enumerate(critical_formula_tokens(value), 1):
        items.append({
            "id": f"critical:{index:04d}", "role": "critical_token",
            "candidate": token, "normalized_candidate": normalize_verified_atom(token),
            "requires_double_read": bool(re.fullmatch(r"\d{8,}", token)),
        })
    return items


def structured_item_map(value: str) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in structured_math_items(value)}


def _bbox_overlap_min(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = left.get("source_position", {}).get("bbox", [])
    b = right.get("source_position", {}).get("bbox", [])
    if len(a) != 4 or len(b) != 4:
        return 0.0
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / min(area_a, area_b)


def load_native_blocks(pdf_path: Path, pages: Iterable[int]) -> dict[int, list[TextBlock]]:
    wanted = sorted({int(page) for page in pages if int(page) > 0})
    result: dict[int, list[TextBlock]] = {}
    document = fitz.open(pdf_path)
    try:
        for page_number in wanted:
            if page_number > len(document):
                continue
            page = document[page_number - 1]
            blocks: list[TextBlock] = []
            raw = page.get_text("blocks", sort=True)
            # Reindex after sorting because the source block number is not a
            # reliable reading-order key across PyMuPDF versions.
            raw.sort(key=lambda item: (round(float(item[1]), 1), round(float(item[0]), 1)))
            for index, item in enumerate(raw):
                text = re.sub(r"\s+", " ", str(item[4])).strip()
                if not text:
                    continue
                blocks.append(TextBlock(
                    page=page_number,
                    index=index,
                    bbox=tuple(float(value) for value in item[:4]),
                    text=text,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                ))
            result[page_number] = blocks
    finally:
        document.close()
    return result


def _content_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Drop likely running headers and page-number-only footers."""
    result = []
    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        # Book running headers occupy roughly the first 30 points.  Test y0,
        # not y1: a two-line header can extend past 31 while still being a
        # header (Guy p.105/B35 is a concrete example).
        if y0 < 32:
            continue
        if y0 > block.page_height - 28 and re.fullmatch(r"[\d\sivxlcdm.-]+", block.text, flags=re.I):
            continue
        result.append(block)
    return result


def _token_similarity(query: list[str], candidate: list[str]) -> tuple[float, int, int]:
    """Return score and candidate token interval corresponding to query."""
    if not query or not candidate:
        return 0.0, 0, 0
    matcher = SequenceMatcher(None, query, candidate, autojunk=False)
    blocks = matcher.get_matching_blocks()
    matched = sum(block.size for block in blocks)
    recall = matched / len(query)
    precision = matched / max(1, min(len(candidate), len(query) * 2))
    score = 0.82 * recall + 0.18 * min(1.0, precision)
    nonempty = [block for block in blocks if block.size]
    if not nonempty:
        return score, 0, 0
    start = min(block.b for block in nonempty)
    end = max(block.b + block.size for block in nonempty)
    return score, start, end


def align_text_to_page(text: str, blocks: list[TextBlock], max_blocks: int = 3) -> dict[str, Any]:
    query = semantic_tokens(text)
    best: dict[str, Any] = {"score": 0.0}
    content = _content_blocks(blocks)
    for left in range(len(content)):
        for width in range(1, max_blocks + 1):
            chosen = content[left:left + width]
            if len(chosen) != width:
                break
            candidate = semantic_tokens(" ".join(block.text for block in chosen))
            score, token_start, token_end = _token_similarity(query, candidate)
            # Prefer the smallest equally good source interval.
            # Round before tie-breaking: otherwise a 0.909999999999 score can
            # lose to a stored 0.91 and leave an unnecessarily wide interval.
            rank = (round(score, 6), -width)
            old_rank = (round(float(best.get("score", 0.0)), 6), -int(best.get("block_count", 999)))
            if rank <= old_rank:
                continue
            x0 = min(block.bbox[0] for block in chosen)
            y0 = min(block.bbox[1] for block in chosen)
            x1 = max(block.bbox[2] for block in chosen)
            y1 = max(block.bbox[3] for block in chosen)
            best = {
                "score": round(score, 6),
                "block_start": chosen[0].index,
                "block_end": chosen[-1].index,
                "block_count": width,
                "bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                "token_start": token_start,
                "token_end": token_end,
                "matched_query_tokens": round(score * len(query), 1),
                "query_tokens": len(query),
            }
    return best


def ground_row(row: dict[str, Any], blocks_by_page: dict[int, list[TextBlock]]) -> dict[str, Any]:
    pages = sorted(set(row.get("source_pages") or ([row.get("source_pdf_page")] if row.get("source_pdf_page") else [])))
    text = row.get("statement", "")
    candidates = []
    for page in pages:
        if page not in blocks_by_page:
            continue
        location = align_text_to_page(text, blocks_by_page[page])
        location.update({"page": page, "method": "native_text_geometry_alignment"})
        candidates.append(location)
    reliable = [item for item in candidates if item.get("score", 0) >= 0.24]
    # A statement starts on its earliest source page.  Only fall forward when
    # that page cannot be aligned at all (common for cross-page continuations).
    earliest_page = min(pages) if pages else None
    earliest = [item for item in reliable if item["page"] == earliest_page]
    chosen = max(earliest or reliable or candidates, key=lambda item: item.get("score", 0), default={})
    value = dict(row)
    value["candidate_locations"] = sorted(candidates, key=lambda item: (-item.get("score", 0), item["page"]))[:4]
    if chosen:
        value["source_position"] = chosen
        value["source_pdf_page"] = chosen["page"]
        value["source_order_key"] = [
            chosen["page"], chosen.get("bbox", [0, 10**9])[1],
            chosen.get("block_start", 10**9), chosen.get("token_start", 0),
        ]
        value["grounding_confidence"] = (
            "high" if chosen.get("score", 0) >= 0.72 else
            "medium" if chosen.get("score", 0) >= 0.48 else "low"
        )
    else:
        page = earliest_page or 10**9
        value["source_position"] = {"page": earliest_page, "method": "page_only_fallback", "score": 0.0}
        fallback_order = value.get("vlm_source_order")
        value["source_order_key"] = [page, 10**9, int(fallback_order) if fallback_order else 10**9, 0]
        value["grounding_confidence"] = "unlocated"
    return value


def _section_query(section: dict[str, Any]) -> str:
    return " ".join(part for part in [section.get("label", ""), section.get("title", "")] if part)


def anchor_sections(
    sections: list[dict[str, Any]], blocks_by_page: dict[int, list[TextBlock]]
) -> list[dict[str, Any]]:
    anchors = []
    for section in sections:
        page = int(section.get("start_page") or section.get("source_pdf_page_start") or 0)
        location = align_text_to_page(_section_query(section), blocks_by_page.get(page, []), max_blocks=1)
        # Heading-only matching can be short; explicit label presence is strong evidence.
        # Search *all* blocks here.  A real heading can start above the ordinary
        # running-header cutoff (Guy p.110/B38 is a regression case).
        label = str(section.get("label") or section.get("section_id") or "")
        explicit = []
        for block in blocks_by_page.get(page, []):
            if not re.search(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", block.text, flags=re.I):
                continue
            score, token_start, token_end = _token_similarity(
                semantic_tokens(_section_query(section)), semantic_tokens(block.text)
            )
            height = max(0.0, block.bbox[3] - block.bbox[1])
            explicit.append((round(score, 6), round(height, 3), -block.bbox[1], block, token_start, token_end))
        if explicit:
            below_header = [item for item in explicit if item[3].bbox[1] >= 32]
            if below_header:
                explicit = below_header
            score, _, _, block, token_start, token_end = max(explicit)
            location = {
                "score": max(0.8, score),
                "block_start": block.index, "block_end": block.index,
                "bbox": [round(value, 3) for value in block.bbox],
                "token_start": token_start, "token_end": token_end,
                "heading_label_exact": True,
            }
        anchor = dict(section)
        anchor["source_position"] = {"page": page, "method": "native_text_heading_alignment", **location}
        anchor["source_order_key"] = [
            page, location.get("bbox", [0, 10**9])[1], location.get("block_start", 10**9), 0,
        ]
        anchors.append(anchor)
    return sorted(anchors, key=lambda row: tuple(row["source_order_key"]))


def section_anchor_order_issues(
    sections: list[dict[str, Any]], anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report structural headings whose localized positions reverse book order."""
    by_identity = {
        (str(anchor.get("label") or anchor.get("section_id") or ""),
         int(anchor.get("start_page") or anchor.get("source_pdf_page_start") or 0)): anchor
        for anchor in anchors
    }
    ordered = []
    for index, section in enumerate(sections):
        identity = (
            str(section.get("label") or section.get("section_id") or ""),
            int(section.get("start_page") or section.get("source_pdf_page_start") or 0),
        )
        if identity in by_identity:
            ordered.append((index, by_identity[identity]))
    labelled = []
    for original_index, anchor in ordered:
        label = str(anchor.get("label") or anchor.get("section_id") or "")
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", label)
        if match:
            labelled.append((match.group(1).casefold(), int(match.group(2)), original_index, anchor))
    if labelled:
        ordered = [
            (original_index, anchor)
            for _, _, original_index, anchor in sorted(labelled, key=lambda item: (item[0], item[1]))
        ]
    issues = []
    for (left_index, left), (right_index, right) in zip(ordered, ordered[1:]):
        left_key, right_key = tuple(left["source_order_key"]), tuple(right["source_order_key"])
        if left_key <= right_key:
            continue
        issues.append({
            "left_structure_index": left_index,
            "right_structure_index": right_index,
            "left_section": left.get("label") or left.get("section_id"),
            "right_section": right.get("label") or right.get("section_id"),
            "left_source_order_key": list(left_key),
            "right_source_order_key": list(right_key),
        })
    return issues


def assign_sections(rows: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    for row in rows:
        key = tuple(row["source_order_key"])
        eligible = [anchor for anchor in anchors if tuple(anchor["source_order_key"]) <= key]
        if eligible:
            row["section_id"] = eligible[-1].get("label") or eligible[-1].get("section_id", "")
            row["section_assignment"] = {
                "method": "preceding_heading_source_position",
                "heading_score": eligible[-1]["source_position"].get("score", 0),
            }


def _intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = left.get("source_position", {})
    b = right.get("source_position", {})
    if a.get("page") != b.get("page") or a.get("block_start") is None or b.get("block_start") is None:
        return False
    if a["block_end"] < b["block_start"] or b["block_end"] < a["block_start"]:
        return False
    if (
        a.get("block_start") == b.get("block_start")
        and a.get("block_end") == b.get("block_end")
        and all(position.get(key) is not None for position in (a, b) for key in ("token_start", "token_end"))
    ):
        # Adjacent statements can share one coarse native-text block. Touching
        # token boundaries are sequential, not overlapping source spans.
        return a["token_end"] > b["token_start"] and b["token_end"] > a["token_start"]
    return True


def _choose_duplicate_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statement = _plain(rows[0].get("statement", "")).lower()
    preferred: list[str] = []
    if re.search(r"\b(?:asks? us to prove|poses? (?:the )?(?:following )?problems?|prove|find|construct|determine)\b", statement):
        preferred = ["exercise"]
    elif re.search(r"\b(?:conjecture[sd]?|asks? (?:if|whether)|is it true|can .*\?|are there|does .*\?)\b", statement):
        preferred = ["conjecture"]
    elif re.search(r"\b(?:denote by|write .* for|put )\b", statement):
        preferred = ["notation", "definition"]
    elif re.search(r"\b(?:define[sd]?|is called|are called)\b", statement):
        preferred = ["definition", "notation"]
    elif re.search(r"\b(?:showed|shown|proved|obtained|established|implies that)\b", statement):
        preferred = ["theorem", "proposition", "lemma", "corollary", "fact"]
    elif re.search(r"\b(?:for example|e\.g\.)\b", statement):
        preferred = ["example"]
    elif re.match(r"(?:take|choose|start with|apply)\b", statement):
        preferred = ["algorithm", "example"]

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        statement = row.get("statement", "")
        cue_bonus = int(bool(re.search(
            r"\b(?:define[sd]?|theorem|conjecture[sd]?|asks?|question|example|exercise|show(?:s|ed)?|prove[sd]?)\b",
            _plain(statement), flags=re.I,
        )))
        return (
            (len(preferred) - preferred.index(row.get("kind"))) if row.get("kind") in preferred else 0,
            cue_bonus,
            float(row.get("source_position", {}).get("score", 0)),
            len(semantic_tokens(statement)),
            -len(row.get("uncertain_spans", [])),
        )
    return max(rows, key=rank)


def merge_format_duplicates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Merge formatting-equivalent entries only when their source spans overlap."""
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row.get("source_pdf_page") or 0), semantic_fingerprint(row.get("statement", "")))].append(row)
    dropped: set[str] = set()
    remap: dict[str, str] = {}
    events = []
    for (_, fingerprint), group in groups.items():
        if not fingerprint or len(group) < 2:
            continue
        # Connected components prevent a same-text occurrence elsewhere on the
        # page from being merged merely because its wording is identical.
        remaining = list(group)
        while remaining:
            component = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for candidate in list(remaining):
                    if any(_intervals_overlap(candidate, member) for member in component):
                        component.append(candidate)
                        remaining.remove(candidate)
                        changed = True
            if len(component) < 2:
                continue
            winner = _choose_duplicate_winner(component)
            source_ids = sorted({row["id"] for row in component})
            winner["candidate_variants"] = [
                {
                    "id": row["id"], "kind": row.get("kind", ""),
                    "statement": row.get("statement", ""),
                    "source_position": row.get("source_position", {}),
                    "canonical_role": "canonical" if row is winner else "superseded",
                }
                for row in component
            ]
            winner["merged_source_ids"] = sorted(set(winner.get("merged_source_ids", []) + source_ids))
            winner["candidate_kinds"] = sorted({row["kind"] for row in component})
            winner["canonicalization_reason"] = "format_equivalent_same_source_span"
            for row in component:
                if row is winner:
                    continue
                dropped.add(row["id"])
                remap[row["id"]] = winner["id"]
            events.append({"winner": winner["id"], "source_ids": source_ids, "reason": winner["canonicalization_reason"]})
    return [row for row in rows if row["id"] not in dropped], remap, events


def merge_near_duplicates(
    rows: list[dict[str, Any]], similarity_threshold: float = 0.94,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Merge near-identical overlapping candidates only when math-critical tokens agree.

    This catches harmless author-name/punctuation variants and one-word truncations,
    while refusing to merge candidates that differ in any mathematical token.
    """
    parent = {row["id"]: row["id"] for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    by_scope: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scope[(int(row.get("source_pdf_page") or 0), row.get("section_id", ""))].append(row)
    pair_scores: dict[tuple[str, str], float] = {}
    for scoped in by_scope.values():
        for index, left in enumerate(scoped):
            for right in scoped[index + 1:]:
                if not _intervals_overlap(left, right):
                    continue
                left_fp = semantic_fingerprint(left.get("statement", ""))
                right_fp = semantic_fingerprint(right.get("statement", ""))
                if not left_fp or not right_fp:
                    continue
                similarity = SequenceMatcher(None, left_fp, right_fp, autojunk=False).ratio()
                if similarity < similarity_threshold:
                    continue
                if critical_formula_tokens(left.get("statement", "")) != critical_formula_tokens(
                    right.get("statement", "")
                ):
                    continue
                pair = tuple(sorted((left["id"], right["id"])))
                pair_scores[pair] = similarity
                union(left["id"], right["id"])
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        components[find(row["id"])].append(row)
    dropped: set[str] = set()
    remap: dict[str, str] = {}
    events = []
    for component in components.values():
        if len(component) < 2:
            continue
        winner = _choose_duplicate_winner(component)
        source_ids = sorted(row["id"] for row in component)
        winner["candidate_variants"] = [
            {
                "id": row["id"], "kind": row.get("kind", ""),
                "statement": row.get("statement", ""),
                "source_position": row.get("source_position", {}),
                "canonical_role": "canonical" if row is winner else "superseded",
            }
            for row in component
        ]
        winner["merged_source_ids"] = sorted(set(winner.get("merged_source_ids", []) + source_ids))
        winner["candidate_kinds"] = sorted({row.get("kind", "") for row in component})
        winner["canonicalization_reason"] = "near_equivalent_same_source_span_math_critical_equal"
        for row in component:
            if row is winner:
                continue
            dropped.add(row["id"]); remap[row["id"]] = winner["id"]
        scores = [
            pair_scores[pair] for pair in pair_scores
            if pair[0] in source_ids and pair[1] in source_ids
        ]
        events.append({
            "winner": winner["id"], "source_ids": source_ids,
            "similarity_min": round(min(scores), 6) if scores else 1.0,
            "reason": winner["canonicalization_reason"],
        })
    return [row for row in rows if row["id"] not in dropped], remap, events


def detect_structured_candidate_conflicts(
    rows: list[dict[str, Any]], excluded_pairs: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Find cell-level disagreements hidden by different object boundaries/kinds."""
    excluded_pairs = excluded_pairs or set()
    by_scope: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scope[(int(row.get("source_pdf_page") or 0), row.get("section_id", ""))].append(row)
    conflicts = []
    for (page, section_id), scoped in by_scope.items():
        for index, left in enumerate(scoped):
            left_map = structured_item_map(left.get("statement", ""))
            left_table = {key: item for key, item in left_map.items() if item["role"] == "table_value"}
            if len(left_table) < 4:
                continue
            for right in scoped[index + 1:]:
                pair = tuple(sorted((left["id"], right["id"])))
                if pair in excluded_pairs or _bbox_overlap_min(left, right) < 0.40:
                    continue
                right_map = structured_item_map(right.get("statement", ""))
                right_table = {key: item for key, item in right_map.items() if item["role"] == "table_value"}
                common = sorted(set(left_table) & set(right_table))
                if len(common) < 4:
                    continue
                differences = []
                checklist = []
                for item_id in common:
                    left_item, right_item = left_table[item_id], right_table[item_id]
                    candidate_values = {
                        left["id"]: left_item["candidate"], right["id"]: right_item["candidate"],
                    }
                    checklist.append({
                        "id": item_id, "role": "table_value",
                        "candidate_values": candidate_values,
                        "requires_double_read": bool(
                            left_item.get("requires_double_read") or right_item.get("requires_double_read")
                        ),
                    })
                    if left_item["normalized_candidate"] != right_item["normalized_candidate"]:
                        differences.append({"item_id": item_id, "candidate_values": candidate_values})
                if not differences:
                    continue
                a, b = left["source_position"]["bbox"], right["source_position"]["bbox"]
                digest = hashlib.sha1(("|".join(pair) + f"|{page}|structured").encode()).hexdigest()[:12]
                conflict = {
                    "conflict_set_id": f"structured:{digest}", "status": "unresolved",
                    "conflict_type": "cross_candidate_table_cell_mismatch",
                    "hard_conflict": True, "pdf_page": page, "section_id": section_id,
                    "kind": "cross_kind_structured_content", "candidate_ids": list(pair),
                    "bbox": [round(min(a[0], b[0]), 3), round(min(a[1], b[1]), 3),
                             round(max(a[2], b[2]), 3), round(max(a[3], b[3]), 3)],
                    "shared_structured_items": len(common),
                    "critical_differences": differences,
                    "verification_checklist": checklist,
                    "candidates": [{
                        "id": row["id"], "kind": row.get("kind", ""),
                        "statement": row.get("statement", ""),
                        "source_position": row.get("source_position", {}),
                    } for row in (left, right)],
                }
                conflicts.append(conflict)
                for row in (left, right):
                    row.setdefault("formula_conflict_set_ids", []).append(conflict["conflict_set_id"])
                    row["canonical_status"] = "unresolved_hard_conflict"
    return conflicts


def detect_formula_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find near-duplicate same-span candidates with semantic-critical formula differences."""
    by_scope: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scope[(int(row.get("source_pdf_page") or 0), row.get("section_id", ""), row.get("kind", ""))].append(row)
    conflicts = []
    seen: set[tuple[str, str]] = set()
    for (page, section_id, kind), scoped in by_scope.items():
        for index, left in enumerate(scoped):
            for right in scoped[index + 1:]:
                pair = tuple(sorted((left["id"], right["id"])))
                if pair in seen or not _intervals_overlap(left, right):
                    continue
                seen.add(pair)
                left_prose, right_prose = prose_skeleton(left.get("statement", "")), prose_skeleton(right.get("statement", ""))
                prose_similarity = SequenceMatcher(None, left_prose, right_prose, autojunk=False).ratio()
                left_sem, right_sem = semantic_fingerprint(left.get("statement", "")), semantic_fingerprint(right.get("statement", ""))
                semantic_similarity = SequenceMatcher(None, left_sem, right_sem, autojunk=False).ratio()
                if left_sem in right_sem or right_sem in left_sem:
                    # One candidate is source context containing the other atomic
                    # statement, not a competing transcription of the same unit.
                    continue
                left_formula, right_formula = formula_tokens(left.get("statement", "")), formula_tokens(right.get("statement", ""))
                if not left_formula or not right_formula or prose_similarity < 0.86 or semantic_similarity < 0.72:
                    continue
                formula_similarity = SequenceMatcher(None, left_formula, right_formula, autojunk=False).ratio()
                left_critical = critical_formula_tokens(left.get("statement", ""))
                right_critical = critical_formula_tokens(right.get("statement", ""))
                if left_critical == right_critical:
                    continue
                matcher = SequenceMatcher(None, left_critical, right_critical, autojunk=False)
                differences = []
                for tag, a0, a1, b0, b1 in matcher.get_opcodes():
                    if tag != "equal":
                        differences.append({
                            "operation": tag,
                            "left": left_critical[a0:a1], "right": right_critical[b0:b1],
                        })
                single_critical_substitution = (
                    len(differences) == 1
                    and differences[0]["operation"] == "replace"
                    and len(differences[0]["left"]) == len(differences[0]["right"]) == 1
                )
                if formula_similarity < 0.90 and not single_critical_substitution:
                    continue
                digest = hashlib.sha1(("|".join(pair) + f"|{page}").encode()).hexdigest()[:12]
                position_left, position_right = left.get("source_position", {}), right.get("source_position", {})
                bbox = [
                    min(position_left["bbox"][0], position_right["bbox"][0]),
                    min(position_left["bbox"][1], position_right["bbox"][1]),
                    max(position_left["bbox"][2], position_right["bbox"][2]),
                    max(position_left["bbox"][3], position_right["bbox"][3]),
                ]
                conflict = {
                    "conflict_set_id": f"formula:{digest}", "status": "unresolved",
                    "conflict_type": "semantic_critical_formula_token_mismatch",
                    "hard_conflict": True, "pdf_page": page, "section_id": section_id,
                    "kind": kind, "bbox": [round(value, 3) for value in bbox],
                    "candidate_ids": list(pair),
                    "prose_similarity": round(prose_similarity, 6),
                    "semantic_similarity": round(semantic_similarity, 6),
                    "formula_similarity": round(formula_similarity, 6),
                    "critical_differences": differences,
                    "candidates": [
                        {"id": row["id"], "statement": row.get("statement", ""),
                         "source_position": row.get("source_position", {})}
                        for row in (left, right)
                    ],
                }
                left_items = structured_item_map(left.get("statement", ""))
                right_items = structured_item_map(right.get("statement", ""))
                common_item_ids = sorted(set(left_items) & set(right_items))
                differing_item_ids = [
                    item_id for item_id in common_item_ids
                    if left_items[item_id].get("normalized_candidate")
                    != right_items[item_id].get("normalized_candidate")
                ]
                # A formula-conflict verifier should inspect the actual candidate
                # delta, not re-audit every identical token in a surrounding
                # matrix.  Whole-table exhaustive checks remain mandatory for
                # cross_candidate_table_cell_mismatch conflicts above.
                if differing_item_ids and len(differing_item_ids) <= 200:
                    conflict["verification_checklist"] = [{
                        "id": item_id,
                        "role": left_items[item_id].get("role", "critical_token"),
                        "candidate_values": {
                            left["id"]: left_items[item_id]["candidate"],
                            right["id"]: right_items[item_id]["candidate"],
                        },
                        "requires_double_read": bool(
                            left_items[item_id].get("requires_double_read")
                            or right_items[item_id].get("requires_double_read")
                        ),
                    } for item_id in differing_item_ids]
                conflicts.append(conflict)
                for row in (left, right):
                    row.setdefault("formula_conflict_set_ids", []).append(conflict["conflict_set_id"])
                    row["canonical_status"] = "unresolved_hard_conflict"
    conflicts.extend(detect_structured_candidate_conflicts(
        rows, {tuple(sorted(conflict["candidate_ids"])) for conflict in conflicts}
    ))
    return sorted(conflicts, key=lambda item: (item["pdf_page"], item["bbox"][1], item["conflict_set_id"]))


def apply_formula_resolutions(
    rows: list[dict[str, Any]], conflicts: list[dict[str, Any]], resolutions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    by_conflict = {item.get("conflict_set_id"): item for item in resolutions if item.get("conflict_set_id")}
    by_candidates = {
        frozenset(item.get("candidate_ids", [])): item for item in resolutions
        if item.get("candidate_ids")
    }
    by_id = {row["id"]: row for row in rows}
    dropped: set[str] = set()
    remap: dict[str, str] = {}
    applied = []

    def canonical_audit_covers_delta(
        conflict: dict[str, Any], resolution: dict[str, Any], selected_id: str,
    ) -> bool:
        """Accept a high-confidence canonical crop audit for a formula-only delta.

        This bridge is deliberately narrow: the canonical verifier must have
        merged the exact same candidate pair, supplied a complete verified
        statement, and the verified formula token stream must contain the
        selected side of every critical difference.  Structured table conflicts
        still require exhaustive cell checks plus an independent observer.
        """
        if not (
            resolution.get("canonical_source_verification") is True
            and conflict.get("conflict_type") == "semantic_critical_formula_token_mismatch"
            and resolution.get("status") == "resolved"
            and resolution.get("confidence") == "high"
            and resolution.get("action") == "merge"
            and str(resolution.get("verified_statement", "")).strip()
            and resolution.get("audit_safety", {}).get("evidence_protocol")
                == "v6_existing_candidate_only"
            and resolution.get("audit_safety", {}).get("model_generated_text_applied") is False
            and resolution.get("canonical_consensus", {}).get("complete") is True
        ):
            return False
        candidate_order = [item.get("id") for item in conflict.get("candidates", [])]
        if selected_id not in candidate_order or len(candidate_order) != 2:
            return False
        selected_side = "left" if candidate_order.index(selected_id) == 0 else "right"
        verified_tokens = critical_formula_tokens(resolution["verified_statement"])
        for difference in conflict.get("critical_differences", []):
            expected = difference.get(selected_side, [])
            if expected and not any(
                verified_tokens[index:index + len(expected)] == expected
                for index in range(len(verified_tokens) - len(expected) + 1)
            ):
                return False
        return bool(conflict.get("critical_differences"))

    for conflict in conflicts:
        resolution = by_conflict.get(conflict["conflict_set_id"])
        candidate_resolution = by_candidates.get(frozenset(conflict.get("candidate_ids", [])))
        if not resolution or resolution.get("status") != "resolved":
            resolution = candidate_resolution or resolution
        if not resolution or resolution.get("status") != "resolved":
            continue
        selected_id = resolution.get("selected_id", "")
        machine_safe = (
            resolution.get("audit_safety", {}).get("evidence_protocol")
            == "v6_existing_candidate_only"
            and resolution.get("audit_safety", {}).get("model_generated_text_applied") is False
        )
        if resolution.get("human_pdf_visual_verification") is not True and not machine_safe:
            continue
        if conflict.get("verification_checklist") and not (
            resolution.get("human_pdf_visual_verification") is True
            or canonical_audit_covers_delta(conflict, resolution, selected_id)
            or (
                resolution.get("structured_verification", {}).get("complete")
                and resolution.get("independent_observer", {}).get("complete")
            )
        ):
            continue
        if selected_id not in conflict["candidate_ids"] or selected_id not in by_id:
            continue
        winner = by_id[selected_id]
        losers = [candidate_id for candidate_id in conflict["candidate_ids"] if candidate_id != selected_id]
        verified_statement = str(resolution.get("verified_statement", "")).strip()
        if (
            resolution.get("human_pdf_visual_verification") is not True
            and verified_statement != str(winner.get("statement", "")).strip()
        ):
            continue
        if verified_statement:
            winner["statement"] = verified_statement
        winner["canonical_status"] = "source_verified"
        winner["source_verification"] = {
            **resolution,
            "method": resolution.get("method", "targeted_vlm_crop"),
            "conflict_set_id": conflict["conflict_set_id"],
            "verified_statement_sha256": hashlib.sha256(
                winner.get("statement", "").encode()
            ).hexdigest(),
        }
        winner["merged_source_ids"] = sorted(set(
            winner.get("merged_source_ids", []) + conflict["candidate_ids"]
        ))
        winner["candidate_variants"] = [
            {**candidate, "canonical_role": (
                "canonical" if candidate.get("id") == selected_id else "superseded"
            )}
            for candidate in conflict["candidates"]
        ]
        winner["canonicalization_reason"] = "source_verified_formula_conflict_resolution"
        winner.pop("formula_conflict_set_ids", None)
        for loser in losers:
            dropped.add(loser); remap[loser] = selected_id
        conflict["status"] = "resolved"
        conflict["resolution"] = resolution
        applied.append(resolution)
    return [row for row in rows if row["id"] not in dropped], remap, applied


def merge_contained_fragments(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Drop a lowercase fragment already contained in a same-span object.

    This is deliberately narrower than ordinary fuzzy/containment deduplication:
    two independently useful mathematical statements are retained even if one
    entails or quotes the other.
    """
    dropped: set[str] = set()
    remap: dict[str, str] = {}
    events = []
    for short in rows:
        short_fp = semantic_fingerprint(short.get("statement", ""))
        if short["id"] in dropped or len(short_fp.split()) < 4 or not _fragment_start(short.get("statement", "")):
            continue
        candidates = [long for long in rows if long["id"] != short["id"]
                      and long["id"] not in dropped and long.get("kind") == short.get("kind")
                      and _intervals_overlap(long, short)]
        candidates = [long for long in candidates
                      if short_fp in semantic_fingerprint(long.get("statement", ""))
                      and len(semantic_tokens(long.get("statement", ""))) > len(semantic_tokens(short.get("statement", "")))]
        if not candidates:
            continue
        winner = max(candidates, key=lambda row: len(semantic_tokens(row.get("statement", ""))))
        winner["merged_source_ids"] = sorted(set(
            winner.get("merged_source_ids", [winner["id"]]) + short.get("merged_source_ids", [short["id"]])
        ))
        winner["canonicalization_reason"] = "lowercase_fragment_contained_in_same_source_span"
        dropped.add(short["id"])
        remap[short["id"]] = winner["id"]
        events.append({"winner": winner["id"], "source_ids": [winner["id"], short["id"]],
                       "reason": winner["canonicalization_reason"]})
    return [row for row in rows if row["id"] not in dropped], remap, events


def _ends_sentence(value: str) -> bool:
    return bool(re.search(r"[.!?][\s)}\]]*(?:\$\$)?$", value.strip()))


def _fragment_start(value: str) -> bool:
    value = re.sub(r"^[#*\s]+", "", _plain(value)).strip()
    if not value:
        return False
    return bool(re.match(
        r"(?:and|or|but|while|whereas|with|whose|which|who|that|the|a|an|its|their|these|those|it|he|she|is|are|has|have)\b",
        value, flags=re.I,
    )) and (value[0].islower() or re.match(r"(?:and|or|but|while|whereas|with|whose|which|who|that)\b", value, flags=re.I))


def merge_sentence_fragments(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: tuple(row["source_order_key"]))
    dropped: set[str] = set()
    remap: dict[str, str] = {}
    events = []
    for left, right in zip(ordered, ordered[1:]):
        if left["id"] in dropped or right["id"] in dropped:
            continue
        a, b = left.get("source_position", {}), right.get("source_position", {})
        same_block = (
            a.get("page") == b.get("page")
            and a.get("block_start") is not None and b.get("block_start") is not None
            and not (a["block_end"] < b["block_start"] or b["block_end"] < a["block_start"])
        )
        close_tokens = int(b.get("token_start", 10**9)) <= int(a.get("token_end", -10)) + 5
        if not (left.get("kind") == right.get("kind")
                and same_block and close_tokens and not _ends_sentence(left.get("statement", ""))
                and _fragment_start(right.get("statement", ""))):
            continue
        left["statement"] = left.get("statement", "").rstrip() + " " + right.get("statement", "").lstrip()
        left["candidate_kinds"] = sorted(set(
            left.get("candidate_kinds", [left["kind"]]) + right.get("candidate_kinds", [right["kind"]])
        ))
        left["merged_source_ids"] = sorted(set(
            left.get("merged_source_ids", [left["id"]]) + right.get("merged_source_ids", [right["id"]])
        ))
        left["canonicalization_reason"] = "adjacent_source_sentence_fragments"
        a["block_end"] = max(a["block_end"], b["block_end"])
        a["token_end"] = max(int(a.get("token_end", 0)), int(b.get("token_end", 0)))
        dropped.add(right["id"])
        remap[right["id"]] = left["id"]
        events.append({"winner": left["id"], "source_ids": [left["id"], right["id"]],
                       "reason": left["canonicalization_reason"]})
    return [row for row in ordered if row["id"] not in dropped], remap, events


def _statement_needs_completion(value: str) -> bool:
    plain = _plain(value).strip()
    return bool(
        not _ends_sentence(value)
        or re.search(r"\b(?:is|are|equals?|given by|not the sum of|specimen is|following)\s*$", plain, flags=re.I)
    )


def attach_statement_completions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        support = row.get("supporting_text", "").strip()
        role = row.get("supporting_text_role", "")
        if not support or not _statement_needs_completion(row.get("statement", "")):
            continue
        if role not in {"", "derivation", "explanation", "answer"}:
            continue
        # A compact display immediately completing an unfinished sentence is
        # statement content, not an answer.  Long prose remains untouched.
        display_like = support.startswith("$$") or support.startswith("\\[") or len(semantic_tokens(support)) <= 35
        if not display_like:
            continue
        row["statement"] = row["statement"].rstrip() + "\n\n" + support
        row["supporting_text"] = ""
        row["supporting_text_role"] = ""
        row["supporting_text_origin"] = ""
        row["support_reclassified_as_statement"] = True
    return rows


def _looks_context_dependent(value: str) -> bool:
    plain = _plain(value).strip()
    if not plain:
        return False
    return bool(re.match(
        r"(?:this|these|those|it|its|they|their|he|she|similarly|otherwise|also|the latter|the former|"
        r"which|who|whose|where|with the same|in this case|in the infinite case|a similar|"
        r"an example|for example|the corresponding|the above|are there|and what)\b",
        plain, flags=re.I,
    ) or re.search(
        r"\b(?:these|those)\s+(?:bounds|results|values|examples|cases|conditions|numbers)|"
        r"\bthe corresponding\s+(?:maximum|minimum|value|number|result|case)\b",
        plain, flags=re.I,
    ) or (plain[0].islower() and len(plain) < 320))


def populate_context(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: tuple(row["source_order_key"]))
    for index, row in enumerate(ordered):
        if not _looks_context_dependent(row.get("statement", "")):
            continue
        previous = next((candidate for candidate in reversed(ordered[:index])
                         if candidate.get("section_id") == row.get("section_id")), None)
        if previous:
            row["context_before"] = previous.get("statement", "")
            row["context_before_id"] = previous["id"]
            row["context_reason"] = "anaphoric_or_lowercase_start"


def factor_repeated_prefix_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a shared example/setup once and make the following question atomic.

    This applies only to literal source-faithful prefixes and an explicit
    question/conjecture suffix.  It never attempts fuzzy semantic subtraction.
    """
    events = []
    host_kinds = {"conjecture", "exercise", "theorem", "proposition", "claim"}
    setup_kinds = {"example", "fact", "notation", "definition", "remark"}
    for host in rows:
        original = str(host.get("statement", "")).strip()
        if host.get("kind") not in host_kinds or len(original) < 80:
            continue
        candidates = []
        for setup in rows:
            prefix = str(setup.get("statement", "")).strip()
            if (
                setup is host or setup.get("kind") not in setup_kinds
                or setup.get("section_id") != host.get("section_id")
                or not _intervals_overlap(setup, host)
                or len(prefix) < 40 or not original.startswith(prefix)
            ):
                continue
            suffix = original[len(prefix):].strip()
            if not re.match(
                r"^(?:are|is|does|do|can|could|will|would|must|should|find|prove|show|determine|"
                r"it is conjectured|the conjecture is|the question is|whether)\b",
                _plain(suffix), flags=re.I,
            ):
                continue
            candidates.append((len(prefix), setup, suffix))
        if not candidates:
            continue
        _, setup, suffix = max(candidates, key=lambda item: item[0])
        host["statement_before_atomic_split"] = original
        host["statement"] = suffix
        host["context_before"] = setup.get("statement", "")
        host["context_before_id"] = setup["id"]
        host["context_reason"] = "literal_repeated_setup_factored_from_atomic_question"
        host["atomic_split"] = {
            "method": "literal_prefix_plus_explicit_question_suffix",
            "setup_id": setup["id"], "source_faithful": True,
        }
        events.append({"id": host["id"], "setup_id": setup["id"]})
    return events


def populate_source_context_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create paragraph/span groups without sacrificing atomic object records."""
    by_scope: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        position = row.get("source_position", {})
        if position.get("block_start") is not None:
            by_scope[(int(position.get("page") or 0), row.get("section_id", ""))].append(row)
    groups = []
    for (page, section_id), scoped in by_scope.items():
        parent = list(range(len(scoped)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = root(left), root(right)
            if a != b:
                parent[b] = a

        for left in range(len(scoped)):
            for right in range(left + 1, len(scoped)):
                if _intervals_overlap(scoped[left], scoped[right]):
                    union(left, right)
        components: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(scoped):
            components[root(index)].append(row)
        for component in components.values():
            if len(component) < 2:
                continue
            component.sort(key=lambda row: tuple(row["source_order_key"]))
            positions = [row["source_position"] for row in component]
            block_start = min(item["block_start"] for item in positions)
            block_end = max(item["block_end"] for item in positions)
            group_id = f"src:p{page}.b{block_start}-{block_end}"
            source_ids = [row["id"] for row in component]
            bbox = [
                min(item["bbox"][0] for item in positions), min(item["bbox"][1] for item in positions),
                max(item["bbox"][2] for item in positions), max(item["bbox"][3] for item in positions),
            ]
            for row in component:
                row["source_context_group_id"] = group_id
                row["source_context_source_ids"] = source_ids
                row_fp = semantic_fingerprint(row.get("statement", ""))
                hosts = [candidate for candidate in component if candidate is not row
                         and row_fp and row_fp in semantic_fingerprint(candidate.get("statement", ""))]
                if hosts:
                    host = max(hosts, key=lambda candidate: len(semantic_tokens(candidate.get("statement", ""))))
                    row["source_context"] = host.get("statement", "")
                    row["source_context_host_id"] = host["id"]
                    row["context_reason"] = "atomic_statement_contained_in_same_source_span"
            groups.append({
                "group_id": group_id, "section_id": section_id, "pdf_page": page,
                "block_start": block_start, "block_end": block_end,
                "bbox": [round(value, 3) for value in bbox], "source_ids": source_ids,
            })
    return sorted(groups, key=lambda group: (group["pdf_page"], group["bbox"][1], group["group_id"]))


def populate_nearby_containment_context(
    rows: list[dict[str, Any]], groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Explain nearby semantic containment without asserting identical geometry."""
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_section[row.get("section_id", "")].append(row)
    added = []
    for section_rows in by_section.values():
        for child in section_rows:
            # A reviewed audit may explicitly clear a misleading host.  Do not
            # let the heuristic containment pass silently recreate that edge.
            if child.get("source_context_suppressed"):
                continue
            if child.get("source_context"):
                continue
            child_fp = semantic_fingerprint(child.get("statement", ""))
            child_compact = semantic_compact(child.get("statement", ""))
            if len(child_fp.split()) < 6:
                continue
            hosts = [host for host in section_rows if host is not child
                     and abs(int(host.get("source_pdf_page") or 0) - int(child.get("source_pdf_page") or 0)) <= 2
                     and (
                         child_fp in semantic_fingerprint(host.get("statement", ""))
                         or child_compact in semantic_compact(host.get("statement", ""))
                     )
                     and len(semantic_tokens(host.get("statement", ""))) > len(semantic_tokens(child.get("statement", "")))]
            if not hosts:
                continue
            host = max(hosts, key=lambda row: len(semantic_tokens(row.get("statement", ""))))
            group_id = "ctx:" + re.sub(r"[^A-Za-z0-9_.-]+", "_", host["id"])
            child["source_context"] = host.get("statement", "")
            child["source_context_host_id"] = host["id"]
            child["semantic_context_group_id"] = group_id
            child["source_context_method"] = "nearby_semantic_containment"
            child["source_context_confidence"] = "medium"
            child["context_reason"] = "atomic_statement_contained_in_nearby_source_passage"
            added.append((group_id, host, child))
    grouped: dict[str, dict[str, Any]] = {}
    for group_id, host, child in added:
        value = grouped.setdefault(group_id, {
            "group_id": group_id, "section_id": host.get("section_id", ""),
            "pdf_page": host.get("source_pdf_page"), "source_pages": sorted(set(
                host.get("source_pages", []) + child.get("source_pages", [])
            )), "bbox": [], "source_ids": [host["id"]],
            "method": "nearby_semantic_containment", "confidence": "medium",
        })
        value["source_ids"] = list(dict.fromkeys(value["source_ids"] + [child["id"]]))
        value["source_pages"] = sorted(set(value["source_pages"] + child.get("source_pages", [])))
    groups.extend(grouped.values())
    return sorted(groups, key=lambda group: (
        int(group.get("pdf_page") or 10**9),
        (group.get("bbox") or [0, 10**9])[1], group["group_id"],
    ))


def ground_canonicalize(
    rows: list[dict[str, Any]], sections: list[dict[str, Any]], pdf_path: Path,
    formula_resolutions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pages = set()
    for row in rows:
        pages.update(page for page in row.get("source_pages", []) if page)
    pages.update(int(section.get("start_page") or 0) for section in sections)
    blocks = load_native_blocks(pdf_path, pages)
    grounded = [ground_row(row, blocks) for row in rows]
    anchors = anchor_sections(sections, blocks)
    anchor_order_issues = section_anchor_order_issues(sections, anchors)
    assign_sections(grounded, anchors)

    grounded, duplicate_remap, duplicate_events = merge_format_duplicates(grounded)
    grounded, near_duplicate_remap, near_duplicate_events = merge_near_duplicates(grounded)
    grounded, contained_remap, contained_events = merge_contained_fragments(grounded)
    grounded = sorted(grounded, key=lambda row: tuple(row["source_order_key"]))
    grounded, fragment_remap, fragment_events = merge_sentence_fragments(grounded)
    grounded = attach_statement_completions(grounded)
    formula_conflicts = detect_formula_conflicts(grounded)
    grounded, formula_remap, applied_resolutions = apply_formula_resolutions(
        grounded, formula_conflicts, formula_resolutions or []
    )
    atomic_prefix_splits = factor_repeated_prefix_context(grounded)
    populate_context(grounded)
    grounded.sort(key=lambda row: tuple(row["source_order_key"]))
    context_groups = populate_source_context_groups(grounded)
    context_groups = populate_nearby_containment_context(grounded, context_groups)

    remap = {
        **duplicate_remap, **near_duplicate_remap, **contained_remap,
        **fragment_remap, **formula_remap,
    }
    report = {
        "schema_version": 1,
        "method": "native_text_geometry_alignment_with_vlm_transcription",
        "objects_in": len(rows),
        "objects_out": len(grounded),
        "source_position_confidence": {
            level: sum(row.get("grounding_confidence") == level for row in grounded)
            for level in ("high", "medium", "low", "unlocated")
        },
        "section_anchors": len(anchors),
        "section_anchor_order_issues": anchor_order_issues,
        "section_ids_used": len({row.get("section_id") for row in grounded if row.get("section_id")}),
        "format_duplicates_merged": duplicate_events,
        "near_duplicates_merged": near_duplicate_events,
        "contained_fragments_merged": contained_events,
        "sentence_fragments_merged": fragment_events,
        "formula_conflicts": formula_conflicts,
        "formula_resolutions_applied": applied_resolutions,
        "atomic_prefix_splits": atomic_prefix_splits,
        "support_reclassified_as_statement": [row["id"] for row in grounded if row.get("support_reclassified_as_statement")],
        "context_links_added": sum(bool(row.get("context_before_id")) for row in grounded),
        "source_context_groups": context_groups,
        "entries_with_source_context": sum(bool(row.get("source_context")) for row in grounded),
        "entries_with_semantic_containment_context": sum(
            row.get("source_context_method") == "nearby_semantic_containment" for row in grounded
        ),
        "id_remap": remap,
        "low_confidence_ids": [row["id"] for row in grounded if row.get("grounding_confidence") in {"low", "unlocated"}],
    }
    return grounded, anchors, report
