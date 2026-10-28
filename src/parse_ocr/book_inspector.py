#!/usr/bin/env python3
"""Render representative textbook pages and optionally suggest a book profile with a VLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz

from parse_ocr.common import call_chat, image_data_url, load_json, resolve_config, write_json


SIGNALS = {
    "theorem": re.compile(r"\b(theorem|lemma|proposition|corollary|definition)\b", re.I),
    "exercise": re.compile(r"\b(exercises?|problems?)\b", re.I),
    "support": re.compile(r"\b(hints?|answers?|solutions?)\b", re.I),
    "example": re.compile(r"\bexamples?\b", re.I),
    "algorithm": re.compile(r"\balgorithms?\b", re.I),
    "appendix": re.compile(r"\bappendi(?:x|ces)\b", re.I),
}

HEADING_SIGNALS = {
    "theorem": re.compile(r"^(theorem|lemma|proposition|corollary|definition)\b", re.I),
    "exercise": re.compile(r"^(exercises?|problems?)\s*$", re.I),
    "support": re.compile(r"^(hints?(?:\s+for\s+.*)?|answers?(?:\s+to\s+.*)?|solutions?(?:\s+to\s+.*)?)\s*$", re.I),
    "example": re.compile(r"^examples?\b", re.I),
    "algorithm": re.compile(r"^algorithms?\b", re.I),
    "appendix": re.compile(r"^appendix\b", re.I),
}


def toc_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    numbered_tails = sum(bool(re.search(r"(?:\.{2,}|\s)\s*\d+\s*$", line)) for line in lines)
    return "contents" in " ".join(lines[:8]).lower() or numbered_tails / len(lines) > 0.28


def choose_pages(document: fitz.Document, limit: int) -> list[tuple[int, list[str]]]:
    count = len(document)
    selected: dict[int, set[str]] = {}
    coverage_slots = max(4, limit - len(SIGNALS))
    for i in range(coverage_slots):
        page = round((count - 1) * i / max(1, coverage_slots - 1))
        selected.setdefault(page, set()).add("coverage")

    candidates: dict[str, list[tuple[float, int]]] = {name: [] for name in SIGNALS}
    body_start = max(6, round(count * 0.04))
    for page_index in range(body_start, count):
        text = document[page_index].get_text("text")[:16000]
        if len(text) < 250 or toc_like(text):
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for name, pattern in SIGNALS.items():
            matches = len(pattern.findall(text))
            if not matches:
                continue
            exact_heading = any(HEADING_SIGNALS[name].search(line) for line in lines[:35])
            if name in {"exercise", "support", "appendix"} and not exact_heading:
                continue
            position_bonus = page_index / max(1, count - 1) if name in {"support", "appendix"} else 0
            score = (100 if exact_heading else 0) + matches * 5 + min(len(text) / 1000, 8) + position_bonus
            candidates[name].append((score, page_index))
    for name, values in candidates.items():
        if values:
            _, page = max(values)
            selected.setdefault(page, set()).add(name)

    signaled = sorted((page, reasons) for page, reasons in selected.items() if reasons != {"coverage"})
    coverage = sorted((page, reasons) for page, reasons in selected.items() if reasons == {"coverage"})
    chosen = signaled[:limit]
    for item in coverage:
        if len(chosen) >= limit:
            break
        if item[0] not in {page for page, _ in chosen}:
            chosen.append(item)
    return sorted((page, sorted(reasons)) for page, reasons in chosen)


def suggest_profile(config: dict[str, Any], manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    prompt = """You are auditing the visual layout of a mathematics textbook before extraction.
Infer only layout and labeling conventions visible in the supplied representative pages. Do not
transcribe or solve mathematics. Return ONLY one JSON object with these keys:
title, languages, include_unnumbered_named, kind_aliases, label_regex, numbering_scope,
question_section_headings, support_section_headings, distant_support_alignment,
preserve_markers, worked_example_policy, figure_policy, segmentation_prompt_overlay,
visual_prompt_overlay, inspection_notes. Keep prompt overlays concise and evidence-based. Preserve
the existing all-mathematical-object scope. If evidence is insufficient, retain the current value
and record uncertainty in inspection_notes.

CURRENT PROFILE:
""" + json.dumps(profile, ensure_ascii=False)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for sample in manifest["samples"]:
        path = Path(sample["image"])
        content.append({"type": "text", "text": f"Physical PDF page {sample['pdf_page']}; selection reasons: {sample['reasons']}"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path), "detail": "high"}})
    raw, usage = call_chat(config, content, max_tokens=12000)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise RuntimeError("VLM profile response did not contain a JSON object")
    result = json.loads(match.group(0))
    result["schema_version"] = 1
    result["book_id"] = profile["book_id"]
    result["kinds"] = profile["kinds"]
    result["expected_counts"] = profile.get("expected_counts", {})
    return {"profile": result, "usage": usage, "raw": raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--zoom", type=float, default=1.6)
    parser.add_argument("--auto", action="store_true", help="ask the configured VLM for a draft profile")
    args = parser.parse_args()
    config = resolve_config(args.config)
    pdf = Path(config["pdf"])
    output = Path(config["output"]) / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf)
    page_count = len(document)
    samples = []
    try:
        for page_index, reasons in choose_pages(document, max(6, args.pages)):
            image = output / f"page_{page_index + 1:04d}.png"
            if not image.exists():
                pixmap = document[page_index].get_pixmap(
                    matrix=fitz.Matrix(args.zoom, args.zoom), alpha=False
                )
                pixmap.save(image)
            samples.append({
                "pdf_page": page_index + 1,
                "reasons": reasons,
                "image": str(image.resolve()),
                "native_text_preview": document[page_index].get_text("text")[:1200],
            })
    finally:
        document.close()
    manifest = {"pdf": str(pdf), "page_count": page_count, "samples": samples}
    write_json(output / "manifest.json", manifest)
    result: dict[str, Any] = {"ok": True, "manifest": str(output / "manifest.json"), "samples": len(samples)}
    if args.auto:
        profile = load_json(Path(config["profile"]))
        suggestion = suggest_profile(config, manifest, profile)
        write_json(output / "profile_suggestion.json", suggestion)
        write_json(output / "book_profile.suggested.json", suggestion["profile"])
        result["suggested_profile"] = str(output / "book_profile.suggested.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
