#!/usr/bin/env python3
"""Visually correct HDP exercise statements against their source PDF crops.

This is a recoverable post-processing layer for the existing PDF-VQA run.  It
reconstructs each selected question's MinerU block IDs from the cached GPT
segmentation responses, joins those IDs to the original MinerU page/bbox data,
renders tight source crops, and asks a vision model for a conservative
transcription.  The existing dataset is never overwritten.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz
import requests


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "HDP-2.pdf"
SOURCE = ROOT / "output/hdp_pdf_vqa"
OUTPUT = ROOT / "output/hdp_pdf_vqa_gpt55_corrected"
API_URL = os.environ.get("VLM_CHAT_URL", "http://127.0.0.1:3000/v1/chat/completions")
DEFAULT_MODEL = "gpt-5.5"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def xml_field(block: str, tag: str) -> str:
    match = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", block, flags=re.S)
    return match.group(1).strip() if match else ""


def normalize_label(label: str) -> str:
    match = re.search(r"(?:\d+|[A-Z])(?:\.\d+)+", label.replace(" ", ""), flags=re.I)
    return match.group(0) if match else ""


def parse_ids(value: str, layout_size: int) -> list[int]:
    ids: list[int] = []
    for token in value.replace(" ", "").split(","):
        if token.isdigit() and int(token) < layout_size:
            ids.append(int(token))
    return ids


def ids_to_text(ids: list[int], layout: list[dict[str, Any]], image_prefix: str) -> str:
    texts: list[str] = []
    for block_id in ids:
        item = layout[block_id]
        if item.get("text"):
            texts.append(item["text"])
        elif item.get("table_body"):
            texts.append(item["table_body"])
        elif item.get("img_path"):
            caption = " ".join(item.get("image_caption", []) or ["image"])
            name = Path(item["img_path"]).name
            texts.append(f"![{caption}]({image_prefix}/{name})")
    return "\n".join(texts)


def reconstruct_question_sources() -> dict[str, dict[str, Any]]:
    """Repeat the original longest-candidate merge while retaining IDs/bboxes."""
    extracted = json.loads((SOURCE / "mineru/extracted.json").read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for part in extracted:
        part_id = str(part["data_id"])
        layout = json.loads((SOURCE / "layouts" / f"part_{part_id}.json").read_text(encoding="utf-8"))
        original = json.loads(Path(part["content_list"]).read_text(encoding="utf-8"))
        if len(layout) != len(original):
            raise RuntimeError(f"layout/content-list length mismatch in part {part_id}")

        for chunk_path in sorted((SOURCE / "responses").glob(f"part_{part_id}_chunk_*.txt")):
            chunk_index = int(re.search(r"chunk_(\d+)", chunk_path.name).group(1))
            raw = chunk_path.read_text(encoding="utf-8")
            for block in re.findall(r"<qa_pair>(.*?)</qa_pair>", raw, flags=re.S):
                label = normalize_label(xml_field(block, "label"))
                ids = parse_ids(xml_field(block, "question"), len(layout))
                if not label or not ids:
                    continue
                image_prefix = str(Path(part["mineru_dir"]) / "images")
                text = ids_to_text(ids, layout, image_prefix)
                sources = []
                for block_id in ids:
                    item = original[block_id]
                    sources.append({
                        "id": block_id,
                        "page_idx": int(item.get("page_idx") or 0),
                        "bbox": item.get("bbox"),
                        "type": item.get("type"),
                    })
                candidates.append({
                    "label": label,
                    "text": text,
                    "ids": ids,
                    "blocks": sources,
                    "source_part": part_id,
                    "part_start_page": int(part["start_page"]),
                    "source_chunk": chunk_index,
                })

    selected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        current = selected.get(candidate["label"])
        if current is None or len(candidate["text"]) > len(current["text"]):
            selected[candidate["label"]] = candidate

    expected = {
        row["label"]: row["question"]
        for row in json.loads((SOURCE / "qa/questions_raw.json").read_text(encoding="utf-8"))
    }
    if set(selected) != set(expected):
        raise RuntimeError(
            f"source mapping labels differ: missing={sorted(set(expected)-set(selected))}, "
            f"extra={sorted(set(selected)-set(expected))}"
        )
    mismatches = [label for label in expected if selected[label]["text"] != expected[label]]
    if mismatches:
        raise RuntimeError(f"source reconstruction differs for {len(mismatches)} labels: {mismatches[:10]}")
    return selected


def crop_specs(source: dict[str, Any], padding: int = 30) -> list[dict[str, Any]]:
    """Return physical PDF pages and padded normalized bboxes in source order."""
    by_page: dict[int, list[list[float]]] = {}
    for block in source["blocks"]:
        bbox = block.get("bbox")
        if bbox and len(bbox) == 4:
            by_page.setdefault(block["page_idx"], []).append([float(x) for x in bbox])
    specs = []
    for page_idx in sorted(by_page):
        boxes = by_page[page_idx]
        x0 = max(0.0, min(box[0] for box in boxes) - padding)
        y0 = max(0.0, min(box[1] for box in boxes) - padding)
        x1 = min(1000.0, max(box[2] for box in boxes) + padding)
        y1 = min(1000.0, max(box[3] for box in boxes) + padding)
        specs.append({
            "page_idx": page_idx,
            "pdf_page": source["part_start_page"] + page_idx,
            "bbox_1000": [x0, y0, x1, y1],
        })
    if not specs:
        raise RuntimeError(f"no usable bboxes for {source['label']}")
    return specs


def render_crops(label: str, specs: list[dict[str, Any]], zoom: float = 3.0) -> list[Path]:
    crop_dir = OUTPUT / "crops" / label
    crop_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    document = fitz.open(PDF)
    try:
        for order, spec in enumerate(specs):
            path = crop_dir / f"{order:02d}_page_{spec['pdf_page']:03d}.png"
            if not path.exists():
                page = document[spec["pdf_page"] - 1]
                x0, y0, x1, y1 = spec["bbox_1000"]
                rect = page.rect
                clip = fitz.Rect(
                    rect.x0 + rect.width * x0 / 1000.0,
                    rect.y0 + rect.height * y0 / 1000.0,
                    rect.x0 + rect.width * x1 / 1000.0,
                    rect.y0 + rect.height * y1 / 1000.0,
                )
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
                pixmap.save(path)
            paths.append(path)
    finally:
        document.close()
    return paths


def native_text_for_specs(specs: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    document = fitz.open(PDF)
    try:
        for spec in specs:
            page = document[spec["pdf_page"] - 1]
            x0, y0, x1, y1 = spec["bbox_1000"]
            rect = page.rect
            clip = fitz.Rect(
                rect.x0 + rect.width * x0 / 1000.0,
                rect.y0 + rect.height * y0 / 1000.0,
                rect.x0 + rect.width * x1 / 1000.0,
                rect.y0 + rect.height * y1 / 1000.0,
            )
            text = page.get_textbox(clip).strip()
            sections.append(f"[physical PDF page {spec['pdf_page']}]\n{text}")
    finally:
        document.close()
    return "\n\n".join(sections)


def extend_bottom_continuation(
    source: dict[str, Any], specs: list[dict[str, Any]], native_text: str
) -> list[dict[str, Any]]:
    """Add the next page when a bottom-of-page block clearly hides a continuation.

    MinerU occasionally emits one logical text block for text that continues onto
    the following PDF page, while retaining only the first page's bbox.  In that
    case a tight visual crop can omit the formula or sentence tail even though it
    is present in the OCR candidate.  Compare the candidate's final prose words
    with the native crop and conservatively include the top of the next page when
    the tail is absent.
    """
    if not specs or specs[-1]["bbox_1000"][3] < 900:
        return specs
    candidate_words = re.findall(r"[A-Za-z]+", source.get("text", "").lower())
    native_words = set(re.findall(r"[A-Za-z]+", native_text.lower()))
    tail = candidate_words[-12:]
    if len(tail) < 6 or sum(word in native_words for word in tail) >= max(5, len(tail) - 3):
        return specs
    next_pdf_page = specs[-1]["pdf_page"] + 1
    with fitz.open(PDF) as document:
        if next_pdf_page > len(document):
            return specs
    extended = list(specs)
    extended.append({
        "page_idx": specs[-1]["page_idx"] + 1,
        "pdf_page": next_pdf_page,
        "bbox_1000": [80.0, 60.0, 920.0, 600.0],
        "auto_continuation": True,
    })
    return extended


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def correction_prompt(label: str, candidate: str, native_text: str, preserve_difficulty: bool) -> str:
    difficulty_rule = (
        "Preserve the candidate's K/KK/KKK difficulty marker as plain K characters; do not turn it into icons."
        if preserve_difficulty else
        "This book has no K/KK/KKK difficulty markers. Do not invent one; a mathematical italic k at the start of a title remains $k$."
    )
    return f"""You are the final visual transcription corrector for a mathematics textbook dataset.

The attached image(s), in order, are tight crops from the authoritative PDF and together contain
exactly Exercise {label}. The OCR/VQA candidate and the PDF's native text layer are independent
secondary evidence. The native text often identifies individual letters correctly but may lose
formula layout. Transcribe the exercise by reconciling all evidence, with the images authoritative.

Hard requirements:
1. Return only Exercise {label}. Exclude page headers, page numbers, footnotes, hints, solutions,
   and adjacent exercises.
2. Be source-faithful. Do not solve, paraphrase, improve, infer a more plausible theorem, change
   dimensions, or repair an apparent mathematical inconsistency that is really printed in the book.
3. Preserve all hypotheses, subparts, primes, transposes, accents, subscripts, superscripts,
   delimiters, norm layers, signs, Greek letters, equation order, and prose order.
4. Use Markdown with $...$ and $$...$$ LaTeX. Normalizing equivalent LaTeX syntax and spacing is OK.
5. Correct unmistakable OCR damage and obvious English spelling errors, but if unusual wording is
   visibly printed, preserve it. Join words split only by OCR or line wrapping.
6. Begin corrected_question with the exact label `{label}`. {difficulty_rule}
7. Never duplicate text. If a symbol remains genuinely unreadable, give the best visual
   transcription and disclose it under uncertain_spans.

Return exactly the following XML-like envelope without a Markdown fence. Write LaTeX normally;
do not JSON-escape backslashes.
<label>{label}</label>
<corrected_question>
complete corrected Markdown/LaTeX
</corrected_question>
<changes>
one concise `FROM => TO | reason` entry per line, or empty
</changes>
<uncertain_spans>
one entry per line, or empty
</uncertain_spans>

Candidate:
---
{candidate}
---

PDF native text layer (secondary evidence; layout may be flattened):
---
{native_text}
---
"""


def verification_prompt(
    label: str, original: str, proposed: str, native_text: str, preserve_difficulty: bool
) -> str:
    difficulty_rule = (
        "preserve the K/KK/KKK difficulty marker as plain K characters"
        if preserve_difficulty else
        "do not invent K/KK/KKK difficulty markers; this book has none"
    )
    return f"""You are the independent final verifier of a mathematics textbook transcription.

The attached source crop(s) are authoritative. Compare BOTH transcriptions below against the
images character by character and return the source-faithful final version of Exercise {label}.
The first version may contain MinerU OCR damage. The second version may contain silent VLM edits.
The PDF native text layer is a third, independent clue: it may flatten formula layout, but often
resolves visually similar letters such as p/n, l/1, and x/X.

Verification rules:
1. Audit every mathematical difference between ORIGINAL and PROPOSED against the visible source,
   especially variables under roots, exponents, nested norms, dimensions, domains, signs, primes,
   transposes, bounds, and the ordering of subparts.
2. Do not prefer PROPOSED merely because it is newer or prettier. Keep ORIGINAL wherever it matches
   the source. If both are wrong, transcribe the visible source directly. When the image is genuinely
   ambiguous, use agreement with the native text layer; if evidence still conflicts, keep ORIGINAL
   and disclose the exact conflict instead of silently changing a mathematical token.
3. Do not solve, paraphrase, repair a printed mathematical inconsistency, or infer what the author
   probably intended.
4. Exclude headers, page numbers, footnotes, adjacent exercises, hints, and solutions. Never
   duplicate text.
5. Use Markdown with $...$ and $$...$$ LaTeX. Correct unmistakable OCR spelling damage. Begin with
   the exact label `{label}` and {difficulty_rule}.
6. If any source symbol remains genuinely unreadable, preserve the best visible transcription and
   disclose it; otherwise leave uncertain_spans empty.

Return exactly this XML-like envelope, without a Markdown fence:
<label>{label}</label>
<final_question>
complete verified Markdown/LaTeX
</final_question>
<verdict_notes>
one concise verified decision per line, especially for rejected proposed changes
</verdict_notes>
<uncertain_spans>
one entry per line, or empty
</uncertain_spans>

ORIGINAL:
---
{original}
---

PROPOSED:
---
{proposed}
---

PDF NATIVE TEXT LAYER (secondary evidence; formula layout may be flattened):
---
{native_text}
---
"""


def extract_content(payload: dict[str, Any]) -> str:
    content = payload["choices"][0]["message"].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def parse_response(raw: str) -> dict[str, Any]:
    def field(name: str) -> str:
        match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", raw, flags=re.S)
        return match.group(1).strip() if match else ""

    label = normalize_label(field("label"))
    question = field("corrected_question")
    if not label or not question:
        raise ValueError("required XML-like fields are missing")
    uncertain = [line.strip() for line in field("uncertain_spans").splitlines() if line.strip()]
    uncertain = [
        line for line in uncertain if line.lower().strip(" .:-") not in {"empty", "none", "n/a"}
    ]
    return {
        "label": label,
        "corrected_question": question,
        "changes": [line.strip() for line in field("changes").splitlines() if line.strip()],
        "uncertain_spans": uncertain,
    }


def parse_verification_response(raw: str) -> dict[str, Any]:
    def field(name: str) -> str:
        match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", raw, flags=re.S)
        return match.group(1).strip() if match else ""

    label = normalize_label(field("label"))
    question = field("final_question")
    if not label or not question:
        raise ValueError("required verification fields are missing")
    uncertain = [line.strip() for line in field("uncertain_spans").splitlines() if line.strip()]
    uncertain = [
        line for line in uncertain if line.lower().strip(" .:-") not in {"empty", "none", "n/a"}
    ]
    return {
        "label": label,
        "corrected_question": question,
        "changes": [line.strip() for line in field("verdict_notes").splitlines() if line.strip()],
        "uncertain_spans": uncertain,
    }


def subparts(text: str) -> set[str]:
    return set(re.findall(r"(?:^|\n)\s*\(([a-z])\)", text))


def validate(label: str, candidate: str, parsed: dict[str, Any]) -> list[str]:
    question = parsed["corrected_question"]
    errors: list[str] = []
    if parsed["label"] != label:
        errors.append(f"response label {parsed['label']} != {label}")
    if not re.match(fr"\s*{re.escape(label)}(?!\d)", question):
        errors.append("corrected text does not begin with the requested label")
    ratio = len(question) / max(1, len(candidate))
    # A valid visual correction may be much shorter when the MinerU candidate
    # contains an embedded-image path, figure caption, or a long footnote.
    if ratio < 0.20 or ratio > 2.2:
        errors.append(f"suspicious length ratio {ratio:.2f}")
    missing_subparts = subparts(candidate) - subparts(question)
    if missing_subparts:
        errors.append(f"missing subparts {sorted(missing_subparts)}")
    if question.count("$$") % 2:
        errors.append("unbalanced display-math delimiters")
    if question.count("$") % 2:
        errors.append("unbalanced dollar delimiters")
    return errors


def request_correction(
    *, label: str, candidate: str, crop_paths: list[Path], specs: list[dict[str, Any]],
    native_text: str, preserve_difficulty: bool, model: str, api_key: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{
        "type": "text", "text": correction_prompt(label, candidate, native_text, preserve_difficulty)
    }]
    for path, spec in zip(crop_paths, specs):
        content.append({
            "type": "text",
            "text": f"Source crop {path.name}, physical PDF page {spec['pdf_page']}:",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url(path), "detail": "high"},
        })
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 12000,
    }
    last_error = ""
    for attempt in range(5):
        started = time.time()
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=1200,
            )
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            payload = response.json()
            raw = extract_content(payload)
            parsed = parse_response(raw)
            validation_errors = validate(label, candidate, parsed)
            result = {
                "label": label,
                "model": model,
                "pages": [spec["pdf_page"] for spec in specs],
                "crop_specs": specs,
                "candidate_question": candidate,
                "raw_response": raw,
                "parsed": parsed,
                "validation_errors": validation_errors,
                "elapsed_seconds": round(time.time() - started, 3),
                "usage": payload.get("usage", {}),
                "attempt": attempt + 1,
            }
            if not validation_errors:
                return result
            last_error = "; ".join(validation_errors)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 4:
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"correction failed validation after retries: {last_error}")


def request_verification(
    *, label: str, original: str, proposed: str, crop_paths: list[Path],
    specs: list[dict[str, Any]], native_text: str, preserve_difficulty: bool,
    model: str, api_key: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": verification_prompt(label, original, proposed, native_text, preserve_difficulty),
    }]
    for path, spec in zip(crop_paths, specs):
        content.append({
            "type": "text",
            "text": f"Authoritative crop {path.name}, physical PDF page {spec['pdf_page']}:",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url(path), "detail": "high"},
        })
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 12000,
    }
    last_error = ""
    for attempt in range(5):
        started = time.time()
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=1200,
            )
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            payload = response.json()
            raw = extract_content(payload)
            parsed = parse_verification_response(raw)
            validation_errors = validate(label, original, parsed)
            result = {
                "raw_response": raw,
                "parsed": parsed,
                "validation_errors": validation_errors,
                "elapsed_seconds": round(time.time() - started, 3),
                "usage": payload.get("usage", {}),
                "attempt": attempt + 1,
            }
            if not validation_errors:
                return result
            last_error = "; ".join(validation_errors)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 4:
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"verification failed after retries: {last_error}")


def label_sort_key(label: str) -> tuple[int, ...]:
    parts = label.upper().split(".")
    head = int(parts[0]) if parts[0].isdigit() else 1000 + ord(parts[0]) - ord("A")
    return (head, *(int(value) for value in parts[1:]))


def build_outputs(model: str, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_rows = json.loads((SOURCE / "qa/questions.json").read_text(encoding="utf-8"))
    standard_rows = []
    audit_rows = []
    failed = []
    changed = 0
    uncertain = 0
    for row in baseline_rows:
        label = row["label"]
        result_path = OUTPUT / "corrections" / model / f"{label}.json"
        if not result_path.exists():
            failed.append(label)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("error") or not result.get("parsed") or result.get("validation_errors"):
            failed.append(label)
            continue
        corrected = result["parsed"]["corrected_question"]
        updated = dict(row)
        updated["question"] = corrected
        standard_rows.append(updated)
        if corrected != row["question"]:
            changed += 1
        if result["parsed"].get("uncertain_spans"):
            uncertain += 1
        audit_rows.append({
            "label": label,
            "pages": result["pages"],
            "question_before": row["question"],
            "question_after": corrected,
            "changes": result["parsed"].get("changes", []),
            "uncertain_spans": result["parsed"].get("uncertain_spans", []),
            "source_part": sources[label]["source_part"],
            "source_block_ids": sources[label]["ids"],
            "elapsed_seconds": result.get("elapsed_seconds"),
            "usage": result.get("usage", {}),
        })

    standard_rows.sort(key=lambda row: label_sort_key(row["label"]))
    audit_rows.sort(key=lambda row: label_sort_key(row["label"]))
    write_json(OUTPUT / "qa/questions.json", standard_rows)
    write_json(OUTPUT / "qa/visual_correction_audit.json", audit_rows)
    (OUTPUT / "qa/questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in standard_rows),
        encoding="utf-8",
    )
    markdown: list[str] = []
    for row in standard_rows:
        markdown.extend([f"### Exercise {row['label']}", "", row["question"], ""])
        if row.get("solution"):
            markdown.extend(["**Solution/Hint:**", "", row["solution"], ""])
        markdown.extend(["---", ""])
    (OUTPUT / "qa/questions.md").write_text("\n".join(markdown), encoding="utf-8")
    summary = {
        "pdf": str(PDF),
        "source_dataset": str(SOURCE / "qa/questions.json"),
        "visual_correction_model": model,
        "questions_expected": len(baseline_rows),
        "questions_written": len(standard_rows),
        "questions_changed": changed,
        "questions_with_uncertainty": uncertain,
        "failed_labels": failed,
    }
    write_json(OUTPUT / "summary.json", summary)
    return summary


def main() -> None:
    global PDF, SOURCE, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument(
        "--no-difficulty-markers", action="store_true",
        help="the source book has no HDP-style K/KK/KKK exercise difficulty markers",
    )
    args = parser.parse_args()

    PDF = args.pdf.resolve()
    SOURCE = args.source.resolve()
    OUTPUT = args.output.resolve()

    sources = reconstruct_question_sources()
    write_json(OUTPUT / "question_sources.json", sources)
    all_labels = sorted(sources, key=label_sort_key)
    labels = args.labels or all_labels
    unknown = sorted(set(labels) - set(all_labels), key=label_sort_key)
    if unknown:
        raise SystemExit(f"unknown labels: {unknown}")

    crop_manifest = {}
    for label in labels:
        specs = crop_specs(sources[label])
        native_text = native_text_for_specs(specs)
        specs = extend_bottom_continuation(sources[label], specs, native_text)
        paths = render_crops(label, specs)
        crop_manifest[label] = {
            "specs": specs,
            "paths": [str(path) for path in paths],
            "native_text": native_text_for_specs(specs),
        }
    write_json(OUTPUT / "crop_manifest.json", crop_manifest)
    if args.prepare_only:
        print(json.dumps({"prepared": len(labels)}, ensure_ascii=False), flush=True)
        return

    api_key = os.environ.get("DF_API_KEY")
    if not api_key:
        raise SystemExit("DF_API_KEY is required unless --prepare-only is used")
    baseline = {
        row["label"]: row["question"]
        for row in json.loads((SOURCE / "qa/questions.json").read_text(encoding="utf-8"))
    }
    jobs = []
    for label in labels:
        result_path = OUTPUT / "corrections" / args.model / f"{label}.json"
        if result_path.exists() and not args.force:
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            complete = (
                not cached.get("error") and cached.get("parsed") and
                not cached.get("validation_errors") and
                (args.single_pass or cached.get("verification"))
            )
            if complete and (not args.retry_errors or complete):
                continue
        jobs.append((label, result_path))

    def run(job: tuple[str, Path]) -> dict[str, Any]:
        label, result_path = job
        manifest = crop_manifest[label]
        try:
            crop_paths = [Path(path) for path in manifest["paths"]]
            cached = None
            if result_path.exists() and not args.force:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached and not cached.get("error") and cached.get("parsed"):
                result = cached
                initial = cached.get("initial_parsed", cached["parsed"])
            else:
                result = request_correction(
                    label=label,
                    candidate=baseline[label],
                    crop_paths=crop_paths,
                    specs=manifest["specs"],
                    native_text=manifest["native_text"],
                    preserve_difficulty=not args.no_difficulty_markers,
                    model=args.model,
                    api_key=api_key,
                )
                initial = result["parsed"]
            if not args.single_pass:
                verification = request_verification(
                    label=label,
                    original=baseline[label],
                    proposed=initial["corrected_question"],
                    crop_paths=crop_paths,
                    specs=manifest["specs"],
                    native_text=manifest["native_text"],
                    preserve_difficulty=not args.no_difficulty_markers,
                    model=args.model,
                    api_key=api_key,
                )
                result["initial_parsed"] = initial
                result["verification"] = verification
                result["parsed"] = verification["parsed"]
                result["validation_errors"] = verification["validation_errors"]
        except Exception as exc:
            result = {"label": label, "model": args.model, "error": f"{type(exc).__name__}: {exc}"}
        write_json(result_path, result)
        return result

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            status = "error" if result.get("error") else "ok"
            print(f"[{completed}/{len(jobs)}] {result['label']} {status}", flush=True)

    summary = build_outputs(args.model, sources)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
