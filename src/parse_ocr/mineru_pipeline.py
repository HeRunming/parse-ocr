#!/usr/bin/env python3
"""Recoverable MinerU + VLM pipeline for all textbook mathematical environments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz

from parse_ocr.common import call_chat, image_data_url, load_json, request, require_secret, resolve_config, write_json


KIND_ORDER = {name: index for index, name in enumerate([
    "definition", "theorem", "lemma", "proposition", "corollary",
    "claim", "fact", "example", "exercise", "algorithm", "remark",
])}
SUPPORT_ROLES = {"", "proof", "derivation", "explanation", "answer", "hint", "solution", "hint_or_solution"}


def choose_mineru_backend(config: dict[str, Any]) -> str:
    mineru = config["mineru"]
    output = Path(config["output"])
    requested = mineru.get("backend", "auto")
    if requested != "auto":
        return requested
    if (output / "mineru" / "extracted.json").is_file():
        return "cache"
    if mineru.get("local_command"):
        return "local"
    if os.environ.get(mineru.get("api_key_env", "MINERU_API_KEY")):
        return "api"
    raise RuntimeError("no usable MinerU cache, local command, or API credential")


def split_pdf(config: dict[str, Any]) -> list[dict[str, Any]]:
    pdf, output = Path(config["pdf"]), Path(config["output"])
    max_pages = int(config["mineru"].get("max_pages_per_part", 200))
    parts_dir = output / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    source = fitz.open(pdf)
    parts = []
    try:
        for start in range(0, len(source), max_pages):
            end = min(start + max_pages, len(source))
            path = parts_dir / f"{pdf.stem}_pages_{start + 1:04d}_{end:04d}.pdf"
            if not path.exists():
                part = fitz.open()
                part.insert_pdf(source, from_page=start, to_page=end - 1)
                part.save(path)
                part.close()
            with fitz.open(path) as check:
                if len(check) != end - start:
                    raise RuntimeError(f"bad PDF part: {path}")
            parts.append({"path": str(path), "start_page": start + 1, "end_page": end})
    finally:
        source.close()
    write_json(output / "parts.json", parts)
    return parts


def mineru_api(config: dict[str, Any], parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output, mineru = Path(config["output"]), config["mineru"]
    token = require_secret(mineru.get("api_key_env", "MINERU_API_KEY"))
    base = mineru.get("api_base", "https://mineru.net/api/v4").rstrip("/")
    state_path = output / "mineru" / "state.json"
    state = load_json(state_path) if state_path.exists() else {}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if "batch_id" not in state:
        payload = {
            "files": [{"name": Path(part["path"]).name, "data_id": str(i)} for i, part in enumerate(parts)],
            "model_version": mineru.get("model_version", "vlm"),
        }
        response = request("POST", f"{base}/file-urls/batch", headers=headers, json=payload, timeout=120).json()
        if response.get("code") != 0:
            raise RuntimeError(f"MinerU upload URL request failed: {response}")
        state.update(response["data"])
        state["model_version"] = payload["model_version"]
        write_json(state_path, state)
    if not state.get("uploaded"):
        for part, upload_url in zip(parts, state["file_urls"]):
            with open(part["path"], "rb") as stream:
                request("PUT", upload_url, data=stream, timeout=300)
        state["uploaded"] = True
        write_json(state_path, state)
    poll_url = f"{base}/extract-results/batch/{state['batch_id']}"
    deadline = time.time() + int(mineru.get("poll_timeout_seconds", 7200))
    while True:
        response = request("GET", poll_url, headers=headers, timeout=120).json()
        if response.get("code") != 0:
            raise RuntimeError(f"MinerU polling failed: {response}")
        results = response.get("data", {}).get("extract_result", [])
        if results and all(item.get("state") == "done" for item in results):
            state["results"] = results
            write_json(state_path, state)
            break
        failed = [item for item in results if item.get("state") == "failed"]
        if failed:
            raise RuntimeError(f"MinerU extraction failed: {failed}")
        if time.time() > deadline:
            raise TimeoutError(f"MinerU polling timed out: {state['batch_id']}")
        print("MinerU states:", [item.get("state") for item in results], flush=True)
        time.sleep(10)
    return materialize_mineru_results(config, parts, state["results"])


def materialize_mineru_results(
    config: dict[str, Any], parts: list[dict[str, Any]], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = Path(config["output"])
    extracted = []
    for result in results:
        data_id = str(result["data_id"])
        part_dir = output / "mineru" / data_id
        content_lists = list(part_dir.rglob("*_content_list.json"))
        if not content_lists:
            archive_path = output / "mineru" / f"{data_id}.zip"
            response = request("GET", result["full_zip_url"], timeout=300)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(response.content)
            part_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(part_dir)
            content_lists = list(part_dir.rglob("*_content_list.json"))
        if not content_lists:
            raise RuntimeError(f"MinerU content list missing under {part_dir}")
        extracted.append({
            **parts[int(data_id)], "data_id": data_id,
            "content_list": str(content_lists[0]), "mineru_dir": str(part_dir),
        })
    write_json(output / "mineru" / "extracted.json", extracted)
    return extracted


def mineru_local(config: dict[str, Any], parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output, command = Path(config["output"]), config["mineru"].get("local_command", [])
    if not isinstance(command, list) or not command:
        raise RuntimeError("mineru.local_command must be a nonempty argv list")
    extracted = []
    for index, part in enumerate(parts):
        part_dir = output / "mineru" / str(index)
        part_dir.mkdir(parents=True, exist_ok=True)
        argv = [str(value).format(input=part["path"], output=str(part_dir)) for value in command]
        subprocess.run(argv, check=True)
        content_lists = list(part_dir.rglob("*_content_list.json"))
        if not content_lists:
            raise RuntimeError(f"local MinerU produced no *_content_list.json under {part_dir}")
        extracted.append({
            **part, "data_id": str(index), "content_list": str(content_lists[0]),
            "mineru_dir": str(part_dir),
        })
    write_json(output / "mineru" / "extracted.json", extracted)
    return extracted


def run_mineru(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = Path(config["output"])
    backend = choose_mineru_backend(config)
    if backend == "cache":
        return load_json(output / "mineru" / "extracted.json")
    parts = split_pdf(config)
    if backend == "api":
        return mineru_api(config, parts)
    if backend == "local":
        return mineru_local(config, parts)
    raise RuntimeError(f"unsupported MinerU backend: {backend}")


def compact_layout(content_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original = load_json(content_path)
    compact = []
    for block_id, item in enumerate(original):
        block = {"id": block_id, "page_idx": int(item.get("page_idx") or 0), "type": item.get("type", "")}
        text = item.get("text") or item.get("table_body")
        if not text and item.get("list_items"):
            text = "\n".join(item["list_items"])
        if text:
            block["text"] = text
        if item.get("image_caption"):
            block["image_caption"] = item["image_caption"]
        if item.get("img_path"):
            block["image"] = Path(item["img_path"]).name
        compact.append(block)
    return compact, original


def page_chunks(layout: list[dict[str, Any]], target_chars: int, overlap: int) -> list[list[dict[str, Any]]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for item in layout:
        by_page.setdefault(item["page_idx"], []).append(item)
    pages = sorted(by_page)
    if not pages:
        return []
    ranges, start, previous, size = [], pages[0], pages[0], 0
    for page in pages:
        page_size = len(json.dumps(by_page[page], ensure_ascii=False))
        if size and size + page_size > target_chars:
            ranges.append((start, previous)); start, size = page, 0
        size += page_size; previous = page
    ranges.append((start, previous))
    low, high = pages[0], pages[-1]
    return [[item for page in range(max(low, start - overlap), min(high, end + overlap) + 1)
             for item in by_page.get(page, [])] for start, end in ranges]


def segmentation_prompt(profile: dict[str, Any]) -> str:
    kinds = ", ".join(profile["kinds"])
    aliases = json.dumps(profile.get("kind_aliases", {}), ensure_ascii=False)
    support_headings = ", ".join(profile.get("support_section_headings", []))
    unnumbered_rule = (
        "For a complete explicitly headed but unnumbered object, leave label empty; the engine "
        "will assign a source-derived label."
        if profile.get("include_unnumbered_named")
        else "Ignore unnumbered objects."
    )
    return f"""You are a source-indexing model for a mathematics textbook. Input is a JSON array
of MinerU layout blocks with stable integer id and page_idx. Select IDs only; never transcribe,
paraphrase, solve, correct, or create mathematical content.

Extract every complete named or numbered mathematical environment of these canonical kinds:
{kinds}. Kind aliases: {aliases}. Ignore references, contents/index entries, equation numbers,
ordinary prose, and orphan fragments.

Return zero or more exact envelopes, without commentary:
<environment>
<kind>canonical lowercase kind</kind>
<label>full printed label without kind word</label>
<title>printed title or empty</title>
<statement_ids>comma-separated block IDs</statement_ids>
<supporting_ids>comma-separated block IDs or empty</supporting_ids>
<supporting_role>proof, derivation, explanation, answer, hint, solution, or empty</supporting_role>
</environment>

Rules:
1. Include the heading and complete statement/setup/question, all subparts, and continuations.
2. Put an explicit proof following a theorem-like object in supporting_ids.
3. Split worked examples into setup/claim/question versus visible calculation/reasoning/answer when
   block boundaries allow; otherwise keep the inseparable block in statement_ids.
4. Definitions and remarks normally have no support. Keep complete algorithm bodies.
5. Never assign one block to adjacent objects. Page overlap may duplicate complete objects; emit
   them normally for downstream deduplication by (kind,label).
6. In distant source support sections ({support_headings}), emit the matching kind/label with empty
   statement_ids and populated supporting_ids. Never invent missing support.
7. Preserve the full label namespace. If the book restarts numbering, include the visible scope
   prefix required by the profile.
8. {unnumbered_rule}

BOOK PROFILE OVERLAY (takes precedence only for book layout facts):
{profile.get('segmentation_prompt_overlay', '')}
"""


def xml_field(block: str, name: str) -> str:
    match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", block, flags=re.S)
    return match.group(1).strip() if match else ""


def parse_ids(value: str, size: int) -> list[int]:
    result = []
    for token in re.findall(r"\d+", value):
        number = int(token)
        if 0 <= number < size and number not in result:
            result.append(number)
    return result


def normalize_label(value: str, profile: dict[str, Any]) -> str:
    compact = value.replace(" ", "").strip()
    match = re.search(profile["label_regex"], compact, flags=re.I)
    if match:
        return match.group(0)
    if profile.get("include_unnumbered_named") and re.fullmatch(r"p\d+\.b\d+", compact, re.I):
        return compact
    return ""


def ids_text(ids: list[int], original: list[dict[str, Any]], image_dir: Path) -> str:
    values = []
    for block_id in ids:
        item = original[block_id]
        text = item.get("text") or item.get("table_body")
        if not text and item.get("list_items"):
            text = "\n".join(item["list_items"])
        if text:
            values.append(text)
        elif item.get("img_path"):
            caption = " ".join(item.get("image_caption") or ["image"])
            values.append(f"![{caption}]({image_dir / Path(item['img_path']).name})")
    return "\n".join(values).strip()


def source_blocks(ids: list[int], original: list[dict[str, Any]], part: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for block_id in ids:
        item = original[block_id]
        value = {
            "part": str(part["data_id"]), "block_id": block_id,
            "page_idx": int(item.get("page_idx") or 0),
            "pdf_page": int(part["start_page"]) + int(item.get("page_idx") or 0),
            "bbox": item.get("bbox"), "type": item.get("type"),
        }
        if item.get("img_path"):
            value["image_source"] = str(Path(part["mineru_dir"]) / "images" / Path(item["img_path"]).name)
            value["image_caption"] = item.get("image_caption") or []
        values.append(value)
    return values


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("source_pdf_page") or 10**6, KIND_ORDER.get(row["kind"], 99), row["label"])


def run_segment(config: dict[str, Any], force: bool = False) -> list[dict[str, Any]]:
    output, profile = Path(config["output"]), load_json(Path(config["profile"]))
    extracted = run_mineru(config)
    jobs, originals = [], {}
    target_chars = int(config["pipeline"].get("segment_chunk_chars", 80000))
    overlap = int(config["pipeline"].get("segment_overlap_pages", 2))
    for part in extracted:
        part_id = str(part["data_id"])
        compact, original = compact_layout(Path(part["content_list"]))
        originals[part_id] = original
        write_json(output / "segmentation" / "layouts" / f"part_{part_id}.json", compact)
        for index, chunk in enumerate(page_chunks(compact, target_chars, overlap)):
            jobs.append({"part": part, "index": index, "chunk": chunk,
                         "path": output / "segmentation" / "responses" / f"part_{part_id}_chunk_{index:03d}.txt"})
    prompt = segmentation_prompt(profile)

    def work(job: dict[str, Any]) -> None:
        if job["path"].exists() and not force:
            cached = job["path"].read_text(encoding="utf-8")
            if "<environment>" in cached or "<no_environments/>" in cached:
                return
        raw, _ = call_chat(config, prompt + "\n\nINPUT BLOCKS:\n" + json.dumps(job["chunk"], ensure_ascii=False))
        if "<environment>" not in raw:
            raw = "<no_environments/>\n" + raw
        job["path"].parent.mkdir(parents=True, exist_ok=True)
        job["path"].write_text(raw, encoding="utf-8")

    workers = max(1, int(config["vlm"].get("segment_workers", 32)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            future.result(); job = futures[future]
            print(f"segment [{done}/{len(jobs)}] part={job['part']['data_id']} chunk={job['index']}", flush=True)
    parsed = []
    aliases = {k.lower(): v.lower() for k, v in profile.get("kind_aliases", {}).items()}
    kinds = set(profile["kinds"])
    for job in jobs:
        raw = job["path"].read_text(encoding="utf-8")
        part, part_id = job["part"], str(job["part"]["data_id"])
        original = originals[part_id]
        for block in re.findall(r"<environment>(.*?)</environment>", raw, flags=re.S):
            kind = aliases.get(xml_field(block, "kind").lower(), xml_field(block, "kind").lower())
            statement_ids = parse_ids(xml_field(block, "statement_ids"), len(original))
            supporting_ids = parse_ids(xml_field(block, "supporting_ids"), len(original))
            label = normalize_label(xml_field(block, "label"), profile)
            if not label and profile.get("include_unnumbered_named") and (statement_ids or supporting_ids):
                first_id = min(statement_ids + supporting_ids)
                pdf_page = int(part["start_page"]) + int(original[first_id].get("page_idx") or 0)
                label = f"p{pdf_page}.b{first_id}"
            if kind not in kinds or not label or not (statement_ids or supporting_ids):
                continue
            image_dir = Path(part["mineru_dir"]) / "images"
            role = xml_field(block, "supporting_role").lower()
            if role not in SUPPORT_ROLES:
                role = ""
            parsed.append({
                "kind": kind, "label": label, "title": xml_field(block, "title"),
                "statement": ids_text(statement_ids, original, image_dir),
                "supporting_text": ids_text(supporting_ids, original, image_dir),
                "supporting_text_role": role,
                "statement_sources": source_blocks(statement_ids, original, part),
                "supporting_sources": source_blocks(supporting_ids, original, part),
                "source_part": part_id, "source_chunk": job["index"],
            })
    write_json(output / "segmentation" / "parsed_with_duplicates.json", parsed)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in parsed:
        key = (candidate["kind"], candidate["label"])
        row = merged.setdefault(key, {
            "id": f"{candidate['kind']}:{candidate['label']}", "kind": candidate["kind"],
            "label": candidate["label"], "title": "", "statement": "", "supporting_text": "",
            "supporting_text_role": "", "statement_sources": [], "supporting_sources": [],
            "candidate_locations": [],
        })
        row["candidate_locations"].append([candidate["source_part"], candidate["source_chunk"]])
        if len(candidate["statement"]) > len(row["statement"]):
            for field in ("title", "statement", "statement_sources"):
                row[field] = candidate[field]
        if candidate["supporting_text"] and (candidate["statement"] or profile.get("distant_support_alignment")):
            if len(candidate["supporting_text"]) > len(row["supporting_text"]):
                for field in ("supporting_text", "supporting_text_role", "supporting_sources"):
                    row[field] = candidate[field]
    rows = []
    for row in merged.values():
        if not row["statement"]:
            continue
        sources = row["statement_sources"] + row["supporting_sources"]
        row["source_pdf_page"] = min((x["pdf_page"] for x in sources), default=None)
        rows.append(row)
    rows.sort(key=sort_key)
    write_json(output / "segmentation" / "environments_raw.json", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    write_json(output / "segmentation" / "summary.json", {
        "model": config["vlm"]["model"], "chunks": len(jobs), "parsed_candidates": len(parsed),
        "environments": len(rows), "counts_by_kind": dict(sorted(counts.items())),
    })
    return rows


def crop_specs(row: dict[str, Any], padding: int) -> list[dict[str, Any]]:
    by_page: dict[int, list[list[float]]] = {}
    for source in row["statement_sources"] + row["supporting_sources"]:
        bbox = source.get("bbox")
        if bbox and len(bbox) == 4:
            by_page.setdefault(int(source["pdf_page"]), []).append([float(x) for x in bbox])
    specs = []
    for page, boxes in sorted(by_page.items()):
        specs.append({"pdf_page": page, "bbox": [
            max(0.0, min(x[0] for x in boxes) - padding), max(0.0, min(x[1] for x in boxes) - padding),
            min(1000.0, max(x[2] for x in boxes) + padding), min(1000.0, max(x[3] for x in boxes) + padding),
        ]})
    return specs


def render_crops(config: dict[str, Any], row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Path], str]:
    output, pdf = Path(config["output"]), Path(config["pdf"])
    padding = int(config["pipeline"].get("crop_padding", 35))
    zoom = float(config["pipeline"].get("crop_zoom", 3.0))
    specs = crop_specs(row, padding)
    if not specs:
        raise RuntimeError("no usable source bbox")
    paths, native = [], []
    document = fitz.open(pdf)
    try:
        for index, spec in enumerate(specs):
            page = document[spec["pdf_page"] - 1]
            x0, y0, x1, y1 = spec["bbox"]
            rect = page.rect
            clip = fitz.Rect(rect.x0 + rect.width*x0/1000, rect.y0 + rect.height*y0/1000,
                             rect.x0 + rect.width*x1/1000, rect.y0 + rect.height*y1/1000)
            path = output / "transcription" / "crops" / re.sub(r"[^A-Za-z0-9_.-]", "_", row["id"]) / f"{index:02d}_page_{spec['pdf_page']:04d}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False).save(path)
            paths.append(path)
            native.append(f"[physical PDF page {spec['pdf_page']}]\n{page.get_textbox(clip).strip()}")
    finally:
        document.close()
    return specs, paths, "\n\n".join(native)


def split_rule(kind: str) -> str:
    if kind == "example":
        return "Separate visible setup/question/claim from visible derivation/explanation/answer when source structure supports it; never invent a question."
    if kind in {"theorem", "lemma", "proposition", "corollary", "claim", "fact"}:
        return "Put only the assertion in statement and its explicit source proof in supporting_text."
    if kind == "exercise":
        return "Put the complete exercise in statement and only an explicit source hint/answer/solution in supporting_text; never solve it."
    return "Keep the complete object in statement; use supporting_text only when explicitly present and structurally separate."


def visual_prompt(row: dict[str, Any], native: str, profile: dict[str, Any], proposed: dict[str, Any] | None) -> str:
    mode = "Independently adjudicate ORIGINAL and PROPOSED against every image." if proposed else "Produce a conservative source-faithful transcription from the images."
    proposal = "" if not proposed else f"\nPROPOSED STATEMENT:\n{proposed['statement']}\n\nPROPOSED SUPPORT:\n{proposed['supporting_text']}\n"
    return f"""You are the final visual transcription editor for a mathematics textbook dataset.
{mode} Images are authoritative; OCR and native PDF text are secondary.

Target: {row['kind']} {row['label']}

Rules:
1. Return only this object. Exclude headers, page numbers, footnotes, adjacent objects, and unrelated exposition.
2. Do not solve, paraphrase, improve, or mathematically repair. Preserve every hypothesis, quantifier,
   symbol, index, sign, dimension, equation tag, subpart, marker, and prose order.
3. Use Markdown with $...$ and $$...$$ LaTeX. Fix unmistakable visible OCR/spelling damage.
4. Preserve kind and label exactly. Do not emit local paths or Markdown image links.
5. {split_rule(row['kind'])}
6. Prefer ORIGINAL whenever an image does not support a change. Report unreadable spans only.

Return only:
<kind>{row['kind']}</kind>
<label>{row['label']}</label>
<statement>complete statement</statement>
<supporting_text>source support or empty</supporting_text>
<supporting_role>proof, derivation, explanation, answer, hint, solution, hint_or_solution, or empty</supporting_role>
<uncertain_spans>one per line or empty</uncertain_spans>

ORIGINAL STATEMENT:\n{row['statement']}\n\nORIGINAL SUPPORT ({row['supporting_text_role']}):\n{row['supporting_text']}
{proposal}
PDF NATIVE TEXT:\n{native}

BOOK PROFILE OVERLAY:\n{profile.get('visual_prompt_overlay', '')}
"""


def parse_visual(raw: str) -> dict[str, Any]:
    uncertain = [line.strip() for line in xml_field(raw, "uncertain_spans").splitlines() if line.strip()]
    uncertain = [x for x in uncertain if x.lower().strip(" .:-") not in {"none", "empty", "n/a"}]
    return {"kind": xml_field(raw, "kind").lower(), "label": xml_field(raw, "label").replace(" ", ""),
            "statement": xml_field(raw, "statement"), "supporting_text": xml_field(raw, "supporting_text"),
            "supporting_text_role": xml_field(raw, "supporting_role").lower(), "uncertain_spans": uncertain}


def validate_visual(row: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    errors = []
    if parsed["kind"] != row["kind"]: errors.append("kind mismatch")
    if parsed["label"].upper() != row["label"].upper(): errors.append("label mismatch")
    if not parsed["statement"]: errors.append("empty statement")
    if parsed["supporting_text_role"] not in SUPPORT_ROLES: errors.append("bad supporting role")
    before = len(row["statement"] + row["supporting_text"])
    ratio = len(parsed["statement"] + parsed["supporting_text"]) / max(1, before)
    if ratio < 0.18 or ratio > (50.0 if row["kind"] == "algorithm" else 3.0): errors.append(f"length ratio {ratio:.2f}")
    if parsed["statement"].count("$") % 2 or parsed["supporting_text"].count("$") % 2: errors.append("unbalanced dollars")
    return errors


def run_transcribe(config: dict[str, Any], force: bool = False, ids: set[str] | None = None) -> list[dict[str, Any]]:
    output, profile = Path(config["output"]), load_json(Path(config["profile"]))
    raw_path = output / "segmentation" / "environments_raw.json"
    rows = load_json(raw_path) if raw_path.exists() else run_segment(config)
    selected = [row for row in rows if ids is None or row["id"] in ids]

    def work(row: dict[str, Any]) -> dict[str, Any]:
        path = output / "transcription" / "results" / (re.sub(r"[^A-Za-z0-9_.-]", "_", row["id"]) + ".json")
        if path.exists() and not force:
            cached = load_json(path)
            if cached.get("final") and not cached.get("error"):
                return cached
        try:
            specs, images, native = render_crops(config, row)
            def one(proposed: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
                content: list[dict[str, Any]] = [{"type": "text", "text": visual_prompt(row, native, profile, proposed)}]
                for image, spec in zip(images, specs):
                    content.extend([{"type": "text", "text": f"Source crop, physical PDF page {spec['pdf_page']}:"},
                                    {"type": "image_url", "image_url": {"url": image_data_url(image), "detail": "high"}}])
                last = ""
                for _ in range(5):
                    raw, usage = call_chat(config, content)
                    parsed = parse_visual(raw); errors = validate_visual(row, parsed)
                    if not errors: return parsed, {"usage": usage, "raw": raw}
                    last = "; ".join(errors)
                raise RuntimeError(last)
            first, first_meta = one(None)
            if config["vlm"].get("double_pass", True):
                final, final_meta = one(first)
            else:
                final, final_meta = first, {}
            result = {"id": row["id"], "pages": [x["pdf_page"] for x in specs], "first": first,
                      "final": final, "first_meta": first_meta, "verification_meta": final_meta}
        except Exception as exc:
            result = {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}
        write_json(path, result)
        return result

    workers = max(1, int(config["vlm"].get("vision_workers", 64)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, row): row for row in selected}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"transcribe [{done}/{len(selected)}] {result['id']} {'error' if result.get('error') else 'ok'}", flush=True)
    return rows


IMAGE_MD = re.compile(r"!\[(.*?)\]\((.*?)\)")


def clean_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", IMAGE_MD.sub("", value)).strip()


def copy_assets(row: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    assets, seen = [], set()
    for source in row["statement_sources"] + row["supporting_sources"]:
        image = source.get("image_source")
        if not image or image in seen or not Path(image).exists(): continue
        seen.add(image); source_path = Path(image)
        name = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', row['id'])}_{len(assets)+1}{source_path.suffix.lower()}"
        destination = output / "dataset" / "assets" / name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source_path, destination)
        assets.append({"image": f"assets/{name}", "caption": " ".join(source.get("image_caption") or []),
                       "pdf_page": source.get("pdf_page")})
    return assets


def portable_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove machine-local extraction paths after referenced assets have been copied."""
    return [{key: value for key, value in source.items() if key != "image_source"} for source in sources]


def run_build(config: dict[str, Any]) -> list[dict[str, Any]]:
    output = Path(config["output"])
    raw_rows = load_json(output / "segmentation" / "environments_raw.json")
    rows, audit, failures = [], [], []
    for raw in raw_rows:
        path = output / "transcription" / "results" / (re.sub(r"[^A-Za-z0-9_.-]", "_", raw["id"]) + ".json")
        if not path.exists(): failures.append(raw["id"]); continue
        result = load_json(path)
        if result.get("error") or not result.get("final"): failures.append(raw["id"]); continue
        final = result["final"]; row = dict(raw)
        row["statement"] = clean_text(final["statement"])
        row["supporting_text"] = clean_text(final["supporting_text"])
        row["supporting_text_role"] = final["supporting_text_role"]
        row["uncertain_spans"] = final.get("uncertain_spans", [])
        row["assets"] = copy_assets(raw, output)
        row["statement_sources"] = portable_sources(row["statement_sources"])
        row["supporting_sources"] = portable_sources(row["supporting_sources"])
        rows.append(row)
        audit.append({"id": row["id"], "statement_before": raw["statement"], "statement_after": row["statement"],
                      "supporting_before": raw["supporting_text"], "supporting_after": row["supporting_text"],
                      "pages": result.get("pages", []), "uncertain_spans": row["uncertain_spans"]})
    rows.sort(key=sort_key)
    dataset = output / "dataset"
    write_json(dataset / "environments.json", rows)
    (dataset / "environments.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")
    qa = [{"id": x["id"], "kind": x["kind"], "label": x["label"], "title": x.get("title", ""),
           "question": x["statement"], "answer": x["supporting_text"],
           "supporting_text_role": x["supporting_text_role"], "assets": x["assets"]} for x in rows]
    write_json(dataset / "qa_pairs.json", qa)
    for kind in sorted({x["kind"] for x in rows}):
        write_json(dataset / "by_kind" / f"{kind}.json", [x for x in rows if x["kind"] == kind])
    markdown = []
    for row in rows:
        markdown += [f"## {row['kind'].title()} {row['label']}", "", row["statement"], ""]
        for asset in row["assets"]: markdown += [f"![{asset['caption']}]({asset['image']})", ""]
        if row["supporting_text"]: markdown += [f"**{row['supporting_text_role'].replace('_',' ').title()}:**", "", row["supporting_text"], ""]
        markdown += ["---", ""]
    (dataset / "environments.md").write_text("\n".join(markdown), encoding="utf-8")
    write_json(dataset / "transcription_audit.json", audit)
    counts = {kind: sum(x["kind"] == kind for x in rows) for kind in sorted({x["kind"] for x in rows})}
    write_json(dataset / "run_summary.json", {"environments": len(rows), "counts_by_kind": counts,
                                               "with_supporting_text": sum(bool(x["supporting_text"]) for x in rows),
                                               "uncertain": sum(bool(x["uncertain_spans"]) for x in rows), "failed": failures})
    return rows


def run_validate(config: dict[str, Any]) -> None:
    subprocess.run([sys.executable, "-m", "parse_ocr.validate_dataset", "--config", config["_config_path"]], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["mineru", "segment", "transcribe", "build", "validate", "run"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args(); config = resolve_config(args.config)
    if args.ids and args.stage != "transcribe":
        raise SystemExit("--ids is supported only by the transcribe stage; build after all records finish")
    Path(config["output"]).mkdir(parents=True, exist_ok=True)
    if args.stage == "mineru": run_mineru(config)
    elif args.stage == "segment": run_segment(config, args.force_stage)
    elif args.stage == "transcribe": run_transcribe(config, args.force_stage, set(args.ids) if args.ids else None)
    elif args.stage == "build": run_build(config)
    elif args.stage == "validate": run_validate(config)
    else:
        run_mineru(config); run_segment(config, args.force_stage)
        run_transcribe(config, args.force_stage, set(args.ids) if args.ids else None)
        run_build(config); run_validate(config)


if __name__ == "__main__":
    main()
