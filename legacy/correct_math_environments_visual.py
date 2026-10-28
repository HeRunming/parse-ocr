#!/usr/bin/env python3
"""Visually transcribe and verify expanded numbered math environments."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz
import requests


API_URL = os.environ.get("VLM_CHAT_URL", "http://127.0.0.1:3000/v1/chat/completions")
MODEL = "gpt-5.5"
KIND_ORDER = {name: index for index, name in enumerate([
    "definition", "theorem", "lemma", "proposition", "corollary",
    "claim", "fact", "example", "exercise", "algorithm", "remark",
])}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def crop_specs(row: dict[str, Any], pdf: Path) -> list[dict[str, Any]]:
    by_page: dict[int, list[list[float]]] = {}
    for source in row.get("statement_sources", []) + row.get("supporting_sources", []):
        bbox = source.get("bbox")
        if bbox and len(bbox) == 4:
            by_page.setdefault(int(source["pdf_page"]), []).append([float(x) for x in bbox])
    specs = []
    for page in sorted(by_page):
        boxes = by_page[page]
        specs.append({
            "pdf_page": page,
            "bbox_1000": [
                max(0.0, min(x[0] for x in boxes) - 30),
                max(0.0, min(x[1] for x in boxes) - 30),
                min(1000.0, max(x[2] for x in boxes) + 30),
                min(1000.0, max(x[3] for x in boxes) + 30),
            ],
        })
    if not specs:
        raise ValueError(f"no crop bboxes for {row['id']}")
    # MinerU sometimes stores a logical block that continues onto the next
    # page but exposes only the first page bbox.  A bottom-edge source block is
    # cheap to handle conservatively: include the next page's upper region and
    # let the visual prompt exclude any adjacent environment.
    if specs[-1]["bbox_1000"][3] >= 780:
        with fitz.open(pdf) as document:
            next_page = specs[-1]["pdf_page"] + 1
            if next_page <= len(document):
                specs.append({
                    "pdf_page": next_page,
                    "bbox_1000": [80.0, 55.0, 920.0, 950.0],
                    "auto_continuation": True,
                })
    return specs


def page_clip(page: fitz.Page, bbox: list[float]) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    rect = page.rect
    return fitz.Rect(
        rect.x0 + rect.width * x0 / 1000,
        rect.y0 + rect.height * y0 / 1000,
        rect.x0 + rect.width * x1 / 1000,
        rect.y0 + rect.height * y1 / 1000,
    )


def render_and_text(pdf: Path, output: Path, row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Path], str]:
    specs = crop_specs(row, pdf)
    directory = output / "crops" / safe_id(row["id"])
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    native: list[str] = []
    with fitz.open(pdf) as document:
        for index, spec in enumerate(specs):
            page = document[spec["pdf_page"] - 1]
            clip = page_clip(page, spec["bbox_1000"])
            bbox_key = "_".join(str(int(value)) for value in spec["bbox_1000"])
            path = directory / f"{index:02d}_page_{spec['pdf_page']:03d}_{bbox_key}.png"
            if not path.exists():
                page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), clip=clip, alpha=False).save(path)
            paths.append(path)
            native.append(f"[physical PDF page {spec['pdf_page']}]\n{page.get_textbox(clip).strip()}")
    return specs, paths, "\n\n".join(native)


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def split_rules(kind: str) -> str:
    if kind == "example":
        return """This is a worked example. Put its setup, question, or stated claim in statement.
Put the source book's worked calculation, proof, reasoning, conclusion, or answer in supporting_text.
If setup and reasoning share one source block, split them semantically using the visible prose. For a
compact illustrative example with no natural split, keep the complete example in statement and leave
supporting_text empty. Never manufacture a question or answer."""
    if kind in {"theorem", "lemma", "proposition", "corollary", "claim", "fact"}:
        return """Put only the numbered assertion in statement. Put its explicit immediately-following
Proof in supporting_text with role proof. Exclude post-QED discussion and the next environment. If the
source crop contains no explicit proof, leave supporting_text empty."""
    if kind == "exercise":
        return "Put the exercise in statement and only an explicit source hint/answer/solution in supporting_text. Never solve it."
    return "Keep the complete numbered environment in statement and leave supporting_text empty."


def prompt(row: dict[str, Any], native: str, verify: bool = False, proposed: dict[str, Any] | None = None) -> str:
    mode = (
        "Independently adjudicate ORIGINAL and PROPOSED against every visible source crop."
        if verify else
        "Produce a conservative source-faithful transcription from the visible source crops."
    )
    proposed_section = ""
    if proposed is not None:
        proposed_section = f"""
PROPOSED STATEMENT:
---
{proposed['statement']}
---
PROPOSED SUPPORTING TEXT:
---
{proposed['supporting_text']}
---
"""
    return f"""You are the final visual transcription editor for a mathematics textbook dataset.
{mode} The images are authoritative; OCR candidates and native PDF text are secondary evidence.

Target: {row['kind']} {row['label']}

Rules:
1. Return only this environment. Exclude page headers/numbers, footnotes, adjacent environments,
   unrelated figure captions, and surrounding textbook exposition.
2. Do not solve, paraphrase, improve, or repair a mathematical statement. Preserve all hypotheses,
   quantifiers, symbols, dimensions, signs, equation tags, subparts, and prose order.
3. Use Markdown with $...$ and $$...$$ LaTeX. Fix unmistakable OCR and spelling damage. Do not emit
   local file paths or Markdown image links; image assets are handled separately.
4. The statement must start with the printed kind/label heading (or the bare exercise label exactly
   as printed). The kind and label themselves may not change.
5. {split_rules(row['kind'])}
6. Prefer ORIGINAL over a prettier rendering whenever the image does not support a change. Report
   genuinely unreadable spans; otherwise leave uncertainties empty.

Return only this envelope, without a Markdown fence:
<kind>{row['kind']}</kind>
<label>{row['label']}</label>
<statement>
complete statement Markdown
</statement>
<supporting_text>
source proof/derivation/answer, or empty
</supporting_text>
<supporting_role>proof, derivation, explanation, answer, hint, solution, or empty</supporting_role>
<uncertain_spans>
one item per line, or empty
</uncertain_spans>

ORIGINAL STATEMENT:
---
{row['statement']}
---
ORIGINAL SUPPORTING TEXT ({row.get('supporting_text_role','')}):
---
{row.get('supporting_text','')}
---
{proposed_section}
PDF NATIVE TEXT (secondary evidence; formula layout may be flattened):
---
{native}
---
"""


def extract_content(payload: dict[str, Any]) -> str:
    value = payload["choices"][0]["message"].get("content", "")
    if isinstance(value, str):
        return value
    return "".join(x.get("text", "") for x in value if isinstance(x, dict))


def field(raw: str, name: str) -> str:
    match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", raw, flags=re.S)
    return match.group(1).strip() if match else ""


def parse(raw: str) -> dict[str, Any]:
    uncertainties = [x.strip() for x in field(raw, "uncertain_spans").splitlines() if x.strip()]
    uncertainties = [x for x in uncertainties if x.lower().strip(" .:-") not in {"none", "empty", "n/a"}]
    return {
        "kind": field(raw, "kind").lower(),
        "label": field(raw, "label").replace(" ", ""),
        "statement": field(raw, "statement"),
        "supporting_text": field(raw, "supporting_text"),
        "supporting_text_role": field(raw, "supporting_role").lower(),
        "uncertain_spans": uncertainties,
    }


def validate(row: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    errors = []
    if parsed["kind"] != row["kind"]: errors.append("kind mismatch")
    if parsed["label"].upper() != row["label"].upper(): errors.append("label mismatch")
    if not parsed["statement"]: errors.append("empty statement")
    before = len(row["statement"] + row.get("supporting_text", ""))
    after = len(parsed["statement"] + parsed["supporting_text"])
    ratio = after / max(1, before)
    upper_ratio = 50.0 if row["kind"] == "algorithm" else 2.5
    if ratio < 0.18 or ratio > upper_ratio: errors.append(f"length ratio {ratio:.2f}")
    if parsed["statement"].count("$") % 2 or parsed["supporting_text"].count("$") % 2:
        errors.append("unbalanced dollar delimiters")
    return errors


def call(row: dict[str, Any], specs: list[dict[str, Any]], paths: list[Path], native: str,
         key: str, proposed: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt(row, native, proposed is not None, proposed)}]
    for path, spec in zip(paths, specs):
        content.append({"type": "text", "text": f"Source crop, physical PDF page {spec['pdf_page']}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url(path), "detail": "high"}})
    last = ""
    for attempt in range(5):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": content}],
                      "temperature": 0, "max_tokens": 32000},
                timeout=1200,
            )
            if not response.ok: raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
            payload = response.json(); raw = extract_content(payload); parsed = parse(raw)
            errors = validate(row, parsed)
            if not errors:
                return parsed, {"raw": raw, "usage": payload.get("usage", {}), "attempt": attempt + 1}
            last = "; ".join(errors)
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < 4: time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(last)


def copy_assets(row: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    assets = []
    seen = set()
    for source in row.get("statement_sources", []) + row.get("supporting_sources", []):
        image = source.get("image_source")
        if not image or image in seen or not Path(image).exists(): continue
        seen.add(image)
        name = f"{safe_id(row['id'])}_{len(assets)+1}{Path(image).suffix.lower()}"
        destination = output / "assets" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
        assets.append({
            "image": f"assets/{name}",
            "caption": " ".join(source.get("image_caption") or []),
            "pdf_page": source.get("pdf_page"),
        })
    return assets


IMAGE_MARKDOWN = re.compile(r"!\[(.*?)\]\((.*?)\)")


def strip_image_markdown(text: str) -> tuple[str, list[tuple[str, str]]]:
    found = IMAGE_MARKDOWN.findall(text)
    cleaned = IMAGE_MARKDOWN.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, found


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("source_pdf_page") or 10**6, KIND_ORDER.get(row["kind"], 99), row["label"])


def build_outputs(raw_rows: list[dict[str, Any]], canonical_path: Path, output: Path) -> dict[str, Any]:
    canonical = {x["label"]: x for x in json.loads(canonical_path.read_text(encoding="utf-8"))}
    raw_exercises = {x["label"]: x for x in raw_rows if x["kind"] == "exercise"}
    rows = []
    audit = []
    failures = []
    for raw in raw_rows:
        if raw["kind"] == "exercise":
            continue
        path = output / "corrections" / f"{safe_id(raw['id'])}.json"
        if not path.exists(): failures.append(raw["id"]); continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("error") or not result.get("final"):
            failures.append(raw["id"]); continue
        updated = dict(raw)
        final = result["final"]
        for key in ["statement", "supporting_text", "supporting_text_role"]:
            updated[key] = final[key]
        updated["assets"] = copy_assets(raw, output)
        rows.append(updated)
        audit.append({
            "id": raw["id"], "kind": raw["kind"], "label": raw["label"],
            "statement_before": raw["statement"], "statement_after": final["statement"],
            "supporting_before": raw.get("supporting_text", ""),
            "supporting_after": final["supporting_text"],
            "supporting_role": final["supporting_text_role"],
            "uncertain_spans": final.get("uncertain_spans", []),
            "pages": result.get("pages", []),
        })
    for label, item in canonical.items():
        raw = raw_exercises.get(label, {})
        statement, statement_images = strip_image_markdown(item["question"])
        supporting, supporting_images = strip_image_markdown(
            item.get("solution", "") or item.get("answer", "")
        )
        row = {
            "id": f"exercise:{label}", "kind": "exercise", "label": label,
            "title": raw.get("title", ""), "statement": statement,
            "supporting_text": supporting,
            "supporting_text_role": "hint_or_solution" if supporting else "",
            "statement_sources": raw.get("statement_sources", []),
            "supporting_sources": raw.get("supporting_sources", []),
            "candidate_locations": raw.get("candidate_locations", []),
            "source_pdf_page": raw.get("source_pdf_page"),
            "assets": copy_assets(raw, output) if statement_images or supporting_images else [],
        }
        # Preserve the carefully curated FOML exercise figure references.
        if item.get("referenced_figures"):
            references = []
            for index, figure in enumerate(item["referenced_figures"], start=1):
                copied = dict(figure)
                source_image = canonical_path.parent / figure["image"]
                if source_image.exists():
                    name = f"exercise_{safe_id(label)}_figure_{index}{source_image.suffix.lower()}"
                    destination = output / "assets" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_image, destination)
                    copied["image"] = f"assets/{name}"
                references.append(copied)
            row["referenced_figures"] = references
        rows.append(row)
    rows.sort(key=sort_key)
    write_json(output / "environments.json", rows)
    (output / "environments.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    qa_pairs = [{
        "id": x["id"], "kind": x["kind"], "label": x["label"], "title": x.get("title", ""),
        "question": x["statement"], "answer": x["supporting_text"],
        "supporting_text_role": x["supporting_text_role"], "assets": x.get("assets", []),
    } for x in rows]
    write_json(output / "qa_pairs.json", qa_pairs)
    by_kind = {}
    for kind in sorted({x["kind"] for x in rows}):
        values = [x for x in rows if x["kind"] == kind]
        write_json(output / "by_kind" / f"{kind}.json", values)
        by_kind[kind] = len(values)
    markdown = []
    for x in rows:
        markdown.extend([f"## {x['kind'].title()} {x['label']}", "", x["statement"], ""])
        for asset in x.get("assets", []):
            markdown.extend([f"![{asset['caption']}]({asset['image']})", ""])
        for figure in x.get("referenced_figures", []):
            markdown.extend([
                f"**Referenced source figure: {figure['label']}**", "",
                f"![{figure.get('caption', figure['label'])}]({figure['image']})", "",
            ])
        if x["supporting_text"]:
            markdown.extend([f"**{x['supporting_text_role'].replace('_',' ').title()}:**", "", x["supporting_text"], ""])
        markdown.extend(["---", ""])
    (output / "environments.md").write_text("\n".join(markdown), encoding="utf-8")
    write_json(output / "visual_correction_audit.json", audit)
    summary = {
        "environments": len(rows), "counts_by_kind": by_kind,
        "with_supporting_text": sum(bool(x["supporting_text"]) for x in rows),
        "worked_examples_with_supporting_text": sum(
            x["kind"] == "example" and bool(x["supporting_text"]) for x in rows),
        "uncertain": sum(bool(x.get("uncertain_spans")) for x in audit),
        "failed": failures,
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--canonical-exercises", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    pdf, source, canonical, output = map(Path.resolve, [args.pdf, args.source, args.canonical_exercises, args.output])
    key = os.environ.get("DF_API_KEY")
    if not key: raise SystemExit("DF_API_KEY is required")
    raw_rows = json.loads((source / "environments_raw.json").read_text(encoding="utf-8"))
    selected = [x for x in raw_rows if x["kind"] != "exercise"]
    if args.ids:
        wanted = set(args.ids); selected = [x for x in selected if x["id"] in wanted]
    jobs = []
    for row in selected:
        path = output / "corrections" / f"{safe_id(row['id'])}.json"
        if path.exists() and not args.force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if not cached.get("error") and cached.get("final"): continue
        jobs.append((row, path))

    def run(job: tuple[dict[str, Any], Path]) -> dict[str, Any]:
        row, path = job
        try:
            specs, images, native = render_and_text(pdf, output, row)
            first, first_meta = call(row, specs, images, native, key)
            if args.single_pass:
                final, verify_meta = first, {}
            else:
                final, verify_meta = call(row, specs, images, native, key, first)
            result = {
                "id": row["id"], "pages": [x["pdf_page"] for x in specs],
                "first": first, "final": final, "first_meta": first_meta,
                "verification_meta": verify_meta,
            }
        except Exception as exc:
            result = {"id": row["id"], "error": f"{type(exc).__name__}: {exc}"}
        write_json(path, result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            result = future.result(); done += 1
            print(f"[{done}/{len(jobs)}] {result['id']} {'error' if result.get('error') else 'ok'}", flush=True)
    summary = build_outputs(raw_rows, canonical, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
