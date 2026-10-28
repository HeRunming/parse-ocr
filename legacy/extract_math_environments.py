#!/usr/bin/env python3
"""Extract all numbered mathematical environments from cached MinerU output.

This is the expanded companion to the exercise-only PDF-VQA runner.  GPT only
selects source block IDs here; it does not transcribe or rewrite mathematics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


GATEWAY = os.environ.get("VLM_CHAT_URL", "http://127.0.0.1:3000/v1/chat/completions")
MODEL = "gpt-5.5"
KINDS = {
    "definition", "theorem", "lemma", "proposition", "corollary",
    "example", "exercise", "remark", "algorithm", "claim", "fact",
}
KIND_ORDER = {name: index for index, name in enumerate([
    "definition", "theorem", "lemma", "proposition", "corollary",
    "claim", "fact", "example", "exercise", "algorithm", "remark",
])}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def request_llm(prompt: str, key: str) -> str:
    last_error = ""
    for attempt in range(6):
        try:
            response = requests.post(
                GATEWAY,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 32000,
                },
                timeout=1200,
            )
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("response content is not text")
            # A chunk made only of front matter, references, or index pages can
            # legitimately contain zero target environments.
            return content if "<environment>" in content else "<no_environments/>\n" + content
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 5:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(last_error)


def prompt_for(book: str) -> str:
    book_rules = (
        r"""
HDP-specific rules:
- Preserve hierarchical labels such as 2.9.1 exactly. Exercises use labels such as 4.31.
- K, KK, or KKK printed immediately after an exercise number is difficulty metadata but remains
  part of the exercise source blocks.
- Under "Hints for the exercises", "Solutions", or "Answers", emit kind=exercise with empty
  statement_ids and put the matching source hint/solution in supporting_ids/supporting_role=hint
  or solution. It will be joined to the question globally by kind+label.
"""
        if book == "hdp" else
        r"""
FOML-specific rules:
- Preserve labels such as 2.4, 10.10, B.10, and E.1 exactly.
- A chapter's worked Example and its Exercise may legally have the same numeric label. They are
  distinct because kind differs; never merge or suppress one because of that collision.
- Numbered exercise entries occur under a section titled "Exercises" even though the individual
  entries do not begin with the word Exercise.
- The public book has no instructor exercise solutions. Do not invent any.
"""
    )
    return r"""
You are a source-indexing model for a mathematics textbook. The input is a JSON array of MinerU
layout blocks. Every block has a stable integer id and page_idx. Your job is only to select IDs;
never transcribe, paraphrase, solve, or create mathematical content.

Extract every complete, explicitly numbered mathematical environment of these kinds:
definition, theorem, lemma, proposition, corollary, example, exercise, remark, algorithm, claim,
and fact. Ignore mere references such as "by Theorem 2.3", table-of-contents entries, index entries,
equation numbers, unnumbered prose, and unnumbered proof fragments whose parent cannot be identified.

For each environment return exactly:
<environment>
<kind>one lowercase kind from the list above</kind>
<label>full source label without the kind word</label>
<title>printed parenthetical/title text, or empty</title>
<statement_ids>comma-separated source block IDs</statement_ids>
<supporting_ids>comma-separated source block IDs, or empty</supporting_ids>
<supporting_role>proof, derivation, explanation, answer, hint, solution, or empty</supporting_role>
</environment>

ID selection rules:
1. statement_ids contain the heading and complete statement/setup/question. Include all lettered
   subparts and continuations. Do not include the next numbered environment.
2. For theorem/lemma/proposition/corollary/claim/fact, an immediately following source proof goes
   in supporting_ids with role=proof.
3. For a worked example, separate its setup/claim/question into statement_ids and its worked
   calculation, reasoning, or answer into supporting_ids. If one MinerU block contains both and
   cannot be split by ID, put that block in statement_ids; a later visual layer will split it.
4. Definitions and remarks normally have no supporting_ids. Algorithms may include the complete
   numbered algorithm/figure blocks needed to define them.
5. Never assign one source block to adjacent environments. Figure captions preceding an environment
   belong only when the environment explicitly depends on that figure or algorithm.
6. A chunk has page overlap. Extract duplicated complete environments normally; downstream merging
   will deduplicate by kind+label. Do not join adjacent labels.
7. If only the supporting proof/hint is visible but its explicit parent label is visible, emit an
   environment with empty statement_ids. Otherwise ignore an orphan continuation.
8. Return only zero or more <environment> blocks, without a Markdown fence or commentary.
""" + book_rules


def compact_layout(content_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original = json.loads(content_path.read_text(encoding="utf-8"))
    compact: list[dict[str, Any]] = []
    for block_id, item in enumerate(original):
        block: dict[str, Any] = {
            "id": block_id,
            "page_idx": int(item.get("page_idx") or 0),
            "type": item.get("type", ""),
        }
        if item.get("text"):
            block["text"] = item["text"]
        elif item.get("table_body"):
            block["table_body"] = item["table_body"]
        elif item.get("list_items"):
            block["text"] = "\n".join(item["list_items"])
        if item.get("image_caption"):
            block["image_caption"] = item["image_caption"]
        if item.get("img_path"):
            block["image"] = Path(item["img_path"]).name
        compact.append(block)
    return compact, original


def page_chunks(layout: list[dict[str, Any]], target_chars: int = 80_000) -> list[list[dict[str, Any]]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for item in layout:
        by_page.setdefault(item["page_idx"], []).append(item)
    pages = sorted(by_page)
    ranges: list[tuple[int, int]] = []
    start = previous = pages[0]
    size = 0
    for page in pages:
        page_size = len(json.dumps(by_page[page], ensure_ascii=False))
        if size and size + page_size > target_chars:
            ranges.append((start, previous))
            start, size = page, 0
        size += page_size
        previous = page
    ranges.append((start, previous))
    low, high = pages[0], pages[-1]
    return [
        [item for page in range(max(low, start - 2), min(high, end + 2) + 1)
         for item in by_page.get(page, [])]
        for start, end in ranges
    ]


def split_chunk(chunk: list[dict[str, Any]], target_chars: int = 30_000) -> list[list[dict[str, Any]]]:
    """Split one pathological chunk into smaller page-aligned overlapping pieces."""
    by_page: dict[int, list[dict[str, Any]]] = {}
    for item in chunk:
        by_page.setdefault(item["page_idx"], []).append(item)
    pages = sorted(by_page)
    ranges: list[tuple[int, int]] = []
    start = previous = pages[0]
    size = 0
    for page in pages:
        page_size = len(json.dumps(by_page[page], ensure_ascii=False))
        if size and size + page_size > target_chars:
            ranges.append((start, previous))
            start, size = page, 0
        size += page_size
        previous = page
    ranges.append((start, previous))
    low, high = pages[0], pages[-1]
    return [
        [item for page in range(max(low, start - 1), min(high, end + 1) + 1)
         for item in by_page.get(page, [])]
        for start, end in ranges
    ]


def xml_field(block: str, name: str) -> str:
    match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", block, flags=re.S)
    return match.group(1).strip() if match else ""


def normalize_label(value: str) -> str:
    value = value.replace(" ", "").strip()
    match = re.search(r"(?:\d+|[A-Z])(?:\.\d+)+", value, flags=re.I)
    return match.group(0) if match else ""


def parse_ids(value: str, size: int) -> list[int]:
    result: list[int] = []
    for token in re.findall(r"\d+", value):
        number = int(token)
        if 0 <= number < size and number not in result:
            result.append(number)
    return result


def ids_text(ids: list[int], original: list[dict[str, Any]], image_dir: Path) -> str:
    result: list[str] = []
    for block_id in ids:
        item = original[block_id]
        text = item.get("text") or item.get("table_body")
        if not text and item.get("list_items"):
            text = "\n".join(item["list_items"])
        if text:
            result.append(text)
        elif item.get("img_path"):
            caption = " ".join(item.get("image_caption") or ["image"])
            result.append(f"![{caption}]({image_dir / Path(item['img_path']).name})")
    return "\n".join(result).strip()


def source_blocks(ids: list[int], original: list[dict[str, Any]], part: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for block_id in ids:
        item = original[block_id]
        value = {
            "part": str(part["data_id"]),
            "block_id": block_id,
            "page_idx": int(item.get("page_idx") or 0),
            "pdf_page": int(part["start_page"]) + int(item.get("page_idx") or 0),
            "bbox": item.get("bbox"),
            "type": item.get("type"),
        }
        if item.get("img_path"):
            value["image_source"] = str(Path(part["mineru_dir"]) / "images" / Path(item["img_path"]).name)
            value["image_caption"] = item.get("image_caption") or []
        values.append(value)
    return values


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    page = row.get("source_pdf_page") or 10**6
    label_parts = row["label"].upper().split(".")
    head = int(label_parts[0]) if label_parts[0].isdigit() else 1000 + ord(label_parts[0][0])
    tail = tuple(int(x) for x in label_parts[1:] if x.isdigit())
    return (page, head, *tail, KIND_ORDER.get(row["kind"], 99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", choices=["hdp", "foml"], required=True)
    parser.add_argument("--mineru-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repair-part")
    parser.add_argument("--repair-chunk", type=int)
    args = parser.parse_args()
    key = os.environ.get("DF_API_KEY")
    if not key:
        raise SystemExit("DF_API_KEY is required")
    source = args.mineru_source.resolve()
    output = args.output.resolve()
    extracted = json.loads((source / "mineru/extracted.json").read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    originals: dict[str, list[dict[str, Any]]] = {}
    for part in extracted:
        part_id = str(part["data_id"])
        compact, original = compact_layout(Path(part["content_list"]))
        originals[part_id] = original
        write_json(output / "layouts" / f"part_{part_id}.json", compact)
        for chunk_index, chunk in enumerate(page_chunks(compact)):
            jobs.append({
                "part": part,
                "chunk_index": chunk_index,
                "chunk": chunk,
                "path": output / "responses" / f"part_{part_id}_chunk_{chunk_index:03d}.txt",
            })

    prompt = prompt_for(args.book)

    if args.repair_part is not None or args.repair_chunk is not None:
        if args.repair_part is None or args.repair_chunk is None:
            raise SystemExit("--repair-part and --repair-chunk must be used together")
        selected = next(
            (job for job in jobs if str(job["part"]["data_id"]) == args.repair_part
             and job["chunk_index"] == args.repair_chunk),
            None,
        )
        if selected is None:
            raise SystemExit("selected repair chunk does not exist")
        pieces = split_chunk(selected["chunk"])
        results: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(pieces), max(1, args.workers))) as pool:
            futures = [
                pool.submit(
                    request_llm,
                    prompt + "\n\nINPUT BLOCKS:\n" + json.dumps(piece, ensure_ascii=False),
                    key,
                )
                for piece in pieces
            ]
            for index, future in enumerate(futures):
                results.append(future.result())
                print(f"repair piece {index + 1}/{len(pieces)} ok", flush=True)
        selected["path"].parent.mkdir(parents=True, exist_ok=True)
        selected["path"].write_text("\n".join(results), encoding="utf-8")
        print(json.dumps({"repaired": str(selected["path"]), "pieces": len(pieces)}))
        return

    def run(job: dict[str, Any]) -> None:
        if job["path"].exists() and not args.force:
            raw = job["path"].read_text(encoding="utf-8")
            if "<environment>" in raw or "<no_environments/>" in raw:
                return
        raw = request_llm(prompt + "\n\nINPUT BLOCKS:\n" + json.dumps(job["chunk"], ensure_ascii=False), key)
        job["path"].parent.mkdir(parents=True, exist_ok=True)
        job["path"].write_text(raw, encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            job = futures[future]
            print(f"[{done}/{len(jobs)}] part {job['part']['data_id']} chunk {job['chunk_index']} ok", flush=True)

    parsed: list[dict[str, Any]] = []
    for job in jobs:
        raw = job["path"].read_text(encoding="utf-8")
        part = job["part"]
        part_id = str(part["data_id"])
        original = originals[part_id]
        image_dir = Path(part["mineru_dir"]) / "images"
        for block in re.findall(r"<environment>(.*?)</environment>", raw, flags=re.S):
            kind = xml_field(block, "kind").lower().strip()
            label = normalize_label(xml_field(block, "label"))
            statement_ids = parse_ids(xml_field(block, "statement_ids"), len(original))
            supporting_ids = parse_ids(xml_field(block, "supporting_ids"), len(original))
            if kind not in KINDS or not label or not (statement_ids or supporting_ids):
                continue
            statement = ids_text(statement_ids, original, image_dir)
            supporting = ids_text(supporting_ids, original, image_dir)
            parsed.append({
                "kind": kind,
                "label": label,
                "title": xml_field(block, "title"),
                "statement": statement,
                "supporting_text": supporting,
                "supporting_text_role": xml_field(block, "supporting_role").lower().strip(),
                "statement_sources": source_blocks(statement_ids, original, part),
                "supporting_sources": source_blocks(supporting_ids, original, part),
                "source_part": part_id,
                "source_chunk": job["chunk_index"],
            })
    write_json(output / "raw/parsed_with_duplicates.json", parsed)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in parsed:
        key_tuple = (candidate["kind"], candidate["label"])
        target = merged.setdefault(key_tuple, {
            "id": f"{candidate['kind']}:{candidate['label']}",
            "kind": candidate["kind"],
            "label": candidate["label"],
            "title": "",
            "statement": "",
            "supporting_text": "",
            "supporting_text_role": "",
            "statement_sources": [],
            "supporting_sources": [],
            "candidate_locations": [],
        })
        target["candidate_locations"].append([candidate["source_part"], candidate["source_chunk"]])
        if len(candidate["statement"]) > len(target["statement"]):
            for field in ["title", "statement", "statement_sources"]:
                target[field] = candidate[field]
        if len(candidate["supporting_text"]) > len(target["supporting_text"]):
            for field in ["supporting_text", "supporting_text_role", "supporting_sources"]:
                target[field] = candidate[field]

    rows = []
    for row in merged.values():
        if not row["statement"]:
            continue
        all_sources = row["statement_sources"] + row["supporting_sources"]
        row["source_pdf_page"] = min((x["pdf_page"] for x in all_sources), default=None)
        rows.append(row)
    rows.sort(key=sort_key)
    write_json(output / "environments_raw.json", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    summary = {
        "book": args.book,
        "model": MODEL,
        "mineru_source": str(source),
        "chunks": len(jobs),
        "parsed_candidates": len(parsed),
        "environments": len(rows),
        "counts_by_kind": dict(sorted(counts.items())),
    }
    write_json(output / "segmentation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
