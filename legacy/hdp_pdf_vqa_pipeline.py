#!/usr/bin/env python3
"""Recoverable HDP runner for DataFlow's PDF_VQA extraction design."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz
import requests


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "HDP-2.pdf"
OUTPUT = ROOT / "output/hdp_pdf_vqa"
BOOK_PROFILE = "hdp"
DATAFLOW = Path(os.environ.get("DATAFLOW_ROOT", "../DataFlow"))
QA_PROMPT_SOURCE = DATAFLOW / "dataflow/prompts/pdf2vqa.py"
GATEWAY = os.environ.get("VLM_CHAT_URL", "http://127.0.0.1:3000/v1/chat/completions")
MODEL = "gpt-5.5"
MINERU_BASE = "https://mineru.net/api/v4"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request_with_retry(method: str, url: str, *, attempts: int = 8, **kwargs) -> requests.Response:
    delay = 2.0
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 1.7, 20)
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last_error}")


def split_pdf(max_pages: int = 200) -> List[Dict[str, Any]]:
    parts_dir = OUTPUT / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    source = fitz.open(PDF)
    parts = []
    for start in range(0, len(source), max_pages):
        end = min(start + max_pages, len(source))
        part_path = parts_dir / f"{PDF.stem}_pages_{start + 1:03d}_{end:03d}.pdf"
        if not part_path.exists():
            part = fitz.open()
            part.insert_pdf(source, from_page=start, to_page=end - 1)
            part.save(part_path)
            part.close()
        check = fitz.open(part_path)
        if len(check) != end - start:
            raise RuntimeError(f"bad split {part_path}: expected {end-start}, got {len(check)}")
        check.close()
        parts.append({"path": str(part_path), "start_page": start + 1, "end_page": end})
    write_json(OUTPUT / "parts.json", parts)
    return parts


def mineru_extract(parts: List[Dict[str, Any]], token: str) -> List[Dict[str, Any]]:
    state_path = OUTPUT / "mineru/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if "batch_id" not in state:
        payload = {
            "files": [
                {"name": Path(part["path"]).name, "data_id": str(index)}
                for index, part in enumerate(parts)
            ],
            "model_version": "vlm",
        }
        response = request_with_retry(
            "POST", f"{MINERU_BASE}/file-urls/batch", headers=headers, json=payload, timeout=120
        ).json()
        if response.get("code") != 0:
            raise RuntimeError(f"MinerU upload URL request failed: {response}")
        state.update(response["data"])
        state["model_version"] = "vlm"
        write_json(state_path, state)

    if not state.get("uploaded"):
        for part, upload_url in zip(parts, state["file_urls"]):
            last_error = None
            for attempt in range(6):
                try:
                    # Reopen on every retry so the upload always starts at byte zero.
                    with open(part["path"], "rb") as stream:
                        response = requests.put(upload_url, data=stream, timeout=300)
                        response.raise_for_status()
                    last_error = None
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    time.sleep(min(2 ** attempt, 20))
            if last_error is not None:
                raise RuntimeError(f"MinerU upload failed for {part['path']}: {last_error}")
        state["uploaded"] = True
        write_json(state_path, state)

    print(f"MinerU batch {state['batch_id']} uploaded; polling...", flush=True)
    poll_url = f"{MINERU_BASE}/extract-results/batch/{state['batch_id']}"
    deadline = time.time() + 3600
    while True:
        response = request_with_retry("GET", poll_url, headers=headers, timeout=120).json()
        if response.get("code") != 0:
            raise RuntimeError(f"MinerU polling failed: {response}")
        results = response["data"].get("extract_result", [])
        states = [item.get("state") for item in results]
        print(f"MinerU states: {states}", flush=True)
        if results and all(item.get("state") == "done" for item in results):
            state["results"] = results
            write_json(state_path, state)
            break
        failed = [item for item in results if item.get("state") == "failed"]
        if failed:
            raise RuntimeError(f"MinerU failed: {failed}")
        if time.time() > deadline:
            raise TimeoutError(f"MinerU polling timed out; batch_id={state['batch_id']}")
        time.sleep(10)

    extracted = []
    for result in state["results"]:
        data_id = str(result["data_id"])
        part_dir = OUTPUT / "mineru" / data_id
        content_lists = list(part_dir.glob("*_content_list.json"))
        if not content_lists:
            zip_path = OUTPUT / "mineru" / f"{data_id}.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            response = request_with_retry("GET", result["full_zip_url"], timeout=300)
            zip_path.write_bytes(response.content)
            part_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(part_dir)
            content_lists = list(part_dir.glob("*_content_list.json"))
        if not content_lists:
            raise RuntimeError(f"content_list missing under {part_dir}")
        extracted.append({
            **parts[int(data_id)],
            "data_id": data_id,
            "content_list": str(content_lists[0]),
            "mineru_dir": str(part_dir),
        })
    write_json(OUTPUT / "mineru/extracted.json", extracted)
    return extracted


def load_dataflow_prompt() -> str:
    tree = ast.parse(QA_PROMPT_SOURCE.read_text(encoding="utf-8"))
    qa_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QAExtractPrompt"
    )
    qa_class.decorator_list = []
    namespace = {"PromptABC": object}
    exec(compile(ast.Module(body=[qa_class], type_ignores=[]), str(QA_PROMPT_SOURCE), "exec"), namespace)
    base = namespace["QAExtractPrompt"]().build_prompt()
    if BOOK_PROFILE == "foml":
        override = r"""

BOOK-SPECIFIC OVERRIDES (these take precedence over conflicting rules above):
1. Preserve the FULL original hierarchical exercise label exactly. Numeric examples include
   "2.1" and "10.3"; appendix examples include "B.1", "C.2", "D.5", and "E.1".
2. Extract every complete numbered exercise under a section titled "Exercises", even when there is
   no answer or solution. Do not treat numbered definitions, theorems, examples, algorithms,
   chapter-note citations, contents entries, or equation numbers as exercises.
3. This public edition does not contain the instructor solution manual. Do not invent answers or
   solutions. Leave <answer></answer> and <solution></solution> empty unless the supplied blocks
   visibly contain an explicit source answer/hint section.
4. A chunk may begin in the middle of an exercise section. Do not require the first label to end in
   .1, and leave <title></title> blank when needed.
5. A page-overlap item may duplicate a complete exercise. Extract it normally; downstream
   deduplication will remove exact duplicates.
6. Never join adjacent labels into one qa_pair. Keep all lettered subparts belonging to one label in
   that label's single qa_pair, including continuations across page boundaries.
7. Ignore isolated continuation fragments at the beginning or end of a chunk unless the complete
   labeled exercise is present in the supplied JSON.
8. Do not paraphrase, solve, or rewrite anything. The output remains ID-only as specified above.
"""
    else:
        override = r"""

BOOK-SPECIFIC OVERRIDES (these take precedence over conflicting rules above):
1. Preserve the FULL original hierarchical exercise label exactly. Examples: use "4.31", "5.18", and "8.4"; never shorten them to "31", "18", or "4".
2. Extract every complete exercise even if it has no answer or solution nearby. Do not require the first label after a title or chunk boundary to be 1. A chunk may begin in the middle of a chapter; leave <title></title> blank when needed.
3. Under an "Exercises" heading, numbered entries are questions. Under "Hints for the exercises", "Solutions", or "Answers", numbered entries are solutions and must use an empty <question></question>.
4. A page-overlap item may duplicate a complete problem. Extract it normally; downstream deduplication will remove exact duplicates.
5. Never join adjacent labels into one qa_pair. Keep all subparts (a), (b), ... belonging to one label in that label's single qa_pair.
6. Ignore isolated continuation fragments at the beginning or end of a chunk unless the complete labeled exercise is present in the supplied JSON.
7. Do not paraphrase or solve anything. The output remains ID-only as specified above.
"""
    return base + "\n" + override


def convert_layout(content_list_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(content_list_path.read_text(encoding="utf-8"))
    converted = []
    next_id = 0
    for original in data:
        item = dict(original)
        item["id"] = next_id
        item.pop("bbox", None)
        # Keep page_idx so safe chunks can be formed at page boundaries.
        if item.get("type") == "list" and item.get("sub_type") == "text":
            entries = item.get("list_items", [])
            for offset, entry in enumerate(entries):
                converted.append({
                    "type": "text", "text": entry, "id": next_id + offset,
                    "page_idx": item.get("page_idx"),
                })
            next_id += len(entries)
        else:
            converted.append(item)
            next_id += 1
    return converted


def page_chunks(layout: List[Dict[str, Any]], target_chars: int = 150_000) -> List[List[Dict[str, Any]]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for item in layout:
        by_page.setdefault(int(item.get("page_idx") or 0), []).append(item)
    pages = sorted(by_page)
    core_ranges: List[Tuple[int, int]] = []
    start = pages[0]
    used = 0
    previous = pages[0]
    for page in pages:
        page_size = len(json.dumps(by_page[page], ensure_ascii=False))
        if used and used + page_size > target_chars:
            core_ranges.append((start, previous))
            start, used = page, 0
        used += page_size
        previous = page
    core_ranges.append((start, previous))

    chunks = []
    min_page, max_page = pages[0], pages[-1]
    for start, end in core_ranges:
        include_start, include_end = max(min_page, start - 1), min(max_page, end + 1)
        chunks.append([
            item for page in range(include_start, include_end + 1) for item in by_page.get(page, [])
        ])
    return chunks


def call_llm(prompt: str, model: str, key: str, *, max_tokens: int = 32000) -> str:
    response = request_with_retry(
        "POST",
        GATEWAY,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=1200,
        attempts=5,
    ).json()
    return response["choices"][0]["message"]["content"]


def extract_all_chunks(extracted: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    prompt = load_dataflow_prompt()
    jobs = []
    for part in extracted:
        layout = convert_layout(Path(part["content_list"]))
        converted_path = OUTPUT / "layouts" / f"part_{part['data_id']}.json"
        write_json(converted_path, layout)
        for chunk_index, chunk in enumerate(page_chunks(layout)):
            response_path = OUTPUT / "responses" / f"part_{part['data_id']}_chunk_{chunk_index:02d}.txt"
            jobs.append({
                "part": part,
                "layout": layout,
                "chunk": chunk,
                "chunk_index": chunk_index,
                "response_path": response_path,
            })

    def run(job: Dict[str, Any]) -> None:
        if job["response_path"].exists() and "<qa_pair>" in job["response_path"].read_text(encoding="utf-8"):
            return
        payload = prompt + "\n" + json.dumps(job["chunk"], ensure_ascii=False)
        result = call_llm(payload, MODEL, key)
        job["response_path"].parent.mkdir(parents=True, exist_ok=True)
        job["response_path"].write_text(result, encoding="utf-8")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            future.result()
            print(f"LLM done: part {job['part']['data_id']} chunk {job['chunk_index']}", flush=True)
    return jobs


def xml_field(block: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", block, flags=re.S)
    return match.group(1).strip() if match else ""


def ids_to_text(ids: str, layout: List[Dict[str, Any]], image_prefix: str) -> str:
    texts = []
    for raw_id in ids.replace(" ", "").split(","):
        if not raw_id.isdigit() or int(raw_id) >= len(layout):
            continue
        item = layout[int(raw_id)]
        if "text" in item and item["text"]:
            texts.append(item["text"])
        elif "table_body" in item:
            texts.append(item["table_body"])
        elif "img_path" in item:
            name = Path(item["img_path"]).name
            caption = " ".join(item.get("image_caption", []) or ["image"])
            texts.append(f"![{caption}]({image_prefix}/{name})")
    return "\n".join(texts)


def normalize_label(label: str) -> str:
    label = label.strip().replace(" ", "")
    match = re.search(r"(?:\d+|[A-Z])(?:\.\d+)+|\d+", label, flags=re.I)
    return match.group(0) if match else ""


def label_sort_key(label: str) -> Tuple[int, ...]:
    parts = label.upper().split(".")
    head = int(parts[0]) if parts[0].isdigit() else 1000 + ord(parts[0]) - ord("A")
    return (head, *(int(value) for value in parts[1:]))


def parse_and_merge(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed = []
    for job in jobs:
        raw = job["response_path"].read_text(encoding="utf-8")
        part = job["part"]
        prefix = str(Path(part["mineru_dir"]) / "images")
        for block in re.findall(r"<qa_pair>(.*?)</qa_pair>", raw, flags=re.S):
            label = normalize_label(xml_field(block, "label"))
            if not label:
                continue
            parsed.append({
                "label": label,
                "question": ids_to_text(xml_field(block, "question"), job["layout"], prefix),
                "answer": xml_field(block, "answer"),
                "solution": ids_to_text(xml_field(block, "solution"), job["layout"], prefix),
                "source_part": part["data_id"],
                "source_chunk": job["chunk_index"],
            })
    write_json(OUTPUT / "qa/parsed_with_duplicates.json", parsed)

    merged: Dict[str, Dict[str, Any]] = {}
    for row in parsed:
        target = merged.setdefault(row["label"], {
            "label": row["label"], "question": "", "answer": "", "solution": "", "sources": []
        })
        target["sources"].append([row["source_part"], row["source_chunk"]])
        if len(row["question"]) > len(target["question"]):
            target["question"] = row["question"]
        if len(row["answer"]) > len(target["answer"]):
            target["answer"] = row["answer"]
        if len(row["solution"]) > len(target["solution"]):
            target["solution"] = row["solution"]

    rows = sorted((row for row in merged.values() if row["question"]), key=lambda row: label_sort_key(row["label"]))
    write_json(OUTPUT / "qa/questions_raw.json", rows)
    return rows


MATH_PATTERN = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)


def mask_math(text: str) -> Tuple[str, List[str]]:
    math = []
    def replace(match: re.Match) -> str:
        math.append(match.group(0))
        return f"<MATH_{len(math) - 1}>"
    return MATH_PATTERN.sub(replace, text), math


def valid_spelling_only(original: str, corrected: str) -> Tuple[bool, List[Dict[str, str]]]:
    original_tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", original)
    corrected_tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", corrected)
    original_words = [token for token in original_tokens if token.isalpha()]
    corrected_words = [token for token in corrected_tokens if token.isalpha()]
    original_other = [token for token in original_tokens if not token.isalpha()]
    corrected_other = [token for token in corrected_tokens if not token.isalpha()]
    if original_other != corrected_other or len(original_words) != len(corrected_words):
        return False, []
    changes = []
    for old, new in zip(original_words, corrected_words):
        if old == new:
            continue
        distance = sum(1 for op in difflib.ndiff(old.lower(), new.lower()) if op[0] in "+-")
        if distance > 2:
            return False, []
        changes.append({"from": old, "to": new})
    if len(changes) > max(3, len(original_words) // 30):
        return False, []
    return True, changes


def proofread_questions(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    proofreading_dir = OUTPUT / "proofreading"
    proofreading_dir.mkdir(parents=True, exist_ok=True)
    batches = [rows[index:index + 20] for index in range(0, len(rows), 20)]

    def run_batch(index: int, batch: List[Dict[str, Any]]) -> None:
        response_path = proofreading_dir / f"batch_{index:03d}.json"
        input_path = proofreading_dir / f"batch_{index:03d}_input.json"
        masked = []
        for row in batch:
            prose, _ = mask_math(row["question"])
            masked.append({"label": row["label"], "question": prose})
        write_json(input_path, masked)
        if response_path.exists():
            return
        prompt = r"""
You are a conservative OCR spelling proofreader for an English mathematics textbook.
Return ONLY a valid JSON array with objects {"label": string, "question": string}.
Correct only unmistakable English OCR spelling errors or broken English words, for example "eficient" to "efficient".
Do not paraphrase, improve style, change terminology, add or delete words, or change capitalization unless it is part of the spelling error.
Keep every digit, punctuation mark, whitespace character, HTML tag, and <MATH_n> placeholder byte-for-byte unchanged.
If uncertain, leave the text unchanged.
Input:
""" + json.dumps(masked, ensure_ascii=False)
        raw = call_llm(prompt, MODEL, key, max_tokens=24000)
        match = re.search(r"\[.*\]", raw, flags=re.S)
        if not match:
            raise RuntimeError(f"invalid proofreading response for batch {index}")
        parsed = json.loads(match.group(0))
        write_json(response_path, parsed)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(run_batch, index, batch): index for index, batch in enumerate(batches)}
        for future in as_completed(futures):
            future.result()
            print(f"Proofreading done: batch {futures[future]}", flush=True)

    audit = []
    corrected_rows = []
    for index, batch in enumerate(batches):
        suggestions = {
            item["label"]: item["question"]
            for item in json.loads((proofreading_dir / f"batch_{index:03d}.json").read_text(encoding="utf-8"))
        }
        for row in batch:
            masked, math = mask_math(row["question"])
            suggestion = suggestions.get(row["label"], masked)
            valid, changes = valid_spelling_only(masked, suggestion)
            restored = suggestion
            if valid:
                for math_index, original_math in enumerate(math):
                    restored = restored.replace(f"<MATH_{math_index}>", original_math)
            if not valid or any(f"<MATH_{i}>" in restored for i in range(len(math))):
                restored, changes, valid = row["question"], [], False
            updated = dict(row)
            updated["question"] = restored
            corrected_rows.append(updated)
            if changes:
                audit.append({"label": row["label"], "changes": changes})
    write_json(OUTPUT / "qa/spelling_corrections.json", audit)
    write_json(OUTPUT / "qa/questions.json", corrected_rows)
    return corrected_rows


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    jsonl = OUTPUT / "qa/questions.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    markdown = []
    for row in rows:
        markdown.extend([
            f"### Exercise {row['label']}", "", row["question"], "",
            f"**Answer:** {row['answer']}" if row["answer"] else "**Answer:**", "",
        ])
        if row["solution"]:
            markdown.extend(["**Solution/Hint:**", "", row["solution"], ""])
        markdown.extend(["---", ""])
    (OUTPUT / "qa/questions.md").write_text("\n".join(markdown), encoding="utf-8")
    summary = {
        "pdf": str(PDF), "pages": len(fitz.open(PDF)), "mineru_model_version": "vlm", "llm_model": MODEL,
        "questions": len(rows), "with_answer": sum(bool(row["answer"]) for row in rows),
        "with_solution_or_hint": sum(bool(row["solution"]) for row in rows),
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def apply_verified_pdf_fixes() -> None:
    """Apply the small set of source-verified corrections after normal output writing."""
    from apply_verified_fixes import audit
    print(f"Applied {len(audit)} source-verified fixes", flush=True)


def main() -> None:
    global PDF, OUTPUT, BOOK_PROFILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-proofreading", action="store_true")
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--profile", choices=["hdp", "foml"], default=BOOK_PROFILE)
    args = parser.parse_args()
    PDF = args.pdf.resolve()
    OUTPUT = args.output.resolve()
    BOOK_PROFILE = args.profile
    mineru_key = os.environ.get("MINERU_API_KEY")
    llm_key = os.environ.get("DF_API_KEY")
    if not mineru_key or not llm_key:
        raise SystemExit("MINERU_API_KEY and DF_API_KEY are required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parts = split_pdf()
    extracted = mineru_extract(parts, mineru_key)
    jobs = extract_all_chunks(extracted, llm_key)
    rows = parse_and_merge(jobs)
    if not args.skip_proofreading:
        rows = proofread_questions(rows, llm_key)
    else:
        write_json(OUTPUT / "qa/questions.json", rows)
    write_outputs(rows)
    if BOOK_PROFILE == "hdp":
        apply_verified_pdf_fixes()


if __name__ == "__main__":
    main()
