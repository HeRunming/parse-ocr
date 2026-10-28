#!/usr/bin/env python3
"""Apply math-frozen spelling repair and deterministic footnote-marker cleanup."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output/hdp_pdf_vqa_gpt55_corrected"
QA = DATASET / "qa"
API_URL = os.environ.get("VLM_CHAT_URL", "http://127.0.0.1:3000/v1/chat/completions")
MODEL = "gpt-5.5"
MATH_PATTERN = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def mask_math(text: str) -> tuple[str, list[str]]:
    math: list[str] = []

    def replace(match: re.Match[str]) -> str:
        math.append(match.group(0))
        return f"<MATH_{len(math)-1}>"

    return MATH_PATTERN.sub(replace, text), math


def restore_math(text: str, math: list[str]) -> str:
    for index, value in enumerate(math):
        text = text.replace(f"<MATH_{index}>", value)
    return text


def valid_spelling_only(original: str, corrected: str) -> tuple[bool, list[dict[str, str]]]:
    old_tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", original)
    new_tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", corrected)
    old_words = [token for token in old_tokens if token.isalpha()]
    new_words = [token for token in new_tokens if token.isalpha()]
    old_other = [token for token in old_tokens if not token.isalpha()]
    new_other = [token for token in new_tokens if not token.isalpha()]
    if old_other != new_other or len(old_words) != len(new_words):
        return False, []
    changes = []
    for old, new in zip(old_words, new_words):
        if old == new:
            continue
        distance = sum(1 for op in difflib.ndiff(old.lower(), new.lower()) if op[0] in "+-")
        if distance > 3:
            return False, []
        changes.append({"from": old, "to": new})
    if len(changes) > max(4, len(old_words) // 25):
        return False, []
    return True, changes


def remove_footnote_markers(text: str) -> tuple[str, list[str]]:
    removed = re.findall(r"\$\^(?:\{\d+\}|\d+)\$", text)
    text = re.sub(r"\$\^(?:\{\d+\}|\d+)\$", "", text)
    if r"\text{satisfies}^{10}" in text:
        removed.append(r"^{10} after \text{satisfies}")
        text = text.replace(r"\text{satisfies}^{10}", r"\text{satisfies}")
    if r"(\mathrm{Type}^9\ p)" in text:
        removed.append(r"^9 after \mathrm{Type}")
        text = text.replace(r"(\mathrm{Type}^9\ p)", r"(\mathrm{Type}\ p)")
    return text, removed


def call_batch(batch: list[dict[str, str]], api_key: str) -> list[dict[str, str]]:
    prompt = r"""
You are a conservative spelling proofreader for English mathematics exercise statements.
The mathematics has been frozen into <MATH_n> placeholders.

Return ONLY a valid JSON array with objects {"label": string, "question": string}, in the same order.
Correct only unmistakable English spelling errors, including obvious printed typos such as
"entires" to "entries", "eficient" to "efficient", and words broken by OCR.
Do not paraphrase, improve style, change terminology, change capitalization, add/delete/reorder words,
or touch any digit, punctuation, whitespace, HTML tag, or <MATH_n> placeholder.
If uncertain, leave the word unchanged.

Input:
""" + json.dumps(batch, ensure_ascii=False)
    last_error = ""
    for attempt in range(5):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 24000,
                },
                timeout=1200,
            )
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            raw = response.json()["choices"][0]["message"]["content"]
            match = re.search(r"\[.*\]", raw, flags=re.S)
            if not match:
                raise ValueError("response contains no JSON array")
            parsed = json.loads(match.group(0))
            if not isinstance(parsed, list):
                raise ValueError("response is not an array")
            return parsed
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 4:
                time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(last_error)


def write_outputs(rows: list[dict[str, Any]]) -> None:
    write_json(QA / "questions.json", rows)
    (QA / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    markdown: list[str] = []
    for row in rows:
        markdown.extend([f"### Exercise {row['label']}", "", row["question"], ""])
        if row.get("solution"):
            markdown.extend(["**Solution/Hint:**", "", row["solution"], ""])
        markdown.extend(["---", ""])
    (QA / "questions.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    global DATASET, QA
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-source", action="store_true",
        help="replace questions_before_spelling.json from the current visual output",
    )
    args = parser.parse_args()
    DATASET = args.dataset.resolve()
    QA = DATASET / "qa"
    api_key = os.environ.get("DF_API_KEY")
    if not api_key:
        raise SystemExit("DF_API_KEY is required")

    source_path = QA / "questions_before_spelling.json"
    if source_path.exists() and not args.refresh_source:
        rows = json.loads(source_path.read_text(encoding="utf-8"))
    else:
        rows = json.loads((QA / "questions.json").read_text(encoding="utf-8"))
        write_json(source_path, rows)

    prepared = []
    footnote_audit = []
    math_by_label: dict[str, list[str]] = {}
    cleaned_by_label: dict[str, str] = {}
    for row in rows:
        cleaned, removed = remove_footnote_markers(row["question"])
        if removed:
            footnote_audit.append({"label": row["label"], "removed": removed})
        masked, math = mask_math(cleaned)
        prepared.append({"label": row["label"], "question": masked})
        math_by_label[row["label"]] = math
        cleaned_by_label[row["label"]] = cleaned
    write_json(QA / "footnote_marker_cleanup.json", footnote_audit)

    batches = [prepared[i:i + args.batch_size] for i in range(0, len(prepared), args.batch_size)]
    proof_dir = DATASET / "proofreading"

    def run(index: int, batch: list[dict[str, str]]) -> None:
        path = proof_dir / f"batch_{index:03d}.json"
        if path.exists() and not args.force:
            return
        write_json(path, call_batch(batch, api_key))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, index, batch): index for index, batch in enumerate(batches)}
        for future in as_completed(futures):
            future.result()
            print(f"spelling batch {futures[future]+1}/{len(batches)} ok", flush=True)

    suggestions: dict[str, str] = {}
    for index in range(len(batches)):
        values = json.loads((proof_dir / f"batch_{index:03d}.json").read_text(encoding="utf-8"))
        suggestions.update({item["label"]: item["question"] for item in values})

    final_rows = []
    audit = []
    rejected = []
    for row in rows:
        label = row["label"]
        original_masked, _ = mask_math(cleaned_by_label[label])
        suggestion = suggestions.get(label, original_masked)
        valid, changes = valid_spelling_only(original_masked, suggestion)
        if not valid:
            suggestion = original_masked
            rejected.append(label)
            changes = []
        restored = restore_math(suggestion, math_by_label[label])
        if re.search(r"<MATH_\d+>", restored):
            restored = cleaned_by_label[label]
            rejected.append(label)
            changes = []
        updated = dict(row)
        updated["question"] = restored
        final_rows.append(updated)
        if changes:
            audit.append({"label": label, "changes": changes})

    write_outputs(final_rows)
    write_json(QA / "spelling_corrections.json", audit)
    summary = {
        "questions": len(final_rows),
        "footnote_markers_removed": sum(len(item["removed"]) for item in footnote_audit),
        "questions_with_spelling_changes": len(audit),
        "word_changes": sum(len(item["changes"]) for item in audit),
        "rejected_labels": sorted(set(rejected)),
    }
    write_json(QA / "cleanup_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
