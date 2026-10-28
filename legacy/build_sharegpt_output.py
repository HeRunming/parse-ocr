#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "--qa-dir",
    type=Path,
    default=ROOT / "output/hdp_pdf_vqa/qa",
    help="directory containing questions.json and receiving sharegpt.json",
)
args = parser.parse_args()
QA = args.qa_dir.resolve()
rows = json.loads((QA / "questions.json").read_text(encoding="utf-8"))
image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
records = []

for row in rows:
    assistant = "\n\n".join(value.strip() for value in (row["answer"], row["solution"]) if value.strip())
    if not assistant:
        continue
    images = image_pattern.findall(row["question"]) + image_pattern.findall(row["solution"])
    question = image_pattern.sub("", row["question"]).strip()
    assistant = image_pattern.sub("", assistant).strip()
    records.append({
        "label": row["label"],
        "messages": [
            {"role": "user", "content": "<image>" * len(images) + question},
            {"role": "assistant", "content": assistant},
        ],
        "images": images,
    })

(QA / "sharegpt.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print({"records": len(records), "with_images": sum(bool(row["images"]) for row in records)})
