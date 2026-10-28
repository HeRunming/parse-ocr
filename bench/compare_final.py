#!/usr/bin/env python3
"""FINAL vs FINAL2: verdict changes, added-assumption catches, exercise merges, and the judge's flagged items.

  python3 compare_final.py [--old FINAL --new FINAL2]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ["vershynin", "perelman", "string_topology", "nesterov", "hdp_ex", "kochenderfer", "dantzig"]


def load(case: str, variant: str) -> list[dict]:
    path = HERE / case / "output/dataset" / f"structured_statements__{variant}.json"
    return json.loads(path.read_text()) if path.is_file() else []


def verdict(row: dict) -> str:
    norm = row.get("normalization") or {}
    return (norm.get("verification") or {}).get("verdict") or row.get("verification", {}).get("verdict", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="FINAL")
    parser.add_argument("--new", default="FINAL2")
    args = parser.parse_args()
    report = {}
    for case in CASES:
        old, new = load(case, args.old), load(case, args.new)
        if not new:
            continue
        # Count catches in the raw verifier checkpoints: a repaired item gets a new uid, so its original
        # flag no longer appears on the final structured row.
        added, final_flags = [], 0
        cache = {"FINAL2": "pc_final2"}.get(args.new, "pc_final2")
        for path in sorted((HERE / case / "output" / cache / "verify").glob("*.json")):
            for value in json.loads(path.read_text()).get("rows", []):
                for phrase in value.get("added_assumptions", []) or []:
                    added.append({"uid": value["id"], "phrase": phrase})
        for row in new:
            ver = (row.get("normalization") or {}).get("verification") or row.get("verification") or {}
            final_flags += bool(ver.get("added_assumptions"))
        summary_path = HERE / case / "output/dataset" / f"prose_claims_summary__{args.new}.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        report[case] = {
            "items": [len(old), len(new)],
            "empty_normalized": [sum(not r.get("normalized_statement") for r in old),
                                 sum(not r.get("normalized_statement") for r in new)],
            "verdicts_old": dict(Counter(verdict(r) for r in old)),
            "verdicts_new": dict(Counter(verdict(r) for r in new)),
            "added_assumption_flags": len(added),
            "final_items_still_flagged": final_flags,
            "examples": added[:6],
            "merged_unmarked_exercises": sum(1 for r in new if r.get("origin", "").endswith("environment")
                                             and r.get("kind") == "exercise"
                                             and "component" in json.dumps(r.get("normalization") or {})),
            "rejected": summary.get("reject_reasons"),
        }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    (HERE / "eval/final_vs_final2.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
