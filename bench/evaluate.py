#!/usr/bin/env python3
"""Score a pipeline output against a gold list of mathematical units.

Gold entries: {gid, page, kind, labeled, label, anchor, source_excerpt, gold_statement, must_have, borderline?}
Predictions: dataset/structured_statements.json when present (baseline + prose layer), otherwise
dataset/environments.json (baseline only), or a mathdoc-parser result.json (input_json list).

Matching: a prediction matches a gold unit when it is on the same/adjacent page and its source text
contains most of the gold source tokens (token containment), greedy best-first, one-to-one.

Usage: evaluate.py GOLD.json PRED.json [--mode auto|baseline|structured|mathdoc] [--pdf PDF]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from parse_ocr.source_grounding import semantic_tokens  # noqa: E402
from parse_ocr.text_spans import has_text_layer, span_from_excerpt, span_iou  # noqa: E402

FAMILY = {
    "definition": "def", "notation": "def",
    "theorem": "result", "lemma": "result", "proposition": "result", "corollary": "result",
    "claim": "result", "fact": "result", "conjecture": "conj", "assumption": "def",
    "example": "example", "exercise": "exercise", "remark": "remark",
    "def": "def", "thm": "result", "prop": "result", "cor": "result", "lem": "result",
}


def containment(small: list[str], large: list[str]) -> float:
    if not small or not large:
        return 0.0
    pool = Counter(large)
    hit = 0
    for token in small:
        if pool[token] > 0:
            pool[token] -= 1
            hit += 1
    return hit / len(small)


def match_score(gold_tokens: list[str], pred_tokens: list[str]) -> float:
    """Gold-in-pred containment, or pred-in-gold when the prediction is a tight sub-excerpt."""
    forward = containment(gold_tokens, pred_tokens)
    if len(pred_tokens) >= max(5, 0.35 * len(gold_tokens)):
        forward = max(forward, containment(pred_tokens, gold_tokens) - 0.05)
    return forward


def norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"\\[a-zA-Z]+", "", value or "").casefold())


def load_predictions(path: Path, mode: str, pdf: Path | None) -> list[dict]:
    data = json.loads(path.read_text())
    if mode == "auto":
        if isinstance(data, dict) and "input_json" in data:
            mode = "mathdoc"
        elif path.name == "structured_statements.json":
            mode = "structured"
        else:
            mode = "baseline"
    preds = []
    if mode == "mathdoc":
        # mathdoc result.json has no pages; locate each statement in the PDF text layer.
        import fitz
        pages_text = []
        if pdf:
            with fitz.open(pdf) as doc:
                pages_text = [semantic_tokens(page.get_text()) for page in doc]
        for item in data["input_json"]:
            tokens = semantic_tokens(item.get("statement", ""))
            page = 0
            if pages_text and tokens:
                scores = [containment(tokens[:60], text) for text in pages_text]
                page = max(range(len(scores)), key=scores.__getitem__) + 1
            preds.append({"id": item.get("label"), "kind": item.get("env", ""), "pages": [page] if page else [],
                          "source": item.get("statement", ""), "normalized": item.get("statement", ""),
                          "origin": "mathdoc"})
        return preds
    for row in data:
        if mode == "structured":
            preds.append({"id": row["uid"], "kind": row["kind"], "pages": row.get("source_pages", []),
                          "source": row.get("source_statement", ""),
                          "normalized": row.get("normalized_statement") or row.get("source_statement", ""),
                          "origin": row.get("origin", "")})
        else:
            preds.append({"id": row["id"], "kind": row["kind"], "pages": row.get("source_pages", []),
                          "source": row.get("statement", ""), "normalized": row.get("statement", ""),
                          "origin": "unlabeled_environment" if re.match(r"p\d+\.", row.get("label", "")) else "labeled_environment"})
    return preds


def must_have_score(gold: dict, text: str) -> float:
    items = gold.get("must_have") or []
    if not items:
        return 1.0
    target = norm_text(text)
    hits = 0
    for item in items:
        key = norm_text(item)
        tokens = semantic_tokens(item)
        if (key and key in target) or containment(tokens, semantic_tokens(text)) >= 0.8:
            hits += 1
    return hits / len(items)


def granularity_view(gold: list[dict], level: str) -> list[dict]:
    """`whole`: drop sub-part entries; `subpart`: replace each parent that has sub-parts by its sub-parts."""
    if level == "whole":
        return [g for g in gold if not g.get("parent_gid")]
    parents = {g["parent_gid"] for g in gold if g.get("parent_gid")}
    return [g for g in gold if g["gid"] not in parents]


def attach_eval_spans(pdf: Path | None, gold: list[dict], preds: list[dict]) -> bool:
    pages = sorted({g["page"] for g in gold if g.get("page")})
    if not pdf or not pages or not has_text_layer(pdf, pages):
        return False
    for g in gold:
        g.setdefault("_span", span_from_excerpt(pdf, [g["page"]], g.get("source_excerpt", "")))
    for p in preds:
        p.setdefault("_span", span_from_excerpt(pdf, p["pages"] or pages[:1], p["source"]))
    return True


def evaluate(gold: list[dict], preds: list[dict], threshold: float, pdf: Path | None = None) -> dict:
    spans = attach_eval_spans(pdf, gold, preds)
    pairs = []
    for gi, g in enumerate(gold):
        gt = semantic_tokens(g.get("source_excerpt", ""))
        for pi, p in enumerate(preds):
            if p["pages"] and g["page"] and min(abs(g["page"] - q) for q in p["pages"]) > 1:
                continue
            score = match_score(gt, semantic_tokens(p["source"]))
            if score >= threshold:
                # Prefer tight predictions: penalize containers much longer than the gold unit.
                ratio = len(semantic_tokens(p["source"])) / max(1, len(gt))
                pairs.append((score - 0.02 * max(0.0, ratio - 3), gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, matches = set(), set(), {}
    for score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); matches[gi] = (pi, score)
    # A container prediction covering several gold units still "covers" them (lenient recall).
    covered_lenient = set(matches)
    for gi, g in enumerate(gold):
        if gi in covered_lenient:
            continue
        gt = semantic_tokens(g.get("source_excerpt", ""))
        for p in preds:
            if (not p["pages"] or min(abs(g["page"] - q) for q in p["pages"]) <= 1) and match_score(gt, semantic_tokens(p["source"])) >= threshold:
                covered_lenient.add(gi)
                break

    def bucket(predicate):
        idx = [i for i, g in enumerate(gold) if predicate(g)]
        return {"n": len(idx), "recall_1to1": round(sum(i in matches for i in idx) / max(1, len(idx)), 3),
                "recall_lenient": round(sum(i in covered_lenient for i in idx) / max(1, len(idx)), 3)}

    ious = [span_iou(gold[gi]["_span"], preds[pi]["_span"]) for gi, (pi, _) in matches.items()] if spans else []
    located = [iou for iou, (gi, (pi, _)) in zip(ious, matches.items())
               if gold[gi]["_span"].get("found") and preds[pi]["_span"].get("found")]
    family_ok = [FAMILY.get(gold[gi]["kind"]) == FAMILY.get(preds[pi]["kind"]) for gi, (pi, _) in matches.items()]
    must = [must_have_score(gold[gi], preds[pi]["normalized"]) for gi, (pi, _) in matches.items()]
    unmatched_preds = [preds[i] for i in range(len(preds)) if i not in used_p]
    return {
        "gold": len(gold), "predicted": len(preds), "matched": len(matches),
        "all": bucket(lambda g: True),
        "labeled": bucket(lambda g: g.get("labeled")),
        "unlabeled": bucket(lambda g: not g.get("labeled")),
        "unlabeled_non_borderline": bucket(lambda g: not g.get("labeled") and not g.get("borderline")),
        "precision_1to1": round(len(matches) / max(1, len(preds)), 3),
        "kind_family_accuracy": round(sum(family_ok) / max(1, len(family_ok)), 3),
        "must_have_mean_on_matched": round(sum(must) / max(1, len(must)), 3),
        "boundary": None if not spans else {
            "located_pairs": len(located),
            "mean_span_iou": round(sum(located) / max(1, len(located)), 3),
            "iou_ge_0.8": round(sum(iou >= 0.8 for iou in located) / max(1, len(located)), 3),
            "iou_lt_0.5": round(sum(iou < 0.5 for iou in located) / max(1, len(located)), 3),
        },
        "missed": [{"gid": g["gid"], "page": g["page"], "kind": g["kind"], "labeled": g.get("labeled"),
                    "anchor": g.get("anchor", "")} for i, g in enumerate(gold) if i not in covered_lenient],
        "unmatched_predictions": [{"id": p["id"], "kind": p["kind"], "pages": p["pages"], "origin": p["origin"],
                                   "text": re.sub(r"\s+", " ", p["source"])[:140]} for p in unmatched_preds],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("pred", type=Path)
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--granularity", choices=["whole", "subpart"], default="whole")
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text())
    preds = load_predictions(args.pred, args.mode, args.pdf)
    report = evaluate(granularity_view(gold, args.granularity), preds, args.threshold, args.pdf)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"missed", "unmatched_predictions"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
