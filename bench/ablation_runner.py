#!/usr/bin/env python3
"""Ablation harness for the prose-claim layer.

Each variant runs prose_claims.py with its own settings, checkpoint directory (cache_dir) and output
suffix, so variants never overwrite each other and reruns resume from checkpoints. Results are scored
with evaluate.py against gold/ and collected in eval/ablation_results.json.

  python3 ablation_runner.py run VARIANT [VARIANT ...] [--cases a,b]
  python3 ablation_runner.py score [--variants ...]
Requires VLM_API_KEY in the environment for `run`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import evaluate, granularity_view, load_predictions  # noqa: E402

CASES = {  # case -> gold file (None: no gold, gold-free evaluation only)
    "vershynin": "vershynin", "perelman": "perelman", "string_topology": "string_topology",
    "nesterov": "nesterov_colleague", "hdp_ex": "hdp_ex", "dantzig": None, "kochenderfer": None,
}
BASE = {"window_pages": 2, "overlap_pages": 1, "sweep": True, "workers": 3}
VARIANTS: dict[str, dict] = {
    # Reference: the v3.3 configuration (existing checkpoints under prose_claims/).
    "A_v33": {"cache_dir": "prose_claims"},
    # A_rep: identical settings to A_v33 with a fresh cache, to measure run-to-run model variance.
    "A_rep": {"cache_dir": "pc_A2"},
    # B: text-layer span grounding + model anchors (miner re-run with anchor fields).
    "B_anchors": {"cache_dir": "pc_B", "anchors": True},
    # B0: span grounding only, reusing A's model outputs (isolates the deterministic part of B).
    "B0_spans_only": {"cache_dir": "prose_claims", "span_grounding": True},
    # C: granularity. C_record keeps one entry per record (no atomization); C_subparts splits exercise
    # sub-parts; both are applied on top of B.
    "C_record": {"cache_dir": "pc_B", "anchors": True, "granularity": "record", "merge_only": True},
    "C_subparts": {"cache_dir": "pc_C", "anchors": True, "subpart_policy": "split",
                   "reuse_from": {"mine": "pc_B", "sweep": "pc_B"}},
    # FINAL: recommended defaults from the ablation (span grounding, sub-part split, no anchors/tiers).
    # Its miner prompt equals A's, so A's mine/sweep checkpoints are reused when present.
    "FINAL": {"cache_dir": "pc_final", "span_grounding": True, "anchors": False, "subpart_policy": "split",
              "granularity": "atomize", "claim_tiers": False,
              "reuse_from": {"mine": "prose_claims", "sweep": "prose_claims"}},
    # FINAL2: FINAL + verifier added-assumption check with book context, unmarked exercises kept whole.
    # Reuses FINAL's miner outputs; normalize/verify/repair re-run because their prompts changed.
    "FINAL2": {"cache_dir": "pc_final2", "span_grounding": True, "anchors": False, "subpart_policy": "split",
               "granularity": "atomize", "claim_tiers": False,
               "reuse_from": {"mine": "pc_final", "sweep": "pc_final"}},
    # D: claim tiers (one miner run with tier labels, three acceptance filters).
    "D1_explicit": {"cache_dir": "pc_D", "anchors": True, "claim_tiers": True, "allowed_tiers": ["explicit"]},
    "D2_explicit_implicit": {"cache_dir": "pc_D", "anchors": True, "claim_tiers": True,
                             "allowed_tiers": ["explicit", "implicit"], "merge_only": True},
    "D3_all": {"cache_dir": "pc_D", "anchors": True, "claim_tiers": True,
               "allowed_tiers": ["explicit", "implicit", "heuristic"], "merge_only": True},
}


def settings_for(variant: str) -> dict:
    value = {**BASE, **VARIANTS[variant]}
    value["output_suffix"] = "" if variant == "A_v33" else f"__{variant}"
    value.pop("reuse_from", None)
    value.pop("merge_only", None)
    return value


def write_variant_config(case: str, variant: str) -> Path:
    base = json.loads((HERE / case / "run.json").read_text())
    base["prose_claims"] = settings_for(variant)
    for key in ("pdf", "output", "profile"):
        value = base.get(key)
        if value and not Path(value).is_absolute():
            base[key] = str((HERE / case / value).resolve())
    path = HERE / case / "ablation" / f"{variant}.run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2) + "\n")
    return path


def seed_reused_stages(case: str, variant: str) -> None:
    """Copy checkpoints of stages a variant shares with another one (e.g. C reuses B's miner)."""
    output = HERE / case / "output"
    target = VARIANTS[variant]["cache_dir"]
    for stage, source in VARIANTS[variant].get("reuse_from", {}).items():
        src, dst = output / source / stage, output / target / stage
        if src.is_dir() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-R", str(src), str(dst)], check=True)


def run_case(case: str, variant: str) -> str:
    config = write_variant_config(case, variant)
    seed_reused_stages(case, variant)
    log = HERE / case / "ablation" / f"{variant}.log"
    # Variants that only change merge-time filtering reuse another variant's model outputs.
    stages = ["merge"] if VARIANTS[variant].get("merge_only") else ["run"]
    with log.open("a") as handle:
        for stage in stages:
            result = subprocess.run([sys.executable, "-m", "parse_ocr.prose_claims", stage, "--config", str(config)],
                                    stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                return f"{case}/{variant}: FAILED ({stage}), see {log}"
    errors = errored_checkpoints(case, variant)
    return f"{case}/{variant}: ok" + (f" but {errors} errored checkpoints remain" if errors else "")


def errored_checkpoints(case: str, variant: str) -> int:
    root = HERE / case / "output" / VARIANTS[variant]["cache_dir"]
    count = 0
    for path in root.glob("*/*.json"):
        try:
            count += bool(json.loads(path.read_text()).get("error"))
        except Exception:
            count += 1
    return count


def token_usage(case: str, variant: str) -> int:
    summary = HERE / case / "output/dataset" / f"prose_claims_summary{settings_for(variant)['output_suffix']}.json"
    if not summary.is_file():
        return 0
    usage = json.loads(summary.read_text()).get("usage", {})
    return int(usage.get("total_tokens") or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))


def score(variants: list[str], cases: list[str]) -> list[dict]:
    rows = []
    for case in cases:
        if not CASES.get(case):
            continue
        gold_all = json.loads((HERE / "gold" / f"{CASES[case]}.json").read_text())
        pdf = HERE / "pdfs" / f"{case}.pdf"
        candidates = [("baseline", HERE / case / "baseline_dataset_snapshot/environments.json", "baseline")]
        candidates += [(v, HERE / case / "output/dataset" / f"structured_statements{settings_for(v)['output_suffix']}.json",
                        "structured") for v in variants]
        for name, path, mode in candidates:
            if not path.is_file():
                continue
            for level in (["whole", "subpart"] if any(g.get("parent_gid") for g in gold_all) else ["whole"]):
                gold = [dict(g) for g in granularity_view(gold_all, level)]
                report = evaluate(gold, load_predictions(path, mode, pdf), 0.6, pdf)
                summary_path = path.parent / f"prose_claims_summary{settings_for(name)['output_suffix']}.json" \
                    if name != "baseline" else None
                summary = json.loads(summary_path.read_text()) if summary_path and summary_path.is_file() else {}
                rows.append({
                    "case": case, "variant": name, "granularity": level,
                    "gold": report["gold"], "pred": report["predicted"],
                    "R": report["all"]["recall_1to1"], "R_lab": report["labeled"]["recall_1to1"],
                    "R_unlab": report["unlabeled"]["recall_1to1"],
                    "R_unlab_nb": report["unlabeled_non_borderline"]["recall_1to1"],
                    "P": report["precision_1to1"],
                    "F1": round(2 * report["all"]["recall_1to1"] * report["precision_1to1"]
                                / max(1e-9, report["all"]["recall_1to1"] + report["precision_1to1"]), 3),
                    "kind": report["kind_family_accuracy"], "must": report["must_have_mean_on_matched"],
                    "iou": (report["boundary"] or {}).get("mean_span_iou"),
                    "iou_ge_0.8": (report["boundary"] or {}).get("iou_ge_0.8"),
                    "tokens": token_usage(case, name) if name != "baseline" else None,
                    "unfaithful": (summary.get("normalization_verdicts") or {}).get("unfaithful"),
                    "span_located": summary.get("span_located"),
                    "missed": [m["gid"] for m in report["missed"]],
                })
    (HERE / "eval").mkdir(exist_ok=True)
    path = HERE / "eval/ablation_results.json"
    previous = json.loads(path.read_text()) if path.is_file() else []
    keys = {(r["case"], r["variant"], r["granularity"]) for r in rows}
    merged = [r for r in previous if (r["case"], r["variant"], r["granularity"]) not in keys] + rows
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    return rows


def print_table(rows: list[dict]) -> None:
    head = f"{'case':16s} {'variant':22s} {'gran':7s} {'gold':>4s} {'pred':>4s} {'R':>5s} {'Rlab':>5s} {'Runl':>5s} {'P':>5s} {'F1':>5s} {'kind':>5s} {'iou':>5s} {'iou8':>5s} {'tok':>7s}"
    print(head)
    for r in rows:
        f = lambda v: "  -  " if v is None else f"{v:5.2f}"
        print(f"{r['case']:16s} {r['variant']:22s} {r['granularity']:7s} {r['gold']:4d} {r['pred']:4d} "
              f"{f(r['R'])} {f(r['R_lab'])} {f(r['R_unlab'])} {f(r['P'])} {f(r['F1'])} {f(r['kind'])} "
              f"{f(r['iou'])} {f(r['iou_ge_0.8'])} {(r['tokens'] or 0) // 1000:6d}k")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "score"])
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--cases", default=",".join(case for case, gold in CASES.items() if gold))
    args = parser.parse_args()
    cases = [case for case in args.cases.split(",") if case]
    variants = args.variants or list(VARIANTS)
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        sys.exit(f"unknown variants: {sorted(unknown)}")
    if args.action == "run":
        if not os.environ.get("VLM_API_KEY"):
            sys.exit("VLM_API_KEY is not set")
        for variant in variants:
            with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                for line in pool.map(lambda case: run_case(case, variant), cases):
                    print(line, flush=True)
    print_table(score(variants, cases))


if __name__ == "__main__":
    main()
