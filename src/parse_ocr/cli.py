"""Single entry point: `parse-ocr COMMAND ...` dispatches to the stage modules.

Each command keeps the argument interface of its module, so
`parse-ocr direct run --config run.json` equals `python -m parse_ocr.direct_pipeline run --config run.json`.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "init": ("init_project", "create run.json and book_profile.json for a PDF"),
    "inspect": ("book_inspector", "render representative pages for profile writing"),
    "preflight": ("preflight", "check dependencies, credentials and endpoints"),
    "direct": ("direct_pipeline", "default route: direct VLM extraction (all stages)"),
    "prose-claims": ("prose_claims", "model-written layer: prose claims + normalized statements"),
    "verify": ("verify_source_conflicts", "source-crop adjudication of formula/span conflicts"),
    "relationships": ("extract_relationships", "cross-references and dependency graph"),
    "validate": ("validate_dataset", "deterministic schema and anomaly audit"),
    "hybrid": ("hybrid_pipeline", "optional fusion of direct output with an independent TeX export"),
    "mineru": ("mineru_pipeline", "legacy MinerU route (explicit opt-in only)"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: parse-ocr COMMAND [ARGS...]", "", "commands:"]
    lines += [f"  {name.ljust(width)}  {help_text}" for name, (_, help_text) in COMMANDS.items()]
    lines += ["", "Run `parse-ocr COMMAND --help` for the arguments of one command."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(usage())
        return
    name = argv[0]
    if name not in COMMANDS:
        raise SystemExit(f"unknown command: {name}\n\n{usage()}")
    module = importlib.import_module(f"parse_ocr.{COMMANDS[name][0]}")
    sys.argv = [f"parse-ocr {name}", *argv[1:]]
    module.main()


if __name__ == "__main__":
    main()
