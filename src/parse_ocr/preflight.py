#!/usr/bin/env python3
"""Secret-safe capability check for VLM and MinerU backends."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from parse_ocr.common import model_url, request, resolve_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--network", action="store_true", help="probe configured remote endpoints")
    args = parser.parse_args()
    config = resolve_config(args.config)
    output = Path(config["output"])
    pdf = Path(config["pdf"])
    vlm = config["vlm"]
    route = config.get("route", "direct")
    mineru = config.get("mineru", {})
    vlm_env = vlm.get("api_key_env", "VLM_API_KEY")
    mineru_env = mineru.get("api_key_env", "MINERU_API_KEY")
    cached_manifest = output / "mineru" / "extracted.json"
    local_candidates = [name for name in ("mineru", "magic-pdf") if shutil.which(name)]

    report = {
        "ok": True,
        "route": route,
        "pdf": {"exists": pdf.is_file(), "path": str(pdf)},
        "python": {
            "requests": True,
            "pymupdf": False,
        },
        "vlm": {
            "base_url": vlm.get("base_url"),
            "model": vlm.get("model"),
            "api_key_env": vlm_env,
            "credential_present": bool(os.environ.get(vlm_env)),
            "network_ok": None,
            "model_visible": None,
        },
        "mineru": {
            "requested_backend": mineru.get("backend", "auto"),
            "cache_present": cached_manifest.is_file(),
            "local_candidates": local_candidates,
            "local_command_configured": bool(mineru.get("local_command")),
            "api_key_env": mineru_env,
            "api_credential_present": bool(os.environ.get(mineru_env)),
        },
        "selected_mineru_backend": None,
        "errors": [],
    }
    try:
        import requests  # noqa: F401
    except ImportError:
        report["python"]["requests"] = False
        report["errors"].append("requests is required for remote backends")
    try:
        import fitz  # noqa: F401
        report["python"]["pymupdf"] = True
    except ImportError:
        report["errors"].append("PyMuPDF (fitz) is required")

    selected = None
    if route == "mineru":
        requested = mineru.get("backend", "auto")
        if requested == "auto":
            if cached_manifest.is_file():
                selected = "cache"
            elif mineru.get("local_command"):
                selected = "local"
            elif os.environ.get(mineru_env):
                selected = "api"
        else:
            selected = requested
        if selected == "cache" and not cached_manifest.is_file():
            report["errors"].append(f"cached MinerU manifest missing: {cached_manifest}")
        if selected == "local" and not mineru.get("local_command"):
            report["errors"].append("mineru.local_command must be configured for local backend")
        if selected == "api" and not os.environ.get(mineru_env):
            report["errors"].append(f"required environment variable is missing: {mineru_env}")
        if selected is None:
            report["errors"].append("no usable MinerU cache, local command, or API credential")
    report["selected_mineru_backend"] = selected
    if not os.environ.get(vlm_env):
        report["errors"].append(f"required environment variable is missing: {vlm_env}")

    if args.network and os.environ.get(vlm_env):
        try:
            response = request(
                "GET", model_url(vlm["base_url"]),
                headers={"Authorization": f"Bearer {os.environ[vlm_env]}"}, timeout=60, attempts=2,
            ).json()
            ids = [item.get("id") for item in response.get("data", [])]
            report["vlm"]["network_ok"] = True
            report["vlm"]["model_visible"] = vlm.get("model") in ids
            report["vlm"]["available_model_count"] = len(ids)
        except Exception as exc:
            report["vlm"]["network_ok"] = False
            report["errors"].append(f"VLM endpoint probe failed: {type(exc).__name__}: {exc}")

    report["ok"] = not report["errors"]
    write_json(output / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
