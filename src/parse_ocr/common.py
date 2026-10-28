#!/usr/bin/env python3
"""Shared helpers for the adaptive mathematics PDF pipeline."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_usage(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively sum numeric usage fields while retaining provider subtrees."""
    keys = sorted({key for value in values if isinstance(value, dict) for key in value})
    total: dict[str, Any] = {}
    for key in keys:
        items = [value.get(key) for value in values if isinstance(value, dict) and value.get(key) is not None]
        if items and all(isinstance(item, dict) for item in items):
            total[key] = aggregate_usage(items)
        elif items and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
            total[key] = sum(items)
    return total


def resolve_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = load_json(path)
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    for name in ("pdf", "output", "profile"):
        value = config.get(name)
        if value and not Path(value).is_absolute():
            config[name] = str((path.parent / value).resolve())
    # Endpoints may be written as "${VLM_BASE_URL}" so shared configs carry no host names.
    vlm = config.get("vlm")
    if isinstance(vlm, dict) and isinstance(vlm.get("base_url"), str):
        expanded = os.path.expandvars(vlm["base_url"])
        if "$" in expanded:
            raise RuntimeError(f"unresolved environment variable in vlm.base_url: {vlm['base_url']}")
        vlm["base_url"] = expanded
    return config


def require_secret(env_name: str) -> str:
    value = os.environ.get(env_name, "")
    if not value:
        raise RuntimeError(f"required environment variable is missing: {env_name}")
    return value


def request(method: str, url: str, *, attempts: int = 6, **kwargs: Any) -> Any:
    # Keep offline commands such as init/inspect/validate independent of the HTTP stack.
    import requests

    delay = 2.0
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last = exc
            if attempt + 1 == attempts:
                break
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 429:
                # Rate limited: honor Retry-After, otherwise back off much longer than for transport errors.
                retry_after = response.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 15.0 * (attempt + 1)
                time.sleep(min(wait, 120.0))
                continue
            time.sleep(delay)
            delay = min(delay * 1.7, 20.0)
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last}")


def chat_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def model_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value + "/models"


def extract_message(payload: dict[str, Any]) -> str:
    value = payload["choices"][0]["message"].get("content", "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item.get("text", "") for item in value if isinstance(item, dict)
        )
    raise ValueError("unsupported response content shape")


def call_chat(
    config: dict[str, Any], content: str | list[dict[str, Any]], *, max_tokens: int = 32000
) -> tuple[str, dict[str, Any]]:
    vlm = config["vlm"]
    key = require_secret(vlm.get("api_key_env", "VLM_API_KEY"))
    response = request(
        "POST",
        chat_url(vlm["base_url"]),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": vlm["model"],
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=int(vlm.get("timeout_seconds", 1200)),
        attempts=int(vlm.get("request_attempts", 6)),
    )
    payload = response.json()
    return extract_message(payload), payload.get("usage", {})


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
