#!/usr/bin/env python3
"""Anchor-based source spans over the native PDF text layer.

A model returns short exact opening/closing anchors; this module finds them among the page words and
cuts the source span deterministically (page/word interval, bbox, plain text). It never trusts the
model's own transcription for location, and it is the boundary oracle used by evaluation (word-span
IoU between a prediction and a gold unit).

The text layer is a positional scaffold: its words are reliable, its mathematics often is not, so the
cut text is stored next to, never instead of, the model's LaTeX excerpt.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz

HEADER_POINTS = 32.0
MAX_SPAN_WORDS = 900


def norm_word(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).replace("_", "")


def plain_words(text: str) -> list[str]:
    """Words of a Markdown/LaTeX excerpt that can appear verbatim in a text layer."""
    text = re.sub(r"\$\$.*?\$\$", " ", text or "", flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", " ", text)
    text = re.sub(r"[*_`#>]", " ", text)
    words = [norm_word(word) for word in text.split()]
    return [word for word in words if word and not word.isdigit() or (word.isdigit() and len(word) > 1)]


@lru_cache(maxsize=64)
def _page_words(pdf: str, page: int) -> tuple[tuple[Any, ...], ...]:
    with fitz.open(pdf) as document:
        if page < 1 or page > len(document):
            return ()
        raw = document[page - 1].get_text("words", sort=True)
    words = []
    pending = None
    for x0, y0, x1, y1, text, *_ in raw:
        if y0 < HEADER_POINTS:
            continue
        if pending is not None:
            # Join end-of-line hyphenation: "deﬁ-" + "nition".
            text = pending[4][:-1] + text
            x0, y0 = min(x0, pending[0]), min(y0, pending[1])
            pending = None
        if text.endswith("-") and len(text) > 2:
            pending = (x0, y0, x1, y1, text)
            continue
        key = norm_word(text)
        if key:
            words.append((key, page, (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)), text))
    return tuple(words)


def has_text_layer(pdf: Path, pages: list[int]) -> bool:
    return any(len(_page_words(str(pdf), page)) > 30 for page in pages)


def _ordered_hits(sequence: list[str], window: list[str]) -> tuple[int, int]:
    """Greedy in-order matches of `sequence` in `window`; returns (hits, index of last hit in window)."""
    hits, position, last = 0, 0, -1
    for token in sequence:
        try:
            found = window.index(token, position)
        except ValueError:
            continue
        hits += 1
        position = found + 1
        last = found
    return hits, last


def _find(sequence: list[str], words: list[tuple], lo: int, hi: int) -> tuple[int, int, float]:
    """Best gap-tolerant occurrence of `sequence` in words[lo:hi].

    Math tokens are stripped from anchors but present (garbled) in the text layer, so the anchor
    words may be separated by extra page words. Returns (first word index, last word index, score)
    where score is the share of anchor words matched in order within a 3x-long window.
    """
    k = len(sequence)
    keys = [word[0] for word in words]
    best = (-1, -1, 0.0)
    if not k:
        return best
    for index in range(lo, min(hi, len(words))):
        if keys[index] != sequence[0] and (k < 2 or keys[index] != sequence[1]):
            continue
        window = keys[index:index + 3 * k + 4]
        hits, last = _ordered_hits(sequence, window)
        score = hits / k
        if score > best[2]:
            best = (index, index + last, score)
            if score == 1.0:
                break
    return best


def locate_span(
    pdf: Path, pages: list[int], anchor_start: str, anchor_end: str, *, min_score: float = 0.7,
) -> dict[str, Any]:
    """Locate [anchor_start ... anchor_end] on the given pages (plus the next page for spill-over)."""
    pages = sorted(set(int(page) for page in pages if page))
    if pages:
        pages = pages + [pages[-1] + 1]
    words: list[tuple] = []
    for page in pages:
        words.extend(_page_words(str(pdf), page))
    start_seq, end_seq = plain_words(anchor_start)[:12], plain_words(anchor_end)[-12:]
    result: dict[str, Any] = {"found": False, "start_score": 0.0, "end_score": 0.0}
    if not words or not start_seq:
        return result
    start, start_last, start_score = _find(start_seq, words, 0, len(words))
    result["start_score"] = round(start_score, 3)
    if start < 0 or start_score < min_score:
        return result
    end_index = start_last
    end_score = 1.0
    if end_seq:
        _, found_last, end_score = _find(end_seq, words, start, start + MAX_SPAN_WORDS)
        if found_last >= 0 and end_score >= min_score:
            end_index = max(start_last, found_last)
        else:
            end_score = round(end_score, 3)
            result["end_score"] = end_score
            return result
    chosen = words[start:end_index + 1]
    first_page = chosen[0][1]
    boxes = [word[2] for word in chosen if word[1] == first_page]
    return {
        "found": True, "start_score": round(start_score, 3), "end_score": round(end_score, 3),
        "page_start": first_page, "page_end": chosen[-1][1],
        "word_start": _global_index(pdf, first_page, words, start),
        "word_end": _global_index(pdf, chosen[-1][1], words, end_index),
        "word_count": len(chosen),
        "bbox": [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)],
        "text": " ".join(word[3] for word in chosen),
    }


def _global_index(pdf: Path, page: int, words: list[tuple], index: int) -> int:
    """Document-wide word index: words on all earlier pages plus the index within this page."""
    first_on_page = next(position for position, word in enumerate(words) if word[1] == page)
    before = sum(len(_page_words(str(pdf), earlier)) for earlier in range(1, page))
    return before + (index - first_on_page)


def span_from_excerpt(pdf: Path, pages: list[int], excerpt: str, *, anchor_words: int = 7) -> dict[str, Any]:
    """Derive anchors deterministically from an excerpt's plain words, then locate the span."""
    words = plain_words(excerpt)
    if len(words) < 3:
        return {"found": False, "start_score": 0.0, "end_score": 0.0}
    return locate_span(pdf, pages, " ".join(words[:anchor_words]), " ".join(words[-anchor_words:]))


def span_bounds(span: dict[str, Any]) -> tuple[int, int] | None:
    if not span or not span.get("found"):
        return None
    return int(span["word_start"]), int(span["word_end"])


def span_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = span_bounds(left), span_bounds(right)
    if not a or not b:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union else 0.0


def span_containment(inner: dict[str, Any], outer: dict[str, Any]) -> float:
    """Share of `inner`'s words that lie inside `outer`."""
    a, b = span_bounds(inner), span_bounds(outer)
    if not a or not b:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    return inter / (a[1] - a[0] + 1)
