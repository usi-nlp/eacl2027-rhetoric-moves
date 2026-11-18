#!/usr/bin/env python3
"""Export the data behind the interactive figure of the project page.

Writes docs/static/js/page_data.js with, for each of the nine sources (the
originals and the eight generated sets), the move structure of every
introduction: its collapsed move sequence and how many words each move unit
takes. The page draws one strip per introduction from it.

Everything follows the conventions of the other analyses in this folder: the
automatic annotation is used on both sides, out-of-scheme segments are dropped
and consecutive segments of the same move are merged (move_data.load_intros).
The per-source summary numbers are checked against the paper's tables before
anything is written.

Usage (from the repo root):
    python comparative_analysis/export_page_data.py
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from move_data import MASTER, SOURCE_KEYS, load_intros, source_key

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "static" / "js" / "page_data.js"

LABELS = {
    "original": ("Human originals", None, None),
    "gpt-5.5|basic": ("ChatGPT, basic", "chatgpt", "basic"),
    "gpt-5.5|cars_guided": ("ChatGPT, CARS-guided", "chatgpt", "cars_guided"),
    "claude-opus-4-8|basic": ("Claude, basic", "claude", "basic"),
    "claude-opus-4-8|cars_guided": ("Claude, CARS-guided", "claude", "cars_guided"),
    "google/gemma-4-31B-it|basic": ("Gemma, basic", "gemma", "basic"),
    "google/gemma-4-31B-it|cars_guided": ("Gemma, CARS-guided", "gemma", "cars_guided"),
    "Qwen/Qwen3.5-27B|basic": ("Qwen, basic", "qwen", "basic"),
    "Qwen/Qwen3.5-27B|cars_guided": ("Qwen, CARS-guided", "qwen", "cars_guided"),
}

# Values printed in the paper (ordering and graph tables): distinct move
# structures, % of introductions that return to a move they had left, mean
# number of distinct moves and of distinct steps per introduction.
PAPER = {
    "original": (36, 66, 2.67, 5.3),
    "gpt-5.5|basic": (18, 70, 2.93, 6.3),
    "gpt-5.5|cars_guided": (9, 63, 3.00, 6.9),
    "claude-opus-4-8|basic": (24, 74, 2.95, 6.2),
    "claude-opus-4-8|cars_guided": (11, 51, 3.00, 6.6),
    "google/gemma-4-31B-it|basic": (14, 67, 2.96, 6.1),
    "google/gemma-4-31B-it|cars_guided": (8, 50, 3.00, 7.4),
    "Qwen/Qwen3.5-27B|basic": (35, 95, 2.97, 6.8),
    "Qwen/Qwen3.5-27B|cars_guided": (25, 87, 2.99, 7.0),
}


def half_up(value: float, digits: int) -> float:
    """Round like the paper's tables do (6.25 prints as 6.3, not 6.2)."""
    return float(Decimal(repr(value)).quantize(Decimal(1).scaleb(-digits),
                                                rounding=ROUND_HALF_UP))


def move_units(annotation: dict) -> list[list[int]]:
    """Collapsed move units of one annotation as [move number, words]."""
    units: list[list[int]] = []
    for seg in annotation["segments"]:
        if not seg.get("in_scheme"):
            continue
        move = int(seg["move"].replace("move_", ""))
        words = len(seg["text"].split())
        if units and units[-1][0] == move:
            units[-1][1] += words
        else:
            units.append([move, words])
    return units


def main() -> None:
    records = {(r["source"], r["paper_id"]): r for r in load_intros()}
    data = json.loads(MASTER.read_text())

    intros: dict[str, list[dict]] = {key: [] for key in SOURCE_KEYS}
    for paper in data["papers"]:
        for intro in paper["intros"]:
            key = source_key(intro)
            units = move_units(intro["annotations"]["chatgpt"])
            record = records[(key, paper["paper_id"])]
            # Same structure as the one every statistic of the paper is built on.
            assert [str(m) for m, _ in units] == record["move_seq"], intro["intro_id"]
            intros[key].append({"id": paper["paper_key"], "q": paper["quartile"],
                                "seq": units})

    sources = []
    for key in SOURCE_KEYS:
        rows = intros[key]
        recs = [records[(key, int(r["id"]))] for r in rows]
        seqs = [tuple(m for m, _ in r["seq"]) for r in rows]
        n = len(rows)
        summary = {
            "structures": len(set(seqs)),
            "canonical": sum(s == (1, 2, 3) for s in seqs),
            "cyclic_pct": int(half_up(100 * sum(len(s) > len(set(s)) for s in seqs) / n, 0)),
            "uniq_moves": half_up(sum(len(set(s)) for s in seqs) / n, 2),
            "uniq_steps": half_up(sum(sum(c > 0 for c in r["counts"].values())
                                      for r in recs) / n, 1),
            "all_three_moves": sum(len(set(s)) == 3 for s in seqs),
        }
        expected = PAPER[key]
        got = (summary["structures"], summary["cyclic_pct"],
               summary["uniq_moves"], summary["uniq_steps"])
        assert got == expected, f"{key}: computed {got}, paper reports {expected}"
        label, model, mode = LABELS[key]
        sources.append({"key": key, "label": label, "model": model, "mode": mode,
                        "n": n, **summary, "intros": rows})

    titles = {p["paper_key"]: p["title"] for p in data["papers"]}
    payload = {"sources": sources, "titles": titles}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "// Generated by comparative_analysis/export_page_data.py. Do not edit by hand.\n"
        "window.PAGE_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB): "
          f"{len(sources)} sources x {len(titles)} papers; summary numbers match the paper")


if __name__ == "__main__":
    main()
