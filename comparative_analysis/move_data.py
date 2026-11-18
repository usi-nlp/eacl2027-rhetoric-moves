"""Shared loading for the Step-9 comparative analyses.

Reads data/master_dataset.json and exposes one record per introduction with
its evaluator (chatgpt) CARS annotation reduced to the pieces the analyses
need: per-sub-step counts / word shares / presence, and the ordered sub-step
and collapsed move sequences. The evaluator annotation is used on BOTH sides
(original and generated) so annotator granularity is not a confound — see the
paper's method section.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master_dataset.json"

# Canonical sub-step order + short display codes.
SUBSTEPS = [
    ("move_1__claiming_centrality",                  "CC"),
    ("move_1__making_topic_generalization",          "MTG"),
    ("move_1__reviewing_items_of_previous_research", "RR"),
    ("move_2__continuing_a_tradition",               "CT"),
    ("move_2__indicating_a_gap",                     "Gap"),
    ("move_2__question_raising",                     "QR"),
    ("move_3__announcing_present_research",          "APR"),
    ("move_3__announcing_principal_findings",        "APF"),
    ("move_3__indicating_the_structure_of_research", "IS"),
    ("move_3__outlining_purpose",                    "OP"),
]
SHORT = dict(SUBSTEPS)
MOVES = ["move_1", "move_2", "move_3"]

# Display order of the generated sets (model, mode) -> column label.
SOURCES = [
    ("original", None, "Original"),
    ("gpt-5.5", "basic", "GPT b"),
    ("gpt-5.5", "cars_guided", "GPT c"),
    ("claude-opus-4-8", "basic", "Claude b"),
    ("claude-opus-4-8", "cars_guided", "Claude c"),
    ("google/gemma-4-31B-it", "basic", "Gemma b"),
    ("google/gemma-4-31B-it", "cars_guided", "Gemma c"),
    ("Qwen/Qwen3.5-27B", "basic", "Qwen b"),
    ("Qwen/Qwen3.5-27B", "cars_guided", "Qwen c"),
]


def source_key(intro: dict) -> str:
    if intro["intro_type"] == "original":
        return "original"
    return f"{intro['model']}|{intro['mode']}"


def source_label(key: str) -> str:
    for model, mode, label in SOURCES:
        if key == "original" and model == "original":
            return label
        if key == f"{model}|{mode}":
            return label
    return key


SOURCE_KEYS = ["original"] + [f"{m}|{mode}" for m, mode, _ in SOURCES[1:]]


def load_intros(annotator: str = "chatgpt",
                collapse_substeps: bool = True) -> list[dict]:
    """One flat record per introduction (76 papers x 9 sources).

    Analysis convention (2026-07-16): consecutive segments that share a
    sub-step label are merged into one unit before ANY calculation, so counts
    and sequences measure rhetorical structure rather than the evaluator's
    segmentation granularity (it splits same-function stretches on 11-14% of
    adjacent segment pairs; the human gold does so on 1.4%). Word shares are
    unaffected by the merge, and incidence and all move-level statistics are
    identical either way (verified in collapse_impact.py). Pass
    collapse_substeps=False only for granularity diagnostics.
    """
    data = json.loads(MASTER.read_text())
    records = []
    for paper in data["papers"]:
        for intro in paper["intros"]:
            ann = intro["annotations"].get(annotator)
            if ann is None:
                continue
            segs = [s for s in ann["segments"] if s.get("in_scheme")]
            n_words = lambda s: len(s["text"].split())
            total_words = sum(n_words(s) for s in segs) or 1
            words = {lbl: 0 for lbl, _ in SUBSTEPS}
            for s in segs:
                words[s["label"]] += n_words(s)
            units = ([s for i, s in enumerate(segs)
                      if i == 0 or s["label"] != segs[i - 1]["label"]]
                     if collapse_substeps else segs)
            counts = {lbl: 0 for lbl, _ in SUBSTEPS}
            for s in units:
                counts[s["label"]] += 1
            sub_seq = [SHORT[s["label"]] for s in units]
            move_seq_full = [s["move"] for s in units]
            move_seq = [m for i, m in enumerate(move_seq_full)
                        if i == 0 or m != move_seq_full[i - 1]]
            records.append({
                "paper_id": paper["paper_id"],
                "quartile": paper["quartile"],
                "nativeness": paper["nativeness"],
                "swales_mention": paper["swales_topic"]["mentions_swales_or_cars"],
                "source": source_key(intro),
                "word_count": intro["word_count"],
                "n_segments": len(units),
                "counts": counts,
                "word_share": {lbl: words[lbl] / total_words for lbl, _ in SUBSTEPS},
                "move_counts": {mv: sum(c for lbl, c in counts.items()
                                        if lbl.startswith(mv)) for mv in MOVES},
                "move_share": {mv: sum(sh for lbl, sh in
                                       ((l, words[l] / total_words) for l, _ in SUBSTEPS)
                                       if lbl.startswith(mv)) for mv in MOVES},
                "sub_seq": sub_seq,
                "move_seq": [m.replace("move_", "") for m in move_seq],
            })
    return records
