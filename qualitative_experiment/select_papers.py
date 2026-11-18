#!/usr/bin/env python3
"""Select the papers for the two qualitative close-reading experiments.

Experiment 1 — human-vs-AI triplets (5 papers x 3 intros):
    For each selected paper the reader sees the original introduction plus the
    two claude-opus-4-8 generations (basic + cars_guided) — Claude is the
    quantitatively most human-similar generator (graph_similarity.json) and the
    discrimination-experiment generator, so the close reading supports both.
    Purposive slots, filled in order from the eligible pool (not Swales-related,
    original >= MIN_WORDS words, prefer papers inside the discrimination
    sample so the close reading can later be triangulated with the blind
    judgments):
      1. best_imitation   — highest mean paired sub-step edit similarity
                            (claude basic+cars vs original; corpus convention:
                            collapsed sequences, chatgpt annotations)
      2. worst_imitation  — lowest mean paired similarity
      3. move_omission    — original omits Move 2 entirely while both Claude
                            versions use all three moves (normalisation case);
                            prefer a quartile not yet selected, then longest
      4. cyclical         — original with the most returns to an already-used
                            move (human cyclicity case)
      5. typical_median   — similarity closest to the corpus median, drawn from
                            a quartile not yet selected if possible
    Soft aim: cover all four quartiles across the five papers.

Experiment 2 — Swales/CARS-aware originals, one per quartile, no controls:
    The master dataset's `swales_topic` screen (whole-paper substring match)
    produced several false positives on manual audit — see EXP2_NOTES below
    for the full account. After auditing every flagged paper's actual context,
    the researcher hand-picked four papers with genuine Swales/CARS
    engagement, one per quartile: 004 (Q1), 036 (Q2), 078 (Q3), 047 (Q4).
    No control/distractor papers — the researcher decided this study doesn't
    need them.

Output: qualitative_experiment/selection.json (picks + per-paper stats table).

Usage:
    python qualitative_experiment/select_papers.py
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "comparative_analysis"))

from move_data import load_intros  # noqa: E402
from graph_analysis import prepare, pair_similarity  # noqa: E402

GENERATOR = "claude-opus-4-8"
MIN_WORDS = 250  # originals shorter than this are too thin for a close read
MATCH_WEIGHTS = {"title": 4, "abstract": 3, "original_intro": 3, "body": 1}


def build_paper_stats() -> list[dict]:
    records = load_intros()  # chatgpt annotations, collapse convention
    prepare(records)
    master = json.loads((REPO / "data" / "master_dataset.json").read_text())
    pmeta = {p["paper_id"]: p for p in master["papers"]}
    alloc = json.loads(
        (REPO / "discrimination_experiment" / "allocation.json").read_text()
    )
    in_sample = set(alloc["meta"]["sampled_paper_ids"])

    by_paper: dict[int, dict[str, dict]] = {}
    for r in records:
        by_paper.setdefault(r["paper_id"], {})[r["source"]] = r

    rows = []
    for pid, srcs in sorted(by_paper.items()):
        orig = srcs["original"]
        cb = srcs[f"{GENERATOR}|basic"]
        cc = srcs[f"{GENERATOR}|cars_guided"]
        sim_b = pair_similarity(cb, orig)["edit_sim_sub"]
        sim_c = pair_similarity(cc, orig)["edit_sim_sub"]
        mseq = orig["move_seq"]
        meta = pmeta[pid]
        rows.append(
            {
                "paper_id": pid,
                "paper_key": meta["paper_key"],
                "title": meta["title"],
                "quartile": meta["quartile"],
                "nativeness": meta["nativeness"],
                "orig_words": orig["word_count"],
                "sim_basic": round(sim_b, 3),
                "sim_cars": round(sim_c, 3),
                "sim_mean": round((sim_b + sim_c) / 2, 3),
                "orig_move_seq": "-".join(mseq),
                "claude_basic_move_seq": "-".join(cb["move_seq"]),
                "claude_cars_move_seq": "-".join(cc["move_seq"]),
                "omitted_moves": [m for m, c in orig["move_counts"].items() if c == 0],
                "move_returns": sum(
                    1 for i, m in enumerate(mseq) if m in mseq[:i] and mseq[i - 1] != m
                ),
                "claude_all_moves": all(
                    r["move_counts"][m] > 0
                    for r in (cb, cc)
                    for m in ("move_1", "move_2", "move_3")
                ),
                "swales_related": meta["swales_topic"]["is_swales_related"],
                "swales_mentions": meta["swales_topic"]["mentions_swales_or_cars"],
                "swales_matched_in": meta["swales_topic"]["matched_in"],
                "swales_matched_terms": meta["swales_topic"]["matched_terms"],
                "in_discrimination_sample": pid in in_sample,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Experiment 1 — triplet slots
# ---------------------------------------------------------------------------

def select_exp1(rows: list[dict]) -> list[dict]:
    eligible = [
        r
        for r in rows
        if not r["swales_related"] and r["orig_words"] >= MIN_WORDS
    ]
    pool = [r for r in eligible if r["in_discrimination_sample"]] or eligible
    median_sim = statistics.median(r["sim_mean"] for r in eligible)
    picked: list[dict] = []

    def take(row: dict, slot: str, reason: str) -> None:
        picked.append({**row, "slot": slot, "reason": reason})
        pool.remove(row)

    best = max(pool, key=lambda r: r["sim_mean"])
    take(best, "best_imitation",
         f"highest mean paired sub-step edit similarity ({best['sim_mean']}) — "
         "what does Claude's closest structural imitation still get wrong?")

    worst = min(pool, key=lambda r: r["sim_mean"])
    take(worst, "worst_imitation",
         f"lowest mean paired similarity ({worst['sim_mean']}) — where the "
         "imitation breaks down")

    omit = [r for r in pool if "move_2" in r["omitted_moves"] and r["claude_all_moves"]]
    new_q = [r for r in omit if r["quartile"] not in {p["quartile"] for p in picked}]
    omit_pick = max(new_q or omit, key=lambda r: r["orig_words"])
    take(omit_pick, "move_omission",
         "original omits Move 2 entirely (move sequence "
         f"{omit_pick['orig_move_seq']}) while both Claude versions realise all "
         "three moves — does the model normalise a non-canonical human structure?")

    cyc = max(pool, key=lambda r: (r["move_returns"], r["orig_words"]))
    take(cyc, "cyclical",
         f"most cyclical original ({cyc['move_returns']} returns to an "
         f"already-used move, sequence {cyc['orig_move_seq']}) — human "
         "back-and-forth vs the models' linear funnel")

    new_q = [r for r in pool if r["quartile"] not in {p["quartile"] for p in picked}]
    med_pool = new_q or pool
    med = min(
        med_pool,
        key=lambda r: (abs(r["sim_mean"] - median_sim), r["nativeness"] != "native", r["paper_id"]),
    )
    take(med, "typical_median",
         f"similarity closest to the corpus median ({median_sim}) in a not-yet-"
         "covered quartile — the unremarkable middle of the distribution")
    return picked


# ---------------------------------------------------------------------------
# Experiment 2 — Swales/CARS-aware originals (no controls)
# ---------------------------------------------------------------------------
#
# The master dataset's `swales_topic` screen is a coarse substring match over
# the whole paper and produced several false positives on manual audit (e.g.
# paper 022's "hit" was the substring "Cars" inside the citation "Carson,
# 2010"; paper 034's intro-level hit was a citation to an unrelated paper
# co-authored by a person named Swales, not the CARS model). After a
# word-boundary, case-correct re-scan of the full corpus and a manual read of
# every candidate's context, only four papers show genuine engagement with
# Swales' CARS model / rhetorical move analysis, one per quartile — the
# researcher selected exactly these after the audit; picked here by paper_key,
# not re-derived, since the selection is now a manual editorial judgment call,
# not a scorable heuristic. No control/distractor papers: the researcher
# decided this study doesn't need them.

EXP2_PAPER_KEYS = ["004", "036", "078", "047"]  # Q1, Q2, Q3, Q4
EXP2_NOTES = {
    "004": (
        "Corpus itself annotated with an adapted Swales CARS scheme (Methods "
        "section); the introduction text does not mention Swales/CARS."
    ),
    "036": (
        "Extensive, explicit engagement with Swales' CARS model inside the "
        "introduction itself (Move 1-3 and their steps discussed by name, "
        "~19 Swales/CARS mentions) — the paper's own topic is CARS-related "
        "promotion strategies."
    ),
    "078": (
        "One self-referential line in the introduction announcing its own "
        "move-structure analysis; Swales' genre theory is cited later in the "
        "body, not in the introduction."
    ),
    "047": (
        "Cites other researchers' CARS-based studies of a different genre "
        "(thesis problem statements) in a literature-review passage; the "
        "introduction text does not mention Swales/CARS directly."
    ),
}


def select_exp2(rows: list[dict]) -> list[dict]:
    by_key = {r["paper_key"]: r for r in rows}
    return [
        {**by_key[pk], "researcher_note": EXP2_NOTES[pk]} for pk in EXP2_PAPER_KEYS
    ]


def main() -> None:
    rows = build_paper_stats()
    exp1 = select_exp1(rows)
    exp2 = select_exp2(rows)

    out = {
        "meta": {
            "built": date.today().isoformat(),
            "generator": GENERATOR,
            "similarity_metric": (
                "edit_sim_sub — 1 - normalised Levenshtein on the collapsed "
                "sub-step sequence (chatgpt annotations both sides, corpus "
                "collapse convention; same metric that ranked the generators)"
            ),
            "min_original_words": MIN_WORDS,
            "corpus_median_sim": statistics.median(
                r["sim_mean"]
                for r in rows
                if not r["swales_related"] and r["orig_words"] >= MIN_WORDS
            ),
            "note": (
                "All exp-1 papers and the exp-2 papers are inside the "
                "discrimination-experiment sample: if the same person does both "
                "studies, the qualitative tasks must run AFTER the "
                "discrimination judgments are collected. Exp-2's four papers "
                "(one per quartile) were picked manually by the researcher "
                "after auditing every master_dataset.json swales_topic flag "
                "for false positives (see EXP2_NOTES in this script); no "
                "control/distractor papers are used."
            ),
        },
        "exp1_close_reading": exp1,
        "exp2_swales_aware": exp2,
        "all_paper_stats": rows,
    }
    path = REPO / "qualitative_experiment" / "selection.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print(f"wrote {path}\n\nExperiment 1 — triplets ({GENERATOR}):")
    for r in exp1:
        print(
            f"  {r['paper_key']} {r['quartile']} {r['nativeness']:<10} "
            f"sim {r['sim_mean']:.3f}  orig {r['orig_move_seq']:<40} [{r['slot']}]"
        )
        print(f"      {r['title'][:88]}")
    print("\nExperiment 2 — Swales/CARS-aware originals (no controls):")
    for p in exp2:
        print(f"  {p['quartile']} {p['paper_key']} ({p['orig_words']}w) — {p['researcher_note']}")
        print(f"      {p['title'][:88]}")


if __name__ == "__main__":
    main()
