#!/usr/bin/env python3
"""Generate the paper's descriptive tables from the master dataset.

Emits the data rows of the paper's descriptive-statistics table (mean words /
segments / words per segment) and of its move/step distribution table
(percentage of each source's
move segments carrying each label, pooled over its 76 introductions), plus
programmatic checks for every number quoted in the Results prose around those
tables, so a prose rewrite never relies on hand-reading the table.

Follows the collapse convention (consecutive same-sub-step segments merged;
see move_data.load_intros). --raw disables it, reproducing the pre-2026-07-16
published tables; that mode exists to validate the aggregation method against
the published numbers and for granularity diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from move_data import SUBSTEPS, MOVES, SOURCE_KEYS, load_intros

OUT = Path(__file__).resolve().parent / "outputs"

MODELS = ["gpt-5.5", "claude-opus-4-8", "google/gemma-4-31B-it", "Qwen/Qwen3.5-27B"]
TEX_ROW = {
    "original": r"\textit{Original (human)}",
    "gpt-5.5|basic": "ChatGPT, basic",
    "gpt-5.5|cars_guided": "ChatGPT, CARS",
    "claude-opus-4-8|basic": "Claude, basic",
    "claude-opus-4-8|cars_guided": "Claude, CARS",
    "google/gemma-4-31B-it|basic": "Gemma, basic",
    "google/gemma-4-31B-it|cars_guided": "Gemma, CARS",
    "Qwen/Qwen3.5-27B|basic": "Qwen, basic",
    "Qwen/Qwen3.5-27B|cars_guided": "Qwen, CARS",
}
SUB_ROW = {  # substeps.tex row labels
    "CC": r"\quad Claiming centrality (CC)",
    "MTG": r"\quad Topic generalization (MTG)",
    "RR": r"\quad Reviewing prev.\ research (RR)",
    "CT": r"\quad Continuing a tradition (CT)",
    "Gap": r"\quad Indicating a gap (Gap)",
    "QR": r"\quad Question raising (QR)",
    "APR": r"\quad Announcing present res.\ (APR)",
    "APF": r"\quad Announcing findings (APF)",
    "IS": r"\quad Indicating structure (IS)",
    "OP": r"\quad Outlining purpose (OP)",
}
MOVE_ROW = {
    "move_1": r"\textbf{Move 1: Establishing a Territory}",
    "move_2": r"\textbf{Move 2: Establishing a Niche}",
    "move_3": r"\textbf{Move 3: Occupying the Niche}",
}


def stats(records: list[dict]) -> dict:
    """Per source: descstats values + pooled label percentages."""
    out = {}
    for key in SOURCE_KEYS:
        rs = [r for r in records if r["source"] == key]
        words = float(np.mean([r["word_count"] for r in rs]))
        segs = float(np.mean([r["n_segments"] for r in rs]))
        tot = sum(r["n_segments"] for r in rs)
        pct = {s: 100 * sum(r["counts"][lbl] for r in rs) / tot
               for lbl, s in SUBSTEPS}
        pct_mv = {mv: 100 * sum(r["move_counts"][mv] for r in rs) / tot
                  for mv in MOVES}
        out[key] = {"words": words, "segs": segs, "wps": words / segs,
                    "pct": pct, "pct_mv": pct_mv}
    return out


def print_tables(st: dict) -> None:
    print("---- descstats.tex rows " + "-" * 40)
    for key in SOURCE_KEYS:
        s = st[key]
        cells = f"{s['words']:.0f} & {s['segs']:.1f} & {s['wps']:.1f}"
        if key == "original":
            cells = " & ".join(rf"\textit{{{c}}}" for c in cells.split(" & "))
        print(f"{TEX_ROW[key]:<26}& {cells} \\\\")

    print("\n---- substeps.tex rows " + "-" * 40)
    for mv in MOVES:
        row = [st[k]["pct_mv"][mv] for k in SOURCE_KEYS]
        cells = " & ".join(
            (rf"\textit{{{v:.1f}}}" if i == 0 else f"{v:4.1f}")
            for i, v in enumerate(row))
        print(f"{MOVE_ROW[mv]} & {cells} \\\\")
        for lbl, s in SUBSTEPS:
            if not lbl.startswith(mv):
                continue
            row = [st[k]["pct"][s] for k in SOURCE_KEYS]
            cells = " & ".join(
                (rf"\textit{{{v:.1f}}}" if i == 0 else f"{v:4.1f}")
                for i, v in enumerate(row))
            print(f"{SUB_ROW[s]} & {cells} \\\\")


def print_claim_checks(st: dict) -> None:
    gen = SOURCE_KEYS[1:]
    hum = st["original"]
    print("\n---- Results-prose claim checks " + "-" * 32)
    print(f"granularity: human segs {hum['segs']:.1f}, generated "
          f"{min(st[k]['segs'] for k in gen):.1f}-"
          f"{max(st[k]['segs'] for k in gen):.1f}; "
          f"all generated > human: "
          f"{all(st[k]['segs'] > hum['segs'] for k in gen)}")
    print(f"words/segment: human {hum['wps']:.1f}, generated "
          f"{min(st[k]['wps'] for k in gen):.1f}-"
          f"{max(st[k]['wps'] for k in gen):.1f}; sets NOT shorter than human: "
          f"{[k for k in gen if st[k]['wps'] >= hum['wps']]}")
    for m in MODELS:
        b, c = st[m + "|basic"], st[m + "|cars_guided"]
        print(f"  {m}: segs {b['segs']:.1f} -> {c['segs']:.1f}")

    print(f"\nmove shares human M1/M2/M3: "
          + "/".join(f"{hum['pct_mv'][m]:.1f}" for m in MOVES))
    above2 = [k for k in gen if st[k]["pct_mv"]["move_2"] > hum["pct_mv"]["move_2"]]
    below1 = [k for k in gen if st[k]["pct_mv"]["move_1"] < hum["pct_mv"]["move_1"]]
    print(f"M2 above human: {len(above2)}/8 (not: "
          f"{[k for k in gen if k not in above2]}); "
          f"max M2 {max(st[k]['pct_mv']['move_2'] for k in gen):.1f}")
    print(f"M1 below human: {len(below1)}/8 (not: "
          f"{[k for k in gen if k not in below1]}); "
          f"min M1 {min(st[k]['pct_mv']['move_1'] for k in gen):.1f}, "
          f"max M1 {max(st[k]['pct_mv']['move_1'] for k in gen):.1f}")
    print(f"M3 generated range "
          f"{min(st[k]['pct_mv']['move_3'] for k in gen):.1f}-"
          f"{max(st[k]['pct_mv']['move_3'] for k in gen):.1f} "
          f"vs human {hum['pct_mv']['move_3']:.1f}")

    print("\nhuman sub-steps: "
          + ", ".join(f"{s} {hum['pct'][s]:.1f}" for _, s in SUBSTEPS))
    top4 = sum(sorted((hum["pct"][s] for _, s in SUBSTEPS), reverse=True)[:4])
    print(f"top-4 human sub-steps cover {top4:.1f}%")
    rr_largest = all(max(st[k]["pct"], key=st[k]["pct"].get) == "RR" for k in gen)
    print(f"RR largest sub-step in every generated set: {rr_largest}")
    for s, name in (("RR", "RR gen range"), ("OP", "OP gen range")):
        print(f"{name}: {min(st[k]['pct'][s] for k in gen):.1f}-"
              f"{max(st[k]['pct'][s] for k in gen):.1f} vs human {hum['pct'][s]:.1f}")
    for s in ("Gap", "QR", "APF"):
        print(f"{s} max generated: {max(st[k]['pct'][s] for k in gen):.1f} "
              f"vs human {hum['pct'][s]:.1f}")
    print(f"GPT_b MTG {st['gpt-5.5|basic']['pct']['MTG']:.1f} "
          f"vs human {hum['pct']['MTG']:.1f}")

    print("\nbasic -> CARS direction per model:")
    m1_subs = [s for lbl, s in SUBSTEPS if lbl.startswith("move_1")]
    for m in MODELS:
        b, c = st[m + "|basic"], st[m + "|cars_guided"]
        shrink = all(c["pct"][s] < b["pct"][s] for s in m1_subs)
        grow = all(c["pct"][s] > b["pct"][s] for s in ("Gap", "QR", "APF"))
        print(f"  {m}: all M1 sub-steps shrink: {shrink}; "
              f"Gap/QR/APF all grow: {grow}; "
              f"M1 {b['pct_mv']['move_1']:.1f}->{c['pct_mv']['move_1']:.1f}, "
              f"M2 {b['pct_mv']['move_2']:.1f}->{c['pct_mv']['move_2']:.1f}, "
              f"M3 {b['pct_mv']['move_3']:.1f}->{c['pct_mv']['move_3']:.1f}, "
              f"IS {b['pct']['IS']:.1f}->{c['pct']['IS']:.1f}")
    c = st["gpt-5.5|cars_guided"]
    print(f"GPT_c triple: "
          + "/".join(f"{c['pct_mv'][m]:.1f}" for m in MOVES))


def print_graphsim() -> None:
    """Rows of the paper's graph-similarity table, from the graph-analysis outputs
    (run graph_analysis.py first)."""
    sim = json.loads((OUT / "graph_similarity.json").read_text())
    sim = sim["similarity"]["per_source"]
    div = json.loads((OUT / "graph_diversity.json").read_text())["sources"]
    short = {
        "gpt-5.5|basic": "ChatGPT$_b$", "gpt-5.5|cars_guided": "ChatGPT$_c$",
        "claude-opus-4-8|basic": "Claude$_b$",
        "claude-opus-4-8|cars_guided": "Claude$_c$",
        "google/gemma-4-31B-it|basic": "Gemma$_b$",
        "google/gemma-4-31B-it|cars_guided": "Gemma$_c$",
        "Qwen/Qwen3.5-27B|basic": "Qwen$_b$",
        "Qwen/Qwen3.5-27B|cars_guided": "Qwen$_c$",
    }
    print("\n---- graphsim.tex rows " + "-" * 40)
    o = div["original"]
    print(rf"\textit{{Original}} & -- & -- & -- & -- & "
          rf"\textit{{{o['mean_pairwise']['edit_sim_sub']:.2f}}} & "
          rf"\textit{{{o['n_move_structure_types']}}} \\")
    for key in SOURCE_KEYS[1:]:
        p = sim[key]["paired_mean"]
        w = div[key]
        print(f"{short[key]:<10} & {p['edit_sim_move']:.2f} & "
              f"{p['edit_sim_sub']:.2f} & {p['bag_sim_move']:.2f} & "
              f"{p['bag_sim_sub']:.2f} & "
              f"{w['mean_pairwise']['edit_sim_sub']:.2f} & "
              f"{w['n_move_structure_types']} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="disable the collapse convention (validation only)")
    args = ap.parse_args()
    st = stats(load_intros(collapse_substeps=not args.raw))
    print(f"mode: {'RAW (published pre-convention)' if args.raw else 'collapsed'}")
    print_tables(st)
    print_claim_checks(st)
    if not args.raw:
        print_graphsim()


if __name__ == "__main__":
    main()
