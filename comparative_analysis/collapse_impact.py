#!/usr/bin/env python3
"""Sensitivity check: how do the Step-9 results change if consecutive
segments with the same sub-step label are collapsed into one?

Motivation: the evaluator splits same-function stretches into multiple
segments (14.3% of adjacent segment pairs on originals, 11.4% on generated;
the human gold does it on only 1.4%), so every count-based measure carries
some of that segmentation granularity. This script recomputes the published
measures in both modes ("raw" = segments as annotated, "collapsed" = runs of
equal sub-step labels merged) and reports the differences.

Invariant by construction, VERIFIED here rather than assumed:
  incidence, word/move share, Kanoksilapatham inventories, McNemar and
  Wilcoxon tests (they use presence and share), all move-level sequence stats
  (structure types, TTR, cycles, canonical %, paired move edit distance), the
  sub-move ordering profile (submove_stats already collapses internally),
  bag-of-classes similarities, and all move-level graph metrics.

Changing:
  mean segments per intro, sub-step (and move) token frequency, the raw
  sub-step n-gram tables in sequence_stats, and the sub-step edit-similarity
  numbers in the graph analysis (A self-similarity + Vendi, B paired /
  mismatched / delta + the generator ranking).

Output: outputs/collapse_impact.json + console report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from move_data import SUBSTEPS, MOVES, SOURCE_KEYS, source_label, load_intros
from sequence_analysis import collapse, seq_stats, submove_stats, top_ngrams
from distribution_analysis import by_source, describe, inventory, paired_tests
import graph_analysis as GA

OUT = Path(__file__).resolve().parent / "outputs"
LONG = {code: lbl for lbl, code in SUBSTEPS}


# --------------------------------------------------------------------------- #
def collapsed_view(records: list[dict]) -> list[dict]:
    """Copy of the records with runs of equal sub-step labels merged into one
    segment: sub_seq collapsed, counts = runs per label. Shares and the (already
    collapsed) move sequence are untouched by construction."""
    out = []
    for r in records:
        c = collapse(r["sub_seq"])
        counts = {lbl: 0 for lbl, _ in SUBSTEPS}
        for code in c:
            counts[LONG[code]] += 1
        r2 = dict(r)
        r2["sub_seq"] = c
        r2["n_segments"] = len(c)
        r2["counts"] = counts
        r2["move_counts"] = {mv: sum(v for l, v in counts.items()
                                     if l.startswith(mv)) for mv in MOVES}
        out.append(r2)
    return out


def graph_variant(records: list[dict], collapse_sub: bool) -> tuple[dict, dict, dict]:
    """Graph analyses with the sub-step path sequence raw or collapsed."""
    recs = [dict(r) for r in records]
    for r in recs:
        r["seqs"] = {"move": r["move_seq"],
                     "sub": collapse(r["sub_seq"]) if collapse_sub else r["sub_seq"]}
        r["bags"] = {"move": np.array([r["move_share"][m] for m in MOVES]),
                     "sub": np.array([r["word_share"][lbl] for lbl, _ in SUBSTEPS])}
    a = {k: GA.within_source_stats([x for x in recs if x["source"] == k])
         for k in SOURCE_KEYS}
    b = GA.similarity_to_original(recs)
    return a, b, GA.select_generator_for_discrimination(b)


def max_diff(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(a - b)


# --------------------------------------------------------------------------- #
def main() -> None:
    # The one legitimate raw consumer: load_intros collapses by default
    # (analysis convention 2026-07-16), so raw must be requested explicitly.
    raw = load_intros(collapse_substeps=False)
    col = collapsed_view(raw)
    g_raw, g_col = by_source(raw), by_source(col)
    report: dict = {"invariance_checks": {}, "changes": {}}

    # ---- 0. verify the invariances ------------------------------------------
    checks: dict[str, float] = {}
    inv_mismatch: list[str] = []
    for key in SOURCE_KEYS:
        dr, dc = describe(g_raw[key]), describe(g_col[key])
        for fam, cats in (("substep", [s for _, s in SUBSTEPS]), ("move", MOVES)):
            for cat in cats:
                checks["incidence"] = max(checks.get("incidence", 0.0), max_diff(
                    dr[fam][cat]["incidence"], dc[fam][cat]["incidence"]))
                checks["word_share"] = max(checks.get("word_share", 0.0), max_diff(
                    dr[fam][cat]["word_share"], dc[fam][cat]["word_share"]))
        if inventory(dr) != inventory(dc):
            inv_mismatch.append(key)
    for key in SOURCE_KEYS[1:]:
        tr = paired_tests(g_raw["original"], g_raw[key])
        tc = paired_tests(g_col["original"], g_col[key])
        for fam in ("substep", "move"):
            for cat, v in tr[fam].items():
                for p in ("mcnemar_p", "wilcoxon_p"):
                    checks[p] = max(checks.get(p, 0.0),
                                    max_diff(v[p], tc[fam][cat][p]))
                checks["share_delta"] = max(checks.get("share_delta", 0.0), max_diff(
                    v["share_delta_mean"], tc[fam][cat]["share_delta_mean"]))
        sr, sc = seq_stats(g_raw[key]), seq_stats(g_col[key])
        for f in ("mean_move_units", "mean_cycles", "pct_canonical",
                  "n_sequence_types", "type_token_ratio"):
            checks["move_seq_stats"] = max(checks.get("move_seq_stats", 0.0),
                                           max_diff(sr[f], sc[f]))
        if submove_stats(g_raw[key]) != submove_stats(g_col[key]):
            inv_mismatch.append(f"submove_stats:{key}")
    report["invariance_checks"] = {
        "max_abs_diffs": checks, "inventory_or_submove_mismatches": inv_mismatch}

    # ---- 1. mean segments per intro ------------------------------------------
    segs = {}
    for key in SOURCE_KEYS:
        segs[key] = {
            "raw": float(np.mean([len(r["sub_seq"]) for r in g_raw[key]])),
            "collapsed": float(np.mean([len(r["sub_seq"]) for r in g_col[key]])),
        }
    report["changes"]["mean_segments"] = segs

    # ---- 2. token frequency ---------------------------------------------------
    tok = {}
    for key in SOURCE_KEYS:
        dr, dc = describe(g_raw[key]), describe(g_col[key])
        tok[key] = {
            "substep": {s: [dr["substep"][s]["token_freq"],
                            dc["substep"][s]["token_freq"]]
                        for _, s in SUBSTEPS},
            "move": {m: [dr["move"][m]["token_freq"], dc["move"][m]["token_freq"]]
                     for m in MOVES},
        }
    report["changes"]["token_freq_raw_vs_collapsed"] = tok

    # ---- 3. top sub-step n-grams ----------------------------------------------
    grams = {}
    for key in SOURCE_KEYS:
        grams[key] = {
            "bigrams_raw": top_ngrams(g_raw[key], 2, k=5),
            "bigrams_collapsed": top_ngrams(g_col[key], 2, k=5),
            "trigrams_raw": top_ngrams(g_raw[key], 3, k=3),
            "trigrams_collapsed": top_ngrams(g_col[key], 3, k=3),
        }
    report["changes"]["top_ngrams"] = grams

    # ---- 4./5. graph analyses --------------------------------------------------
    a_raw, b_raw, sel_raw = graph_variant(raw, collapse_sub=False)
    a_col, b_col, sel_col = graph_variant(raw, collapse_sub=True)
    inv_graph = 0.0
    for key in SOURCE_KEYS:
        inv_graph = max(inv_graph,
                        max_diff(a_raw[key]["mean_pairwise"]["edit_sim_move"],
                                 a_col[key]["mean_pairwise"]["edit_sim_move"]),
                        max_diff(a_raw[key]["mean_pairwise"]["bag_sim_sub"],
                                 a_col[key]["mean_pairwise"]["bag_sim_sub"]),
                        max_diff(a_raw[key]["vendi"]["move"],
                                 a_col[key]["vendi"]["move"]))
    report["invariance_checks"]["max_abs_diffs"]["graph_move_level_and_bags"] = inv_graph
    report["changes"]["graph_A_substep_edit"] = {
        key: {"self_sim_raw": a_raw[key]["mean_pairwise"]["edit_sim_sub"],
              "self_sim_collapsed": a_col[key]["mean_pairwise"]["edit_sim_sub"],
              "vendi_raw": a_raw[key]["vendi"]["sub"],
              "vendi_collapsed": a_col[key]["vendi"]["sub"]}
        for key in SOURCE_KEYS}
    report["changes"]["graph_B_substep_edit"] = {
        key: {"paired_raw": b_raw["per_source"][key]["paired_mean"]["edit_sim_sub"],
              "paired_collapsed": b_col["per_source"][key]["paired_mean"]["edit_sim_sub"],
              "delta_raw": b_raw["per_source"][key]["delta"]["edit_sim_sub"],
              "delta_collapsed": b_col["per_source"][key]["delta"]["edit_sim_sub"]}
        for key in SOURCE_KEYS[1:]}
    report["changes"]["generator_ranking"] = {
        "raw": sel_raw["ranking"], "collapsed": sel_col["ranking"]}

    OUT.mkdir(exist_ok=True)
    (OUT / "collapse_impact.json").write_text(json.dumps(report, indent=2))

    # ---- console ----------------------------------------------------------------
    print("=== 0. Invariance checks (max |raw - collapsed| over all cells) ===")
    for k, v in report["invariance_checks"]["max_abs_diffs"].items():
        print(f"  {k:<28} {v:.3e}")
    print(f"  inventory/submove mismatches: "
          f"{inv_mismatch if inv_mismatch else 'none'}")

    print("\n=== 1. Mean segments per introduction (raw -> collapsed) ===")
    for key in SOURCE_KEYS:
        s = segs[key]
        print(f"  {source_label(key):<10} {s['raw']:5.1f} -> {s['collapsed']:5.1f}"
              f"   ({s['collapsed'] - s['raw']:+.1f})")

    print("\n=== 2. Sub-step token frequency (raw -> collapsed), biggest movers ===")
    print(f"{'':6}" + "".join(f"{source_label(k):>12}" for k in SOURCE_KEYS))
    for _, s in SUBSTEPS:
        deltas = [tok[k]["substep"][s][1] - tok[k]["substep"][s][0]
                  for k in SOURCE_KEYS]
        if max(abs(d) for d in deltas) < 0.15:
            continue
        cells = [f"{tok[k]['substep'][s][0]:.1f}>{tok[k]['substep'][s][1]:.1f}"
                 for k in SOURCE_KEYS]
        print(f"{s:<6}" + "".join(f"{c:>12}" for c in cells))

    print("\n=== 3. Top sub-step bigram (raw -> collapsed) ===")
    for key in SOURCE_KEYS:
        br = grams[key]["bigrams_raw"][0]
        bc = grams[key]["bigrams_collapsed"][0]
        print(f"  {source_label(key):<10} {br['ngram']:<10} ({br['count']:>3}) "
              f"-> {bc['ngram']:<10} ({bc['count']:>3})")

    print("\n=== 4. Graph A: within-source sub-step edit self-similarity / Vendi ===")
    print(f"  {'source':<10} {'selfSim raw>col':>18} {'Vendi raw>col':>16}")
    for key in SOURCE_KEYS:
        c = report["changes"]["graph_A_substep_edit"][key]
        print(f"  {source_label(key):<10} "
              f"{c['self_sim_raw']:.3f} > {c['self_sim_collapsed']:.3f}      "
              f"{c['vendi_raw']:5.2f} > {c['vendi_collapsed']:5.2f}")

    print("\n=== 5. Graph B: paired sub-step edit similarity (raw > col; delta) ===")
    for key in SOURCE_KEYS[1:]:
        c = report["changes"]["graph_B_substep_edit"][key]
        print(f"  {source_label(key):<10} paired {c['paired_raw']:.3f} > "
              f"{c['paired_collapsed']:.3f}   delta {c['delta_raw']:+.3f} > "
              f"{c['delta_collapsed']:+.3f}")
    for tag, sel in (("raw", sel_raw), ("collapsed", sel_col)):
        order = " > ".join(r["model"] for r in sel["ranking"])
        print(f"  ranking ({tag}): {order}")

    print(f"\nSaved -> {OUT / 'collapse_impact.json'}")


if __name__ == "__main__":
    main()
