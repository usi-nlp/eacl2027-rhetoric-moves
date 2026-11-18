#!/usr/bin/env python3
"""Move-ordering / sequence analysis: original vs generated, by quartile.

On the evaluator's annotations (same annotator both sides):
  * top sub-step n-grams (bigrams / trigrams) per corpus and per quartile
  * move-structure types (collapsed move sequence per intro) + diversity
  * cyclicity (Crookes-style: move units per intro, returns to an earlier move)
  * canonical-order conformity (weakly monotone 1 -> 2 -> 3)
  * sub-move ordering profile per source (paper's merged ordering table):
    modal collapsed sub-step trigram, move cycles (the table's Cyc. column:
    % of intros that return to an already-left move), sub-step cycles as a
    secondary diagnostic, within-move sub-step chaining, mean unique moves /
    sub-steps per intro, initial sub-step distribution
  * paired per-paper edit distance between the original's and each generated
    set's collapsed move sequence (normalised Levenshtein)

Output: outputs/sequence_stats.json + console tables.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from move_data import SHORT, SOURCE_KEYS, source_label, load_intros

# sub-step display code -> its move (e.g. "CC" -> "move_1")
MOVE_OF = {code: lbl.split("__")[0] for lbl, code in SHORT.items()}

OUT = Path(__file__).resolve().parent / "outputs"
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]


def ngrams(seq: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def top_ngrams(intros: list[dict], n: int, k: int = 8) -> list[dict]:
    """Most frequent sub-step n-grams; paper_pct = % of intros containing it."""
    tokens = Counter()
    papers = Counter()
    for r in intros:
        grams = ngrams(r["sub_seq"], n)
        tokens.update(grams)
        papers.update(set(grams))
    return [{"ngram": " > ".join(g), "count": c,
             "paper_pct": papers[g] / len(intros)}
            for g, c in tokens.most_common(k)]


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_canonical(move_seq: list[str]) -> bool:
    """Weakly monotone 1 -> 2 -> 3 (each move may repeat, never goes back)."""
    return all(move_seq[i] <= move_seq[i + 1] for i in range(len(move_seq) - 1))


def cycles(move_seq: list[str]) -> int:
    """Returns to a move already visited earlier (after an intervening move)."""
    seen = set()
    c = 0
    for i, m in enumerate(move_seq):
        if i > 0 and m in seen and move_seq[i - 1] != m:
            c += 1
        seen.add(m)
    return c


def has_move_cycle(move_seq: list[str]) -> bool:
    """Does the introduction return to a move it had already left?

    move_seq is already collapsed of adjacent repeats, so a repeated label is
    a return. This is the cyclicity the paper's ordering table reports; on
    this corpus it is the exact complement of is_canonical (no introduction
    departs from the canonical order without returning to an earlier move).
    """
    return len(move_seq) > len(set(move_seq))


def collapse(seq: list[str]) -> list[str]:
    """Merge runs of consecutive identical labels."""
    return [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]


def submove_stats(intros: list[dict], k: int = 5) -> dict:
    """Sub-move ordering profile on collapsed sub-step sequences."""
    n = len(intros)
    tri_tok, tri_pap = Counter(), Counter()
    n_cycle_intros = 0
    ret_trans = new_trans = 0
    within_trans = across_trans = 0
    inits = Counter()
    for r in intros:
        c = collapse(r["sub_seq"])
        inits[c[0]] += 1
        grams = ngrams(c, 3)
        tri_tok.update(grams)
        tri_pap.update(set(grams))
        if len(c) > len(set(c)):  # some sub-step recurs non-adjacently
            n_cycle_intros += 1
        seen = {c[0]}
        for a, b in zip(c, c[1:]):
            if b in seen:
                ret_trans += 1
            else:
                new_trans += 1
            seen.add(b)
            if MOVE_OF[a] == MOVE_OF[b]:
                within_trans += 1
            else:
                across_trans += 1
    total_trans = ret_trans + new_trans
    return {
        "n": n,
        "mean_unique_moves": float(np.mean(
            [len(set(r["move_seq"])) for r in intros])),
        "mean_unique_substeps": float(np.mean(
            [len(set(r["sub_seq"])) for r in intros])),
        "pct_intros_all_three_moves": float(np.mean(
            [len(set(r["move_seq"])) == 3 for r in intros])),
        "pct_intros_with_substep_cycle": n_cycle_intros / n,
        "pct_return_transitions": ret_trans / total_trans,
        "pct_within_move_transitions": within_trans / total_trans,
        "top_collapsed_trigrams": [
            {"ngram": " > ".join(g), "count": c, "paper_pct": tri_pap[g] / n}
            for g, c in tri_tok.most_common(k)],
        "initial_substeps": [
            {"substep": s, "count": c, "pct": c / n}
            for s, c in inits.most_common(k)],
    }


def seq_stats(intros: list[dict]) -> dict:
    types = Counter("-".join(r["move_seq"]) for r in intros)
    return {
        "n": len(intros),
        "mean_move_units": float(np.mean([len(r["move_seq"]) for r in intros])),
        "mean_cycles": float(np.mean([cycles(r["move_seq"]) for r in intros])),
        "pct_intros_with_move_cycle": float(np.mean(
            [has_move_cycle(r["move_seq"]) for r in intros])),
        "pct_canonical": float(np.mean([is_canonical(r["move_seq"])
                                        for r in intros])),
        "n_sequence_types": len(types),
        "type_token_ratio": len(types) / len(intros),
        "top_move_structures": [{"type": t, "count": c}
                                for t, c in types.most_common(6)],
        "top_bigrams": top_ngrams(intros, 2),
        "top_trigrams": top_ngrams(intros, 3),
    }


def main() -> None:
    records = load_intros()
    OUT.mkdir(exist_ok=True)
    by_src: dict[str, list[dict]] = {k: [] for k in SOURCE_KEYS}
    for r in records:
        by_src[r["source"]].append(r)

    summary: dict = {"sources": {}, "submoves": {}, "quartiles": {},
                     "edit_distance": {}}

    for key in SOURCE_KEYS:
        summary["sources"][key] = seq_stats(by_src[key])
        summary["submoves"][key] = submove_stats(by_src[key])

    # Quartile view: originals per quartile, plus all generated pooled per
    # quartile and the two guidance modes pooled separately.
    for q in QUARTILES:
        orig_q = [r for r in by_src["original"] if r["quartile"] == q]
        gen_q = [r for r in records
                 if r["source"] != "original" and r["quartile"] == q]
        # TTR is n-sensitive, so the pooled generated TTR is not comparable to
        # the (much smaller) original quartile cell; also report the mean TTR
        # across the 8 generated sets, whose per-quartile n matches the original.
        per_set_ttr = [seq_stats([r for r in gen_q if r["source"] == k])
                       ["type_token_ratio"] for k in SOURCE_KEYS[1:]]
        summary["quartiles"][q] = {
            "original": seq_stats(orig_q),
            "generated_all": seq_stats(gen_q),
            "generated_ttr_per_set_mean": float(np.mean(per_set_ttr)),
            "generated_basic": seq_stats(
                [r for r in gen_q if r["source"].endswith("|basic")]),
            "generated_cars": seq_stats(
                [r for r in gen_q if r["source"].endswith("|cars_guided")]),
        }

    # Paired normalised edit distance on collapsed move sequences.
    orig_by_pid = {r["paper_id"]: r for r in by_src["original"]}
    for key in SOURCE_KEYS[1:]:
        dists, dists_q = [], {q: [] for q in QUARTILES}
        for r in by_src[key]:
            o = orig_by_pid.get(r["paper_id"])
            if o is None:
                continue
            d = levenshtein(o["move_seq"], r["move_seq"]) / max(
                len(o["move_seq"]), len(r["move_seq"]))
            dists.append(d)
            dists_q[r["quartile"]].append(d)
        summary["edit_distance"][key] = {
            "mean": float(np.mean(dists)),
            "by_quartile": {q: float(np.mean(v)) for q, v in dists_q.items()},
        }

    (OUT / "sequence_stats.json").write_text(json.dumps(summary, indent=2))

    # ---- console report -----------------------------------------------------
    print("=== Sequence statistics per source ===")
    print(f"{'source':<10} {'units':>6} {'cycles':>7} {'canon%':>7} "
          f"{'types':>6} {'TTR':>6}")
    for key in SOURCE_KEYS:
        s = summary["sources"][key]
        print(f"{source_label(key):<10} {s['mean_move_units']:>6.1f} "
              f"{s['mean_cycles']:>7.1f} {100 * s['pct_canonical']:>6.0f}% "
              f"{s['n_sequence_types']:>6} {s['type_token_ratio']:>6.2f}")

    print("\n=== Sub-move ordering profile per source (collapsed sub-steps) ===")
    print(f"{'source':<10} {'top trigram (paper%)':<26} {'Mcyc%':>6} {'scyc%':>5} "
          f"{'ret%':>5} {'w/in%':>6} {'uniqM':>6} {'uniqS':>6} "
          f"{'first (pct)':>12}")
    for key in SOURCE_KEYS:
        s = summary["submoves"][key]
        t = s["top_collapsed_trigrams"][0]
        first = s["initial_substeps"][0]
        tri = "{} ({:.0f}%)".format(t["ngram"], 100 * t["paper_pct"])
        ini = "{} ({:.0f}%)".format(first["substep"], 100 * first["pct"])
        print(f"{source_label(key):<10} {tri:<26} "
              f"{100 * summary['sources'][key]['pct_intros_with_move_cycle']:>5.0f}% "
              f"{100 * s['pct_intros_with_substep_cycle']:>4.0f}% "
              f"{100 * s['pct_return_transitions']:>4.0f}% "
              f"{100 * s['pct_within_move_transitions']:>5.0f}% "
              f"{s['mean_unique_moves']:>6.2f} {s['mean_unique_substeps']:>6.2f} "
              f"{ini:>12}")

    print("\n=== Top move structures (collapsed) ===")
    for key in SOURCE_KEYS:
        s = summary["sources"][key]
        tops = ", ".join(f"{t['type']} ({t['count']})"
                         for t in s["top_move_structures"][:4])
        print(f"{source_label(key):<10} {tops}")

    for q in QUARTILES:
        print(f"\n=== {q} — original vs generated ===")
        for side in ("original", "generated_all"):
            s = summary["quartiles"][q][side]
            print(f"  [{side}] n={s['n']} units={s['mean_move_units']:.1f} "
                  f"cycles={s['mean_cycles']:.1f} "
                  f"canonical={100 * s['pct_canonical']:.0f}% "
                  f"TTR={s['type_token_ratio']:.2f}")
            print(f"    top trigrams: " + "; ".join(
                f"{t['ngram']} ({100 * t['paper_pct']:.0f}%)"
                for t in s["top_trigrams"][:5]))
            print(f"    top structures: " + "; ".join(
                f"{t['type']} ({t['count']})"
                for t in s["top_move_structures"][:3]))

    print("\n=== Paired edit distance (original vs generated move sequence) ===")
    print(f"{'source':<10} {'mean':>6}" + "".join(f"{q:>7}" for q in QUARTILES))
    for key in SOURCE_KEYS[1:]:
        e = summary["edit_distance"][key]
        print(f"{source_label(key):<10} {e['mean']:>6.2f}" +
              "".join(f"{e['by_quartile'][q]:>7.2f}" for q in QUARTILES))

    print(f"\nSaved -> {OUT / 'sequence_stats.json'}")


if __name__ == "__main__":
    main()
