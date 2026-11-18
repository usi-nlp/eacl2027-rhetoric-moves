#!/usr/bin/env python3
"""Graph-structure analysis of the CARS annotations (Step-9, new line).

Treats each annotated introduction as a graph whose nodes are the CARS move
CLASSES (segments in reading order, labelled by move / sub-step) and computes:

  A. WITHIN-source diversity  -> mean pairwise similarity among one source's
     own intros (self-similarity / homogeneity; high = templated) + the Vendi
     Score (effective number of distinct move-structures) + structure types.
  B. Paired similarity to the ORIGINAL -> each generated intro vs the SAME
     paper's original intro, averaged per model (and per model x mode).

Both use the SAME two class-based similarity measures (locked 2026-07-16):

  * EDIT similarity  -- 1 - normalised Levenshtein on the collapsed class
    sequence (graph edit distance on a labelled path). Sequences are collapsed
    (consecutive duplicate labels merged) at BOTH levels so the measure is
    invariant to segmentation granularity -- the evaluator splits generated
    intros more finely than originals (mean 15.2 vs 10.9 segments), and an
    uncollapsed distance would score that split, not the structure.
  * BAG-of-classes similarity -- histogram intersection (1 - total variation)
    between the two intros' class WORD-SHARE distributions: the share of text
    the two intros allocate to the same classes, order ignored. Word share
    (not segment counts) is also granularity-invariant.

Each measure runs at the move level (3 classes) and sub-step level (10
classes). Transition-matrix JSD was evaluated and DROPPED:
per-intro sub-step transition graphs are too sparse (~7.8 of 100 cells) so the
JSD saturates near ceiling, and the move-level chain is too coarse.

Analysis B also reports each source's MISMATCHED baseline -- the mean
similarity of its intros to all OTHER papers' originals -- because most of the
raw paired similarity is generic CARS-ness; the paired-minus-mismatched delta
is the paper-specific signal. Reading guide for A vs B: within-model >
within-human means the model is more templated than the genre; within-model >
model-to-original means the model resembles itself more than the humans it
imitates (house style).

The model with the highest similarity (B) is the recommended generator for the
discrimination experiment (see discrimination_experiment/build_allocation.py).

The measures are defined in the paper's appendix (move-structure similarity
measures). Still stubbed: WL graph kernel / graph embeddings (metric upgrades).
Bootstrap CIs and the Wilcoxon + FDR tests live in significance_tests.py.

Output: outputs/graph_diversity.json, outputs/graph_similarity.json + console.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from move_data import SUBSTEPS, MOVES, SOURCES, SOURCE_KEYS, source_label, load_intros
from sequence_analysis import levenshtein, collapse

OUT = Path(__file__).resolve().parent / "outputs"
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
MODELS = list(dict.fromkeys(m for m, _, _ in SOURCES[1:]))
LEVELS = ("move", "sub")
METRICS = ["edit_sim_move", "edit_sim_sub", "bag_sim_move", "bag_sim_sub"]


# --------------------------------------------------------------------------- #
# graph representation: collapsed class sequence + class word-share bag
# --------------------------------------------------------------------------- #
def prepare(records: list[dict]) -> None:
    """Precompute each intro's class sequences and word-share bags in place."""
    for r in records:
        r["seqs"] = {"move": r["move_seq"],           # already collapsed
                     "sub": collapse(r["sub_seq"])}
        r["bags"] = {
            "move": np.array([r["move_share"][m] for m in MOVES]),
            "sub": np.array([r["word_share"][lbl] for lbl, _ in SUBSTEPS]),
        }


def norm_lev(a: list[str], b: list[str]) -> float:
    """Length-normalised Levenshtein distance in [0, 1]."""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def bag_sim(p: np.ndarray, q: np.ndarray) -> float:
    """Histogram intersection (= 1 - total variation distance) between two
    class share distributions."""
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0
    return float(np.minimum(p / sp, q / sq).sum())


def pair_similarity(a: dict, b: dict) -> dict:
    """The four locked class-based similarities between two intros."""
    out = {}
    for lvl in LEVELS:
        out[f"edit_sim_{lvl}"] = 1.0 - norm_lev(a["seqs"][lvl], b["seqs"][lvl])
        out[f"bag_sim_{lvl}"] = bag_sim(a["bags"][lvl], b["bags"][lvl])
    return out


def vendi_score(structures: list[tuple[str, ...]]) -> float:
    """Vendi Score (Friedman & Dieng 2023) under the exact-match kernel
    k(a, b) = 1 iff the collapsed class sequences a and b are identical.
    This kernel is PSD by construction and the nonzero eigenvalues of K/n
    are exactly the structure-type frequencies, so the score reduces to
    exp(Shannon entropy) of the type distribution: the effective number of
    distinct structures, no eigendecomposition or clipping needed."""
    n = len(structures)
    if n <= 1:
        return float(n)
    p = np.array(list(Counter(structures).values()), dtype=float) / n
    return float(np.exp(-np.sum(p * np.log(p))))


# --------------------------------------------------------------------------- #
# advanced upgrades (STUBS -- not used for any number in the paper)
# --------------------------------------------------------------------------- #
def wl_kernel_matrix(records: list[dict], level: str = "sub") -> np.ndarray:  # noqa: ARG001
    """TODO: Weisfeiler-Lehman subtree kernel on the class-labelled graphs ->
    a PSD similarity matrix usable for a kernel-based Vendi variant and pair
    similarity. (Shervashidze et al. 2011; grakel or custom WL.)"""
    raise NotImplementedError("wl_kernel_matrix is a stub (not used in the paper)")


def graph_embedding(records: list[dict], level: str = "sub") -> np.ndarray:  # noqa: ARG001
    """TODO: graph2vec / GNN embedding of the class-labelled graphs, then
    cosine / Vendi in embedding space. (Narayanan et al. 2017.)"""
    raise NotImplementedError("graph_embedding is a stub (not used in the paper)")


# --------------------------------------------------------------------------- #
# Analysis A -- within-source diversity / homogeneity
# --------------------------------------------------------------------------- #
def within_source_stats(records: list[dict]) -> dict:
    """Mean pairwise self-similarity (all intro pairs within the source, both
    locked metrics x both levels), exact-match-kernel Vendi scores of the
    class sequences, and move-structure type counts."""
    n = len(records)
    sims: dict[str, list[float]] = {m: [] for m in METRICS}
    for a, b in itertools.combinations(records, 2):
        s = pair_similarity(a, b)
        for m in METRICS:
            sims[m].append(s[m])
    seqs = {lvl: [tuple(r["seqs"][lvl]) for r in records] for lvl in LEVELS}
    types = Counter("-".join(r["seqs"]["move"]) for r in records)
    return {
        "n": n,
        "mean_pairwise": {m: float(np.mean(sims[m])) if sims[m] else 1.0
                          for m in METRICS},
        "vendi": {lvl: vendi_score(seqs[lvl]) for lvl in LEVELS},
        "vendi_ratio": {lvl: vendi_score(seqs[lvl]) / n if n else 0.0
                        for lvl in LEVELS},
        "n_move_structure_types": len(types),
        "move_structure_ttr": (len(types) / n) if n else 0.0,
    }
    # TODO: bootstrap CIs on mean_pairwise / vendi (resample intros).


def diversity_analysis(records: list[dict]) -> dict:
    """Analysis A over every source, plus mode and quartile splits."""
    by_src: dict[str, list[dict]] = {k: [] for k in SOURCE_KEYS}
    for r in records:
        by_src[r["source"]].append(r)

    out: dict = {"sources": {}, "by_mode": {}, "by_quartile": {}}
    for key in SOURCE_KEYS:
        out["sources"][key] = within_source_stats(by_src[key])

    out["by_mode"]["original"] = out["sources"]["original"]
    for mode in ("basic", "cars_guided"):
        pool = [r for r in records if r["source"].endswith("|" + mode)]
        out["by_mode"][mode] = within_source_stats(pool)

    for q in QUARTILES:
        out["by_quartile"][q] = {
            "original": within_source_stats(
                [r for r in by_src["original"] if r["quartile"] == q]),
            "generated": within_source_stats(
                [r for r in records
                 if r["source"] != "original" and r["quartile"] == q]),
        }
    return out


# --------------------------------------------------------------------------- #
# Analysis B -- paired similarity to the paper's original
# --------------------------------------------------------------------------- #
def _mean_rows(rows: list[dict]) -> dict:
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def similarity_to_original(records: list[dict]) -> dict:
    """Per generated intro vs the SAME paper's original, averaged per source
    and per model; plus the mismatched baseline (vs all OTHER originals) and
    the paired-minus-mismatched delta = the paper-specific signal."""
    originals = {r["paper_id"]: r for r in records if r["source"] == "original"}

    per_source: dict = {}
    model_rows: dict[str, dict[str, list[dict]]] = {
        m: {"paired": [], "mismatched": []} for m in MODELS}
    for key in SOURCE_KEYS[1:]:
        gens = [r for r in records if r["source"] == key]
        paired = [pair_similarity(g, originals[g["paper_id"]])
                  for g in gens if g["paper_id"] in originals]
        mism = [pair_similarity(g, o)
                for g in gens for pid, o in originals.items()
                if pid != g["paper_id"]]
        model = key.split("|")[0]
        model_rows[model]["paired"] += paired
        model_rows[model]["mismatched"] += mism
        pm, mm = _mean_rows(paired), _mean_rows(mism)
        per_source[key] = {
            "n": len(paired), "paired_mean": pm, "mismatched_mean": mm,
            "delta": {m: pm[m] - mm[m] for m in METRICS},
        }

    per_model: dict = {}
    for model, rows in model_rows.items():
        pm, mm = _mean_rows(rows["paired"]), _mean_rows(rows["mismatched"])
        per_model[model] = {
            "n": len(rows["paired"]), "paired_mean": pm, "mismatched_mean": mm,
            "delta": {m: pm[m] - mm[m] for m in METRICS},
        }
    # TODO: Wilcoxon signed-rank across models on per-paper similarity + BH-FDR;
    #       Kruskal-Wallis on per-paper similarity across quartiles.
    return {"primary_metric": "edit_sim_sub",
            "per_source": per_source, "per_model": per_model}


def select_generator_for_discrimination(sim: dict) -> dict:
    """Rank the models by the primary paired similarity and name the winner
    (the recommended discrimination-experiment generator)."""
    metric = sim["primary_metric"]
    ranked = sorted(sim["per_model"].items(),
                    key=lambda kv: kv[1]["paired_mean"][metric], reverse=True)
    return {
        "primary_metric": metric,
        "ranking": [{"model": m, metric: v["paired_mean"][metric],
                     "delta_over_mismatched": v["delta"][metric]}
                    for m, v in ranked],
        "recommended_generator": ranked[0][0],
        "current_default": "claude-opus-4-8",
        "note": ("PLACEHOLDER selection -- significance testing (Wilcoxon+FDR) "
                 "still to run before overriding the default and rebuilding "
                 "discrimination_experiment/allocation.json."),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    records = load_intros()
    prepare(records)
    OUT.mkdir(exist_ok=True)

    diversity = diversity_analysis(records)
    similarity = similarity_to_original(records)
    selection = select_generator_for_discrimination(similarity)

    (OUT / "graph_diversity.json").write_text(json.dumps(diversity, indent=2))
    (OUT / "graph_similarity.json").write_text(
        json.dumps({"similarity": similarity, "selection": selection}, indent=2))

    # ---- console report ---------------------------------------------------- #
    print("=== A. Within-source self-similarity / diversity ===")
    print("(pairSim: high = homogeneous/templated; Vendi: effective # of "
          "distinct structures, high = diverse)")
    print(f"{'source':<10} {'n':>3} {'editM':>6} {'editS':>6} {'bagM':>6} "
          f"{'bagS':>6} {'VendiM':>7} {'VendiS':>7} {'types':>6} {'TTR':>5}")
    for key in SOURCE_KEYS:
        d = diversity["sources"][key]
        mp = d["mean_pairwise"]
        print(f"{source_label(key):<10} {d['n']:>3} "
              f"{mp['edit_sim_move']:>6.3f} {mp['edit_sim_sub']:>6.3f} "
              f"{mp['bag_sim_move']:>6.3f} {mp['bag_sim_sub']:>6.3f} "
              f"{d['vendi']['move']:>7.2f} {d['vendi']['sub']:>7.2f} "
              f"{d['n_move_structure_types']:>6} {d['move_structure_ttr']:>5.2f}")

    print("\n=== B. Paired similarity to the paper's original ===")
    print("(delta = paired minus mismatched-original baseline, "
          "sub-step edit metric)")
    print(f"{'model x mode':<12} {'n':>3} {'editM':>6} {'editS':>6} "
          f"{'bagM':>6} {'bagS':>6} {'dEditS':>7}")
    for key in SOURCE_KEYS[1:]:
        s = per = similarity["per_source"][key]
        p = per["paired_mean"]
        print(f"{source_label(key):<12} {s['n']:>3} "
              f"{p['edit_sim_move']:>6.3f} {p['edit_sim_sub']:>6.3f} "
              f"{p['bag_sim_move']:>6.3f} {p['bag_sim_sub']:>6.3f} "
              f"{per['delta']['edit_sim_sub']:>+7.3f}")

    print("\n=== Per-model (pooled over modes) + generator selection "
          f"(primary = {selection['primary_metric']}) ===")
    for row in selection["ranking"]:
        star = ("  <-- recommended"
                if row["model"] == selection["recommended_generator"] else "")
        print(f"  {row['model']:<24} {row[selection['primary_metric']]:.3f} "
              f"(delta {row['delta_over_mismatched']:+.3f}){star}")
    print(f"  current default: {selection['current_default']}")

    print(f"\nSaved -> {OUT / 'graph_diversity.json'}, "
          f"{OUT / 'graph_similarity.json'}")


if __name__ == "__main__":
    main()
