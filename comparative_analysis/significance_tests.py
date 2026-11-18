#!/usr/bin/env python3
"""Significance tests + bootstrap CIs for the Step-9 comparative analyses.

Fills the stubs left in graph_analysis.py ("bootstrap CIs and Wilcoxon+FDR")
and sequence/distribution analyses. Everything is PAIRED BY PAPER (n = 76):
each generated introduction is compared with the SAME paper's original, so the
paper is the unit of analysis and between-paper variation is differenced out.

Three families, each BH-FDR corrected within the family:

  A. Model ranking -- is Claude really closest to the human structures?
     Per-paper sub-step edit similarity to the paper's own original, averaged
     over the two prompting modes, compared between models (6 pairwise
     Wilcoxon signed-rank tests). This is the ranking the discrimination
     experiment's generator choice rests on, so it needs a test.

  B. Original vs generated -- every cell of the paper's distribution table
     (3 moves + 10 sub-steps) and of its ordering table (distinct moves,
     distinct sub-steps, move cyclicity), tested per generated set against the
     originals. These are the tests the stars in Tables 2 and 3 report; the
     FDR family is one table, so a star means "significant among the
     comparisons shown in this table".

  C. Basic vs CARS-guided -- the "checklist effect": does guidance widen the
     repertoire and move the structure away from the original?

Effect size is the matched-pairs rank-biserial correlation r_rb (the
Wilcoxon-native effect size, in [-1, 1]); CIs are percentile bootstrap.

Note on quantities: the paper's Table 2 reports POOLED shares (share of all
segments a source produces). A paired test needs a per-paper statistic, so
family B tests the MEAN OF PER-INTRODUCTION shares, which is the paired analog
and can differ from the pooled value by a little when introductions differ in
length. Both are reported side by side.

Output: outputs/significance_tests.json + a console report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, false_discovery_control, binomtest

from move_data import SUBSTEPS, MOVES, SHORT, load_intros
from graph_analysis import prepare, pair_similarity

OUT = Path(__file__).resolve().parent / "outputs"
MODELS = ["gpt-5.5", "claude-opus-4-8", "google/gemma-4-31B-it", "Qwen/Qwen3.5-27B"]
MODEL_LABEL = {"gpt-5.5": "GPT-5.5", "claude-opus-4-8": "Claude",
               "google/gemma-4-31B-it": "Gemma", "Qwen/Qwen3.5-27B": "Qwen"}
MODES = ["basic", "cars_guided"]
MODE_LABEL = {"basic": "b", "cars_guided": "c"}
N_BOOT = 10000
SEED = 20260721


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: (T+ - T-) / (T+ + T-) over the
    signed ranks of the non-zero differences. +1 = every pair favours x."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    t_plus = ranks[d > 0].sum()
    t_minus = ranks[d < 0].sum()
    total = t_plus + t_minus
    return float((t_plus - t_minus) / total) if total else 0.0


def boot_ci(x: np.ndarray, stat=np.mean, n_boot: int = N_BOOT,
            seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of `stat` over the sample."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = stat(x[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_test(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Wilcoxon signed-rank on x - y, with effect size and a bootstrap CI on
    the mean difference."""
    diff = x - y
    if np.allclose(diff, 0):
        stat, p = float("nan"), 1.0
    else:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        stat = float(stat)
    lo, hi = boot_ci(diff)
    return {
        "comparison": label, "n": int(len(x)),
        "mean_x": float(x.mean()), "mean_y": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "diff_ci95": [lo, hi],
        "W": stat, "p_raw": float(p), "r_rb": rank_biserial(diff),
    }


def apply_fdr(tests: list[dict]) -> list[dict]:
    """Benjamini-Hochberg across a family of tests, in place."""
    if not tests:
        return tests
    p = np.array([t["p_raw"] for t in tests])
    q = false_discovery_control(p, method="bh")
    for t, qq in zip(tests, q):
        t["p_fdr"] = float(qq)
        t["sig_fdr"] = bool(qq < 0.05)
    return tests


def stars(q: float) -> str:
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"


# --------------------------------------------------------------------------- #
# per-introduction statistics
# --------------------------------------------------------------------------- #
def intro_stats(r: dict) -> dict:
    """Per-introduction shares and repertoire counts (collapsed units)."""
    units = sum(r["counts"].values()) or 1
    out = {f"share_{SHORT[lbl]}": r["counts"][lbl] / units for lbl, _ in SUBSTEPS}
    mv_total = sum(r["move_counts"].values()) or 1
    for mv in MOVES:
        out[f"share_{mv}"] = r["move_counts"][mv] / mv_total
    out["n_distinct_substeps"] = sum(1 for c in r["counts"].values() if c > 0)
    out["n_distinct_moves"] = sum(1 for c in r["move_counts"].values() if c > 0)
    out["n_segments"] = r["n_segments"]
    seq = r["sub_seq"]
    out["returning_share"] = (
        sum(1 for i in range(1, len(seq)) if seq[i] in seq[:i]) / (len(seq) - 1)
        if len(seq) > 1 else 0.0)
    # Cyclicity as the paper's Table 3 defines it: does the introduction ever
    # return to a move it had already left? (binary, per introduction; the
    # move sequence is already collapsed of adjacent repeats). Measured on
    # moves rather than sub-steps because the sub-step version is saturated
    # (82-100% of introductions) and separates no source from the originals.
    mseq = r["move_seq"]
    out["cyclic"] = int(len(mseq) > len(set(mseq)))
    out["cyclic_substep"] = int(any(seq[i] in seq[:i]
                                    for i in range(1, len(seq))))
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    records = load_intros()          # collapse convention enforced by default
    prepare(records)
    originals = {r["paper_id"]: r for r in records if r["source"] == "original"}
    papers = sorted(originals)
    stats = {r["paper_id"]: intro_stats(r) for r in records
             if r["source"] == "original"}

    # paired similarity + per-intro stats for every generated intro
    sim: dict[tuple[str, str, str], float] = {}
    gstats: dict[tuple[str, str, str], dict] = {}
    for r in records:
        if r["source"] == "original":
            continue
        o = originals.get(r["paper_id"])
        if o is None:
            continue
        model, mode = r["source"].split("|")
        sim[(model, mode, r["paper_id"])] = pair_similarity(r, o)["edit_sim_sub"]
        gstats[(model, mode, r["paper_id"])] = intro_stats(r)

    results: dict = {
        "n_papers": len(papers), "n_boot": N_BOOT, "seed": SEED,
        "primary_metric": "edit_sim_sub (paired, per paper)",
    }

    # ---------------- A. model ranking ------------------------------------- #
    pooled = {m: np.array([np.mean([sim[(m, md, p)] for md in MODES])
                           for p in papers]) for m in MODELS}
    results["A_model_means"] = {
        MODEL_LABEL[m]: {"mean": float(v.mean()),
                         "ci95": list(boot_ci(v)),
                         "median": float(np.median(v))}
        for m, v in pooled.items()}

    a_tests = []
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            a_tests.append(paired_test(
                pooled[m1], pooled[m2],
                f"{MODEL_LABEL[m1]} vs {MODEL_LABEL[m2]}"))
    results["A_pairwise"] = apply_fdr(a_tests)

    # per-mode robustness check
    a_mode = []
    for mode in MODES:
        per = {m: np.array([sim[(m, mode, p)] for p in papers]) for m in MODELS}
        for i, m1 in enumerate(MODELS):
            for m2 in MODELS[i + 1:]:
                a_mode.append(paired_test(
                    per[m1], per[m2],
                    f"[{MODE_LABEL[mode]}] {MODEL_LABEL[m1]} vs {MODEL_LABEL[m2]}"))
    results["A_pairwise_by_mode"] = apply_fdr(a_mode)

    # ---------------- B1. distribution table (Table 2) ---------------------- #
    # One row per move / sub-step, exactly the rows the paper's table shows.
    rows_b1 = ([(f"share_{mv}", mv) for mv in MOVES]
               + [(f"share_{SHORT[lbl]}", SHORT[lbl]) for lbl, _ in SUBSTEPS])
    b1 = []
    for m in MODELS:
        for mode in MODES:
            src = f"{MODEL_LABEL[m]}{MODE_LABEL[mode]}"
            for k, row in rows_b1:
                g = np.array([gstats[(m, mode, p)][k] for p in papers])
                h = np.array([stats[p][k] for p in papers])
                t = paired_test(g, h, f"{src}: {row}")
                t["source"], t["row"] = src, row
                b1.append(t)
    results["B1_distribution_table"] = apply_fdr(b1)

    # ---------------- B2. ordering table (Table 3) -------------------------- #
    b2 = []
    for m in MODELS:
        for mode in MODES:
            src = f"{MODEL_LABEL[m]}{MODE_LABEL[mode]}"
            for k, row in [("n_distinct_moves", "Moves"),
                           ("n_distinct_substeps", "Steps")]:
                g = np.array([gstats[(m, mode, p)][k] for p in papers])
                h = np.array([stats[p][k] for p in papers])
                t = paired_test(g, h, f"{src}: {row}")
                t["source"], t["row"] = src, row
                b2.append(t)
            # Cyclicity is a per-introduction BINARY (does it return to an
            # already-left move?), so the paired test is exact McNemar on the
            # discordant pairs, not Wilcoxon.
            gc = np.array([gstats[(m, mode, p)]["cyclic"] for p in papers])
            hc = np.array([stats[p]["cyclic"] for p in papers])
            nb = int(np.sum((gc == 1) & (hc == 0)))
            nc = int(np.sum((gc == 0) & (hc == 1)))
            p_mc = (binomtest(nb, nb + nc, 0.5).pvalue if nb + nc else 1.0)
            b2.append({"comparison": f"{src}: Cyc.", "source": src, "row": "Cyc.",
                       "n": len(papers), "test": "exact McNemar",
                       "mean_x": float(gc.mean()), "mean_y": float(hc.mean()),
                       "mean_diff": float(gc.mean() - hc.mean()),
                       "median_diff": 0.0, "diff_ci95": [float("nan")] * 2,
                       "discordant_b": nb, "discordant_c": nc,
                       "W": float("nan"), "p_raw": float(p_mc), "r_rb": 0.0})
    results["B2_ordering_table"] = apply_fdr(b2)

    # ---------------- C. basic vs CARS-guided ------------------------------ #
    def mcnemar(x: np.ndarray, y: np.ndarray, label: str) -> dict:
        """Exact McNemar for a paired per-introduction binary."""
        nb = int(np.sum((x == 1) & (y == 0)))
        nc = int(np.sum((x == 0) & (y == 1)))
        p = binomtest(nb, nb + nc, 0.5).pvalue if nb + nc else 1.0
        return {"comparison": label, "n": int(len(x)), "test": "exact McNemar",
                "mean_x": float(x.mean()), "mean_y": float(y.mean()),
                "mean_diff": float(x.mean() - y.mean()), "median_diff": 0.0,
                "diff_ci95": [float("nan")] * 2,
                "discordant_b": nb, "discordant_c": nc,
                "W": float("nan"), "p_raw": float(p), "r_rb": 0.0}

    c_tests = []
    for m in MODELS:
        c_tests.append(paired_test(
            np.array([sim[(m, "cars_guided", p)] for p in papers]),
            np.array([sim[(m, "basic", p)] for p in papers]),
            f"{MODEL_LABEL[m]}: similarity to original (guided - basic)"))
        for k, lab in [("n_distinct_substeps", "distinct sub-steps"),
                       ("returning_share", "returning transitions"),
                       ("share_move_2", "Move 2 share")]:
            c_tests.append(paired_test(
                np.array([gstats[(m, "cars_guided", p)][k] for p in papers]),
                np.array([gstats[(m, "basic", p)][k] for p in papers]),
                f"{MODEL_LABEL[m]}: {lab} (guided - basic)"))
        c_tests.append(mcnemar(
            np.array([gstats[(m, "cars_guided", p)]["cyclic"] for p in papers]),
            np.array([gstats[(m, "basic", p)]["cyclic"] for p in papers]),
            f"{MODEL_LABEL[m]}: move cyclicity (guided - basic)"))
    results["C_guided_vs_basic"] = apply_fdr(c_tests)

    OUT.mkdir(exist_ok=True)
    (OUT / "significance_tests.json").write_text(json.dumps(results, indent=2))

    # ---------------- console report --------------------------------------- #
    def report(title, tests, note=""):
        print(f"\n=== {title} ===")
        if note:
            print(note)
        print(f"{'comparison':<52} {'diff':>7} {'95% CI':>17} "
              f"{'r_rb':>6} {'p_FDR':>9}  ")
        for t in tests:
            ci = f"[{t['diff_ci95'][0]:+.3f},{t['diff_ci95'][1]:+.3f}]"
            print(f"{t['comparison']:<52} {t['mean_diff']:>+7.3f} {ci:>17} "
                  f"{t['r_rb']:>+6.2f} {t['p_fdr']:>9.4f} {stars(t['p_fdr'])}")

    print(f"Paired by paper, n = {len(papers)}. "
          f"BH-FDR within each family; {N_BOOT} bootstrap resamples.")

    print("\n=== A0. Per-paper similarity to the original (sub-step edit) ===")
    print(f"{'model':<10} {'mean':>6} {'95% CI':>17} {'median':>7}")
    for m in MODELS:
        r = results["A_model_means"][MODEL_LABEL[m]]
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
        print(f"{MODEL_LABEL[m]:<10} {r['mean']:>6.3f} {ci:>17} {r['median']:>7.3f}")

    report("A. Model ranking (pooled over modes)", results["A_pairwise"],
           "Does the 'Claude is closest to the originals' ranking hold up?")
    report("A'. Same, per prompting mode", results["A_pairwise_by_mode"])
    report("B1. Distribution table: generated vs original",
           [t for t in results["B1_distribution_table"] if t["sig_fdr"]],
           "(significant rows only; full matrix in the JSON)")
    report("B2. Ordering table: generated vs original",
           results["B2_ordering_table"])
    report("C. CARS-guided vs basic (the checklist effect)",
           results["C_guided_vs_basic"])

    # ---- star marks, ready to transcribe into the LaTeX tables ------------ #
    marks = {}
    for name, fam in [("substeps", results["B1_distribution_table"]),
                      ("subseq", results["B2_ordering_table"])]:
        marks[name] = {}
        for t in fam:
            marks[name].setdefault(t["source"], {})[t["row"]] = t["sig_fdr"]
    results["table_marks"] = marks
    (OUT / "significance_tests.json").write_text(json.dumps(results, indent=2))

    for name in ("substeps", "subseq"):
        print(f"\n=== star marks for the {name} table (row: sources) ===")
        rows = list(next(iter(marks[name].values())).keys())
        for row in rows:
            hits = [s for s in marks[name] if marks[name][s][row]]
            print(f"  {row:<8} {len(hits)}/8  {' '.join(hits) if hits else '--'}")

    print(f"\nWrote {OUT / 'significance_tests.json'}")


if __name__ == "__main__":
    main()
