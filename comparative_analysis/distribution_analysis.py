#!/usr/bin/env python3
"""Move / sub-step distribution comparison: original vs generated intros.

For every source (original + 4 models x 2 modes), at sub-step and move level:
  * incidence          — % of intros containing the category at least once
  * token frequency    — mean occurrences per intro
  * text share         — mean proportion of annotated words devoted to it
  * Kanoksilapatham inventory — obligatory (100%) / conventional (>=60%) /
    optional (<60%) by incidence

Paired statistics (original vs each generated set, same 76 papers):
  * incidence:  exact McNemar (binomial test on discordant pairs)
  * text share: Wilcoxon signed-rank
  * Benjamini–Hochberg FDR across the 10 sub-steps within each family

Output: outputs/distribution_stats.json + console tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from move_data import (SUBSTEPS, SHORT, MOVES, SOURCE_KEYS,
                       source_label, load_intros)

OUT = Path(__file__).resolve().parent / "outputs"


def by_source(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {k: [] for k in SOURCE_KEYS}
    for r in records:
        grouped[r["source"]].append(r)
    return grouped


def describe(intros: list[dict]) -> dict:
    n = len(intros)
    out = {"n": n, "substep": {}, "move": {}}
    for lbl, short in SUBSTEPS:
        pres = [r["counts"][lbl] > 0 for r in intros]
        out["substep"][short] = {
            "incidence": sum(pres) / n,
            "token_freq": float(np.mean([r["counts"][lbl] for r in intros])),
            "word_share": float(np.mean([r["word_share"][lbl] for r in intros])),
        }
    for mv in MOVES:
        pres = [r["move_counts"][mv] > 0 for r in intros]
        out["move"][mv] = {
            "incidence": sum(pres) / n,
            "token_freq": float(np.mean([r["move_counts"][mv] for r in intros])),
            "word_share": float(np.mean([r["move_share"][mv] for r in intros])),
        }
    return out


def inventory(desc: dict) -> dict:
    """Kanoksilapatham (2005) status by incidence."""
    def status(inc: float) -> str:
        if inc >= 0.999:
            return "obligatory"
        return "conventional" if inc >= 0.60 else "optional"
    return {short: status(v["incidence"]) for short, v in desc["substep"].items()}


def bh_fdr(pvals: dict[str, float | None], alpha: float = 0.05) -> dict[str, bool]:
    items = [(k, p) for k, p in pvals.items() if p is not None]
    m = len(items)
    sig = {k: False for k in pvals}
    for rank, (k, p) in enumerate(sorted(items, key=lambda kv: kv[1]), start=1):
        if p <= alpha * rank / m:
            sig[k] = True
        else:
            break
    # BH: all hypotheses with p <= largest passing threshold are significant
    passing = [p for _, p in items if any(
        p <= pp for kk, pp in items if sig[kk])]
    thresh = max((p for k, p in items if sig[k]), default=None)
    if thresh is not None:
        for k, p in items:
            sig[k] = bool(p <= thresh)
    return sig


def paired_tests(orig: list[dict], gen: list[dict]) -> dict:
    """Original vs one generated set, paired on paper_id."""
    o = {r["paper_id"]: r for r in orig}
    g = {r["paper_id"]: r for r in gen}
    common = sorted(set(o) & set(g))
    res = {"n_pairs": len(common), "substep": {}, "move": {}}

    def tests(get_count, get_share):
        b = sum(1 for pid in common
                if get_count(o[pid]) > 0 and get_count(g[pid]) == 0)
        c = sum(1 for pid in common
                if get_count(o[pid]) == 0 and get_count(g[pid]) > 0)
        mcnemar_p = (stats.binomtest(min(b, c), b + c, 0.5).pvalue
                     if b + c > 0 else None)
        d = [get_share(g[pid]) - get_share(o[pid]) for pid in common]
        if any(abs(x) > 0 for x in d):
            wil = stats.wilcoxon(d, zero_method="wilcox", method="auto")
            wilcoxon_p = float(wil.pvalue)
        else:
            wilcoxon_p = None
        return {"mcnemar_b_c": [b, c], "mcnemar_p": mcnemar_p,
                "share_delta_mean": float(np.mean(d)), "wilcoxon_p": wilcoxon_p}

    for lbl, short in SUBSTEPS:
        res["substep"][short] = tests(lambda r, l=lbl: r["counts"][l],
                                      lambda r, l=lbl: r["word_share"][l])
    for mv in MOVES:
        res["move"][mv] = tests(lambda r, m=mv: r["move_counts"][m],
                                lambda r, m=mv: r["move_share"][m])

    for family in ("substep", "move"):
        for test in ("mcnemar_p", "wilcoxon_p"):
            sig = bh_fdr({k: v[test] for k, v in res[family].items()})
            for k in res[family]:
                res[family][k][test + "_fdr_sig"] = sig[k]
    return res


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}"


def main() -> None:
    records = load_intros()
    grouped = by_source(records)
    OUT.mkdir(exist_ok=True)

    summary = {"sources": {}, "inventories": {}, "tests_vs_original": {}}
    for key in SOURCE_KEYS:
        desc = describe(grouped[key])
        summary["sources"][key] = desc
        summary["inventories"][key] = inventory(desc)
    for key in SOURCE_KEYS[1:]:
        summary["tests_vs_original"][key] = paired_tests(
            grouped["original"], grouped[key])

    out_path = OUT / "distribution_stats.json"
    out_path.write_text(json.dumps(summary, indent=2))

    # ---- console report -----------------------------------------------------
    labels = [source_label(k) for k in SOURCE_KEYS]
    for measure, fmt in (("incidence", fmt_pct), ("token_freq", "{:.1f}".format),
                         ("word_share", fmt_pct)):
        print(f"\n=== Sub-step {measure} ===")
        print(f"{'':6}" + "".join(f"{l:>9}" for l in labels))
        for _, short in SUBSTEPS:
            row = [summary["sources"][k]["substep"][short][measure]
                   for k in SOURCE_KEYS]
            print(f"{short:<6}" + "".join(f"{fmt(v):>9}" for v in row))
        print(f"{'':6}--- moves ---")
        for mv in MOVES:
            row = [summary["sources"][k]["move"][mv][measure]
                   for k in SOURCE_KEYS]
            print(f"{mv:<6}" + "".join(f"{fmt(v):>9}" for v in row))

    print("\n=== Kanoksilapatham inventory (obligatory / conventional / optional) ===")
    print(f"{'':6}" + "".join(f"{l:>9}" for l in labels))
    for _, short in SUBSTEPS:
        row = [summary["inventories"][k][short][:4] for k in SOURCE_KEYS]
        print(f"{short:<6}" + "".join(f"{v:>9}" for v in row))

    print("\n=== Paired tests vs original (word share; * = BH-FDR significant) ===")
    print(f"{'':6}" + "".join(f"{source_label(k):>11}" for k in SOURCE_KEYS[1:]))
    for _, short in SUBSTEPS:
        cells = []
        for k in SOURCE_KEYS[1:]:
            t = summary["tests_vs_original"][k]["substep"][short]
            star = "*" if t["wilcoxon_p_fdr_sig"] else " "
            cells.append(f"{100 * t['share_delta_mean']:+.1f}{star}")
        print(f"{short:<6}" + "".join(f"{c:>11}" for c in cells))

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
