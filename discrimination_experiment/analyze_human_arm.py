#!/usr/bin/env python3
"""Signal-detection analysis of the human arm of the discrimination experiment.

Joins the blinded response export (responses.json, produced by
convert_labelstudio_export.py) with the sealed key (allocation.json) and runs
the analyses of the study design (recapped below) plus the follow-ups the
data supports.

Design recap: 60 papers x 2 items (the paper's own original + one
claude-opus-4-8 generation), 2x2 between-paper design -- generation mode
(basic / cars_guided) x label condition (CARS move highlights shown / not),
15 papers (30 items) per cell. One expert judge, 120 items, 5-point
provenance rating (1 sure human ... 5 sure AI) + required decision basis.

Sections
  1  Response profile           rating + basis distributions, undecided rate
  2  Overall discrimination     confusion matrix, d', criterion, AUC, kappa
  3  RQ1 labels                 labeled vs unlabeled sensitivity
  4  Human vs AI items          hit rate vs correct-rejection rate
  5  RQ2 generation mode        basic vs cars_guided evasion
  6  2x2 cells + regressions    ordinal / logistic models with the interaction
  7  Paired within-paper        does the judge order a paper's two intros right
  8  Confidence & calibration
  9  Decision basis             what the judge says they used, and whether it pays
  10 Time on task
  11 Order / practice effects
  12 Item covariates            quartile, nativeness, length
  13 CARS structure             do the ratings track the structural tells?
  14 Pair similarity            do the closest generations evade detection?
  15 Error inventory

Unit of analysis is the item; the paper is the cluster (two items share a
paper and a cell), so CIs use a cluster bootstrap that resamples papers within
each of the four cells, and the regressions use cluster-robust SEs.

Usage:
    python discrimination_experiment/analyze_human_arm.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import (binomtest, fisher_exact, mannwhitneyu, norm,
                         spearmanr, wilcoxon)

EXP = Path(__file__).resolve().parent
REPO = EXP.parent
OUT = EXP / "outputs"
sys.path.insert(0, str(REPO / "comparative_analysis"))

from graph_analysis import pair_similarity, prepare  # noqa: E402
from move_data import MOVES, SUBSTEPS, load_intros  # noqa: E402

GENERATOR = "claude-opus-4-8"
N_BOOT = 10000
SEED = 20260727
CELLS = [(m, l) for m in ("basic", "cars_guided") for l in ("labeled", "unlabeled")]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_rows() -> list[dict]:
    alloc = json.loads((EXP / "allocation.json").read_text())
    key = {i["item_code"]: i for i in alloc["items"]}
    resp = json.loads((EXP / "responses.json").read_text())["responses"]
    if len(resp) != len(key):
        raise SystemExit(f"{len(resp)} responses for {len(key)} items -- incomplete export")

    rows = []
    for code, r in resp.items():
        a = key[code]
        if r["label_condition"] != a["label_condition"]:
            raise SystemExit(f"{code}: label_condition disagrees with the key")
        rating = r["provenance"]
        rows.append({
            "item_code": code,
            "position": a["position"],
            "block": a["block"],
            "paper": a["paper_key"],
            "intro_id": a["intro_id"],
            "is_ai": a["intro_type"] == "generated",
            # The 2x2 cell is assigned to the PAPER, so cell_mode is the mode its
            # generation was written in. An original has no mode of its own -- it is
            # the paired control inside that cell. Never present it as "a basic original".
            "cell_mode": a["mode"],
            "labeled": a["label_condition"] == "labeled",
            "quartile": a["quartile"],
            "nativeness": a["nativeness"],
            "word_count": a["word_count"],
            "rating": rating,
            "said_ai": rating >= 4,
            "undecided": rating == 3,
            "confidence": abs(rating - 3),   # 2 = "sure", 1 = "most likely", 0 = can't decide
            "basis": r["basis"] or [],
            "comment": r["comment"],
            "lead_time": r["lead_time"],
        })
    for row in rows:
        row["correct"] = (row["said_ai"] == row["is_ai"]) and not row["undecided"]
    return sorted(rows, key=lambda r: r["position"])


def attach_structure(rows: list[dict]) -> None:
    """Attach each item's collapsed CARS annotation (same auto-labeller both sides)."""
    records = load_intros()
    prepare(records)
    by_key = {}
    for rec in records:
        by_key[(f"{rec['paper_id']:03d}", rec["source"])] = rec
    for row in rows:
        src = "original" if not row["is_ai"] else f"{GENERATOR}|{row['cell_mode']}"
        rec = by_key[(row["paper"], src)]
        row["rec"] = rec
        row["n_units"] = rec["n_segments"]
        row["repertoire"] = sum(1 for v in rec["counts"].values() if v)
        row["gap_units"] = rec["counts"]["move_2__indicating_a_gap"]
        row["opens_centrality"] = rec["sub_seq"][:1] == ["CC"]
        row["move_cycles"] = max(len(rec["move_seq"]) - 1, 0)
        for mv in MOVES:
            row[f"{mv}_share"] = rec["move_share"][mv]


# --------------------------------------------------------------------------- #
# signal-detection helpers
# --------------------------------------------------------------------------- #
def rates(rows: list[dict]) -> dict:
    ai = [r for r in rows if r["is_ai"]]
    hu = [r for r in rows if not r["is_ai"]]
    hits = sum(r["said_ai"] for r in ai)
    fas = sum(r["said_ai"] for r in hu)
    return {"n_ai": len(ai), "n_human": len(hu), "hits": hits, "false_alarms": fas,
            "hit_rate": hits / len(ai) if ai else float("nan"),
            "fa_rate": fas / len(hu) if hu else float("nan")}


def sdt(rows: list[dict]) -> dict:
    """d' and criterion with the log-linear (Hautus 1995) correction.

    The correction is applied unconditionally rather than only to the 0/1
    cells, so d' stays comparable across conditions -- cars_guided hits at
    30/30 and would otherwise be infinite.
    """
    r = rates(rows)
    h = (r["hits"] + 0.5) / (r["n_ai"] + 1)
    f = (r["false_alarms"] + 0.5) / (r["n_human"] + 1)
    zh, zf = norm.ppf(h), norm.ppf(f)
    return {**r,
            "hit_rate_adj": h, "fa_rate_adj": f,
            "d_prime": zh - zf,
            "criterion_c": -0.5 * (zh + zf),
            "accuracy": np.mean([x["correct"] for x in rows]) if rows else float("nan")}


def auc(rows: list[dict]) -> float:
    """Mann-Whitney AUC of the 5-point rating for AI vs human items (ties = 0.5)."""
    ai = np.array([r["rating"] for r in rows if r["is_ai"]], float)
    hu = np.array([r["rating"] for r in rows if not r["is_ai"]], float)
    if not len(ai) or not len(hu):
        return float("nan")
    diff = ai[:, None] - hu[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(ai) * len(hu)))


def kappa(rows: list[dict]) -> float:
    """Cohen's kappa between the judge's binary verdict and the truth."""
    n = len(rows)
    obs = np.mean([r["correct"] for r in rows])
    p_ai_true = np.mean([r["is_ai"] for r in rows])
    p_ai_said = np.mean([r["said_ai"] for r in rows])
    exp = p_ai_true * p_ai_said + (1 - p_ai_true) * (1 - p_ai_said)
    return float((obs - exp) / (1 - exp)) if n and exp < 1 else float("nan")


def mcc(rows: list[dict]) -> float:
    tp = sum(r["is_ai"] and r["said_ai"] for r in rows)
    tn = sum(not r["is_ai"] and not r["said_ai"] for r in rows)
    fp = sum(not r["is_ai"] and r["said_ai"] for r in rows)
    fn = sum(r["is_ai"] and not r["said_ai"] for r in rows)
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den else float("nan")


def f1_ai(rows: list[dict]) -> float:
    """F1 for the 'this is AI-generated' class (the reported headline metric)."""
    tp = sum(r["is_ai"] and r["said_ai"] for r in rows)
    fp = sum(not r["is_ai"] and r["said_ai"] for r in rows)
    fn = sum(r["is_ai"] and not r["said_ai"] for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else float("nan")


def summarise(rows: list[dict]) -> dict:
    s = sdt(rows)
    s["auc"] = auc(rows)
    s["f1_ai"] = f1_ai(rows)
    s["mean_rating_ai"] = float(np.mean([r["rating"] for r in rows if r["is_ai"]]))
    s["mean_rating_human"] = float(np.mean([r["rating"] for r in rows if not r["is_ai"]]))
    s["n"] = len(rows)
    return s


# --------------------------------------------------------------------------- #
# cluster bootstrap: resample papers with replacement inside each design cell
# --------------------------------------------------------------------------- #
def cluster_bootstrap(rows: list[dict], stat, n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile CI for any statistic of the row set.

    Papers are the resampling unit and cell sizes are held fixed, matching the
    stratified paper-level randomisation.
    """
    rng = np.random.default_rng(seed)
    by_paper: dict[str, list[dict]] = {}
    for r in rows:
        by_paper.setdefault(r["paper"], []).append(r)
    cell_papers: dict[tuple, list[str]] = {}
    for p, items in by_paper.items():
        cell_papers.setdefault((items[0]["cell_mode"], items[0]["labeled"]), []).append(p)

    point = stat(rows)
    draws = []
    for _ in range(n_boot):
        sample = []
        for papers in cell_papers.values():
            picked = rng.choice(papers, size=len(papers), replace=True)
            for p in picked:
                sample.extend(by_paper[p])
        try:
            v = stat(sample)
        except (ValueError, ZeroDivisionError):
            v = float("nan")
        draws.append(v)
    draws = np.array([d for d in draws if np.isfinite(d)], float)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "boot_sd": float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")}


def diff_bootstrap(rows_a: list[dict], rows_b: list[dict], stat,
                   n_boot: int = N_BOOT, seed: int = SEED):
    """CI and bootstrap p for stat(a) - stat(b) when a and b are disjoint cells."""
    rng = np.random.default_rng(seed)

    def cellmap(rows):
        by_paper: dict[str, list[dict]] = {}
        for r in rows:
            by_paper.setdefault(r["paper"], []).append(r)
        cells: dict[tuple, list[str]] = {}
        for p, items in by_paper.items():
            cells.setdefault((items[0]["cell_mode"], items[0]["labeled"]), []).append(p)
        return by_paper, cells

    pa, ca = cellmap(rows_a)
    pb, cb = cellmap(rows_b)

    def draw(by_paper, cells):
        sample = []
        for papers in cells.values():
            for p in rng.choice(papers, size=len(papers), replace=True):
                sample.extend(by_paper[p])
        return sample

    point = stat(rows_a) - stat(rows_b)
    draws = []
    for _ in range(n_boot):
        try:
            draws.append(stat(draw(pa, ca)) - stat(draw(pb, cb)))
        except (ValueError, ZeroDivisionError):
            draws.append(float("nan"))
    draws = np.array([d for d in draws if np.isfinite(d)], float)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p_two = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "boot_p": float(min(p_two, 1.0))}


def prop_test(a_succ, a_n, b_succ, b_n) -> dict:
    """Fisher exact on a 2x2 of successes, plus both rates."""
    table = [[a_succ, a_n - a_succ], [b_succ, b_n - b_succ]]
    odds, p = fisher_exact(table)
    return {"rate_a": a_succ / a_n if a_n else float("nan"),
            "rate_b": b_succ / b_n if b_n else float("nan"),
            "n_a": a_n, "n_b": b_n, "odds_ratio": float(odds), "p_fisher": float(p)}


def feature_auc(rows: list[dict], feature: str, target: str) -> float:
    """AUC of a continuous item feature for separating target-true from target-false."""
    pos = np.array([float(r[feature]) for r in rows if r[target]], float)
    neg = np.array([float(r[feature]) for r in rows if not r[target]], float)
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
# analysis sections
# --------------------------------------------------------------------------- #
def sec_profile(rows: list[dict]) -> dict:
    scale = {1: "sure human", 2: "most likely human", 3: "can't decide",
             4: "most likely AI", 5: "sure AI"}
    dist = Counter(r["rating"] for r in rows)
    by_group = {}
    for is_ai in (False, True):
        for lab in (False, True):
            sub = [r for r in rows if r["is_ai"] == is_ai and r["labeled"] == lab]
            key = f"{'generated' if is_ai else 'original'}|{'labeled' if lab else 'unlabeled'}"
            by_group[key] = {scale[k]: sum(1 for r in sub if r["rating"] == k)
                             for k in sorted(scale)}
    basis_sets = Counter(tuple(sorted(r["basis"])) for r in rows)
    return {
        "n_items": len(rows),
        "rating_distribution": {scale[k]: dist.get(k, 0) for k in sorted(scale)},
        "rating_distribution_by_group": by_group,
        "undecided": {"n": sum(r["undecided"] for r in rows),
                      "share": float(np.mean([r["undecided"] for r in rows]))},
        "confidence": {"sure (1 or 5)": sum(r["confidence"] == 2 for r in rows),
                       "most likely (2 or 4)": sum(r["confidence"] == 1 for r in rows)},
        "said_ai": {"n": sum(r["said_ai"] for r in rows),
                    "share": float(np.mean([r["said_ai"] for r in rows]))},
        "basis_marginal": dict(Counter(b for r in rows for b in r["basis"]).most_common()),
        "basis_combinations": {"+".join(k) if k else "(none)": v
                               for k, v in basis_sets.most_common()},
        "n_comments": sum(1 for r in rows if r["comment"]),
    }


def sec_overall(rows: list[dict]) -> dict:
    s = summarise(rows)
    n_correct = sum(r["correct"] for r in rows)
    bt = binomtest(n_correct, len(rows), 0.5)
    tp = sum(r["is_ai"] and r["said_ai"] for r in rows)
    fn = sum(r["is_ai"] and not r["said_ai"] for r in rows)
    fp = sum(not r["is_ai"] and r["said_ai"] for r in rows)
    tn = sum(not r["is_ai"] and not r["said_ai"] for r in rows)
    return {
        **s,
        "confusion": {"AI item -> said AI (hit)": tp, "AI item -> said human (miss)": fn,
                      "original -> said AI (false alarm)": fp,
                      "original -> said human (correct rejection)": tn},
        "n_correct": n_correct,
        "accuracy_ci": cluster_bootstrap(rows, lambda rs: float(np.mean([r["correct"] for r in rs]))),
        "accuracy_vs_chance_p": float(bt.pvalue),
        "auc_ci": cluster_bootstrap(rows, auc),
        "d_prime_ci": cluster_bootstrap(rows, lambda rs: sdt(rs)["d_prime"]),
        "criterion_ci": cluster_bootstrap(rows, lambda rs: sdt(rs)["criterion_c"]),
        "cohens_kappa": kappa(rows),
        "mcc": mcc(rows),
        "precision_ai": tp / (tp + fp) if tp + fp else float("nan"),
        "recall_ai": tp / (tp + fn) if tp + fn else float("nan"),
        "f1_ai": 2 * tp / (2 * tp + fp + fn) if tp else float("nan"),
    }


def sec_rq1(rows: list[dict]) -> dict:
    lab = [r for r in rows if r["labeled"]]
    unlab = [r for r in rows if not r["labeled"]]
    out = {"labeled": summarise(lab), "unlabeled": summarise(unlab)}
    for k, sub in (("labeled", lab), ("unlabeled", unlab)):
        out[k]["n_papers"] = len({r["paper"] for r in sub})
    out["diff_auc"] = diff_bootstrap(lab, unlab, auc)
    out["diff_d_prime"] = diff_bootstrap(lab, unlab, lambda rs: sdt(rs)["d_prime"])
    out["diff_accuracy"] = diff_bootstrap(
        lab, unlab, lambda rs: float(np.mean([r["correct"] for r in rs])))
    out["accuracy_fisher"] = prop_test(sum(r["correct"] for r in lab), len(lab),
                                       sum(r["correct"] for r in unlab), len(unlab))
    out["hit_fisher"] = prop_test(
        sum(r["said_ai"] for r in lab if r["is_ai"]), sum(r["is_ai"] for r in lab),
        sum(r["said_ai"] for r in unlab if r["is_ai"]), sum(r["is_ai"] for r in unlab))
    out["fa_fisher"] = prop_test(
        sum(r["said_ai"] for r in lab if not r["is_ai"]), sum(not r["is_ai"] for r in lab),
        sum(r["said_ai"] for r in unlab if not r["is_ai"]), sum(not r["is_ai"] for r in unlab))
    # equivalence framing: how large a sensitivity gain would the data still hide?
    out["note"] = ("The CI on the labeled-minus-unlabeled difference is the informative "
                   "quantity: a null result here is 'no detectable effect at n=60/condition', "
                   "not 'no effect'.")
    return out


def sec_sides(rows: list[dict]) -> dict:
    ai = [r for r in rows if r["is_ai"]]
    hu = [r for r in rows if not r["is_ai"]]
    hit = sum(r["said_ai"] for r in ai)
    cr = sum(not r["said_ai"] for r in hu)
    u = mannwhitneyu([r["rating"] for r in ai], [r["rating"] for r in hu],
                     alternative="two-sided")
    return {
        "generated": {"n": len(ai), "hit_rate": hit / len(ai),
                      "mean_rating": float(np.mean([r["rating"] for r in ai])),
                      "ci": cluster_bootstrap(ai, lambda rs: float(np.mean([r["said_ai"] for r in rs])))},
        "original": {"n": len(hu), "correct_rejection_rate": cr / len(hu),
                     "mean_rating": float(np.mean([r["rating"] for r in hu])),
                     "ci": cluster_bootstrap(hu, lambda rs: float(np.mean([not r["said_ai"] for r in rs])))},
        "asymmetry_fisher": prop_test(hit, len(ai), cr, len(hu)),
        "rating_mannwhitney": {"U": float(u.statistic), "p": float(u.pvalue),
                               "rank_biserial": float(2 * u.statistic / (len(ai) * len(hu)) - 1)},
    }


def sec_rq2(rows: list[dict]) -> dict:
    gen = [r for r in rows if r["is_ai"]]
    orig = [r for r in rows if not r["is_ai"]]
    out = {}
    for mode in ("basic", "cars_guided"):
        g = [r for r in gen if r["cell_mode"] == mode]
        o = [r for r in orig if r["cell_mode"] == mode]
        out[mode] = {
            "n_generated": len(g),
            "hit_rate": float(np.mean([r["said_ai"] for r in g])),
            "mean_rating_generated": float(np.mean([r["rating"] for r in g])),
            "n_original": len(o),
            "fa_rate_paired_originals": float(np.mean([r["said_ai"] for r in o])),
            "mean_rating_original": float(np.mean([r["rating"] for r in o])),
            "d_prime_within_cell": sdt(g + o)["d_prime"],
            "auc_within_cell": auc(g + o),
            "f1_within_cell": f1_ai(g + o),
            "accuracy_within_cell": float(np.mean([r["correct"] for r in g + o])),
        }
    b = [r for r in gen if r["cell_mode"] == "basic"]
    c = [r for r in gen if r["cell_mode"] == "cars_guided"]
    ub = mannwhitneyu([r["rating"] for r in c], [r["rating"] for r in b], alternative="two-sided")
    out["hit_rate_fisher_guided_vs_basic"] = prop_test(
        sum(r["said_ai"] for r in c), len(c), sum(r["said_ai"] for r in b), len(b))
    out["rating_mannwhitney_guided_vs_basic"] = {"U": float(ub.statistic), "p": float(ub.pvalue)}
    out["diff_hit_rate"] = diff_bootstrap(
        c, b, lambda rs: float(np.mean([r["said_ai"] for r in rs])))
    out["fa_rate_fisher_guided_vs_basic"] = prop_test(
        sum(r["said_ai"] for r in orig if r["cell_mode"] == "cars_guided"),
        sum(1 for r in orig if r["cell_mode"] == "cars_guided"),
        sum(r["said_ai"] for r in orig if r["cell_mode"] == "basic"),
        sum(1 for r in orig if r["cell_mode"] == "basic"))
    out["diff_auc_within_cell"] = diff_bootstrap(
        [r for r in rows if r["cell_mode"] == "cars_guided"],
        [r for r in rows if r["cell_mode"] == "basic"], auc)
    return out


def sec_mode_structure(rows: list[dict]) -> dict:
    """Why cars_guided is easier to spot: structural uniformity of the judged items.

    RQ2 says guided generations were caught more often; this measures the
    mechanism on the very same 120 introductions -- how far each group
    collapses onto the canonical Move 1-2-3 funnel and how many distinct move
    structures the group actually contains (Vendi score = effective number of
    distinct collapsed move sequences under the exact-match kernel).
    """
    from graph_analysis import vendi_score

    def canonical(r):
        return r["rec"]["move_seq"] == ["1", "2", "3"]

    def cyclic(r):
        seq = r["rec"]["move_seq"]
        return len(seq) != len(set(seq))

    out = {}
    for is_ai in (False, True):
        for mode in ("basic", "cars_guided"):
            sub = [r for r in rows if r["is_ai"] == is_ai and r["cell_mode"] == mode]
            seqs = [tuple(r["rec"]["move_seq"]) for r in sub]
            out[f"{'generated' if is_ai else 'original'}|{mode}"] = {
                "n": len(sub),
                "pct_canonical_1_2_3": float(np.mean([canonical(r) for r in sub])),
                "pct_move_cyclic": float(np.mean([cyclic(r) for r in sub])),
                "distinct_move_sequences": len(set(seqs)),
                "vendi_move_structures": float(vendi_score(seqs)),
                "mean_repertoire": float(np.mean([r["repertoire"] for r in sub])),
                "mean_move_2_share": float(np.mean([r["move_2_share"] for r in sub])),
                "mean_units": float(np.mean([r["n_units"] for r in sub])),
                "top_move_sequences": [["-".join(s), n] for s, n in Counter(seqs).most_common(3)],
            }
    b = [r for r in rows if r["is_ai"] and r["cell_mode"] == "basic"]
    c = [r for r in rows if r["is_ai"] and r["cell_mode"] == "cars_guided"]
    out["guided_vs_basic_generated"] = {
        "canonical_funnel": prop_test(sum(canonical(r) for r in c), len(c),
                                      sum(canonical(r) for r in b), len(b)),
        "move_cyclic": prop_test(sum(cyclic(r) for r in c), len(c),
                                 sum(cyclic(r) for r in b), len(b)),
    }
    orig = [r for r in rows if not r["is_ai"]]
    out["all_originals"] = {
        "n": len(orig),
        "distinct_move_sequences": len({tuple(r["rec"]["move_seq"]) for r in orig}),
        "pct_canonical_1_2_3": float(np.mean([canonical(r) for r in orig])),
    }
    out["note"] = ("The two modes sit on different papers, so compare generated-with-generated; "
                   "the originals' rows are the paper-set baseline.")
    return out


def sec_cells(rows: list[dict]) -> dict:
    cells = {}
    for mode, lab in CELLS:
        sub = [r for r in rows if r["cell_mode"] == mode and r["labeled"] == (lab == "labeled")]
        cells[f"{mode}|{lab}"] = {**summarise(sub),
                                  "n_papers": len({r['paper'] for r in sub})}
    return cells


def sec_models(rows: list[dict]) -> dict:
    """Ordinal and logistic models with the design interactions, cluster-robust by paper.

    Imports the submodules directly: `statsmodels.api` pulls in the tsa/graphics
    tree, which is broken in this environment.
    """
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    is_ai = np.array([r["is_ai"] for r in rows], float)
    labeled = np.array([r["labeled"] for r in rows], float)
    guided = np.array([r["cell_mode"] == "cars_guided" for r in rows], float)
    said_ai = np.array([r["said_ai"] for r in rows], float)
    rating = np.array([r["rating"] for r in rows], float)
    paper = np.array([r["paper"] for r in rows])
    out = {}

    def named(fit, names, drop_const=True):
        params = np.asarray(fit.params)[:len(names)]
        pvals = np.asarray(fit.pvalues)[:len(names)]
        keep = [(n, c, p) for n, c, p in zip(names, params, pvals)
                if not (drop_const and n == "const")]
        return ({n: float(c) for n, c, _ in keep}, {n: float(p) for n, _, p in keep})

    # (a) logistic: verdict ~ truth * labels. The is_ai:labeled interaction is RQ1.
    names = ["const", "is_ai", "labeled", "is_ai_x_labeled"]
    X = np.column_stack([np.ones(len(rows)), is_ai, labeled, is_ai * labeled])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = Logit(said_ai, X).fit(disp=0, cov_type="cluster",
                                        cov_kwds={"groups": paper})
        coef, p = named(fit, names)
        out["logit_said_ai ~ is_ai * labeled"] = {
            "coef": coef, "p": p,
            "note": "is_ai coefficient = log odds ratio of a correct AI call (sensitivity); "
                    "the interaction is the RQ1 effect of showing move labels.",
        }
    except Exception as exc:                                   # pragma: no cover
        out["logit_said_ai ~ is_ai * labeled"] = {"error": str(exc)}

    # (b) ordinal: the full 5-point rating, keeps the confidence information
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            om = OrderedModel(rating, X[:, 1:], distr="logit").fit(method="bfgs", disp=0)
        coef, p = named(om, names[1:], drop_const=False)
        out["ordinal_rating ~ is_ai * labeled"] = {"coef": coef, "p": p}
    except Exception as exc:                                   # pragma: no cover
        out["ordinal_rating ~ is_ai * labeled"] = {"error": str(exc)}

    # (c) RQ2 on generated items only. cars_guided sits at 30/30 -> the logit
    #     separates; report that and fall back to the ordinal rating model.
    g = is_ai == 1
    gnames = ["const", "guided", "labeled", "guided_x_labeled"]
    Xg = np.column_stack([np.ones(int(g.sum())), guided[g], labeled[g],
                          guided[g] * labeled[g]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitg = Logit(said_ai[g], Xg).fit(disp=0)
        coef, p = named(fitg, gnames)
        out["logit_said_ai ~ guided * labeled (generated only)"] = {"coef": coef, "p": p}
    except Exception as exc:
        out["logit_said_ai ~ guided * labeled (generated only)"] = {
            "error": f"complete separation ({type(exc).__name__}): every cars_guided item "
                     f"was called AI, so the logistic model is undefined -- see the exact "
                     f"(Fisher) test in section 5",
            "detail": str(exc)[:200]}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            omg = OrderedModel(rating[g], Xg[:, 1:], distr="logit").fit(method="bfgs", disp=0)
        coef, p = named(omg, gnames[1:], drop_const=False)
        out["ordinal_rating ~ guided * labeled (generated only)"] = {"coef": coef, "p": p}
    except Exception as exc:                                   # pragma: no cover
        out["ordinal_rating ~ guided * labeled (generated only)"] = {"error": str(exc)}
    return out


def sec_paired(rows: list[dict]) -> dict:
    """Within-paper: the judge sees both intros of a paper (far apart, blind to pairing)."""
    by_paper: dict[str, dict] = {}
    for r in rows:
        by_paper.setdefault(r["paper"], {})["gen" if r["is_ai"] else "orig"] = r
    pairs = [(p, v["gen"], v["orig"]) for p, v in by_paper.items() if len(v) == 2]
    diffs = np.array([g["rating"] - o["rating"] for _, g, o in pairs], float)
    ordered = sum(d > 0 for d in diffs)
    tied = sum(d == 0 for d in diffs)
    reversed_ = sum(d < 0 for d in diffs)
    n_eff = ordered + reversed_
    sign_p = binomtest(ordered, n_eff, 0.5).pvalue if n_eff else float("nan")
    try:
        w = wilcoxon(diffs, zero_method="wilcox")
        wil = {"W": float(w.statistic), "p": float(w.pvalue)}
    except ValueError as exc:
        wil = {"error": str(exc)}
    both = sum(g["correct"] and o["correct"] for _, g, o in pairs)
    one = sum(g["correct"] != o["correct"] for _, g, o in pairs)
    neither = sum(not g["correct"] and not o["correct"] for _, g, o in pairs)
    out = {
        "n_pairs": len(pairs),
        "generated_rated_more_AI": ordered, "tied": tied, "reversed": reversed_,
        "pairwise_auc": float((ordered + 0.5 * tied) / len(pairs)),
        "sign_test_p": float(sign_p),
        "wilcoxon": wil,
        "mean_rating_gap": float(diffs.mean()),
        "pair_outcome": {"both correct": both, "one correct": one, "neither": neither},
    }
    for mode in ("basic", "cars_guided"):
        sub = [(g, o) for _, g, o in pairs if g["cell_mode"] == mode]
        d = np.array([g["rating"] - o["rating"] for g, o in sub], float)
        out[f"pairwise_auc_{mode}"] = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))
    for lab in (True, False):
        sub = [(g, o) for _, g, o in pairs if g["labeled"] == lab]
        d = np.array([g["rating"] - o["rating"] for g, o in sub], float)
        out[f"pairwise_auc_{'labeled' if lab else 'unlabeled'}"] = float(
            ((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))
    return out


def sec_confidence(rows: list[dict]) -> dict:
    out = {}
    for conf, name in ((2, "sure (1/5)"), (1, "most likely (2/4)")):
        sub = [r for r in rows if r["confidence"] == conf]
        out[name] = {"n": len(sub),
                     "accuracy": float(np.mean([r["correct"] for r in sub])) if sub else None,
                     "on_generated": sum(1 for r in sub if r["is_ai"]),
                     "on_original": sum(1 for r in sub if not r["is_ai"])}
    hi = [r for r in rows if r["confidence"] == 2]
    lo = [r for r in rows if r["confidence"] == 1]
    if hi and lo:
        out["calibration_fisher"] = prop_test(sum(r["correct"] for r in hi), len(hi),
                                              sum(r["correct"] for r in lo), len(lo))
    out["note"] = ("The judge almost never used the extreme anchors, so the 5-point scale "
                   "behaved like a 2-point forced choice with a confidence tag.")
    return out


def sec_basis(rows: list[dict]) -> dict:
    out = {"marginal": dict(Counter(b for r in rows for b in r["basis"]).most_common())}
    for name, key, pos, neg in (("by_label_condition", "labeled", "labeled", "unlabeled"),
                                ("by_item_type", "is_ai", "generated", "original"),
                                ("by_verdict", "said_ai", "said AI", "said human")):
        grp = {}
        for val, lbl in ((True, pos), (False, neg)):
            sub = [r for r in rows if r[key] == val]
            grp[lbl] = {"n": len(sub),
                        **{b: sum(1 for r in sub if b in r["basis"])
                           for b in ("argumentation_structure", "writing_style", "guess")}}
        out[name] = grp
    style = [r for r in rows if "writing_style" in r["basis"]]
    arg_only = [r for r in rows if r["basis"] == ["argumentation_structure"]]
    out["accuracy_by_basis"] = {
        "mentions_writing_style": {"n": len(style),
                                   "accuracy": float(np.mean([r["correct"] for r in style]))},
        "argumentation_structure_only": {"n": len(arg_only),
                                         "accuracy": float(np.mean([r["correct"] for r in arg_only]))},
        "fisher": prop_test(sum(r["correct"] for r in style), len(style),
                            sum(r["correct"] for r in arg_only), len(arg_only)),
    }
    out["style_use_by_label_condition_fisher"] = prop_test(
        sum(1 for r in rows if r["labeled"] and "writing_style" in r["basis"]),
        sum(1 for r in rows if r["labeled"]),
        sum(1 for r in rows if not r["labeled"] and "writing_style" in r["basis"]),
        sum(1 for r in rows if not r["labeled"]))
    return out


def sec_basis_subgroup(rows: list[dict]) -> dict:
    """Re-run the core contrasts on the items decided on structure alone.

    IMPORTANT: the decision basis is reported *after* seeing the item, so it is
    a post-treatment variable. Filtering on it breaks the randomisation -- the
    dropped items are not a random subset of the conditions (style is cited
    more when labels are absent and more on items called human), so the
    labeled-vs-unlabeled contrast within the subgroup is no longer a clean
    experimental comparison. Reported as a robustness check, with the dropped
    items characterised so the selection is visible.
    """
    keep = [r for r in rows if r["basis"] == ["argumentation_structure"]]
    drop = [r for r in rows if r["basis"] != ["argumentation_structure"]]

    def block(sub: list[dict]) -> dict:
        s = summarise(sub)
        s["n_correct"] = sum(r["correct"] for r in sub)
        s["n_papers"] = len({r["paper"] for r in sub})
        return s

    lab = [r for r in keep if r["labeled"]]
    unlab = [r for r in keep if not r["labeled"]]
    gen_b = [r for r in keep if r["is_ai"] and r["cell_mode"] == "basic"]
    gen_c = [r for r in keep if r["is_ai"] and r["cell_mode"] == "cars_guided"]

    out = {
        "definition": "basis == ['argumentation_structure'] exactly (writing style not cited)",
        "n_kept": len(keep), "n_dropped": len(drop),
        "overall": block(keep),
        "labeled": block(lab), "unlabeled": block(unlab),
        "diff_auc_labeled_minus_unlabeled": diff_bootstrap(lab, unlab, auc),
        "diff_accuracy_labeled_minus_unlabeled": diff_bootstrap(
            lab, unlab, lambda rs: float(np.mean([r["correct"] for r in rs]))),
        "accuracy_fisher_labeled_vs_unlabeled": prop_test(
            sum(r["correct"] for r in lab), len(lab),
            sum(r["correct"] for r in unlab), len(unlab)),
        "hit_rate_fisher_guided_vs_basic": prop_test(
            sum(r["said_ai"] for r in gen_c), len(gen_c),
            sum(r["said_ai"] for r in gen_b), len(gen_b)),
        "generated_basic": {"n": len(gen_b),
                            "hit_rate": float(np.mean([r["said_ai"] for r in gen_b]))},
        "generated_cars_guided": {"n": len(gen_c),
                                  "hit_rate": float(np.mean([r["said_ai"] for r in gen_c]))},
    }
    # what exactly was removed, and was it removed evenly?
    out["dropped_composition"] = {
        "by_label_condition": dict(Counter("labeled" if r["labeled"] else "unlabeled" for r in drop)),
        "by_item_type": dict(Counter("generated" if r["is_ai"] else "original" for r in drop)),
        "by_verdict": dict(Counter("said AI" if r["said_ai"] else "said human" for r in drop)),
        "n_correct": sum(r["correct"] for r in drop),
        "accuracy": float(np.mean([r["correct"] for r in drop])) if drop else None,
        "selection_fisher_drop_rate_labeled_vs_unlabeled": prop_test(
            sum(1 for r in rows if r["labeled"] and r["basis"] != ["argumentation_structure"]),
            sum(1 for r in rows if r["labeled"]),
            sum(1 for r in rows if not r["labeled"] and r["basis"] != ["argumentation_structure"]),
            sum(1 for r in rows if not r["labeled"])),
    }
    # complementary subgroup, for completeness
    out["style_mentioned_subgroup"] = block(drop) if drop else None
    return out


def sec_time(rows: list[dict]) -> dict:
    t = np.array([r["lead_time"] for r in rows], float)
    if np.isnan(t).any():
        return {"error": "lead_time missing -- re-run convert_labelstudio_export.py"}

    def med(rs):
        return float(np.median([r["lead_time"] for r in rs]))

    out = {"median_s": float(np.median(t)), "iqr_s": [float(np.percentile(t, 25)),
                                                     float(np.percentile(t, 75))],
           "min_s": float(t.min()), "max_s": float(t.max()),
           "total_hours": float(t.sum() / 3600),
           "note": ("lead_time is wall-clock from opening a task to submitting it, so long "
                    "tails are tasks left open rather than time on task; rank statistics and "
                    "medians are used throughout.")}
    for key, pos, neg in (("labeled", "labeled", "unlabeled"), ("is_ai", "generated", "original"),
                          ("correct", "correct", "wrong")):
        a = [r for r in rows if r[key]]
        b = [r for r in rows if not r[key]]
        u = mannwhitneyu([r["lead_time"] for r in a], [r["lead_time"] for r in b],
                         alternative="two-sided")
        out[f"median_by_{key}"] = {pos: med(a), neg: med(b), "n": [len(a), len(b)],
                                   "mannwhitney_p": float(u.pvalue)}
    rho, p = spearmanr([r["position"] for r in rows], [r["lead_time"] for r in rows])
    out["speedup_over_session"] = {"spearman_rho_position_vs_time": float(rho), "p": float(p)}
    out["median_by_quarter_of_session"] = {
        f"items {q * 30 + 1}-{q * 30 + 30}": med([r for r in rows if q * 30 < r["position"] <= (q + 1) * 30])
        for q in range(4)}
    return out


def sec_order(rows: list[dict]) -> dict:
    out = {"accuracy_by_quarter_of_session": {}}
    for q in range(4):
        sub = [r for r in rows if q * 30 < r["position"] <= (q + 1) * 30]
        out["accuracy_by_quarter_of_session"][f"items {q * 30 + 1}-{q * 30 + 30}"] = {
            "n": len(sub), "accuracy": float(np.mean([r["correct"] for r in sub])),
            "auc": auc(sub)}
    first, second = rows[:60], rows[60:]
    out["first_half_vs_second_half"] = prop_test(
        sum(r["correct"] for r in first), len(first),
        sum(r["correct"] for r in second), len(second))
    rho, p = spearmanr([r["position"] for r in rows], [int(r["correct"]) for r in rows])
    out["spearman_position_vs_correct"] = {"rho": float(rho), "p": float(p)}
    out["accuracy_by_design_block"] = {
        str(b): float(np.mean([r["correct"] for r in rows if r["block"] == b]))
        for b in sorted({r["block"] for r in rows})}
    return out


def sec_covariates(rows: list[dict]) -> dict:
    out = {}
    for key in ("quartile", "nativeness"):
        grp = {}
        for val in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == val]
            grp[val] = {"n": len(sub), "accuracy": float(np.mean([r["correct"] for r in sub])),
                        "auc": auc(sub), "f1_ai": f1_ai(sub),
                        "hit_rate": float(np.mean([r["said_ai"] for r in sub if r["is_ai"]])),
                        "fa_rate": float(np.mean([r["said_ai"] for r in sub if not r["is_ai"]]))}
        out[key] = grp
    nat = [r for r in rows if r["nativeness"] == "native"]
    non = [r for r in rows if r["nativeness"] == "non-native"]
    out["nativeness_accuracy_fisher"] = prop_test(
        sum(r["correct"] for r in nat), len(nat), sum(r["correct"] for r in non), len(non))
    # length: is the judge simply calling long intros AI?
    rho_r, p_r = spearmanr([r["word_count"] for r in rows], [r["rating"] for r in rows])
    out["word_count"] = {
        "spearman_with_rating": {"rho": float(rho_r), "p": float(p_r)},
        "auc_word_count_predicts_truth": feature_auc(rows, "word_count", "is_ai"),
        "auc_word_count_predicts_verdict": feature_auc(rows, "word_count", "said_ai"),
        "median_generated": float(np.median([r["word_count"] for r in rows if r["is_ai"]])),
        "median_original": float(np.median([r["word_count"] for r in rows if not r["is_ai"]])),
        "note": "length was matched by design (generator chosen for length fidelity); these "
                "confirm length is not the cue.",
    }
    return out


def sec_structure(rows: list[dict]) -> dict:
    """Do the ratings track the CARS features the judge says they used?

    For every structural feature: how well it separates real provenance (the
    feature's own diagnosticity) versus how well it separates the judge's
    verdict (how much the judge behaved as if using it). The gap between the
    two is the interesting quantity.
    """
    feats = ["n_units", "repertoire", "gap_units", "move_cycles", "word_count",
             "move_1_share", "move_2_share", "move_3_share"]
    out = {"features": {}}
    for f in feats:
        gen = [r[f] for r in rows if r["is_ai"]]
        orig = [r[f] for r in rows if not r["is_ai"]]
        rho, p = spearmanr([r[f] for r in rows], [r["rating"] for r in rows])
        out["features"][f] = {
            "mean_generated": float(np.mean(gen)), "mean_original": float(np.mean(orig)),
            "auc_predicts_truth": feature_auc(rows, f, "is_ai"),
            "auc_predicts_verdict": feature_auc(rows, f, "said_ai"),
            "spearman_with_rating": {"rho": float(rho), "p": float(p)},
        }
    # binary tell: does the intro open by claiming centrality?
    cc = prop_test(sum(r["opens_centrality"] for r in rows if r["is_ai"]),
                   sum(r["is_ai"] for r in rows),
                   sum(r["opens_centrality"] for r in rows if not r["is_ai"]),
                   sum(not r["is_ai"] for r in rows))
    cc_verdict = prop_test(sum(r["opens_centrality"] for r in rows if r["said_ai"]),
                           sum(r["said_ai"] for r in rows),
                           sum(r["opens_centrality"] for r in rows if not r["said_ai"]),
                           sum(not r["said_ai"] for r in rows))
    out["opens_with_claiming_centrality"] = {"generated_vs_original": cc,
                                             "said_AI_vs_said_human": cc_verdict}
    # how far could a purely structural rule have got the judge?
    best = max(out["features"].items(),
               key=lambda kv: abs(kv[1]["auc_predicts_truth"] - 0.5))
    out["most_diagnostic_single_feature"] = {"feature": best[0],
                                             "auc_predicts_truth": best[1]["auc_predicts_truth"]}
    # residual check: among misses, were the structural tells actually absent?
    misses = [r for r in rows if r["is_ai"] and not r["said_ai"]]
    caught = [r for r in rows if r["is_ai"] and r["said_ai"]]
    fas = [r for r in rows if not r["is_ai"] and r["said_ai"]]
    crs = [r for r in rows if not r["is_ai"] and not r["said_ai"]]
    out["missed_generations_vs_caught"] = {
        f: {"missed": float(np.mean([r[f] for r in misses])) if misses else None,
            "caught": float(np.mean([r[f] for r in caught]))}
        for f in feats}
    out["false_alarms_vs_correct_rejections"] = {
        f: {"false_alarm": float(np.mean([r[f] for r in fas])) if fas else None,
            "correct_rejection": float(np.mean([r[f] for r in crs]))}
        for f in feats}
    out["n_missed"] = len(misses)
    out["n_false_alarms"] = len(fas)
    out["structure_only_detector"] = structure_detector(rows, feats)
    return out


def structure_detector(rows: list[dict], feats: list[str]) -> dict:
    """How far do the CARS features alone get you, and how much of the judge do they explain?

    Two logistic models on the same feature set: one predicting the truth (the
    ceiling for a purely structural detector) and one predicting the judge's
    verdict (how much of the judge's behaviour the structure accounts for).
    Both are reported in-sample and leave-one-out cross-validated, since with
    n=120 and 5 predictors the in-sample AUC is optimistic.
    """
    from statsmodels.discrete.discrete_model import Logit

    use = ["move_2_share", "repertoire", "n_units", "word_count"]
    X = np.column_stack([np.ones(len(rows))]
                        + [np.array([float(r[f]) for r in rows]) for f in use]
                        + [np.array([float(r["opens_centrality"]) for r in rows])])
    # standardise the continuous columns so the fit is well conditioned
    for j in range(1, X.shape[1] - 1):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() or 1.0)
    names = use + ["opens_centrality"]

    def run(target: str) -> dict:
        y = np.array([float(r[target]) for r in rows])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = Logit(y, X).fit(disp=0)
            p_in = np.asarray(fit.predict(X))
            p_cv = np.empty(len(rows))
            for i in range(len(rows)):
                m = np.ones(len(rows), bool)
                m[i] = False
                try:
                    p_cv[i] = Logit(y[m], X[m]).fit(disp=0).predict(X[i:i + 1])[0]
                except Exception:                              # pragma: no cover
                    p_cv[i] = y[m].mean()

        def _auc(p):
            pos, neg = p[y == 1], p[y == 0]
            d = pos[:, None] - neg[None, :]
            return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))

        return {"auc_in_sample": _auc(p_in), "auc_loo_cv": _auc(p_cv),
                "accuracy_loo_cv": float(np.mean((p_cv >= 0.5) == (y == 1))),
                "coef": {n: float(c) for n, c in zip(names, np.asarray(fit.params)[1:])},
                "p": {n: float(v) for n, v in zip(names, np.asarray(fit.pvalues)[1:])}}

    out = {"features_used": names,
           "predicting_truth": run("is_ai"),
           "predicting_judge_verdict": run("said_ai")}
    out["note"] = ("predicting_truth is the ceiling for a detector that sees only the "
                   "auto-CARS structure; compare its AUC with the judge's. "
                   "predicting_judge_verdict is how much of the judge's decisions the same "
                   "structure reproduces. Coefficients are on standardised predictors.")
    return out


def sec_similarity(rows: list[dict]) -> dict:
    """Does structural closeness to the paper's own original predict evasion?"""
    by_paper: dict[str, dict] = {}
    for r in rows:
        by_paper.setdefault(r["paper"], {})["gen" if r["is_ai"] else "orig"] = r
    recs = []
    for paper, v in by_paper.items():
        if len(v) != 2:
            continue
        sim = pair_similarity(v["gen"]["rec"], v["orig"]["rec"])
        recs.append({"paper": paper, **sim,
                     "gen_rating": v["gen"]["rating"], "orig_rating": v["orig"]["rating"],
                     "gen_caught": v["gen"]["said_ai"],
                     "mode": v["gen"]["cell_mode"], "labeled": v["gen"]["labeled"],
                     "pair_correct": v["gen"]["correct"] and v["orig"]["correct"]})
    out = {"n_pairs": len(recs)}
    for metric in ("edit_sim_sub", "edit_sim_move", "bag_sim_sub", "bag_sim_move"):
        vals = [r[metric] for r in recs]
        caught = [r[metric] for r in recs if r["gen_caught"]]
        missed = [r[metric] for r in recs if not r["gen_caught"]]
        rho, p = spearmanr(vals, [r["gen_rating"] for r in recs])
        entry = {"mean": float(np.mean(vals)),
                 "mean_when_generation_caught": float(np.mean(caught)),
                 "mean_when_generation_missed": float(np.mean(missed)) if missed else None,
                 "spearman_with_generated_rating": {"rho": float(rho), "p": float(p)}}
        if missed and caught:
            u = mannwhitneyu(missed, caught, alternative="two-sided")
            entry["mannwhitney_missed_vs_caught_p"] = float(u.pvalue)
        out[metric] = entry
    out["mean_edit_sim_sub_by_mode"] = {
        m: float(np.mean([r["edit_sim_sub"] for r in recs if r["mode"] == m]))
        for m in ("basic", "cars_guided")}
    out["note"] = ("Similarity is the paired class-based similarity from "
                   "comparative_analysis/graph_analysis.py (collapsed sub-step edit "
                   "similarity is the locked primary metric).")
    return out


def sec_errors(rows: list[dict]) -> dict:
    errs = [r for r in rows if not r["correct"]]
    return {
        "n_errors": len(errs),
        "by_cell": dict(Counter(
            f"{r['cell_mode']}|{'labeled' if r['labeled'] else 'unlabeled'}" for r in errs)),
        "by_direction": dict(Counter(
            "generated passed as human" if r["is_ai"] else "original wrongly flagged as AI"
            for r in errs)),
        "by_nativeness": dict(Counter(r["nativeness"] for r in errs)),
        "items": [{"item_code": r["item_code"], "position": r["position"],
                   "paper": r["paper"], "intro_id": r["intro_id"],
                   "truth": "generated" if r["is_ai"] else "original",
                   "rating": r["rating"], "basis": r["basis"],
                   "cell": f"{r['cell_mode']}|{'labeled' if r['labeled'] else 'unlabeled'}",
                   "quartile": r["quartile"], "nativeness": r["nativeness"],
                   "word_count": r["word_count"], "n_units": r["n_units"],
                   "repertoire": r["repertoire"], "gap_units": r["gap_units"],
                   "lead_time_s": round(r["lead_time"], 1) if r["lead_time"] else None,
                   "comment": r["comment"]}
                  for r in sorted(errs, key=lambda x: x["position"])],
        "comments": [{"item_code": r["item_code"],
                      "truth": "generated" if r["is_ai"] else "original",
                      "rating": r["rating"], "comment": r["comment"]}
                     for r in rows if r["comment"]],
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def disp(key: str) -> str:
    """Render a compound group key inside a markdown table cell.

    Group keys join their facets with '|' ("generated|cars_guided"), which
    markdown reads as a column separator -- the row then renders one cell wide
    and every value after it shifts left.
    """
    return key.replace("|", " · ")


def check_tables(text: str) -> None:
    """Fail loudly if any markdown table has rows that disagree with its header."""
    def width(line: str) -> int:
        return len(line.strip().strip("|").split("|"))

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|") or i + 1 >= len(lines):
            continue
        if not set(lines[i + 1].replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue                                  # not a header/separator pair
        if not lines[i + 1].startswith("|"):
            continue
        head = width(line)
        for j in range(i + 2, len(lines)):
            if not lines[j].startswith("|"):
                break
            if width(lines[j]) != head:
                raise SystemExit(
                    f"malformed table at report line {j + 1}: {width(lines[j])} cells vs "
                    f"{head} in the header -- an unescaped '|' in a cell?\n  {lines[j]}")


def pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def f3(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_report(res: dict, path: Path) -> None:
    """Reader-facing report.

    Reporting convention: the report shows counts,
    accuracy and F1 for the AI class only. d', criterion, AUC, false-alarm
    rates and bootstrap CIs are still computed and are all in
    human_arm_results.json -- they are simply not printed here.
    """
    L: list[str] = []
    A = L.append
    m = res["meta"]
    o = res["overall"]
    r1 = res["rq1_labels"]
    r2 = res["rq2_mode"]
    ms = res["mode_structure"]
    pr = res["paired"]
    bs = res["basis_subgroup"]
    cv = res["covariates"]
    st = res["structure"]
    sd = st["structure_only_detector"]
    nf = cv["nativeness_accuracy_fisher"]
    t = res["time"]

    A("# Discrimination experiment - human arm results\n")
    A(f"One expert judge, {m['n_items']} items ({m['n_papers']} papers x "
      f"[own original + {m['generator']} generation]), 2x2 paper-level design: "
      "generation mode (basic / cars_guided) x CARS move labels shown (labeled / unlabeled), "
      "15 papers per cell. Rating scale 1 = sure human ... 5 = sure AI.\n")
    A(f"Source export: `{m['export']}`; key: `allocation.json` (seed {m['seed']}). "
      f"All {m['n_items']} rendered item texts were verified against the intro the key names.\n")
    A("**F1 is for the AI class** (positive = the introduction is generated). Signal-detection "
      "quantities (d', criterion, AUC), false-alarm rates and bootstrap confidence intervals "
      "are computed for every split and stored in `human_arm_results.json`; this report keeps "
      "to counts, accuracy and F1.\n")

    # ---------------------------------------------------------------- 1
    A("## 1. Headline\n")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| items | {o['n']} ({m['n_papers']} papers) |")
    A(f"| correct | {o['n_correct']} / {o['n']} |")
    A(f"| accuracy | {pct(o['accuracy'])} |")
    A(f"| **F1 (AI class)** | **{f3(o['f1_ai'])}** |")
    A(f"| precision (AI) | {f3(o['precision_ai'])} |")
    A(f"| recall (AI) | {f3(o['recall_ai'])} |")
    A(f"\nAccuracy vs. chance: binomial p = {o['accuracy_vs_chance_p']:.3g}.\n")
    A("Confusion matrix:\n")
    c = o["confusion"]
    A("| | said AI | said human |")
    A("|---|---|---|")
    A(f"| **generated** | {c['AI item -> said AI (hit)']} | {c['AI item -> said human (miss)']} |")
    A(f"| **original** | {c['original -> said AI (false alarm)']} | "
      f"{c['original -> said human (correct rejection)']} |")
    A("\nThe judge called 60 of the 120 items AI - exactly the true base rate, so the errors "
      "are not a guessing bias in either direction.\n")

    # ---------------------------------------------------------------- 2
    p = res["profile"]
    A("\n## 2. Response profile\n")
    A("| rating | n |")
    A("|---|---|")
    for k, v in p["rating_distribution"].items():
        A(f"| {k} | {v} |")
    A(f"\n\"Can't decide\" was used **{p['undecided']['n']} times** - the judge committed on "
      f"every item. The extreme anchors were used {p['confidence']['sure (1 or 5)']} times; "
      f"{p['confidence']['most likely (2 or 4)']} ratings were the hedged 2/4, so the 5-point "
      f"scale behaved as a forced binary with a confidence tag.\n")
    A("Rating distribution by group:\n")
    A("| group | 1 sure human | 2 most likely human | 3 can't decide | 4 most likely AI | 5 sure AI |")
    A("|---|---|---|---|---|---|")
    for g, d in p["rating_distribution_by_group"].items():
        A(f"| {disp(g)} | " + " | ".join(str(d[k]) for k in
          ["sure human", "most likely human", "can't decide", "most likely AI", "sure AI"]) + " |")
    A("\nDecision basis (multi-select, required):\n")
    A("| basis | n items |")
    A("|---|---|")
    for k, v in p["basis_marginal"].items():
        A(f"| {k} | {v} |")
    A("\nCombinations: " + ", ".join(f"{k} ({v})" for k, v in p["basis_combinations"].items())
      + f". \"Just a guess\" was never selected. Free-text comments: {p['n_comments']}.\n")

    # ---------------------------------------------------------------- 3
    A("\n## 3. RQ1 - do CARS move labels help?\n")
    A("Each paper contributes two items - its own original and its generation - so an item "
      "count is always twice the paper count, half original and half generated.\n")
    A("| condition | papers | items | originals | generated | correct | accuracy | F1 (AI) |")
    A("|---|---|---|---|---|---|---|---|")
    for k in ("labeled", "unlabeled"):
        s = r1[k]
        n_ok = round(s["accuracy"] * s["n"])
        A(f"| {k} | {s['n_papers']} | {s['n']} | {s['n_human']} | {s['n_ai']} | "
          f"{n_ok}/{s['n']} | {pct(s['accuracy'])} | {f3(s['f1_ai'])} |")
    A(f"\nFisher p = {r1['accuracy_fisher']['p_fisher']:.3g}. **Showing the move annotation "
      "changed nothing measurable.**\n")
    A("The two conditions match cell for cell (both 27/30 generated caught and 3/30 originals "
      "wrongly flagged). That is coincidence, not duplicated data: the conditions cover "
      "disjoint papers, and every one of the 120 rendered item texts was checked against the "
      "intro the sealed key names before any statistic was computed.\n")
    A("A null at this sample size is 'no detectable effect at 60 items per condition', not a "
      "proven zero - the interval around the difference (in the JSON) still admits a moderate "
      "effect.\n")

    # ---------------------------------------------------------------- 4
    s = res["sides"]
    A("\n## 4. Generated vs. original items\n")
    A("| item type | items | correctly identified | rate | mean rating |")
    A("|---|---|---|---|---|")
    A(f"| generated | {s['generated']['n']} | "
      f"{round(s['generated']['hit_rate'] * s['generated']['n'])}/{s['generated']['n']} | "
      f"{pct(s['generated']['hit_rate'])} | {s['generated']['mean_rating']:.2f} |")
    A(f"| original | {s['original']['n']} | "
      f"{round(s['original']['correct_rejection_rate'] * s['original']['n'])}/"
      f"{s['original']['n']} | {pct(s['original']['correct_rejection_rate'])} | "
      f"{s['original']['mean_rating']:.2f} |")
    A(f"\nPerfectly symmetric (Fisher p = {s['asymmetry_fisher']['p_fisher']:.3g}): the judge is "
      f"no better at spotting machines than at recognising humans. Rating separation "
      f"Mann-Whitney p = {s['rating_mannwhitney']['p']:.3g}.\n")

    # ---------------------------------------------------------------- 5
    A("\n## 5. RQ2 - does CARS-guided generation evade detection?\n")
    A("| generation mode | generated intros | caught | rate | F1 (AI, whole cell) | accuracy (whole cell) |")
    A("|---|---|---|---|---|---|")
    for mode in ("basic", "cars_guided"):
        e = r2[mode]
        A(f"| {mode} | {e['n_generated']} | "
          f"{round(e['hit_rate'] * e['n_generated'])}/{e['n_generated']} | "
          f"{pct(e['hit_rate'])} | {f3(e['f1_within_cell'])} | {pct(e['accuracy_within_cell'])} |")
    A(f"\nFisher p = {r2['hit_rate_fisher_guided_vs_basic']['p_fisher']:.3g}. The originals "
      f"paired with each mode were misjudged at the same rate (Fisher p = "
      f"{r2['fa_rate_fisher_guided_vs_basic']['p_fisher']:.3g}), so the difference is the "
      f"generation mode, not the papers that landed in each cell.\n")
    A("**The effect runs opposite to the pre-registered RQ2: CARS-guided generations were "
      "*easier* to spot, not harder.**\n")
    A("Why, measured on these same 120 introductions. Only the generated rows have a "
      "generation mode; the originals carry none, and appear as the baseline for the papers "
      "each mode was applied to:\n")
    group_label = {
        "original|basic": "originals of the basic papers",
        "generated|basic": "**generated · basic**",
        "original|cars_guided": "originals of the cars_guided papers",
        "generated|cars_guided": "**generated · cars_guided**",
    }
    A("| group | intros | exactly Move 1-2-3 | returns to an earlier move | distinct move sequences | "
      "effective structures (Vendi) | repertoire | Move-2 share |")
    A("|---|---|---|---|---|---|---|---|")
    for k in ("original|basic", "generated|basic", "original|cars_guided", "generated|cars_guided"):
        e = ms[k]
        A(f"| {group_label[k]} | {e['n']} | {pct(e['pct_canonical_1_2_3'])} | {pct(e['pct_move_cyclic'])} | "
          f"{e['distinct_move_sequences']} | {e['vendi_move_structures']:.2f} | "
          f"{e['mean_repertoire']:.2f} | {e['mean_move_2_share']:.3f} |")
    gv = ms["guided_vs_basic_generated"]
    A(f"\nAmong the generated items only, {pct(gv['canonical_funnel']['rate_a'])} of cars_guided "
      f"intros are exactly the canonical funnel vs {pct(gv['canonical_funnel']['rate_b'])} of basic "
      f"(Fisher p = {gv['canonical_funnel']['p_fisher']:.3g}), and only "
      f"{pct(gv['move_cyclic']['rate_a'])} return to an earlier move vs "
      f"{pct(gv['move_cyclic']['rate_b'])} (p = {gv['move_cyclic']['p_fisher']:.3g}). "
      f"The 30 cars_guided intros use just {ms['generated|cars_guided']['distinct_move_sequences']} "
      f"distinct move sequences against {ms['generated|basic']['distinct_move_sequences']} for basic "
      f"and {ms['all_originals']['distinct_move_sequences']} across the "
      f"{ms['all_originals']['n']} originals - human introductions are nearly all structurally "
      f"unique, and only {pct(ms['all_originals']['pct_canonical_1_2_3'])} of them are the textbook "
      f"funnel.\n")
    A("So the guided prompt does not make the output more human, it makes it more *uniform*: "
      "the generator writes the model answer instead of an introduction, and that regularity is "
      "the tell. " + ms["note"] + " The two originals rows also differ from each other "
      "(Move-2 share 0.069 vs 0.147), which is ordinary paper-set variation - lead with the "
      "generated-vs-generated comparison, not the raw guided-minus-basic difference.\n")

    # ---------------------------------------------------------------- 6
    A("\n## 6. Design cells\n")
    A("The 2x2 cell is a property of the **paper**, not of the introduction: `basic` / "
      "`cars_guided` names the mode the paper's generation was written in, and the paper's own "
      "original sits in the same cell as its paired control. There is no such thing as a "
      "\"basic original\".\n")
    A("15 papers per cell, each contributing its original **and** its generation, so a cell "
      "holds 30 items of which 15 are generated. The 60 generated introductions are therefore "
      "30 basic (15 labeled + 15 unlabeled) and 30 cars_guided - the split section 5 tests.\n")
    A("| paper cell | papers | items | originals | generated | correct | accuracy | F1 (AI) |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in res["cells"].items():
        A(f"| {disp(k)} | {v['n_papers']} | {v['n']} | {v['n_human']} | {v['n_ai']} | "
          f"{round(v['accuracy'] * v['n'])}/{v['n']} | {pct(v['accuracy'])} | {f3(v['f1_ai'])} |")
    A(f"| **total** | {m['n_papers']} | {m['n_items']} | "
      f"{sum(v['n_human'] for v in res['cells'].values())} | "
      f"{sum(v['n_ai'] for v in res['cells'].values())} | "
      f"{o['n_correct']}/{o['n']} | {pct(o['accuracy'])} | {f3(o['f1_ai'])} |")
    A("\nRegression models (cluster-robust by paper where estimable; full output in the JSON):\n")
    for name, mod in res["models"].items():
        if "error" in mod:
            A(f"- `{name}`: {mod['error']}")
        else:
            terms = ", ".join(f"{k} = {v:+.2f} (p = {mod['p'][k]:.3g})"
                              for k, v in mod["coef"].items() if k != "const")
            A(f"- `{name}`: {terms}")

    # ---------------------------------------------------------------- 7
    A("\n## 7. Overall insights\n")
    A(f"- **An informed expert separates the two sources reliably but not perfectly**: "
      f"{o['n_correct']}/{o['n']} correct, F1 {f3(o['f1_ai'])}, and the errors are balanced "
      f"({c['AI item -> said human (miss)']} generated intros passed as human, "
      f"{c['original -> said AI (false alarm)']} real ones were flagged as machine). Even the "
      f"structurally closest generator is not indistinguishable to someone who knows the CARS "
      f"model - but roughly one introduction in ten fools them.")
    A(f"- **CARS guidance backfires.** Guided generations were caught "
      f"{pct(r2['cars_guided']['hit_rate'])} of the time against "
      f"{pct(r2['basic']['hit_rate'])} for the plain prompt "
      f"(p = {r2['hit_rate_fisher_guided_vs_basic']['p_fisher']:.3g}). Giving the model the "
      f"rhetorical schema does not make it write like a human, it makes it write the schema: "
      f"{pct(ms['generated|cars_guided']['pct_canonical_1_2_3'])} of guided intros are exactly "
      f"Move 1-2-3, and 30 of them use only "
      f"{ms['generated|cars_guided']['distinct_move_sequences']} distinct move sequences where "
      f"{ms['all_originals']['n']} human introductions use "
      f"{ms['all_originals']['distinct_move_sequences']}.")
    A("- **Uniformity is the tell, not fidelity.** Structural similarity to the paper's *own* "
      "original was unrelated to detection; what gave the generations away was looking like "
      "each other. Human introductions are almost all structurally unique.")
    A(f"- **Showing the move annotation did not help** ({pct(r1['labeled']['accuracy'])} vs "
      f"{pct(r1['unlabeled']['accuracy'])}, identical cell for cell), and the reason is visible "
      f"in the basis data: the judge already reasoned from move structure on "
      f"{p['basis_marginal']['argumentation_structure']}/{m['n_items']} items, including the "
      f"unlabeled ones. The labels were redundant for an expert, not uninformative in principle "
      f"- and the design has 60 items per condition, so a moderate effect could hide.")
    A("- **The judge really did use structure, but structure is not all of it.** Move-2 (niche) "
      "share, sub-step repertoire and gap count are the features that correlate with the "
      "ratings, and the errors sit exactly where those cues misfire. Yet a detector built on "
      f"those features alone reaches {pct(sd['predicting_truth']['accuracy_loo_cv'])} against "
      f"the judge's {pct(o['accuracy'])}.")
    A(f"- **The characteristic failure is calling a real human introduction machine-written**, "
      f"and it concentrates on native-authored papers ({pct(cv['nativeness']['native']['accuracy'])} "
      f"vs {pct(cv['nativeness']['non-native']['accuracy'])} accuracy, "
      f"p = {nf['p_fisher']:.3g}). Fluent, conventionally-structured human prose reads as AI. "
      f"Twelve native papers only, so this is a lead rather than a result.")
    A(f"- **The task was easy and fast**: median {t['median_s']:.0f} s per item, "
      f"\"can't decide\" never used, and generated items judged *faster* than originals. "
      f"Whatever the cue is, it is available on a first read.")

    # ---------------------------------------------------------------- 8
    A("\n## 8. Suggested framing for the paper\n")
    A("**Lead with the reversal, not the accuracy.** \"An expert detects our AI introductions "
      "90% of the time\" is a weak, expected result and invites the objection that a different "
      "generator or a naive reader would score differently. The finding that carries the paper "
      "is that *prompting with the rhetorical model made the output easier to detect*. That is "
      "counter-intuitive, it is actionable for anyone building CARS-aware generators, and it is "
      "the claim the corpus analysis independently supports.\n")
    A("**Suggested claim ladder** (each step licensed by a different piece of evidence, so the "
      "argument does not rest on the single discrimination result):\n")
    A("1. Corpus level (already in Results): CARS guidance straightens move ordering past the "
      "human rate and widens the sub-step repertoire.")
    A("2. Item level (section 5 here): within the 120 judged introductions, guidance collapses "
      "30 texts onto a handful of move sequences while human introductions are nearly all unique.")
    A("3. Reader level (section 5 here): that uniformity is detectable - guided intros were "
      "caught more often than plain ones by an expert reader.")
    A("4. Mechanism (`human_arm_results.json`, `structure` and `similarity` blocks): the features "
      "that predict the judge's ratings are the same ones that differ between sources, and "
      "similarity to a specific original does not matter.\n")
    A("**Where each result belongs.**\n")
    A("- *Results*: one compact table - condition x (n, accuracy, F1) - plus the structure table "
      "from section 5. Keep d'/AUC out of the paper; they are in the JSON if a reviewer asks.")
    A("- *Results*: the within-paper pairing result "
      f"({pr['generated_rated_more_AI']}/{pr['n_pairs']} papers ordered correctly, `paired` "
      "block in the JSON) is the cleanest single number, because it controls for topic.")
    A("- *Analysis/Discussion*: the RQ1 null, framed as redundancy - an expert who already reads "
      "move structure gains nothing from being shown it. Note this also validates the automatic "
      "annotation as *consistent with* expert reading rather than as a crutch.")
    A("- *Discussion/Limitations*: the native-author false alarms, as a caution about "
      "structure-based AI detection penalising fluent conventional writing. This is the most "
      "quotable secondary finding and connects to the paper's applied-linguistics audience.")
    A("- *Limitations*: one judge, one generator, informed condition. State plainly that this is "
      "an expert upper bound and that the LLM-judge arm supplies the scale.\n")
    A("**Claims to avoid.** Do not write \"AI introductions are detectable\" without the "
      "qualifiers - it is one informed expert, one generator chosen for being the hardest case, "
      "and the reader knew the corpus-level tendencies. Do not read the RQ1 null as \"move "
      "annotations are useless\"; the honest statement is \"no detectable benefit for an expert "
      "at this sample size\". Do not present guided-minus-basic as a clean experimental "
      "contrast without noting it is between papers, and cite the null on the paired originals "
      f"(p = {r2['fa_rate_fisher_guided_vs_basic']['p_fisher']:.3g}) as the control.\n")
    A("**A title-level sentence you could use:** *prompting a language model with the rhetorical "
      "schema of a genre makes its output more schema-conformant than the genre itself, and "
      "therefore easier to identify as machine-written.*\n")

    # ---------------------------------------------------------------- 9
    A("\n## 9. Caveats\n")
    A("- **One judge.** Every rate is a single expert's; there is no inter-rater term and the "
      "per-cell n is 30 items (15 papers). Main effects are well estimated, the three-way "
      "interaction is not.")
    A("- **The judge was informed.** The guidelines carried the corpus-level human-vs-AI move "
      "findings, and those were computed on a corpus that includes these items. This arm "
      "measures an *informed* expert by design, so it is an upper bound on structure-aware "
      "human detection, not a naive-reader baseline.")
    A("- **One generator.** claude-opus-4-8 was selected as the structurally closest model, so "
      "these rates are the hardest case among the four generators, not an average over them.")
    A("- **Mode is between papers.** basic and cars_guided were assigned to different papers, so "
      "the RQ2 contrast carries paper variance; the paired-originals check is the control.")
    A("- **Reporting.** This report shows counts, accuracy and F1 only. d', criterion, AUC, "
      "false-alarm rates, bootstrap CIs and the full regression output are in "
      "`human_arm_results.json`.")

    text = "\n".join(L) + "\n"
    check_tables(text)
    path.write_text(text)


# --------------------------------------------------------------------------- #
def main() -> None:
    rows = load_rows()
    attach_structure(rows)

    res = {
        "meta": {
            "n_items": len(rows),
            "n_papers": len({r["paper"] for r in rows}),
            "generator": GENERATOR,
            "export": "labelstudio_export.json",
            "seed": json.loads((EXP / "allocation.json").read_text())["meta"]["seed"],
            "n_boot": N_BOOT,
            "bootstrap_seed": SEED,
        },
        "profile": sec_profile(rows),
        "overall": sec_overall(rows),
        "rq1_labels": sec_rq1(rows),
        "sides": sec_sides(rows),
        "rq2_mode": sec_rq2(rows),
        "mode_structure": sec_mode_structure(rows),
        "cells": sec_cells(rows),
        "models": sec_models(rows),
        "paired": sec_paired(rows),
        "confidence": sec_confidence(rows),
        "basis": sec_basis(rows),
        "basis_subgroup": sec_basis_subgroup(rows),
        "time": sec_time(rows),
        "order": sec_order(rows),
        "covariates": sec_covariates(rows),
        "structure": sec_structure(rows),
        "similarity": sec_similarity(rows),
        "errors": sec_errors(rows),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "human_arm_results.json").write_text(json.dumps(res, indent=1, default=float))
    write_report(res, OUT / "human_arm_report.md")

    o = res["overall"]
    print(f"{o['n_correct']}/{o['n']} correct ({100 * o['accuracy']:.1f}%), "
          f"F1 (AI) = {o['f1_ai']:.3f}")
    print(f"labeled {100 * res['rq1_labels']['labeled']['accuracy']:.1f}% vs unlabeled "
          f"{100 * res['rq1_labels']['unlabeled']['accuracy']:.1f}%   |   cars_guided caught "
          f"{100 * res['rq2_mode']['cars_guided']['hit_rate']:.0f}% vs basic "
          f"{100 * res['rq2_mode']['basic']['hit_rate']:.0f}%")
    print(f"-> {OUT / 'human_arm_results.json'}")
    print(f"-> {OUT / 'human_arm_report.md'}")


if __name__ == "__main__":
    main()
