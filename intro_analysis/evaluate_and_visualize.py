#!/usr/bin/env python3
"""
Evaluation and Visualization for Intro Segmentation Experiment

Compares human (gold) vs AI boundary placements and, for classification runs,
also evaluates Swales CARS label accuracy.

Metrics:
  1. Boundary F1     — precision/recall/F1 on detected boundary positions
                       with a configurable character tolerance (default ±5;
                       boundaries are sentence-aligned, so jitter is sub-char).
  2. WindowDiff      — standard segmentation metric; lower is better.
  3. Pk              — probabilistic segmentation metric; lower is better.
  4. Segment count   — human vs AI number of segments per introduction.
  5. Coverage        — fraction of the text covered by resolved AI segments.
  6. Label accuracy  — per-sentence CARS label accuracy (classification runs only).
  7. Label macro-F1  — macro-averaged per-class F1, per-sentence (classification only).

Usage:
    python intro_analysis/evaluate_and_visualize.py \
        --results intro_analysis/results/results.json \
        --output-dir intro_analysis/results/
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Label palette — used only for the gold (human) side
# ---------------------------------------------------------------------------

MOVE_COLORS = {
    "move_1_claiming_centrality":                  "#4e79a7",
    "move_1_making_topic_generalization":          "#76b7b2",
    "move_1_reviewing_items_of_previous_research": "#59a14f",
    "moves_2_indicating_a_gap":                    "#e15759",
    "moves_2_question_raising":                    "#ff9da7",
    "moves_2_continuing_a_tradition":              "#f28e2b",
    "move_3_announcing_present_research":          "#b07aa1",
    "move_3_outlining_purpose":                    "#d4a6c8",
    "move_3_announcing_principal_findings":        "#9c755f",
    "move_3_indicating_the_structure_of_research": "#bab0ac",
}

SHORT_LABEL = {
    "move_1_claiming_centrality":                  "M1-CC",
    "move_1_making_topic_generalization":          "M1-MTG",
    "move_1_reviewing_items_of_previous_research": "M1-RR",
    "moves_2_indicating_a_gap":                    "M2-Gap",
    "moves_2_question_raising":                    "M2-QR",
    "moves_2_continuing_a_tradition":              "M2-CT",
    "move_3_announcing_present_research":          "M3-APR",
    "move_3_outlining_purpose":                    "M3-OP",
    "move_3_announcing_principal_findings":        "M3-APF",
    "move_3_indicating_the_structure_of_research": "M3-IS",
}

# Stable 1-based class number per CARS label (order = MOVE_COLORS order).
# Shown as a badge at the end of every segment so gold and AI can be compared
# at a glance even where colours are hard to distinguish.
CLASS_NUMBER = {lbl: i + 1 for i, lbl in enumerate(MOVE_COLORS)}

# Alternating greys for the AI segments (no semantic meaning)
AI_SEGMENT_COLORS = ["#c8d8e8", "#d8e8c8"]

# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------


def gold_boundaries(segments: list[dict]) -> list[int]:
    """Interior boundary positions (end of each segment except the last)."""
    ends = sorted(s["end"] for s in segments if s.get("end") is not None)
    return ends[:-1] if ends else []


def ai_boundaries(segments: list[dict]) -> list[int]:
    ends = sorted(s["end"] for s in segments if s.get("end") is not None)
    return ends[:-1] if ends else []


# ---------------------------------------------------------------------------
# Metric 1: Boundary F1
# ---------------------------------------------------------------------------


def boundary_f1(gold_segs: list[dict], pred_segs: list[dict],
                tolerance: int = 5) -> dict:
    """
    Boundary-level precision / recall / F1 with ±tolerance character window.
    Interior boundaries only (the final segment end is not a boundary).
    """
    gold_b = gold_boundaries(gold_segs)
    pred_b = ai_boundaries(pred_segs)

    matched_gold = set()
    tp = 0
    for pb in pred_b:
        for i, gb in enumerate(gold_b):
            if i in matched_gold:
                continue
            if abs(pb - gb) <= tolerance:
                tp += 1
                matched_gold.add(i)
                break

    fp = len(pred_b) - tp
    fn = len(gold_b) - len(matched_gold)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_gold_boundaries": len(gold_b),
        "n_pred_boundaries": len(pred_b),
        "tolerance_chars": tolerance,
    }


# ---------------------------------------------------------------------------
# Segmentation sequences (sentence-level binary boundary lists)
# ---------------------------------------------------------------------------


# Tokens after which a "." does NOT end a sentence.
_ABBREVIATIONS = {
    "e.g", "i.e", "cf", "etc", "al", "vs", "viz", "fig", "no", "eq", "pp",
    "ed", "eds", "vol", "ch", "sec", "p", "dr", "prof", "mr", "mrs", "ms",
    "st", "approx",
}


def _sentence_ends(text: str) -> list[int]:
    """Rough sentence-end positions (char offset after terminal punctuation).

    Guards against splitting on common abbreviations ("p.", "pp.", "e.g.") and
    single-letter initials. Without this, a reference like
    "(Graham, 2019, p. 1)." gets split at "p." into a dangling "1)." fragment
    that is then scored as its own (mis-labelled) sentence — unfairly penalising
    the model. Paragraph breaks ("\\n\\n") always end a sentence.
    """
    ends = []
    for m in re.finditer(r"[.!?][\"')\]]?\s+|\n\n", text):
        e = m.end()
        if not (0 < e < len(text)):
            continue
        # Abbreviation/initial guard applies only to punctuation-based splits.
        if text[m.start()] in ".!?":
            prev = text[:m.start()]
            last_word = re.split(r"[\s(]", prev)[-1].lower().rstrip(".")
            if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word):
                continue
        ends.append(e)
    ends.append(len(text))
    return ends


def to_boundary_sequence(text: str, segments: list[dict]) -> list[int]:
    """
    Binary list of length (n_sentences - 1).
    boundary_seq[i] == 1 means there is a segment boundary between sentence i
    and sentence i+1, within ±40 chars of that sentence end.
    """
    sent_ends = _sentence_ends(text)
    seg_ends = {s["end"] for s in segments if s.get("end") is not None}

    seq = []
    for end in sent_ends[:-1]:
        b = any(abs(end - se) <= 40 for se in seg_ends)
        seq.append(1 if b else 0)
    return seq


# ---------------------------------------------------------------------------
# Metric 2: WindowDiff  (Pevzner & Hearst 2002)
# ---------------------------------------------------------------------------


def windowdiff(ref: list[int], hyp: list[int], k: int | None = None) -> float:
    """
    WindowDiff — lower is better (0 = perfect agreement).
    k defaults to half the mean reference segment length (in sentences).
    """
    n = len(ref)
    if n == 0:
        return 0.0
    if k is None:
        n_segs = max(1, sum(ref) + 1)
        k = max(1, n // (2 * n_segs))
    k = min(k, n - 1)
    if k == 0:
        return 0.0

    errors = sum(
        1 for i in range(n - k)
        if sum(ref[i:i + k]) != sum(hyp[i:i + k])
    )
    return errors / (n - k)


# ---------------------------------------------------------------------------
# Metric 3: Pk  (Beeferman et al. 1999)
# ---------------------------------------------------------------------------


def pk_score(ref: list[int], hyp: list[int], k: int | None = None) -> float:
    """
    Pk segmentation error metric — lower is better.
    """
    n = len(ref)
    if n == 0:
        return 0.0
    if k is None:
        n_segs = max(1, sum(ref) + 1)
        k = max(1, n // (2 * n_segs))
    k = min(k, n - 1)
    if k == 0:
        return 0.0

    errors = 0
    total = 0
    for i in range(n - k):
        ref_same = sum(ref[i:i + k]) == 0
        hyp_same = sum(hyp[i:i + k]) == 0
        if ref_same != hyp_same:
            errors += 1
        total += 1
    return errors / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate over tasks
# ---------------------------------------------------------------------------


def evaluate_task(result: dict, boundary_tolerance: int = 5) -> dict:
    text      = result["text"]
    gold      = result["gold"]
    pred      = [s for s in result["ai"] if s.get("start") is not None]

    bnd = boundary_f1(gold, pred, tolerance=boundary_tolerance)

    ref_seq = to_boundary_sequence(text, gold)
    hyp_seq = to_boundary_sequence(text, pred)
    min_len = min(len(ref_seq), len(hyp_seq))
    wd = windowdiff(ref_seq[:min_len], hyp_seq[:min_len]) if min_len > 1 else None
    pk = pk_score(ref_seq[:min_len], hyp_seq[:min_len])    if min_len > 1 else None

    covered = sum(s["end"] - s["start"] for s in pred if s.get("end") is not None)
    coverage = covered / len(text) if len(text) > 0 else 0.0

    return {
        "task_id":        result["task_id"],
        "n_gold_segs":    len(gold),
        "n_ai_segs":      len(pred),
        "boundary_f1":    bnd,
        "windowdiff":     wd,
        "pk":             pk,
        "coverage":       coverage,
        "label_metrics":  classification_metrics(text, gold, pred),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _norm_label(label: str) -> str:
    """Normalise AI label (double-underscore) to gold format (single underscore).

    e.g. "move_1__claiming_centrality" → "move_1_claiming_centrality"
    """
    for prefix in ("move_1__", "moves_2__", "move_3__"):
        if label.startswith(prefix):
            return prefix.replace("__", "_") + label[len(prefix):]
    return label


def _move_of(label: str) -> str:
    """Collapse a (normalised) CARS sub-step label to its parent move.

    e.g. "move_1_claiming_centrality" → "move_1";
         "moves_2_indicating_a_gap"   → "move_2" (note the gold "moves_2" prefix
                                                  is normalised to "move_2");
         the uncovered/miss sentinel ("none") is kept as-is so a gold sentence the
         system never covers is still a miss at the move level too.
    """
    if label.startswith("move_1_"):
        return "move_1"
    if label.startswith("moves_2_") or label.startswith("move_2_"):
        return "move_2"
    if label.startswith("move_3_"):
        return "move_3"
    return label  # _NO_LABEL or an out-of-scheme prediction


def _prf_per_class(y_true: list[str], y_pred: list[str]) -> dict:
    """Per-class precision/recall/F1/support over the gold (true) classes only.

    Predictions into a non-true label are penalised as false negatives on the
    true class; they do not appear as their own zero-support class.
    """
    per_class = {}
    for cls in sorted(set(y_true)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": tp + fn}
    return per_class


def _overlap(a: dict, b: dict) -> int:
    """Character overlap (intersection length) between two span dicts."""
    lo = max(a["start"], b["start"])
    hi = min(a["end"],   b["end"])
    return max(0, hi - lo)


def _sentence_spans(text: str) -> list[dict]:
    """Sentence spans [{start, end}] over `text`, using the same rough
    sentence-end detector as the segmentation metrics. Leading whitespace is
    trimmed from each span so that overlap reflects sentence content."""
    ends = _sentence_ends(text)
    spans = []
    start = 0
    for e in ends:
        s = start
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            spans.append({"start": s, "end": e})
        start = e
    return spans


def _label_at(sent: dict, segs: list[dict]) -> str | None:
    """Label of the segment with maximum character overlap with `sent`
    (None if no labelled segment overlaps it)."""
    best_ov, best_lbl = 0, None
    for g in segs:
        if g.get("start") is None or g.get("end") is None or not g.get("label"):
            continue
        ov = _overlap(sent, g)
        if ov > best_ov:
            best_ov, best_lbl = ov, g["label"]
    return best_lbl


# Sentinel for a gold-annotated sentence the system left unlabelled/uncovered.
_NO_LABEL = "none"


def classification_metrics(text: str, gold: list[dict], pred: list[dict]) -> dict | None:
    """
    Per-sentence CARS label metrics for classification runs.

    The introduction is split into sentences; each sentence is assigned the
    label of the gold (resp. predicted) segment it most overlaps, and labels
    are compared sentence-by-sentence. Unlike the earlier span-matched metric,
    this is symmetric to over- AND under-segmentation: when the system merges
    two gold spans the swallowed sentences are still scored against their true
    label, and when it splits one span each piece is scored. A gold sentence the
    system never covers is scored as a miss (`_NO_LABEL`), penalising the recall
    of its true class. Scored over gold-labelled sentences only.

    Returns None if neither gold nor pred carry labels.
    """
    gold_has = any(s.get("label") for s in gold)
    pred_has = any(s.get("label") for s in pred)
    if not gold_has or not pred_has:
        return None

    sents = _sentence_spans(text)
    if not sents:
        return None

    y_true, y_pred = [], []
    for sent in sents:
        g = _label_at(sent, gold)
        if not g:
            continue  # only score sentences the human annotated
        p = _label_at(sent, pred)
        y_true.append(_norm_label(g))
        y_pred.append(_norm_label(p) if p else _NO_LABEL)

    if not y_true:
        return None

    # Macro/per-class over the gold (reference) classes only. Predictions into a
    # non-gold label (the _NO_LABEL miss sentinel, or an out-of-scheme class the
    # model invents) are already penalised as false negatives on the true class,
    # so they must not also appear as their own zero-support class.
    per_class = _prf_per_class(y_true, y_pred)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    n_uncovered = sum(1 for p in y_pred if p == _NO_LABEL)

    # --- Move-level view (collapse sub-steps to their parent move) ----------
    # Lets us separate "found the right move" from "found the right sub-step":
    # move_accuracy >> accuracy means the model places the sentence in the right
    # move but picks the wrong sub-step inside it.
    yt_move = [_move_of(t) for t in y_true]
    yp_move = [_move_of(p) for p in y_pred]
    per_move = _prf_per_class(yt_move, yp_move)
    move_accuracy = sum(1 for t, p in zip(yt_move, yp_move) if t == p) / len(yt_move)
    move_macro_f1 = float(np.mean([v["f1"] for v in per_move.values()])) if per_move else 0.0
    # Conditional: among sentences whose MOVE is correct, how often is the
    # SUB-STEP also correct? This isolates the within-move sub-step difficulty.
    move_ok = [(t, p, tm) for t, p, tm, pm in zip(y_true, y_pred, yt_move, yp_move)
               if tm == pm]
    substep_acc_given_move = (sum(1 for t, p, _ in move_ok if t == p) / len(move_ok)
                              if move_ok else None)

    return {
        "accuracy":            accuracy,
        "macro_f1":            macro_f1,
        "n_sentences_eval":    len(y_true),
        "n_uncovered":         n_uncovered,
        "per_class":           per_class,
        "move_accuracy":       move_accuracy,
        "move_macro_f1":       move_macro_f1,
        "per_move":            per_move,
        "substep_acc_given_move": substep_acc_given_move,
        "y_true":              y_true,
        "y_pred":              y_pred,
    }

def aggregate_metrics(per_task: list[dict]) -> dict:
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    bnd_keys = ["precision", "recall", "f1"]
    bnd_agg = {k: _mean([t["boundary_f1"][k] for t in per_task]) for k in bnd_keys}

    agg: dict = {
        "boundary_f1":  bnd_agg,
        "windowdiff":   _mean([t["windowdiff"] for t in per_task]),
        "pk":           _mean([t["pk"]         for t in per_task]),
        "coverage":     _mean([t["coverage"]   for t in per_task]),
        "mean_n_gold":  _mean([t["n_gold_segs"] for t in per_task]),
        "mean_n_ai":    _mean([t["n_ai_segs"]   for t in per_task]),
    }

    # Aggregate label metrics across tasks (only for classification runs)
    lm_list = [t["label_metrics"] for t in per_task if t.get("label_metrics")]
    if lm_list:
        agg["label_accuracy"] = _mean([m["accuracy"]  for m in lm_list])
        agg["label_macro_f1"] = _mean([m["macro_f1"]  for m in lm_list])

        # Pool all sentence labels for aggregate per-class F1 (gold classes only)
        all_true = [lbl for m in lm_list for lbl in m["y_true"]]
        all_pred = [lbl for m in lm_list for lbl in m["y_pred"]]
        agg["label_per_class"] = _prf_per_class(all_true, all_pred)

        # Move-level aggregates (sub-steps collapsed to parent move).
        agg["label_move_accuracy"] = _mean([m.get("move_accuracy") for m in lm_list])
        agg["label_move_macro_f1"] = _mean([m.get("move_macro_f1") for m in lm_list])
        all_true_move = [_move_of(t) for t in all_true]
        all_pred_move = [_move_of(p) for p in all_pred]
        agg["label_per_move"] = _prf_per_class(all_true_move, all_pred_move)
        # Pooled conditional sub-step accuracy given the move is correct.
        move_ok = [(t, p) for t, p, tm, pm
                   in zip(all_true, all_pred, all_true_move, all_pred_move)
                   if tm == pm]
        agg["label_substep_acc_given_move"] = (
            sum(1 for t, p in move_ok if t == p) / len(move_ok) if move_ok else None)
    else:
        agg["label_accuracy"] = None
        agg["label_macro_f1"] = None
        agg["label_per_class"] = None
        agg["label_move_accuracy"] = None
        agg["label_move_macro_f1"] = None
        agg["label_per_move"] = None
        agg["label_substep_acc_given_move"] = None

    return agg


# ---------------------------------------------------------------------------
# HTML Visualization
# ---------------------------------------------------------------------------

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Intro Segmentation: Human vs AI</title>
<style>
  body { font-family: Georgia, serif; max-width: 1400px; margin: 0 auto; padding: 20px;
         background: #f9f9f9; }
  h1   { color: #222; }
  h2   { color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  .paper-title { font-style: italic; color: #666; margin-bottom: 10px; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.82em; }
  .legend-swatch { width: 16px; height: 16px; border-radius: 3px; display: inline-block; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px;
          background: white; border-radius: 8px; padding: 16px;
          box-shadow: 0 1px 4px rgba(0,0,0,.12); }
  .col-header { font-weight: bold; font-size: 1.1em; margin-bottom: 8px; }
  .text-box { line-height: 1.9; font-size: 0.92em; white-space: normal;
              word-wrap: break-word; }
  .seg { border-radius: 3px; padding: 1px 0; }
  .seg:hover { outline: 2px solid #555; cursor: default; }
  .cls-num { display: inline-block; min-width: 1.1em; text-align: center;
             font-size: 0.7em; font-weight: bold; color: #fff; line-height: 1.3;
             border-radius: 3px; padding: 0 3px; margin: 0 1px; vertical-align: middle;
             font-family: Arial, sans-serif; }
  .metrics-table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }
  .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 5px 10px; }
  .metrics-table th { background: #eee; text-align: left; }
  .note { color: #888; font-size: 0.82em; margin-top: 4px; }
</style>
</head>
<body>
<h1>Intro Analysis: Human vs AI</h1>
<p>Both columns use the <b>same</b> colour scheme: each segment is coloured by its
Swales' CARS move and tagged with a class-number badge (1–10, see legend) at its end.
Compare the badges side by side to see where the AI prediction matches or differs from
the human annotation. Segmentation-only runs (no labels) fall back to alternating grey
blocks numbered by position.</p>
"""

_HTML_FOOT = "</body></html>"


def _hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _merge_adjacent_gaps(segments: list[dict], text: str, max_gap: int = 4) -> list[dict]:
    """Extend each span's end to absorb tiny whitespace-only gaps before the next span."""
    segs = sorted(segments, key=lambda s: s["start"])
    result = []
    for i, seg in enumerate(segs):
        s = dict(seg)
        if i + 1 < len(segs):
            gap_start = s["end"]
            gap_end   = segs[i + 1]["start"]
            if 0 < gap_end - gap_start <= max_gap and text[gap_start:gap_end].strip() == "":
                s["end"] = gap_end
        result.append(s)
    return result


def _annotate(text: str, segments: list[dict]) -> str:
    """Render segments identically for both gold and AI columns.

    Each segment is shown as a translucent fill + solid underline in its CARS
    class colour, followed by a class-number badge (1–10) so the two columns
    can be compared at a glance. Segmentation-only runs (no labels) fall back
    to alternating grey blocks numbered by position.
    """
    import html as hl
    valid = [s for s in segments if s.get("start") is not None and s.get("end") is not None]
    valid = _merge_adjacent_gaps(valid, text)
    valid.sort(key=lambda s: s["start"])

    has_labels = any(s.get("label") for s in valid)

    events = []
    for i, seg in enumerate(valid):
        if has_labels:
            lbl   = _norm_label(seg.get("label") or "unknown")
            color = MOVE_COLORS.get(lbl, "#cccccc")
            short = SHORT_LABEL.get(lbl, lbl)
            num   = CLASS_NUMBER.get(lbl)
            badge_txt = str(num) if num is not None else "?"
            title = f"{lbl} ({short})"
            style = (f"background:{_hex_to_rgba(color, 0.4)}; "
                     f"border-bottom:2px solid {color};")
            badge_color = color
        else:
            color = AI_SEGMENT_COLORS[i % len(AI_SEGMENT_COLORS)]
            badge_txt = str(i + 1)
            title = "segment"
            style = f"background:{color};"
            badge_color = "#888"
        open_html  = f'<span class="seg" style="{style}" title="{hl.escape(title)}">'
        badge_html = f'<span class="cls-num" style="background:{badge_color}">{badge_txt}</span>'
        events.append((seg["start"], "open",  open_html))
        events.append((seg["end"],   "close", badge_html))

    events.sort(key=lambda x: (x[0], 0 if x[1] == "close" else 1))

    parts, prev = [], 0
    for pos, etype, info in events:
        if pos > prev:
            parts.append(hl.escape(text[prev:pos]))
        prev = pos
        if etype == "open":
            parts.append(info)
        else:
            parts.append("</span>")
            parts.append(info)  # class-number badge, just after the segment
    if prev < len(text):
        parts.append(hl.escape(text[prev:]))
    return "".join(parts)


def _metrics_table(task_metrics: dict) -> str:
    bnd = task_metrics["boundary_f1"]
    rows = [
        ("Gold segments",        task_metrics["n_gold_segs"]),
        ("AI segments",          task_metrics["n_ai_segs"]),
        ("Boundary Precision",   f"{bnd['precision']:.3f}"),
        ("Boundary Recall",      f"{bnd['recall']:.3f}"),
        ("Boundary F1",          f"{bnd['f1']:.3f}"),
        ("WindowDiff ↓",         f"{task_metrics['windowdiff']:.3f}"
                                 if task_metrics["windowdiff"] is not None else "N/A"),
        ("Pk ↓",                 f"{task_metrics['pk']:.3f}"
                                 if task_metrics["pk"] is not None else "N/A"),
        ("AI text coverage",     f"{task_metrics['coverage']:.1%}"),
    ]
    lm = task_metrics.get("label_metrics")
    if lm:
        rows += [
            ("── Label metrics ──",   "──"),
            ("Label accuracy",        f"{lm['accuracy']:.3f}"),
            ("Label macro-F1",        f"{lm['macro_f1']:.3f}"),
            ("Sentences scored",      lm["n_sentences_eval"]),
            ("Uncovered by AI",       lm["n_uncovered"]),
        ]
    html = ['<table class="metrics-table"><tr><th>Metric</th><th>Value</th></tr>']
    for name, val in rows:
        html.append(f"<tr><td>{name}</td><td>{val}</td></tr>")
    html.append("</table>")
    tol = task_metrics["boundary_f1"].get("tolerance_chars", "?")
    html.append(f'<p class="note">Boundary tolerance: ±{tol} chars. '
                'WindowDiff and Pk: lower = better agreement. '
                'Label metrics: per-sentence (each sentence scored by its '
                'max-overlap gold/predicted label).</p>')
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Tasks-based text & gold remapping
# ---------------------------------------------------------------------------


def _normalize_with_map(text: str):
    """Return (norm_text, fwd_map) where fwd_map[i] = original char index for norm_text[i]."""
    norm_chars = []
    fwd_map    = []
    in_space   = False
    for i, c in enumerate(text):
        if c in " \t\n\r":
            if not in_space:
                norm_chars.append(" ")
                fwd_map.append(i)
            in_space = True
        else:
            norm_chars.append(c)
            fwd_map.append(i)
            in_space = False
    # Strip leading/trailing space to match " ".join(text.split())
    norm = "".join(norm_chars)
    lstrip = len(norm) - len(norm.lstrip())
    rstrip = len(norm.rstrip())
    norm   = norm.strip()
    fwd_map = fwd_map[lstrip: rstrip + 1]
    return norm, fwd_map


def _gold_from_task(task: dict, disp_text: str) -> list[dict]:
    """Extract gold spans from a tasks.json task, mapped into disp_text (raw.replace('\\n',' '))."""
    segs = []
    for ann in task.get("annotations", []):
        for r in sorted(ann.get("result", []), key=lambda x: x["value"]["start"]):
            v   = r["value"]
            s   = v["start"]
            e   = v["end"]
            lbl = v["labels"][0] if v.get("labels") else "unknown"
            segs.append({"start": s, "end": e, "text": disp_text[s:e], "label": lbl})
    return sorted(segs, key=lambda x: x["start"])


def _remap_ai_to_disp(ai_segs: list[dict], disp_text: str) -> list[dict]:
    """Re-locate AI segments (found in clean_text) inside disp_text via normalized search."""
    norm, fwd = _normalize_with_map(disp_text)
    remapped   = []
    search_from = 0
    for seg in ai_segs:
        norm_seg = " ".join(seg["text"].split())
        idx = norm.find(norm_seg, search_from)
        if idx == -1:
            idx = norm.find(norm_seg)          # retry from start
        if idx == -1:
            idx = norm.find(norm_seg[:80], search_from)  # prefix fallback
        if idx == -1:
            remapped.append(dict(seg, start=None, end=None, match_ok=False))
            continue
        end = idx + len(norm_seg)
        orig_start = fwd[idx]
        orig_end   = fwd[end - 1] + 1 if end - 1 < len(fwd) else fwd[-1] + 1
        remapped.append(dict(seg, start=orig_start, end=orig_end, match_ok=True))
        search_from = end
    return remapped


def remap_results_from_tasks(results: list[dict], tasks_path: str) -> list[dict]:
    """
    Replace each result's text and gold with values derived from tasks.json.
    Text = raw.replace('\\n', ' ')   (preserves original char offsets).
    Gold = annotations loaded directly from tasks.json (no re-search needed).
    AI   = spans re-located in the new display text via normalized search.
    """
    with open(tasks_path) as f:
        tasks = json.load(f)
    tasks_by_id = {t["id"]: t for t in tasks}

    updated = []
    for r in results:
        tid  = r["task_id"]
        task = tasks_by_id.get(tid)
        if task is None:
            updated.append(r)
            continue
        raw_text  = task["data"]["text"]
        disp_text = raw_text.replace("\n", " ")
        gold      = _gold_from_task(task, disp_text)
        ai        = _remap_ai_to_disp(r.get("ai", []), disp_text)
        updated.append(dict(r, text=disp_text, gold=gold, ai=ai))
    return updated


def build_html(results: list[dict], per_task_metrics: list[dict]) -> str:
    parts = [_HTML_HEAD]

    # Legend — class number badge + colour + label (shared by both columns)
    parts.append('<div class="legend">')
    for lbl, color in MOVE_COLORS.items():
        short = SHORT_LABEL.get(lbl, lbl)
        num   = CLASS_NUMBER.get(lbl, "?")
        parts.append(
            f'<div class="legend-item">'
            f'<span class="cls-num" style="background:{color}">{num}</span>'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f'<span>{short}: {lbl}</span></div>'
        )
    parts.append("</div>")

    metrics_by_id = {m["task_id"]: m for m in per_task_metrics}

    for result in results:
        tid  = result["task_id"]
        text = result["text"]
        gold = result["gold"]
        ai   = result["ai"]
        m    = metrics_by_id.get(tid, {})

        unmatched = [s for s in ai if not s.get("match_ok", True)]

        parts.append(f'<h2>Paper {tid}</h2>')
        parts.append(f'<p class="paper-title">{result["source_file"]}</p>')
        parts.append(_metrics_table(m))

        parts.append('<div class="pair">')

        parts.append('<div>')
        parts.append('<div class="col-header">Human annotation (move labels)</div>')
        parts.append(f'<div class="text-box">{_annotate(text, gold)}</div>')
        parts.append('</div>')

        parts.append('<div>')
        ai_valid = [s for s in ai if s.get("start") is not None]
        n_ai = len(ai_valid)
        ai_labeled = any(s.get("label") for s in ai_valid)
        ai_header = ("AI prediction (move labels)" if ai_labeled
                     else "AI segmentation (boundaries only)")
        parts.append(f'<div class="col-header">{ai_header} ({n_ai} segments)</div>')
        if unmatched:
            parts.append(
                f'<p style="color:red;font-size:0.85em">⚠ {len(unmatched)} '
                f'segment(s) could not be located in the text.</p>'
            )
        parts.append(f'<div class="text-box">{_annotate(text, ai)}</div>')
        parts.append('</div>')

        parts.append('</div>')

    parts.append(_HTML_FOOT)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Matplotlib figures
# ---------------------------------------------------------------------------


def plot_segment_counts(results: list[dict], per_task: list[dict], out_path: Path):
    """Bar chart of human vs AI segment counts per paper."""
    task_ids    = [m["task_id"]    for m in per_task]
    gold_counts = [m["n_gold_segs"] for m in per_task]
    ai_counts   = [m["n_ai_segs"]  for m in per_task]

    x = np.arange(len(task_ids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(task_ids) * 1.1), 5))
    ax.bar(x - width / 2, gold_counts, width, label="Human",   color="#4e79a7", alpha=0.85)
    ax.bar(x + width / 2, ai_counts,   width, label="AI (GPT)", color="#f28e2b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in task_ids])
    ax.set_ylabel("Number of segments")
    ax.set_title("Segment count: Human vs AI")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_boundary_metrics(per_task: list[dict], out_path: Path):
    """Per-task boundary precision, recall, F1."""
    task_ids = [m["task_id"] for m in per_task]
    prec = [m["boundary_f1"]["precision"] for m in per_task]
    rec  = [m["boundary_f1"]["recall"]    for m in per_task]
    f1   = [m["boundary_f1"]["f1"]        for m in per_task]

    x = np.arange(len(task_ids))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(8, len(task_ids) * 1.1), 5))
    ax.bar(x - width, prec, width, label="Precision", color="#59a14f", alpha=0.85)
    ax.bar(x,          rec,  width, label="Recall",    color="#4e79a7", alpha=0.85)
    ax.bar(x + width,  f1,   width, label="F1",        color="#e15759", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in task_ids])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    _tol = per_task[0]["boundary_f1"].get("tolerance_chars", "?") if per_task else "?"
    ax.set_title(f"Boundary detection metrics per paper (tolerance ±{_tol} chars)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_span_map(result: dict, out_path: Path):
    """Horizontal bar showing human and AI segment spans for one paper."""
    text = result["text"]
    gold = result["gold"]
    ai   = [s for s in result["ai"] if s.get("start") is not None]
    text_len = len(text)

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

    # Human (colour-coded by label; labelled with the CARS class number)
    for seg in gold:
        s, e = seg["start"], seg["end"]
        lbl  = _norm_label(seg.get("label", "unknown"))
        color = MOVE_COLORS.get(lbl, "#ccc")
        axes[0].barh(0, e - s, left=s, height=0.5, color=color,
                     alpha=0.85, edgecolor="white", linewidth=0.5)
        num = CLASS_NUMBER.get(lbl)
        if num is not None and (e - s) > text_len * 0.02:
            axes[0].text((s + e) / 2, 0, str(num),
                         ha="center", va="center", fontsize=7, color="white",
                         fontweight="bold")

    # AI (same colour + class-number scheme as the human side)
    ai_has_labels = any(s.get("label") for s in ai)
    for i, seg in enumerate(ai):
        s, e = seg["start"], seg["end"]
        if ai_has_labels:
            lbl = _norm_label(seg.get("label") or "unknown")
            color = MOVE_COLORS.get(lbl, "#cccccc")
            num = CLASS_NUMBER.get(lbl)
            txt = str(num) if num is not None else "?"
            txt_color = "white"
        else:
            color = AI_SEGMENT_COLORS[i % len(AI_SEGMENT_COLORS)]
            txt = str(i + 1)
            txt_color = "#333"
        axes[1].barh(0, e - s, left=s, height=0.5, color=color,
                     alpha=0.9, edgecolor="#999", linewidth=0.5)
        if (e - s) > text_len * 0.02:
            axes[1].text((s + e) / 2, 0, txt,
                         ha="center", va="center", fontsize=7, color=txt_color,
                         fontweight="bold")

    ai_side_label = "AI (CARS labels)" if ai_has_labels else "AI (boundaries only)"
    for ax, label in zip(axes, ["Human (CARS labels)", ai_side_label]):
        ax.set_xlim(0, text_len)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=55, va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].set_xlabel("Character position")
    fig.suptitle(f"Segment map — Task {result['task_id']}", y=1.01)

    # Legend (shared) — number. short-label
    handles = [mpatches.Patch(color=MOVE_COLORS[l],
                              label=f"{CLASS_NUMBER[l]}. {SHORT_LABEL[l]}")
               for l in MOVE_COLORS]
    fig.legend(handles=handles, loc="upper right", fontsize=6.5,
               bbox_to_anchor=(1.0, 1.0), ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_windowdiff_pk(per_task: list[dict], out_path: Path):
    """WindowDiff and Pk per task."""
    task_ids = [m["task_id"] for m in per_task]
    wd = [m["windowdiff"] if m["windowdiff"] is not None else 0 for m in per_task]
    pk = [m["pk"]         if m["pk"]         is not None else 0 for m in per_task]

    x = np.arange(len(task_ids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(task_ids) * 1.1), 5))
    ax.bar(x - width / 2, wd, width, label="WindowDiff", color="#b07aa1", alpha=0.85)
    ax.bar(x + width / 2, pk, width, label="Pk",          color="#9c755f", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in task_ids])
    ax.set_ylabel("Error (lower = better)")
    ax.set_title("WindowDiff and Pk per paper (lower = closer to human segmentation)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_label_metrics(aggregated: dict, per_task: list[dict], out_path: Path):
    """Per-class F1 bar chart + per-task accuracy/macro-F1 (classification runs only)."""
    per_cls = aggregated.get("label_per_class")
    lm_list = [t["label_metrics"] for t in per_task if t.get("label_metrics")]
    if not per_cls or not lm_list:
        return

    fig, (ax_cls, ax_task) = plt.subplots(1, 2, figsize=(16, 5))

    # ── Left: per-class F1 ──────────────────────────────────────────
    labels   = [SHORT_LABEL.get(c, c) for c in per_cls]
    f1_vals  = [per_cls[c]["f1"]      for c in per_cls]
    supports = [per_cls[c]["support"] for c in per_cls]
    colors   = [MOVE_COLORS.get(c, "#aaaaaa") for c in per_cls]

    x = np.arange(len(labels))
    bars = ax_cls.bar(x, f1_vals, color=colors, alpha=0.85)
    for bar, sup in zip(bars, supports):
        ax_cls.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={sup}", ha="center", va="bottom", fontsize=7)
    ax_cls.set_xticks(x)
    ax_cls.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax_cls.set_ylim(0, 1.15)
    ax_cls.set_ylabel("F1")
    ax_cls.set_title("Per-class label F1 (per-sentence, pooled across tasks)")
    ax_cls.axhline(aggregated["label_macro_f1"], color="black", linewidth=1,
                   linestyle="--", label=f"macro-F1 = {aggregated['label_macro_f1']:.3f}")
    ax_cls.legend(fontsize=8)
    ax_cls.grid(axis="y", alpha=0.3)

    # ── Right: per-task accuracy and macro-F1 ───────────────────────
    task_ids   = [t["task_id"]                 for t in per_task if t.get("label_metrics")]
    acc_vals   = [t["label_metrics"]["accuracy"]  for t in per_task if t.get("label_metrics")]
    mf1_vals   = [t["label_metrics"]["macro_f1"]  for t in per_task if t.get("label_metrics")]
    matched    = [t["label_metrics"]["n_sentences_eval"] for t in per_task if t.get("label_metrics")]

    x2 = np.arange(len(task_ids))
    w  = 0.38
    b1 = ax_task.bar(x2 - w / 2, acc_vals, w, label="Accuracy",  color="#4e79a7", alpha=0.85)
    b2 = ax_task.bar(x2 + w / 2, mf1_vals, w, label="Macro-F1",  color="#f28e2b", alpha=0.85)
    for bar, n in zip(list(b1) + list(b2),
                      matched + matched):
        ax_task.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"n={n}", ha="center", va="bottom", fontsize=7)
    ax_task.set_xticks(x2)
    ax_task.set_xticklabels([f"T{t}" for t in task_ids])
    ax_task.set_ylim(0, 1.15)
    ax_task.set_ylabel("Score")
    ax_task.set_title("Label accuracy and macro-F1 per task\n(per-sentence)")
    ax_task.legend()
    ax_task.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")





def latest_run_dir(base: Path) -> Path:
    """Return the most recently created timestamped subfolder inside base."""
    candidates = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No run folders found inside {base}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="intro_analysis/results",
        help="Base results directory; defaults to latest timestamped subfolder",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Specific run folder name (e.g. 2026-05-15_12-34-56); defaults to latest",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=30,
        help="Character window for boundary matching (default: 30)",
    )
    parser.add_argument(
        "--tasks-path",
        default="data/annotation/human_intros/tasks.json",
        help="Path to Label Studio tasks.json; used to load display text and gold spans "
             "directly from source (preserves all annotations, removes only newlines). "
             "Pass 'none' to skip remapping and use whatever is stored in results.json.",
    )
    args = parser.parse_args()

    base = Path(args.results_dir)
    if args.run:
        out_dir = base / args.run
    else:
        out_dir = latest_run_dir(base)

    results_file = out_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.json found in {out_dir}")

    meta_file = out_dir / "run_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    variant = meta.get("prompt_variant", "unknown")
    model   = meta.get("model", "unknown")

    print(f"Using run: {out_dir}")
    print(f"  model={model}  prompt_variant={variant}")

    with open(results_file) as f:
        results = json.load(f)

    # Remap text/gold/ai from tasks.json (uses raw text with newlines → spaces,
    # so all human-annotated spans map directly without re-search).
    tasks_path = args.tasks_path
    if tasks_path.lower() != "none" and Path(tasks_path).exists():
        print(f"Remapping from tasks.json: {tasks_path}")
        results = remap_results_from_tasks(results, tasks_path)
    else:
        print("Skipping tasks remapping (--tasks-path none or file not found)")

    print(f"Evaluating {len(results)} result(s)…")

    per_task   = [evaluate_task(r, args.boundary_tolerance) for r in results]
    aggregated = aggregate_metrics(per_task)

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"per_task": per_task, "aggregate": aggregated}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Print summary
    print("\n── Aggregate Metrics ───────────────────────────────")
    bnd = aggregated["boundary_f1"]
    print(f"Boundary Precision   : {bnd['precision']:.3f}")
    print(f"Boundary Recall      : {bnd['recall']:.3f}")
    print(f"Boundary F1          : {bnd['f1']:.3f}")
    wd = aggregated["windowdiff"]
    pk = aggregated["pk"]
    print(f"WindowDiff (↓)       : {wd:.3f}" if wd is not None else "WindowDiff: N/A")
    print(f"Pk         (↓)       : {pk:.3f}" if pk is not None else "Pk: N/A")
    print(f"AI coverage          : {aggregated['coverage']:.1%}")
    print(f"Mean gold segments   : {aggregated['mean_n_gold']:.1f}")
    print(f"Mean AI segments     : {aggregated['mean_n_ai']:.1f}")
    if aggregated.get("label_accuracy") is not None:
        print(f"\n── Label Metrics (per-sentence) ────────────────────")
        print(f"Label accuracy       : {aggregated['label_accuracy']:.3f}")
        print(f"Label macro-F1       : {aggregated['label_macro_f1']:.3f}")
        if aggregated.get("label_per_class"):
            print("\nPer-class F1:")
            for cls, m in sorted(aggregated["label_per_class"].items()):
                short = SHORT_LABEL.get(cls, cls)
                print(f"  {short:<12}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
                      f"F1={m['f1']:.2f}  (n={m['support']})")

    # HTML
    html_path = out_dir / "visualization.html"
    html_path.write_text(build_html(results, per_task), encoding="utf-8")
    print(f"\nHTML visualization saved to {html_path}")

    # Plots
    plot_segment_counts(results, per_task, out_dir / "segment_counts.png")
    plot_boundary_metrics(per_task, out_dir / "boundary_metrics.png")
    plot_windowdiff_pk(per_task, out_dir / "windowdiff_pk.png")
    plot_label_metrics(aggregated, per_task, out_dir / "label_metrics.png")
    for result in results:
        plot_span_map(result, out_dir / f"span_map_task{result['task_id']}.png")


if __name__ == "__main__":
    main()
