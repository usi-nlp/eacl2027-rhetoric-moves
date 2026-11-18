#!/usr/bin/env python3
"""Evaluator-vs-human validation on the double-annotated papers.

Scores, against the expert gold and on identical canonical texts:
  (a) the human second annotators (annotator_1 + annotator_2, one per paper), and
  (b) the automatic evaluator's annotations (`annotations.chatgpt` in
      data/master_dataset.json — the same annotations every downstream
      analysis uses),
with the per-sentence label metrics and boundary metrics of
`evaluate_and_visualize.py` (boundary tolerance ±5).

Produces the numbers for the paper's evaluator-validation table
(tab:evaluator-vs-human) and the per-class appendix table.

Usage:
    python intro_analysis/validate_evaluator.py \
        [--dataset data/master_dataset.json] \
        [--out intro_analysis/results/evaluator_validation_32papers.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluate_and_visualize import (
    evaluate_task,
    aggregate_metrics,
)

REPO = Path(__file__).resolve().parent.parent

# Short class names for display, keyed by normalised (single-underscore) label.
CLASS_SHORT = {
    "move_1_claiming_centrality":                 "M1-CC",
    "move_1_making_topic_generalization":         "M1-MTG",
    "move_1_reviewing_items_of_previous_research": "M1-RR",
    "move_2_continuing_a_tradition":              "M2-CT",
    "move_2_indicating_a_gap":                    "M2-Gap",
    "move_2_question_raising":                    "M2-QR",
    "move_3_announcing_present_research":         "M3-APR",
    "move_3_announcing_principal_findings":       "M3-APF",
    "move_3_indicating_the_structure_of_research": "M3-IS",
    "move_3_outlining_purpose":                   "M3-OP",
}


def _segments(ann: dict) -> list[dict]:
    """Master-dataset annotation block -> [{start, end, label}] (in-scheme only kept
    labelled; out-of-scheme spans keep their raw label so they count as wrong)."""
    return [
        {"start": s["start"], "end": s["end"], "label": s["label"]}
        for s in ann["segments"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "data" / "master_dataset.json"))
    ap.add_argument("--out",
                    default=str(REPO / "intro_analysis" / "results" /
                                "evaluator_validation_32papers.json"))
    ap.add_argument("--tolerance", type=int, default=5)
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text())

    human_results, model_results = [], []
    paper_rows = []
    for paper in data["papers"]:
        orig = next(i for i in paper["intros"] if i["intro_type"] == "original")
        anns = orig["annotations"]
        second = next((a for a in ("annotator_1", "annotator_2") if a in anns), None)
        if second is None:
            continue
        if "chatgpt" not in anns:
            raise SystemExit(f"paper {paper['paper_id']}: no evaluator annotation")
        gold = _segments(anns["expert"])
        base = {"task_id": paper["paper_id"], "text": orig["text"], "gold": gold}
        human_results.append({**base, "ai": _segments(anns[second])})
        model_results.append({**base, "ai": _segments(anns["chatgpt"])})
        paper_rows.append({"paper_id": paper["paper_id"],
                           "quartile": paper["quartile"],
                           "second_annotator": second})

    human_per_task = [evaluate_task(r, args.tolerance) for r in human_results]
    model_per_task = [evaluate_task(r, args.tolerance) for r in model_results]
    human_agg = aggregate_metrics(human_per_task)
    model_agg = aggregate_metrics(model_per_task)

    n_sents = sum(t["label_metrics"]["n_sentences_eval"] for t in human_per_task)
    n_sents_m = sum(t["label_metrics"]["n_sentences_eval"] for t in model_per_task)

    def headline(agg: dict) -> dict:
        return {
            "label_accuracy":  agg["label_accuracy"],
            "label_macro_f1":  agg["label_macro_f1"],
            "move_accuracy":   agg["label_move_accuracy"],
            "boundary_f1":     agg["boundary_f1"]["f1"],
            "boundary_p":      agg["boundary_f1"]["precision"],
            "boundary_r":      agg["boundary_f1"]["recall"],
            "windowdiff":      agg["windowdiff"],
            "pk":              agg["pk"],
            "substep_acc_given_move": agg["label_substep_acc_given_move"],
            "mean_segments":   agg["mean_n_ai"],
        }

    summary = {
        "n_papers": len(paper_rows),
        "papers": paper_rows,
        "n_gold_sentences": {"human_side": n_sents, "model_side": n_sents_m},
        "boundary_tolerance": args.tolerance,
        "evaluator_source": "master_dataset annotations.chatgpt "
                            "(v8 frontrunner, 2026-07-10 batch runs)",
        "human": headline(human_agg),
        "evaluator": headline(model_agg),
        "human_per_class": human_agg["label_per_class"],
        "evaluator_per_class": model_agg["label_per_class"],
        "human_per_move": human_agg["label_per_move"],
        "evaluator_per_move": model_agg["label_per_move"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    # ---- console report ----------------------------------------------------
    print(f"{len(paper_rows)} papers, {n_sents} gold sentences "
          f"(tolerance ±{args.tolerance})\n")
    hdr = f"{'Source':<12} {'Acc':>6} {'MacroF1':>8} {'MoveAcc':>8} {'B-F1':>6} {'Pk':>6} {'Segs':>6}"
    print(hdr)
    for name, agg in (("Human", human_agg), ("Evaluator", model_agg)):
        print(f"{name:<12} {agg['label_accuracy']:>6.3f} {agg['label_macro_f1']:>8.3f} "
              f"{agg['label_move_accuracy']:>8.3f} {agg['boundary_f1']['f1']:>6.3f} "
              f"{agg['pk']:>6.3f} {agg['mean_n_ai']:>6.1f}")

    def _class_lookup(per_class: dict, lbl: str) -> dict | None:
        """Master-dataset labels are canonical `move_2__x`; gold-normalised keys
        are `move_2_x`/`moves_2_x` — accept any spelling."""
        variants = {lbl,
                    lbl.replace("move_2_", "moves_2_"),
                    re.sub(r"^(moves?_\d)_", r"\1__", lbl)}
        for v in variants:
            if v in per_class:
                return per_class[v]
        return None

    print(f"\n{'Class':<8} {'Human F1':>9} {'Eval F1':>9} {'support':>8}")
    for lbl, short in CLASS_SHORT.items():
        h = _class_lookup(human_agg["label_per_class"], lbl)
        m = _class_lookup(model_agg["label_per_class"], lbl)
        if h is None and m is None:
            continue
        print(f"{short:<8} {h['f1'] if h else 0:>9.2f} {m['f1'] if m else 0:>9.2f} "
              f"{(h or m)['support']:>8}")

    print(f"\n{'Move':<8} {'Human F1':>9} {'Eval F1':>9}")
    for mv in ("move_1", "move_2", "move_3"):
        h = human_agg["label_per_move"].get(mv)
        m = model_agg["label_per_move"].get(mv)
        print(f"{mv:<8} {h['f1'] if h else 0:>9.2f} {m['f1'] if m else 0:>9.2f}")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
