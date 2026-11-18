#!/usr/bin/env python3
"""
Build evaluation runs for the two second human annotators (annotator_1,
annotator_2) scored against the expert gold, in the SAME on-disk format as the
model runs so that
evaluate_and_visualize.py treats them identically.

Each record's `task_id` is the CANONICAL gold task id (not the annotator's own
task id), because
evaluate_and_visualize.remap_results_from_tasks() keys on task_id to load the
gold spans and relocate the annotator's spans in the gold display text.

Reuses the title→gold mapping from build_unified_annotations.py.

Produces three runs under intro_analysis/results/:
  <ts>_annotator_1-vs-gold — annotator 1's papers
  <ts>_annotator_2-vs-gold — annotator 2's papers
  <ts>_human-combined-vs-gold — one human per paper over a chosen task-id set
                                (annotator 1 preferred where both annotated)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from build_unified_annotations import (  # type: ignore
    GOLD, SOURCES, annotated, segments, title_of, strip_pilot,
)
import json as _json
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "intro_analysis/results"


def load_mapping():
    gold = _json.load(open(GOLD))
    gold_by_title = {title_of(t["data"]["source_file"]): t for t in gold}
    gold_texts = [(strip_pilot(t["data"]["text"]), t) for t in gold]

    def map_to_gold(task):
        gt = gold_by_title.get(title_of(task["data"]["source_file"]))
        if gt is not None:
            return gt
        tn = strip_pilot(task["data"]["text"])[:400]
        best, best_r = None, 0.0
        for gtext, g in gold_texts:
            r = SequenceMatcher(None, tn, gtext[:400]).ratio()
            if r > best_r:
                best, best_r = g, r
        return best if best_r >= 0.85 else None

    # annotator -> { gold_id : segments }
    per = {}
    for name, path in SOURCES.items():
        d = {}
        for task in annotated(_json.load(open(path))):
            g = map_to_gold(task)
            if g is None:
                continue
            segs = [{"text": s["text"], "label": s["labels"][0],
                     "start": s["start"], "end": s["end"], "match_ok": True}
                    for s in segments(task) if s.get("labels")]
            d[g["id"]] = {"segs": segs, "source_file": g["data"]["source_file"]}
        per[name] = d
    return per


def write_run(records, tag):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULTS / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, indent=2, ensure_ascii=False)
    for n in ("results.json", "ai_segments.json"):
        (run_dir / n).write_text(payload)
    (run_dir / "run_meta.json").write_text(json.dumps({
        "run_tag": tag, "n_tasks": len(records),
        "task_ids": sorted(r["task_id"] for r in records),
    }, indent=2))
    print(f"  {tag}: {len(records)} papers -> {run_dir.name}")
    return run_dir


def records_for(mapping_d, annotator):
    out = []
    for gid, blk in sorted(mapping_d.items()):
        out.append({
            "task_id": gid,
            "source_file": blk["source_file"],
            "text": "", "gold": [], "ai": blk["segs"],
            "model": annotator,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-task-ids", nargs="+", type=int,
                    help="Restrict the combined human run to these gold task ids "
                         "(e.g. the model run's evaluated papers).")
    a = ap.parse_args()

    per = load_mapping()
    print("Mapped papers:", {k: len(v) for k, v in per.items()})

    write_run(records_for(per["annotator_1"], "annotator_1"), "annotator_1-vs-gold")
    write_run(records_for(per["annotator_2"], "annotator_2"), "annotator_2-vs-gold")

    # Combined: one human per paper, annotator 1 preferred where both annotated.
    combined = {}
    for gid, blk in per["annotator_2"].items():
        combined[gid] = (blk, "annotator_2")
    for gid, blk in per["annotator_1"].items():
        combined[gid] = (blk, "annotator_1")  # annotator 1 overrides on overlap
    if a.combined_task_ids:
        keep = set(a.combined_task_ids)
        combined = {g: v for g, v in combined.items() if g in keep}
    recs = [{"task_id": g, "source_file": blk["source_file"], "text": "",
             "gold": [], "ai": blk["segs"], "model": f"human:{who}"}
            for g, (blk, who) in sorted(combined.items())]
    write_run(recs, "human-combined-vs-gold")


if __name__ == "__main__":
    main()
