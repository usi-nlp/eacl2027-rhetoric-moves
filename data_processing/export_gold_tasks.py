#!/usr/bin/env python3
"""Rebuild the expert gold annotation tasks from the master dataset.

The annotator scripts in `intro_analysis/` (`run_experiment.py`,
`evaluate_and_visualize.py`, `annotate_master_dataset.py`,
`generate_fewshot_rationales.py`) read the expert gold annotations as a Label
Studio task file, `data/annotation/human_intros/tasks.json`. They need it for
two things: the worked examples of the few-shot prompt, and the gold spans the
predictions are scored against.

That file is not shipped separately, because `data/master_dataset.json`
already holds everything in it: every original introduction (`text`) and its
expert gold segments (`annotations.expert`, with the character offsets and the
original label spelling in `label_raw`). This script writes the task file back
out, one task per paper, with the paper id as the task id.

Usage (from the repo root):
    python data_processing/export_gold_tasks.py
    python data_processing/export_gold_tasks.py --annotator expert \
        --output data/annotation/human_intros/tasks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "master_dataset.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "annotation" / "human_intros" / "tasks.json"


def build_tasks(dataset: dict, annotator: str) -> list[dict]:
    """One Label Studio task per paper that `annotator` labelled."""
    tasks = []
    for paper in dataset["papers"]:
        original = next(i for i in paper["intros"] if i["intro_type"] == "original")
        annotation = original["annotations"].get(annotator)
        if annotation is None:
            continue
        text = original["text"]
        results = []
        for n, seg in enumerate(annotation["segments"], start=1):
            results.append({
                "id": f"r{paper['paper_id']}_{n}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
                "value": {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text[seg["start"]:seg["end"]],
                    # `label_raw` keeps the spelling of the source annotation
                    # files, which is what the downstream scripts normalise.
                    "labels": [seg.get("label_raw") or seg["label"]],
                },
            })
        tasks.append({
            "id": paper["paper_id"],
            "data": {"text": text, "source_file": f"{paper['filename']}.xml"},
            "annotations": [{"result": results}],
            "meta": {"paper_key": paper["paper_key"], "title": paper["title"],
                     "annotator": annotator},
        })
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="master dataset JSON (default: data/master_dataset.json)")
    ap.add_argument("--annotator", default="expert",
                    help="annotation key to export (default: expert, the gold annotation)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="where to write the Label Studio task file")
    args = ap.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    tasks = build_tasks(dataset, args.annotator)
    if not tasks:
        ap.error(f"no original introduction carries an annotation named {args.annotator!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    n_segments = sum(len(t["annotations"][0]["result"]) for t in tasks)
    print(f"wrote {len(tasks)} tasks ({n_segments} gold segments) -> {args.output}")


if __name__ == "__main__":
    main()
