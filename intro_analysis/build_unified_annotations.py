#!/usr/bin/env python3
"""
Build a single unified annotation file combining the two second-annotators
(annotator_1 + annotator_2), each segment tagged with the annotator's label, and every
paper keyed to its canonical gold task id (from human_intros/tasks.json).

Mapping strategy (filenames are inconsistent: gold uses `-Swales_Moves.xml`,
annotator 1 uses `-Inter-rater_Agreement.xml`, etc.):
  1. strip the known suffixes from each source_file → title
  2. match the title to the gold paper's title (exact, then fuzzy)
  3. fall back to introduction-text similarity (difflib) when titles disagree

Output: data/annotation/combined_second_annotators.json
  [ { paper_task_id, title, gold_source_file,
      annotators: { <name>: { source_file, text, num_segments, result[] } } } ]
Each result item is the Label Studio span (value:{start,end,text,labels}) plus
`annotator` so the name travels with every annotation.
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/annotation/human_intros/tasks.json"
SOURCES = {
    "annotator_1": ROOT / "data/annotation/human_intros_annotator_1/tasks.json",
    "annotator_2": ROOT / "data/annotation/human_intros_annotator_2/current_final_version.json",
}
OUT = ROOT / "data/annotation/combined_second_annotators.json"

SUFFIXES = ("-Swales_Moves.xml", "-Inter-rater_Agreement.xml",
            "-Inter-rater_agreement.xml", ".xml")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def strip_pilot(s: str) -> str:
    return re.sub(r"^\[pilot\]\s*", "", norm(s))


def title_of(source_file: str) -> str:
    s = source_file or ""
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return norm(s)


def annotated(tasks):
    return [t for t in tasks
            if t.get("annotations") and any(a.get("result") for a in t["annotations"])]


def segments(task):
    segs = []
    for ann in task.get("annotations", []):
        for r in sorted(ann.get("result", []), key=lambda x: x["value"].get("start", 0)):
            v = r["value"]
            if not v.get("labels"):
                continue
            segs.append({
                "start": v.get("start"),
                "end": v.get("end"),
                "text": v.get("text"),
                "labels": v.get("labels"),
            })
    return segs


def main():
    gold = json.load(open(GOLD))
    gold_by_title = {title_of(t["data"]["source_file"]): t for t in gold}
    gold_texts = [(strip_pilot(t["data"]["text"]), t) for t in gold]

    def map_to_gold(task):
        # 1) title match
        gt = gold_by_title.get(title_of(task["data"]["source_file"]))
        if gt is not None:
            return gt
        # 2) fuzzy text similarity
        tn = strip_pilot(task["data"]["text"])[:400]
        best, best_r = None, 0.0
        for gtext, g in gold_texts:
            r = SequenceMatcher(None, tn, gtext[:400]).ratio()
            if r > best_r:
                best, best_r = g, r
        return best if best_r >= 0.85 else None

    unified = {}  # paper_task_id -> record
    for name, path in SOURCES.items():
        for task in annotated(json.load(open(path))):
            g = map_to_gold(task)
            if g is None:
                print(f"  ⚠ [{name}] could not map: {task['data']['source_file']!r}")
                continue
            pid = g["id"]
            rec = unified.setdefault(pid, {
                "paper_task_id": pid,
                "title": title_of(g["data"]["source_file"]),
                "gold_source_file": g["data"]["source_file"],
                "annotators": {},
            })
            segs = segments(task)
            rec["annotators"][name] = {
                "annotator": name,
                "source_file": task["data"]["source_file"],
                "text": task["data"]["text"],
                "num_segments": len(segs),
                "result": [{**s, "annotator": name} for s in segs],
            }

    records = [unified[k] for k in sorted(unified)]
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    both = [r for r in records if len(r["annotators"]) == 2]
    print(f"Wrote {len(records)} papers to {OUT.relative_to(ROOT)}")
    print(f"  papers with both annotators: {len(both)} "
          f"({sorted(r['paper_task_id'] for r in both)})")
    for name in SOURCES:
        ids = sorted(r["paper_task_id"] for r in records if name in r["annotators"])
        print(f"  {name}: {len(ids)} papers -> {ids}")
    print(f"  union task ids ({len(records)}): {sorted(unified)}")


if __name__ == "__main__":
    main()
