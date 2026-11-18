#!/usr/bin/env python3
"""
Citation-based post-processing of CARS classification output.

The gold annotator's Move-1 labelling follows a near-mechanical surface rule:
a sentence that contains an in-text citation is *reviewing items of previous
research* (RIPR); an uncited general statement is *making a topic
generalization* (MTG); *claiming centrality* (CC) is an explicit value claim
about the topic. The model, reasoning semantically, breaks this rule — it calls
cited sentences CC when they carry evaluative tone, and calls uncited
exposition RIPR.

This module enforces the surface rule deterministically as a post-processing
step on the model's segments (operates on Move 1 only; Moves 2/3 untouched):

  * citation present  + label is CC or MTG  ->  RIPR
  * citation absent    + label is RIPR        ->  MTG

Use as a library (apply_citation_rules) from run_experiment.py, or as a CLI to
re-process an existing run's ai_segments.json into a sibling run dir for
re-evaluation without re-calling the API.

    python intro_analysis/postprocess.py <run_dir> [--out <new_dir>]
"""
import argparse
import json
import re
from pathlib import Path

# 4-digit publication year, optional disambiguation letter (2019a).
_YEAR = r"(?:1[5-9]\d{2}|20\d{2})[a-z]?"

# An in-text citation is present if the text has either:
#   - a parenthetical containing a year:  (Smith, 2019)  (Smith & Jones 2019; ...)
#   - an integral cite:                   Smith (2019)   Smith et al. (2019)
#   - an explicit "et al."
_CITATION_RE = re.compile(
    r"\([^()]*" + _YEAR + r"[^()]*\)"
    + r"|[A-Z][A-Za-z.'’\-]+\s+(?:(?:and|&)\s+[A-Z][A-Za-z.'’\-]+\s+|et\s+al\.?,?\s+)?\(\s*" + _YEAR
    + r"|\bet\s+al\.",
    re.UNICODE,
)


def has_citation(text: str) -> bool:
    """True if `text` contains an in-text academic citation."""
    return bool(_CITATION_RE.search(text or ""))


def map_numeric_labels(segments: list[dict]) -> int:
    """Map v4 numeric class labels (1–11) back to canonical CARS label strings.

    The v4 prompt asks the model to predict an integer per segment. This rewrites
    each segment's `label` to the corresponding `move_*__*` string so the rest of
    the pipeline (citation rules, evaluation) is unchanged. Segments whose label
    is already a label string pass through untouched. Returns the count mapped.
    A label that is out of range / unparseable is left as-is (and flagged).
    """
    from intro_analysis.prompts import CLASS_NUMBER_TO_LABEL
    mapped = 0
    for seg in segments:
        label = seg.get("label")
        if isinstance(label, str) and label.startswith("move"):
            continue  # already a canonical label
        num = None
        if isinstance(label, bool):
            num = None
        elif isinstance(label, int):
            num = label
        elif isinstance(label, str):
            m = re.match(r"\s*(\d{1,2})", label)
            num = int(m.group(1)) if m else None
        if num in CLASS_NUMBER_TO_LABEL:
            seg["label_number"] = num
            seg["label"] = CLASS_NUMBER_TO_LABEL[num]
            mapped += 1
        else:
            seg["label_unmapped"] = label
    return mapped


def _ends_with(label: str, step: str) -> bool:
    return label.endswith(step)


def _ripr_like(label: str) -> str:
    """Return the RIPR label in the same separator style as `label`."""
    sep = "__" if "__" in label else "_"
    return f"move_1{sep}reviewing_items_of_previous_research"


def _mtg_like(label: str) -> str:
    sep = "__" if "__" in label else "_"
    return f"move_1{sep}making_topic_generalization"


def apply_citation_rules(segments: list[dict]) -> int:
    """Relabel Move-1 segments in place by the citation surface rule.

    Returns the number of segments changed. Each changed segment keeps its
    original label under `label_raw` and is flagged `citation_relabeled`.
    """
    changed = 0
    for seg in segments:
        label = seg.get("label")
        if not label or not label.startswith("move_1"):
            continue
        text = seg.get("text", "")
        cite = has_citation(text)
        new = None
        if cite and (_ends_with(label, "claiming_centrality")
                     or _ends_with(label, "making_topic_generalization")):
            new = _ripr_like(label)
        elif not cite and _ends_with(label, "reviewing_items_of_previous_research"):
            new = _mtg_like(label)
        if new and new != label:
            seg["label_raw"] = label
            seg["label"] = new
            seg["citation_relabeled"] = True
            changed += 1
    return changed


def process_run(run_dir: Path, out_dir: Path) -> dict:
    results = json.load(open(run_dir / "ai_segments.json"))
    total = 0
    for r in results:
        total += apply_citation_rules(r.get("ai", []))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ai_segments.json", "results.json"):
        (out_dir / name).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    meta_src = run_dir / "run_meta.json"
    meta = json.load(open(meta_src)) if meta_src.exists() else {}
    meta["citation_postproc"] = True
    meta["citation_postproc_source"] = str(run_dir)
    meta["citation_relabeled_count"] = total
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return {"changed": total, "n_tasks": len(results), "out": str(out_dir)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Run directory containing ai_segments.json")
    ap.add_argument("--out", help="Output directory (default: <run_dir>-citationpp)")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    out_dir = Path(a.out) if a.out else run_dir.with_name(run_dir.name + "-citationpp")
    info = process_run(run_dir, out_dir)
    print(f"Relabeled {info['changed']} Move-1 segment(s) across {info['n_tasks']} task(s)")
    print(f"Wrote {info['out']}")


if __name__ == "__main__":
    main()
