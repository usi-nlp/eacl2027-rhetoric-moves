#!/usr/bin/env python3
"""Splice regenerated papers into a canonical generation run (in place).

The master-dataset build keeps only the LATEST run per (model, mode), so a
partial rerun (e.g. the 9 regenerated Qwen3.5-27B cars_guided intros,
2026-07-10) must be patched INTO the original full run dir rather than left as
its own run. For every paper that has a GOOD entry (real prose, not a JSON
artifact) in one of the patch runs, the canonical run's generated_intros.json
entry and texts/{id}.txt are replaced; run_meta.json records the provenance.

Usage:
    python intro_generation/patch_run_papers.py <canonical_run_dir> <patch_run_dir> [...]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def is_broken(text: str) -> bool:
    t = (text or "").lstrip()
    return not t or t.startswith("{") or '"internal_plan"' in t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("canonical", type=Path)
    ap.add_argument("patches", type=Path, nargs="+",
                    help="Patch run dirs, applied in order (later good entries win)")
    args = ap.parse_args()

    canon_meta = json.load(open(args.canonical / "run_meta.json"))
    entries = json.load(open(args.canonical / "generated_intros.json"))
    by_id = {e["paper_id"]: e for e in entries}

    fixes: dict[str, tuple[dict, str]] = {}  # paper_id -> (entry, source run name)
    for run_dir in args.patches:
        meta = json.load(open(run_dir / "run_meta.json"))
        if (meta["model"], meta["mode"]) != (canon_meta["model"], canon_meta["mode"]):
            sys.exit(f"{run_dir.name}: model/mode {(meta['model'], meta['mode'])} "
                     f"!= canonical {(canon_meta['model'], canon_meta['mode'])}")
        for e in json.load(open(run_dir / "generated_intros.json")):
            if not is_broken(e["introduction"]):
                fixes[e["paper_id"]] = (e, run_dir.name)

    patched, still_broken = [], []
    for pid, entry in by_id.items():
        if not is_broken(entry["introduction"]):
            continue
        if pid in fixes:
            new, src = fixes[pid]
            by_id[pid] = {**new, "patched_from": src}
            text_file = args.canonical / "texts" / f"{pid}.txt"
            if (args.canonical / "texts").is_dir():
                text_file.write_text(new["introduction"])
            patched.append(pid)
        else:
            still_broken.append(pid)

    with open(args.canonical / "generated_intros.json", "w") as f:
        json.dump([by_id[e["paper_id"]] for e in entries], f,
                  ensure_ascii=False, indent=2)
    canon_meta.setdefault("patches", []).append({
        "date": datetime.now().isoformat(timespec="seconds"),
        "paper_ids": sorted(patched),
        "from_runs": sorted({src for _, src in fixes.values()}),
        "reason": "regenerated papers whose output was a JSON artifact "
                  "(no introduction emitted)",
    })
    with open(args.canonical / "run_meta.json", "w") as f:
        json.dump(canon_meta, f, ensure_ascii=False, indent=2)

    print(f"Patched {len(patched)} paper(s) into {args.canonical.name}: {sorted(patched)}")
    if still_broken:
        print(f"⚠ still broken (no good replacement found): {sorted(still_broken)}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
