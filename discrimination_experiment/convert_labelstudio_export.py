#!/usr/bin/env python3
"""Convert Label Studio JSON exports into
discrimination_experiment/responses.json — the same blinded shape as the other
media's exports, so the SDT analysis is agnostic to the medium.

Feed the annotator's exported files (Export → JSON, the full format; they send
a fresh export of the whole project every ~30 items plus a final one — feed
them all, newest submitted_at wins per item). Join-free: never reads
allocation.json.

Usage:
    python discrimination_experiment/convert_labelstudio_export.py returned/*.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

BASIS_MAP = {
    "Argumentation / Swales CARS structure": "argumentation_structure",
    "Writing style": "writing_style",
    "Just a guess": "guess",
}


def parse_annotation(annotation: dict) -> dict:
    provenance = None
    basis = None
    comment = None
    for result in annotation.get("result", []):
        name = result.get("from_name")
        value = result.get("value", {})
        if name == "provenance" and value.get("choices"):
            provenance = int(value["choices"][0].split("—")[0].strip())
        elif name == "basis" and value.get("choices"):
            basis = [BASIS_MAP.get(c, c) for c in value["choices"]]
        elif name == "comment" and value.get("text"):
            comment = "\n".join(t for t in value["text"] if t and t.strip()) or None
    return {
        "provenance": provenance,
        "basis": basis,
        "comment": comment,
        # Label Studio's own time-on-task (seconds from opening the task to
        # submitting). Annotator-side telemetry, never item-key information.
        "lead_time": annotation.get("lead_time"),
        "submitted_at": annotation.get("updated_at") or annotation.get("created_at"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Label Studio JSON export files")
    parser.add_argument("--out", default=str(EXP_DIR / "responses.json"))
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    for path in args.files:
        with open(path) as f:
            tasks = json.load(f)
        for task in tasks:
            data = task.get("data", {})
            item_code = data.get("item_code")
            if not item_code:
                raise SystemExit(f"{path}: task without item_code — wrong file?")
            annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
            for annotation in annotations:
                parsed = parse_annotation(annotation)
                if parsed["provenance"] is None:
                    continue
                entry = {
                    "position": data.get("position"),
                    "label_condition": data.get("label_condition"),
                    **parsed,
                }
                old = merged.get(item_code)
                if old and (old.get("submitted_at") or "") >= (entry.get("submitted_at") or ""):
                    continue
                merged[item_code] = entry

    out = {
        "meta": {
            "exported": datetime.now().isoformat(timespec="seconds"),
            "source_files": [Path(p).name for p in args.files],
            "medium": "labelstudio",
            "n_answered": len(merged),
        },
        "responses": dict(sorted(merged.items())),
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(merged)}/120 answered -> {args.out}")


if __name__ == "__main__":
    main()
