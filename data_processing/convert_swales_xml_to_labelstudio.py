#!/usr/bin/env python3
"""Convert Swales XML segment annotations into Label Studio task JSON.

Input format assumptions:
- Root contains `body/segment` elements.
- Each segment has `features` like: swales_moves;move_1;reviewing_items_of_previous_research
- Segment text is the annotated span text.

Output:
- tasks.json: Label Studio importable tasks containing full document text and predictions.
- labels.txt: one label per line discovered from the corpus.
- label_config_snippet.xml: a ready-to-paste Label Studio labeling config snippet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from lxml import etree as LET
from label_studio_converter.converter import Converter


def normalize_label(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def label_from_features(features: str | None) -> str:
    if not features:
        return "unknown"

    parts = [p.strip() for p in features.split(";") if p.strip()]
    if not parts:
        return "unknown"

    # Typical pattern: swales_moves;move_1;reviewing_items_of_previous_research
    if len(parts) >= 3 and parts[0].lower() == "swales_moves":
        return normalize_label(f"{parts[1]}__{parts[2]}")

    if len(parts) >= 2:
        return normalize_label("__".join(parts[-2:]))

    return normalize_label(parts[0])


def parse_xml_document(xml_path: Path) -> tuple[str, list[dict], set[str]]:
    parser = LET.XMLParser(recover=True, huge_tree=True)
    tree = LET.parse(str(xml_path), parser=parser)
    root = tree.getroot()
    body = root.find("body")
    if body is None:
        raise ValueError(f"Missing <body> in {xml_path}")

    full_text_parts: list[str] = []
    spans: list[dict] = []
    labels_seen: set[str] = set()

    cursor = 0
    segment_index = 0

    for seg in body.findall("segment"):
        seg_text = "".join(seg.itertext())
        if not seg_text:
            continue

        if full_text_parts:
            separator = "\n\n"
            full_text_parts.append(separator)
            cursor += len(separator)

        start = cursor
        full_text_parts.append(seg_text)
        cursor += len(seg_text)
        end = cursor

        label = label_from_features(seg.attrib.get("features"))
        labels_seen.add(label)

        seg_id = seg.attrib.get("id", str(segment_index + 1))
        spans.append(
            {
                "id": f"r{segment_index}_{seg_id}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
                "value": {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                    "labels": [label],
                },
            }
        )
        segment_index += 1

    full_text = "".join(full_text_parts)
    return full_text, spans, labels_seen


def build_task(xml_path: Path, task_id: int) -> tuple[dict, set[str]]:
    text, spans, labels_seen = parse_xml_document(xml_path)
    task = {
        "id": task_id,
        "data": {
            "text": text,
            "source_file": xml_path.name,
        },
        "annotations": [
            {
                "completed_by": 1,  # Admin user ID
                "result": spans,
            }
        ],
        "meta": {
            "source_path": str(xml_path),
            "num_segments": len(spans),
        },
    }
    return task, labels_seen


def write_label_config(labels: list[str], output_path: Path) -> None:
    lines = [
        "<View>",
        "  <Labels name=\"label\" toName=\"text\">",
    ]
    for label in labels:
        lines.append(f"    <Label value=\"{label}\" />")
    lines.extend(
        [
            "  </Labels>",
            "  <Text name=\"text\" value=\"$text\" />",
            "</View>",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="LREC2026-academic_nativeness/Annotated_corpus",
        help="Directory containing XML files",
    )
    parser.add_argument(
        "--output-dir",
        default="LREC2026-academic_nativeness/labelstudio_import",
        help="Directory for generated Label Studio files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in: {input_dir}")

    tasks: list[dict] = []
    all_labels: set[str] = set()

    for task_id, xml_file in enumerate(xml_files, start=1):
        task, labels = build_task(xml_file, task_id)
        tasks.append(task)
        all_labels.update(labels)

    tasks_path = output_dir / "tasks.json"
    tasks_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sorted_labels = sorted(all_labels)

    labels_txt_path = output_dir / "labels.txt"
    labels_txt_path.write_text("\n".join(sorted_labels) + "\n", encoding="utf-8")

    label_config_path = output_dir / "label_config_snippet.xml"
    write_label_config(sorted_labels, label_config_path)

    # Use label-studio-converter to generate an additional normalized export.
    json_min_dir = output_dir / "json_min_export"
    json_min_dir.mkdir(parents=True, exist_ok=True)
    converter = Converter(config=label_config_path.read_text(encoding="utf-8"), project_dir=str(output_dir))
    converter.convert_to_json_min(str(tasks_path), str(json_min_dir), is_dir=False)

    print(f"Converted {len(xml_files)} XML files")
    print(f"Tasks file: {tasks_path}")
    print(f"Labels file: {labels_txt_path}")
    print(f"Label config snippet: {label_config_path}")
    print(f"label-studio-converter json_min export: {json_min_dir}")


if __name__ == "__main__":
    main()
