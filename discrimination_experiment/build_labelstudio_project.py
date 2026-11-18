#!/usr/bin/env python3
"""Build the Label Studio project files for the discrimination experiment.

Free route: the annotator runs Label Studio locally (`pip install label-studio`;
same tool this project's earlier CARS annotation rounds used) and imports the
files generated here — no hosting needed (Label Studio on HF Spaces is a Docker
Space, which HF now paywalls behind PRO, same as Argilla).

Outputs (in discrimination_experiment/labelstudio/):
- config.xml        — labeling interface (paste into project settings)
- tasks.json        — all 120 tasks, in the intended presentation order
- instructions.html — guidelines incl. insights (paste into the project's
                      "Labeling Instructions" setting)
- README.md         — step-by-step setup for the annotator

ONE Label Studio project; tasks import in array order, and the guidelines tell
the annotator to use "Label All Tasks" (sequential, resumes where they left
off — they work at their own pace). Blinding: task data carries ONLY
item_code / position / label_condition + the rendered HTML (same audit as the
other builders). Convert the annotator's JSON exports back with
convert_labelstudio_export.py.

Usage:
    python discrimination_experiment/build_labelstudio_project.py
"""

import argparse
import html as html_lib
import json
from pathlib import Path

from build_argilla_datasets import (
    build_record_payloads,
    load_intro_index,
    load_items,
)
from build_static_ui import guidelines_html

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent

# Choice strings shown in the UI; convert_labelstudio_export.py maps them back.
PROVENANCE_CHOICES = [
    "1 — Sure it's human",
    "2 — Most likely human",
    "3 — Can't decide",
    "4 — Most likely AI",
    "5 — Sure it's AI",
]
BASIS_CHOICES = {
    "Argumentation / Swales CARS structure": "argumentation_structure",
    "Writing style": "writing_style",
    "Just a guess": "guess",
}

CONFIG_XML = """<View>
  <HyperText name="intro" value="$html"/>
  <Header value="Q1 — Who wrote this introduction?"/>
  <Choices name="provenance" toName="intro" choice="single-radio" required="true"
           requiredMessage="Please answer Q1.">
{provenance_choices}
  </Choices>
  <Header value="Q2 — What did you base your decision on? (tick all that apply)"/>
  <Choices name="basis" toName="intro" choice="multiple" required="true"
           requiredMessage="Please tick at least one option in Q2.">
{basis_choices}
  </Choices>
  <Header value="Optional comment"/>
  <TextArea name="comment" toName="intro" rows="3" maxSubmissions="1"
            placeholder="anything that struck you…"/>
</View>
"""

README = """# Discrimination study — Label Studio setup (annotator)

1. Install and start Label Studio (Python 3.9+):

       pip install label-studio
       label-studio

   It opens at http://localhost:8080 — create a local account (stays on your machine).

2. Set up the project (once):
   - Create Project → name it `Introduction Study`.
   - Labeling Setup → Custom template → paste the contents of `config.xml` → Save.
   - Settings → Instructions → paste the contents of `instructions.html`
     (tick "show before labeling"), and read it before starting.
   - Import → upload `tasks.json` (120 items).

3. Annotate: click **Label All Tasks** and work through the items in the order
   they appear — it resumes where you left off, so stop and continue whenever
   suits you. Submit each item once; please don't revisit or change earlier
   answers, and don't skip, filter, or reorder.

4. Every ~30 items (and when you finish): Export → **JSON** (the full one, not
   JSON-MIN) and send the downloaded file to the research team. Intermediate
   exports are just backups — export again over the same project any time.

Notes: judge each introduction on its own; please do NOT search the web for any
passage of the texts. Minor formatting glitches are processing artifacts and not
a signal either way.
"""


def build_config() -> str:
    prov = "\n".join(f'    <Choice value="{html_lib.escape(c)}"/>' for c in PROVENANCE_CHOICES)
    basis = "\n".join(f'    <Choice value="{html_lib.escape(c)}"/>' for c in BASIS_CHOICES)
    return CONFIG_XML.format(provenance_choices=prov, basis_choices=basis)


def task_header(item_code: str, label_condition: str) -> str:
    """White pill so it stays readable on Label Studio's dark theme too."""
    tag = "with move annotations" if label_condition == "labeled" else "plain text"
    return (
        '<div style="margin-bottom:8px;"><span style="background:#ffffff;color:#555555;'
        "font-family:sans-serif;font-size:0.9em;display:inline-block;"
        'padding:4px 12px;border-radius:6px;">'
        f"Item {item_code[-3:]} &nbsp;·&nbsp; {tag}</span></div>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allocation", default=str(EXP_DIR / "allocation.json"))
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "master_dataset.json"))
    parser.add_argument("--out-dir", default=str(EXP_DIR / "labelstudio"))
    args = parser.parse_args()

    items = load_items(Path(args.allocation))
    intro_index = load_intro_index(Path(args.dataset))
    records = build_record_payloads(items, intro_index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.xml").write_text(build_config())
    (out_dir / "instructions.html").write_text(guidelines_html(medium="labelstudio"))
    (out_dir / "README.md").write_text(README)

    forbidden = {"intro_type", "mode", "quartile", "nativeness", "intro_id", "paper_key", "block", "session"}
    tasks = []
    for rec in records:
        data = {
            "item_code": rec["item_code"],
            "position": rec["metadata"]["position"],
            "label_condition": rec["metadata"]["label_condition"],
            "html": task_header(rec["item_code"], rec["metadata"]["label_condition"]) + rec["html"],
        }
        leaked = set(data) & forbidden
        if leaked:
            raise SystemExit(f"blinding violation in {rec['item_code']}: {leaked}")
        tasks.append({"data": data})
    path = out_dir / "tasks.json"
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
    print(f"wrote {path.name}: {len(tasks)} tasks")
    print(f"\nLabel Studio bundle in {out_dir} — send the whole folder to the annotator.")


if __name__ == "__main__":
    main()
