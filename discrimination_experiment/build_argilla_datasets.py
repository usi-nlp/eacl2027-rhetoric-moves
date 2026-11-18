#!/usr/bin/env python3
"""Build the Argilla annotation dataset for the discrimination experiment.

Creates ONE Argilla dataset (120 records in the designed presentation order;
the annotator works through it at their own pace) on the Argilla server (an HF
Space), inside a dedicated workspace that
is NOT OAuth-joinable, plus an annotator user. Each record shows one
introduction (blinded id item_001..item_120); items in the "labeled" condition
are rendered with colored CARS move highlights from `annotations.chatgpt` in
data/master_dataset.json (the same auto-labeller on both sides — never gold vs
auto). This module is also the shared renderer + guidelines source for the
Label Studio and static-app builders.

Blinding: record ids and metadata carry ONLY the judge_manifest fields
(item_code, position, label_condition). intro_type / mode / quartile /
nativeness / intro_id never leave allocation.json.

Presentation order: Argilla's annotation queue is a per-user seeded shuffle,
NOT insertion order — the guidelines instruct the annotator to sort by
"inserted at" ascending, and every record carries its intended `position`.

Usage (credentials from .env: ARGILLA_API_URL, ARGILLA_API_KEY):
    python discrimination_experiment/build_argilla_datasets.py --dry-run
    python discrimination_experiment/build_argilla_datasets.py \
        --annotator-password <pw-for-the-annotator-account>
    python discrimination_experiment/build_argilla_datasets.py --recreate --force
"""

import argparse
import html
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CARS palette — canonical master-dataset labels (move_N__step).
# Colors follow intro_analysis/evaluate_and_visualize.py MOVE_COLORS; the one
# out-of-scheme label seen in the data (counter_claiming) gets its own color.
# ---------------------------------------------------------------------------

CARS_LABELS = {
    # label: (short, display name, hex color)
    "move_1__claiming_centrality":                  ("M1-CC",      "Claiming centrality",            "#4e79a7"),
    "move_1__making_topic_generalization":          ("M1-MTG",     "Making topic generalization(s)", "#76b7b2"),
    "move_1__reviewing_items_of_previous_research": ("M1-RR",      "Reviewing previous research",    "#59a14f"),
    "move_2__indicating_a_gap":                     ("M2-Gap",     "Indicating a gap",               "#e15759"),
    "move_2__question_raising":                     ("M2-QR",      "Question-raising",               "#ff9da7"),
    "move_2__continuing_a_tradition":               ("M2-CT",      "Continuing a tradition",         "#f28e2b"),
    "move_2__counter_claiming":                     ("M2-Counter", "Counter-claiming",               "#17becf"),
    "move_3__announcing_present_research":          ("M3-APR",     "Announcing present research",    "#b07aa1"),
    "move_3__outlining_purpose":                    ("M3-OP",      "Outlining purpose(s)",           "#d4a6c8"),
    "move_3__announcing_principal_findings":        ("M3-APF",     "Announcing principal findings",  "#9c755f"),
    "move_3__indicating_the_structure_of_research": ("M3-IS",      "Indicating article structure",   "#bab0ac"),
}

MOVE_NAMES = {
    "move_1": "Move 1 — Establishing a territory",
    "move_2": "Move 2 — Establishing a niche",
    "move_3": "Move 3 — Occupying the niche",
}

WORKSPACE = "discrimination"
DATASET_NAME = "intro_study"
N_ITEMS = 120
FORBIDDEN_METADATA = {"intro_type", "mode", "quartile", "nativeness", "intro_id", "paper_key", "word_count", "block"}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _pastel(hex_color: str, strength: float = 0.35) -> str:
    """Solid light tint of the color (readable with dark text in any UI theme)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c * strength + 255 * (1 - strength))
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _chip_text_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def _esc(text: str) -> str:
    """Escape and flatten ALL newlines to spaces (2026-07-17): generated intros
    carry many paragraph breaks while extracted originals mostly don't, so line
    breaks are themselves a tell. Presentation-level only — segment character
    offsets into the source text are untouched."""
    return html.escape(text).replace("\n", " ")


def collapse_segments(segments: list[dict], text: str) -> list[dict]:
    """Merge consecutive same-label segments (corpus collapse convention, so
    auto-labeller granularity is not itself a tell), then absorb tiny
    whitespace-only gaps before the next span."""
    segs = sorted(
        (dict(s) for s in segments if s.get("start") is not None and s.get("end") is not None),
        key=lambda s: s["start"],
    )
    merged: list[dict] = []
    for seg in segs:
        if merged and seg["label"] == merged[-1]["label"] and text[merged[-1]["end"] : seg["start"]].strip() == "":
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    for cur, nxt in zip(merged, merged[1:]):
        gap = text[cur["end"] : nxt["start"]]
        if 0 < len(gap) <= 4 and gap.strip() == "":
            cur["end"] = nxt["start"]
    return merged


def _chip(label: str) -> str:
    short, name, color = CARS_LABELS[label]
    return (
        f'<span title="{html.escape(MOVE_NAMES[label.split("__")[0]] + ": " + name)}" '
        f'style="display:inline-block;padding:0 5px;margin:0 4px 0 0;border-radius:3px;'
        f'font-size:0.75em;font-weight:bold;white-space:nowrap;'
        f'background:{color};color:{_chip_text_color(color)};">{short}</span>'
    )


def _legend(labels_present: list[str]) -> str:
    rows = []
    for move in ("move_1", "move_2", "move_3"):
        chips = [
            _chip(lbl) + f'<span style="font-size:0.85em;">{html.escape(CARS_LABELS[lbl][1])}</span>&nbsp;&nbsp; '
            for lbl in CARS_LABELS
            if lbl.startswith(move) and lbl in labels_present
        ]
        if chips:
            rows.append(
                f'<div style="margin:2px 0;"><b style="font-size:0.85em;">{MOVE_NAMES[move]}:</b> '
                + "".join(chips)
                + "</div>"
            )
    return (
        '<div style="border:1px solid #999;border-radius:6px;padding:6px 10px;margin-bottom:12px;">'
        '<div style="font-size:0.8em;margin-bottom:2px;"><i>Rhetorical moves in this text '
        "(automatic CARS annotation):</i></div>" + "".join(rows) + "</div>"
    )


# Explicit white card + dark text so the intro reads in dark-themed viewers too
# (Label Studio dark mode otherwise leaves it dark-on-dark); the highlight
# pastels and chip colors are designed for this white background.
INTRO_CARD_STYLE = (
    "background:#ffffff;color:#1c1c1c;line-height:1.7;"
    "padding:18px 22px;border-radius:10px;"
)


def render_intro_html(text: str, segments: list[dict] | None) -> str:
    """Whole field is one top-level <div> of escaped text so the intro can
    never be re-parsed as markdown; labeled items get a legend + colored spans."""
    if not segments:
        return f'<div style="{INTRO_CARD_STYLE}">{_esc(text)}</div>'

    units = collapse_segments(segments, text)
    parts, prev = [], 0
    for seg in units:
        if seg["start"] > prev:
            parts.append(_esc(text[prev : seg["start"]]))
        short, name, color = CARS_LABELS[seg["label"]]
        move_name = MOVE_NAMES[seg["label"].split("__")[0]]
        parts.append(_chip(seg["label"]))
        parts.append(
            f'<span title="{html.escape(move_name + ": " + name)}" '
            f'style="background:{_pastel(color)};color:#111111;'
            f'border-bottom:2px solid {color};padding:1px 0;">'
            f'{_esc(text[seg["start"] : seg["end"]])}</span>'
        )
        prev = seg["end"]
    if prev < len(text):
        parts.append(_esc(text[prev:]))

    labels_present = sorted({s["label"] for s in units}, key=list(CARS_LABELS).index)
    return f'<div style="{INTRO_CARD_STYLE}">{_legend(labels_present)}{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Guidelines (shown in the dataset UI)
# ---------------------------------------------------------------------------

ORDER_INSTRUCTIONS = {
    "argilla": (
        '**Order:** annotate the items in the intended order — in the record list, set '
        '**Sort → "inserted at" → ascending** once. (Each item also carries its intended '
        "`position` in its metadata, counting 1, 2, 3, …)"
    ),
    "static": (
        "**Order:** the app presents the items in a fixed order and unlocks the next item "
        "once you submit the current one — just work through the list top to bottom. "
        "You can stop at any point and continue later; your progress is saved."
    ),
    "labelstudio": (
        "**Order:** click **Label All Tasks** — it walks the items in the intended order "
        "and resumes where you left off. Work at your own pace, in as many sittings as "
        "you like. Please do not skip, filter, or reorder tasks."
    ),
}


def build_guidelines(medium: str = "argilla") -> str:
    cars_lines = []
    for move in ("move_1", "move_2", "move_3"):
        steps = ", ".join(
            f"**{short}** {name}" for lbl, (short, name, _) in CARS_LABELS.items() if lbl.startswith(move)
        )
        cars_lines.append(f"- **{MOVE_NAMES[move]}**: {steps}")
    cars_block = "\n".join(cars_lines)

    return f"""# Who wrote this introduction?

Each item is the introduction section of a published research article in applied \
linguistics / EFL. Some introductions are the **original**, written by the article's \
authors. Others were **generated by a large language model (LLM)** that was given the \
article's content (but not its introduction) and the works it cites.

Your task, for every item: judge whether the introduction was written by the human \
authors or generated by an LLM.

## How to work

1. {ORDER_INSTRUCTIONS[medium]}
2. Read the whole introduction, then answer:
   - **Q1 — Who wrote this introduction?** (required): *Sure it's human · Most likely \
human · Can't decide · Most likely AI · Sure it's AI*.
   - **Q2 — What did you base your decision on?** (required, tick all that apply): \
*Argumentation / Swales CARS structure*, *Writing style*, *Just a guess*.
   - **Comment** (optional): anything that struck you — a phrase, a structural oddity, \
a suspicious citation.
3. Press **Submit** and move on. Please judge each introduction on its own and do not \
go back to revise earlier answers.
4. Notes: minor formatting glitches (spacing, symbols, citation punctuation) can come \
from text processing and are **not** a reliable signal either way; all texts are shown \
as a single continuous paragraph on purpose.

## Colored move annotations

Some introductions are shown with colored highlights of their rhetorical moves \
(Swales' CARS model); others are plain text. The highlights were produced by the \
**same automatic annotation system for every introduction, regardless of origin**.

CARS model quick reference:

{cars_block}

## What our analysis found about LLM-generated introductions

Corpus-level tendencies from our quantitative comparison of human-written vs \
LLM-generated introductions (move structures, same auto-annotation on both sides). \
These are statistical tendencies, **not rules** — use them as one source of evidence.

1. **Move balance.** Human intros spend on average **67%** of their words on Move 1 \
(territory), **11%** on Move 2 (niche), **22%** on Move 3 (occupying). LLM intros \
shift weight from Move 1 (**58–62%**) to Move 2 (**16–21%**).
2. **Sub-steps.** LLMs review the literature less (**34–36%** of words vs **41%** for \
humans) and state explicit gaps about **twice** as much (**11–15%** vs **6%**); they \
raise questions more, and almost never outline the article's structure.
3. **Openings.** **46%** of human intros open with a centrality claim ("X has become \
increasingly important…") and **32%** open directly with the literature; LLM intros \
open with a centrality claim about **70%** of the time and almost never open with the \
literature (**<10%**).
4. **Signature funnel.** The opening pattern *importance → literature review → gap* \
is far more common in LLM intros (up to **42%**) than any single pattern in human \
intros (the most common human pattern reaches **25%**).
5. **Repertoire.** **28%** of human intros omit an entire move (often Move 2); LLM \
intros use all three moves in **more than 90%** of cases.
6. **Variety.** 76 human intros show **36** distinct move sequences; the same number \
of LLM intros show only **11–24**. A strict canonical Move 1 → 2 → 3 ordering occurs \
in **16%** of human intros but in up to **49%** of LLM intros.
7. **Granularity.** Human intros contain about **9.4** move segments on average; LLM \
intros about **10–11.5** — they switch rhetorical function more often.

In short: LLM introductions tend to be more *complete*, more *canonical*, and more \
*alike* — an explicit gap statement, a centrality-claim opening, all three moves \
ticked off in order. Human introductions are messier: they may dwell on the \
literature, skip a move entirely, or cycle back and forth between moves.
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_items(allocation_path: Path) -> list[dict]:
    """Blinded items + the intro_id join key (used locally, never uploaded)."""
    with open(allocation_path) as f:
        allocation = json.load(f)
    return [
        {
            "item_code": it["item_code"],
            "position": it["position"],
            "label_condition": it["label_condition"],
            "intro_id": it["intro_id"],
        }
        for it in allocation["items"]
    ]


def load_intro_index(dataset_path: Path) -> dict[str, dict]:
    with open(dataset_path) as f:
        data = json.load(f)
    return {intro["intro_id"]: intro for paper in data["papers"] for intro in paper["intros"]}


def build_record_payloads(items: list[dict], intro_index: dict[str, dict]) -> list[dict]:
    """Position-ordered list of {item_code, html, metadata} for all 120 items."""
    records = []
    for item in items:
        intro = intro_index[item["intro_id"]]
        segments = None
        if item["label_condition"] == "labeled":
            segments = intro["annotations"]["chatgpt"]["segments"]
            unknown = {s["label"] for s in segments} - set(CARS_LABELS)
            if unknown:
                raise SystemExit(f"{item['item_code']}: unknown CARS labels {unknown}")
        records.append(
            {
                "item_code": item["item_code"],
                "html": render_intro_html(intro["text"], segments),
                "metadata": {
                    "item_code": item["item_code"],
                    "position": item["position"],
                    "label_condition": item["label_condition"],
                },
            }
        )
    records.sort(key=lambda r: r["metadata"]["position"])
    return records


# ---------------------------------------------------------------------------
# Argilla provisioning
# ---------------------------------------------------------------------------

def build_settings(rg, client):
    return rg.Settings(
        guidelines=build_guidelines(),
        allow_extra_metadata=False,
        distribution=rg.TaskDistribution(min_submitted=1),
        fields=[rg.TextField(name="intro", title="Introduction", use_markdown=True, required=True, client=client)],
        questions=[
            rg.LabelQuestion(
                name="provenance",
                title="Q1 — Who wrote this introduction?",
                labels={
                    "1": "1 — Sure it's human",
                    "2": "2 — Most likely human",
                    "3": "3 — Can't decide",
                    "4": "4 — Most likely AI",
                    "5": "5 — Sure it's AI",
                },
                required=True,
                client=client,
            ),
            rg.MultiLabelQuestion(
                name="basis",
                title="Q2 — What did you base your decision on? (tick all that apply)",
                labels={
                    "argumentation_structure": "Argumentation / Swales CARS structure",
                    "writing_style": "Writing style",
                    "guess": "Just a guess",
                },
                labels_order="natural",
                required=True,
                client=client,
            ),
            rg.TextQuestion(name="comment", title="Optional comment", required=False, client=client),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="item_code", title="Item", client=client),
            rg.IntegerMetadataProperty(name="position", title="Position", min=1, max=N_ITEMS, client=client),
            rg.TermsMetadataProperty(
                name="label_condition", title="Move annotations", options=["labeled", "unlabeled"], client=client
            ),
        ],
    )


def get_client(rg):
    api_url = os.environ.get("ARGILLA_API_URL")
    api_key = os.environ.get("ARGILLA_API_KEY")
    if not api_url or not api_key:
        raise SystemExit(
            "ARGILLA_API_URL / ARGILLA_API_KEY not set (add them to .env)"
        )
    return rg.Argilla(api_url=api_url, api_key=api_key)


def ensure_workspace(rg, client, name: str):
    ws = client.workspaces(name)
    if ws is None:
        ws = rg.Workspace(name=name, client=client).create()
        print(f"created workspace '{name}' (not OAuth-joinable: keep it out of .oauth.yml)")
    return ws


def ensure_annotator(rg, client, workspace, username: str, password: str | None):
    user = client.users(username)
    if user is None:
        if not password:
            raise SystemExit(
                f"annotator user '{username}' does not exist — pass --annotator-password "
                "(or set ANNOTATOR_PASSWORD) to create it"
            )
        user = rg.User(username=username, password=password, role="annotator", client=client).create()
        print(f"created annotator user '{username}'")
    try:
        user.add_to_workspace(workspace)
    except Exception:
        pass  # already a member
    return user


def dataset_has_responses(dataset) -> bool:
    for record in dataset.records(with_responses=True):
        if record.responses and len(list(record.responses)) > 0:
            return True
    return False


def push_dataset(rg, client, records_payload, recreate, force):
    workspace = client.workspaces(WORKSPACE)
    dataset = client.datasets(name=DATASET_NAME, workspace=workspace)
    if dataset is not None:
        if dataset_has_responses(dataset) and not force:
            print(f"SKIP {DATASET_NAME}: already has responses (use --force to touch it)")
            return
        if recreate:
            dataset.delete()
            dataset = None
            print(f"deleted existing dataset {DATASET_NAME}")
    if dataset is None:
        dataset = rg.Dataset(name=DATASET_NAME, workspace=workspace, settings=build_settings(rg, client), client=client)
        dataset.create()
    records = [
        rg.Record(id=r["item_code"], fields={"intro": r["html"]}, metadata=r["metadata"])
        for r in records_payload
    ]
    dataset.records.log(records)
    print(f"pushed {len(records)} records -> {WORKSPACE}/{DATASET_NAME}")


def verify_dataset(client, records_payload):
    """Post-push audit: count, ids, and no forbidden metadata anywhere."""
    workspace = client.workspaces(WORKSPACE)
    dataset = client.datasets(name=DATASET_NAME, workspace=workspace)
    if dataset is None:
        raise SystemExit(f"VERIFY FAIL {DATASET_NAME}: dataset missing")
    records = list(dataset.records())
    expected = {r["item_code"] for r in records_payload}
    got = {str(r.id) for r in records}
    leaked = {k for r in records for k in (r.metadata or {})} & FORBIDDEN_METADATA
    ok = got == expected and not leaked and len(records) == N_ITEMS
    if got != expected:
        print(f"VERIFY FAIL: ids mismatch (missing {expected - got}, extra {got - expected})")
    if leaked:
        print(f"VERIFY FAIL: forbidden metadata keys leaked: {leaked}")
    if len(records) != N_ITEMS:
        print(f"VERIFY FAIL: {len(records)} records, expected {N_ITEMS}")
    if not ok:
        raise SystemExit("verification failed")
    print(f"verified {DATASET_NAME}: {N_ITEMS} records, ids + metadata clean")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def write_dry_run(records_payload, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "guidelines.md").write_text(build_guidelines())
    sections = []
    for cond in ("labeled", "unlabeled"):
        for rec in [r for r in records_payload if r["metadata"]["label_condition"] == cond][:2]:
            sections.append(
                f'<h2 style="font-family:sans-serif">{rec["item_code"]} ({cond})</h2>{rec["html"]}<hr>'
            )
    page = (
        "<!doctype html><meta charset='utf-8'><body style='max-width:800px;"
        "margin:2em auto;font-family:Georgia,serif'>" + "".join(sections) + "</body>"
    )
    (out_dir / "preview.html").write_text(page)
    print(f"dry run: {len(records_payload)} records rendered; preview.html + guidelines.md in {out_dir}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allocation", default=str(EXP_DIR / "allocation.json"))
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "master_dataset.json"))
    parser.add_argument("--annotator", default="annotator", help="annotator username to create/use")
    parser.add_argument("--annotator-password", default=os.environ.get("ANNOTATOR_PASSWORD"))
    parser.add_argument("--recreate", action="store_true", help="delete + recreate existing datasets")
    parser.add_argument("--force", action="store_true", help="allow touching datasets that already have responses")
    parser.add_argument("--dry-run", action="store_true", help="render locally, no server calls")
    parser.add_argument("--dry-run-dir", default=str(EXP_DIR / "dry_run"))
    args = parser.parse_args()

    items = load_items(Path(args.allocation))
    intro_index = load_intro_index(Path(args.dataset))
    records_payload = build_record_payloads(items, intro_index)

    if args.dry_run:
        write_dry_run(records_payload, Path(args.dry_run_dir))
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    import argilla as rg

    client = get_client(rg)
    workspace = ensure_workspace(rg, client, WORKSPACE)
    ensure_annotator(rg, client, workspace, args.annotator, args.annotator_password)
    push_dataset(rg, client, records_payload, args.recreate, args.force)
    verify_dataset(client, records_payload)
    print(f"\nannotator URL: {os.environ.get('ARGILLA_API_URL')}/sign-in  (user: {args.annotator})")


if __name__ == "__main__":
    main()
