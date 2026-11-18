#!/usr/bin/env python3
"""
Prompts for AI introduction *generation* (Step 7).

Two generation modes share the same inputs (the article's content + the
abstracts of the works the original authors cited in their introduction) and
differ only in the system prompt:

  * **basic**        — write an introduction, no rhetorical scaffolding.
  * **cars_guided**  — additionally given Swales' CARS move/step definitions and
                       asked to (1) plan the rhetorical moves, then (2) write the
                       introduction that realises the plan.

This mirrors the convention in `intro_analysis/prompts.py`: the large, static
part of every call (role framing + CARS definitions for cars_guided) lives in
the system prompt so it caches cheaply, and only the short per-paper material
(title, body, references) varies in the user message.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical CARS labels — the gold scheme from
# data/annotation/human_intros/labels.txt. Kept here for reference and for
# downstream analysis of generated intros (cars_guided plans the moves
# internally and does not emit labels). Note the `moves_2_` plural prefix and
# that counter-claiming is NOT one of the gold 10.
# ---------------------------------------------------------------------------

CANONICAL_LABELS = [
    "move_1_claiming_centrality",
    "move_1_making_topic_generalization",
    "move_1_reviewing_items_of_previous_research",
    "moves_2_continuing_a_tradition",
    "moves_2_indicating_a_gap",
    "moves_2_question_raising",
    "move_3_outlining_purpose",
    "move_3_announcing_present_research",
    "move_3_announcing_principal_findings",
    "move_3_indicating_the_structure_of_research",
]

GENERATION_MODES = ["basic", "cars_guided"]


# ---------------------------------------------------------------------------
# CARS move/step definitions — Section 1 of the annotation manual (single
# source of truth; the segmentation rules in Section 2 are annotation-only and
# not relevant to generation, so they are sliced off).
# ---------------------------------------------------------------------------

_MANUAL_PATH = Path(__file__).parent.parent / "guidelines" / "Swales_CARS_Annotation_Manual.md"


def _load_cars_definitions() -> str:
    text = _MANUAL_PATH.read_text()
    marker = "## Section 2"
    idx = text.find(marker)
    return (text[:idx] if idx != -1 else text).strip()


CARS_DEFINITIONS = _load_cars_definitions()


# ---------------------------------------------------------------------------
# Shared role framing + inputs description
# ---------------------------------------------------------------------------

_ROLE = """You are an expert academic writer specialising in applied linguistics and English language teaching (EFL/ESL). Your task is to write the **Introduction** section of a research article."""

_INPUTS = """You are given:
1. The article's content — its abstract and body sections (literature review, methodology, results, discussion). This tells you what the study is about, what it did, and what it found. IMPORTANT: the original Introduction has been withheld on purpose; do not try to recall or reconstruct it — write a fresh one from the content you are given.
2. The works the original authors cited in their introduction. Each entry gives the authors, year and title, and includes an abstract where one was available (some entries have only the title and authors — judge from the title what the work is about)."""


# Shared output rules. Two deliberate choices:
#   * we do NOT tell the model how to structure the introduction — that
#     rhetorical scaffolding is exactly what separates `cars_guided` from
#     `basic`, so `basic` must stay free of it for the comparison to be clean;
#   * we do NOT forbid citing works beyond the provided list — a real author
#     cites far more than the handful we can surface, and the citations are
#     themselves a CARS signal (the cue for "reviewing previous research") that
#     we want to emerge naturally, so the model may invent plausible citations.
_RULES = """Rules:
- Do NOT output a section heading — only the introduction prose.
- Cite sources in author–year format, e.g. "Smith & Jones, 2019". Use the works listed above where relevant; you may also cite other works where the writing calls for it.
- Length: write to the target length stated with the article below and stay within the given range. Do not pad or over-elaborate to reach it, and do not substantially exceed it."""

_JSON_SHAPE = """Return ONLY a JSON object with this exact shape (no markdown, no commentary):
{"introduction": "<the full introduction text>"}"""


# ---------------------------------------------------------------------------
# basic mode
# ---------------------------------------------------------------------------

SYSTEM_BASIC = f"""{_ROLE}

{_INPUTS}

{_RULES}

{_JSON_SHAPE}"""


# ---------------------------------------------------------------------------
# cars_guided mode
# ---------------------------------------------------------------------------

_CARS_FRAMING = f"""{_ROLE}

You will write the introduction following **Swales' CARS (Create A Research Space) model**, whose moves and steps are defined below.

Before writing, PLAN THE MOVES INTERNALLY: think through which CARS moves and steps the introduction will use and in what sequence. The moves do NOT have to appear in 1 → 2 → 3 order — they recur and interleave freely, and any ordering is acceptable (a move can be revisited, e.g. returning to a Move 1 review after a Move 2 gap, and a real introduction may open or close on any move). Keep this plan entirely to yourself.

Then WRITE the introduction that realises your internal plan. Output ONLY the introduction prose — never the plan, and no move/step labels or headings in the text.

{_INPUTS}

# CARS move and step definitions
{CARS_DEFINITIONS}

{_RULES}

{_JSON_SHAPE}"""

SYSTEM_CARS_GUIDED = _CARS_FRAMING


# ---------------------------------------------------------------------------
# Per-paper user message
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """# Article title
{title}

# Article content (abstract + body; the original introduction is withheld)
{body}

# Key prior works cited in the original introduction
{references}

# Target length
{length_instruction}

Now produce the Introduction. Return only the JSON object described in your instructions."""


def get_system_prompt(mode: str) -> str:
    if mode == "basic":
        return SYSTEM_BASIC
    if mode == "cars_guided":
        return SYSTEM_CARS_GUIDED
    raise ValueError(f"Unknown mode {mode!r}. Known: {GENERATION_MODES}")
