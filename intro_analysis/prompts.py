#!/usr/bin/env python3
"""
Prompt templates for the intro segmentation/classification experiments.

This module exposes:
- The four assembled system prompts (seg/clf x builtin/manual)
- A USER_PROMPT_TEMPLATE for the per-task user message
- get_system_prompt(mode, few_shot, source, all_tasks) — single entry point

Mode = "segmentation" | "classification"
Source = "builtin" | "manual"
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------
# Three granularity groups, so the few-shot prompt always shows the full range:
#   - large-span     (few segments)     — group same-function sentences
#   - average         (~median segments) — the typical introduction
#   - fine-grained    (many segments)    — split when the function changes
# The default few-shot block picks ONE example from each group (3 examples).
# Each group holds two reserved candidates (swappable); all are excluded from
# evaluation runs via FEW_SHOT_EXAMPLE_IDS to avoid train/test leakage.
# Average ids (16, 28) sit outside the second annotators' eval union and both have
# 9 gold segments (the corpus median) covering all three moves.
FEW_SHOT_LARGE_SPAN_IDS = [17, 40]      # 3 gold segments each
FEW_SHOT_AVERAGE_IDS = [16, 28]         # 9 gold segments each (corpus median)
FEW_SHOT_FINE_GRAINED_IDS = [34, 66]    # 17–18 gold segments each
FEW_SHOT_EXAMPLE_IDS = set(FEW_SHOT_LARGE_SPAN_IDS
                           + FEW_SHOT_AVERAGE_IDS
                           + FEW_SHOT_FINE_GRAINED_IDS)


# ---------------------------------------------------------------------------
# Guiding principle (prepended to every system prompt)
# ---------------------------------------------------------------------------
COMMUNICATIVE_FUNCTION_REMINDER = """**Guiding principle.** Always classify segments by the **communicative function** of each sentence — what the author is doing rhetorically at that point in the introduction — not by surface cues, sentence boundaries, the presence of connectors, or paragraph layout. A new segment is warranted only when the communicative function genuinely changes.

"""


# ---------------------------------------------------------------------------
# Built-in segmentation prompt (rules only)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are an expert in academic discourse analysis, specialising in the rhetorical structure of research article introductions.

## Task
Split the given research article introduction into coherent argumentative segments. Each segment should correspond to a distinct **communicative step** — a stretch of text that continuously pursues one rhetorical goal (e.g. establishing the importance of a topic, reviewing a body of research, identifying a gap in the literature, announcing the study's purpose).
You do NOT need to name or classify the segments. Only segment the text.

## Output Format
Return a JSON object with a "segments" array. Each element contains only the verbatim text of one segment:
  "segments": [
    {"text": "<verbatim text copied from the input>"},

## Requirements
1. Cover the **entire** introduction — every word must appear in exactly one segment. Do not skip any text.
2. Each segment must contain at least one **complete sentence**. Never cut mid-sentence.
3. The "text" value must be **verbatim** text copied character-for-character from the input — do not paraphrase, reorder, summarise, or alter it in any way.
4. Segments must appear in the **same sequential order** as in the text.
5. Together, all segments must reconstruct the full input text (gaps of whitespace between paragraphs are acceptable).

## Segmentation principle
The boundary of a segment is determined by a **change in communicative step**, not by individual sentences or discourse markers.

A segment continues as long as the text keeps serving the same rhetorical goal — even if it spans many sentences or contains connectors like "However", "Nevertheless", "Furthermore", "Correspondingly", "Hence", "Therefore", "Meanwhile", "In line with this", "Besides", "Moreover".

These connectors do NOT automatically start a new segment. They can elaborate, contrast, or extend within the same step.

## When to start a new segment
Start a new segment when the communicative step genuinely changes. The most reliable signals are:

**1. Shift from reviewing the field → identifying a gap**
The sentence explicitly states that research is missing, limited, or absent:
- "No study has...", "Little/limited attention has been paid to...", "X remains unexplored/understudied/underexamined..."
- "X has received little/no scholarly attention", "there is a lack of research on..."
- A contrastive statement of insufficiency: "studies on X are scarce/in short supply"

**2. Purpose/aim statement (announcing this paper)**
The sentence announces what this specific paper does:
- "This study aims/investigates/seeks to...", "The present paper..."
- "Therefore/Hence, this study..." when followed by a purpose verb

**3. Shift to a genuinely different sub-topic within the literature review**
The text moves from reviewing one research strand to reviewing a clearly distinct one (different concept, population, context, or question). A new topic sentence that opens a new sub-topic signals this.

**4. Shift to announcing findings, structure, or research questions**
- "The results show...", "We found that...", "The paper is organised as follows..."
- Listing the paper's research questions as a block

## When to keep sentences in the same segment
Keep sentences together when they all serve the same step, even across connectors:

- Sentences that **add, qualify, exemplify, or contrast within the same research topic** (e.g., multiple cited studies on the same question, even with "However" between them)
- Sentences connected by **anaphoric or cataphoric references** (pronouns, "ibid.", "the former/latter", "such", "this") that point to the preceding/following content — unless the step changes
- Sentences with **hierarchical/chronological frame markers** ("first", "second", "then", "finally") that structure a single communicative act
- A sentence that **defines or exemplifies** a term just introduced in the preceding sentence
- A conclusion sentence ("hence", "therefore", "thus") that follows directly from the preceding 1-2 sentences **within the same step** — but NOT when it opens a purpose statement for this paper

## Granularity note
A single sentence is a valid segment when it clearly serves a distinct step (e.g., a gap statement or purpose statement). A segment can also span many sentences if they all serve the same step without switching.
"""


# ---------------------------------------------------------------------------
# Few-shot mini-examples from the guidelines
# ---------------------------------------------------------------------------

GUIDELINE_MINI_EXAMPLES = """
## Segmentation Rule Examples (from the annotation guidelines)

**Rule 1 — Hierarchical/chronological markers → one segment**
The following should NOT be split, because "First", "Then", "Finally" are frame markers
signalling a structured list that belongs to a single communicative act:

  CORRECT (one segment):
  "This article is structured as follows. First, I will provide a review of the current
  literature associated with self-mention in academic writing in English and in
  computer-mediated discourse in section 2. Then, I will introduce and discuss the present
  study with its research questions, corpus, methodology, and results, and provide a
  detailed discussion of the major findings in section 3. Finally, the article will be
  concluded with a summary of the results in section 4."

  CORRECT (one segment — "Another" continues "One problem"):
  "One problem is the issue of sample-dependent statistics, that is, the person parameters
  depend on the selection of items and item parameters depend on the sample (Embretson,
  1996). Another limitation of CTT is that the data should be continuous and normally
  distributed."

**Rule 2 — Argumentation markers → one segment**
"Hence", "therefore", "so", "because" link a claim to its consequence — keep together:

  CORRECT (one segment):
  "As the demand and participation in English-medium higher education increase, so does the
  expectation for students to have academic literacy skills. Hence, it is necessary to
  understand academic writing improvements that are made in academic literacy classes."

**Rule 3 — Exemplification markers → continuation, NOT a new segment**
"For instance", "for example" at the start of a sentence means it extends the previous one:

  CORRECT (one segment):
  "However, it is not evident that an early start of learning foreign languages is an
  advantage in and of itself (Nikolov and Djigunović 2011). For instance, Hanušová and
  Najvar (2005) investigated the long-term effects of an early start by examining university
  entrance exam language test scores. They found no evidence of any relationship between
  test score results and the start of language learning."
"""


# ---------------------------------------------------------------------------
# User prompt template (same for all variants)
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """Please segment the following research article introduction into coherent argumentative units. Return only the JSON — no explanation.

---
{text}
---"""



# ---------------------------------------------------------------------------
# Built-in classification prompt (segmentation + CARS labelling)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CLASSIFICATION = """You are an expert in academic discourse analysis, specialising in Swales' CARS (Create a Research Space) model for research article introductions.

## Task
Split the given research article introduction into coherent argumentative segments AND assign each segment a Swales CARS label. Each segment corresponds to one **communicative step** — a stretch of text that continuously pursues the same rhetorical goal.

## Output Format
Return a JSON object with a "segments" array. Each element has "text" and "label":
  "segments": [
    {"text": "<verbatim text copied from the input>", "label": "<label>"},
    ...
  ]

## Requirements
1. Cover the **entire** introduction — every word must appear in exactly one segment.
2. Each segment must contain at least one **complete sentence**. Never cut mid-sentence.
3. The "text" value must be **verbatim** text copied character-for-character from the input.
4. Segments must appear in the **same sequential order** as in the text.
5. A segment ends and a new one begins whenever the **label would change**.

## Segmentation principle
The boundary of a segment is determined by a **change in communicative step** (i.e., a change in the label). Connectors like "However", "Nevertheless", "Furthermore", "Moreover", "Therefore" do NOT automatically start a new segment — only a genuine shift in rhetorical function does.

---

## Swales CARS Label Taxonomy

Use exactly one of the following labels per segment:

### Move 1 — Establishing a Territory
**This move answers: "Why is this general research area important?"**
It sets the background, establishes the field, and motivates why the topic matters.

- `move_1__claiming_centrality`
  The author argues that the research topic is important, significant, central, or widely relevant.
  *Argumentative role:* "This topic MATTERS" — evaluative claim about the field's interest in the topic.
  *Key signals:* adjectives (important, vital, significant, essential, necessary); interest/popularity markers ("many studies have focused on…", "there has been growing interest in…"); statements of usefulness or impact.
  *NB:* If a sentence cites prior work AND claims centrality at the same time, label as claiming_centrality.

- `move_1__making_topic_generalization`
  The author provides a general statement, definition, or description of the topic WITHOUT citing prior work.
  *Argumentative role:* "Here is WHAT this topic is" — defines or describes the territory.
  *Key signals:* definitions, general truths, background descriptions, no in-text citations.
  *NB:* If a definition or description IS accompanied by an in-text citation, label as reviewing_items_of_previous_research instead.

- `move_1__reviewing_items_of_previous_research`
  The author reports what prior researchers have found, studied, or argued about the topic.
  *Argumentative role:* "Here is WHAT OTHERS have found" — maps the existing research landscape.
  *Key signals:* ANY in-text citation (integral: "Smith (2010) found…" or non-integral: "(Smith, 2010)"); verbs of reporting (found, showed, argued, investigated, examined, concluded).

### Move 2 — Establishing a Niche
**This move answers: "What specific problem or gap does this research address?"**
It shows why existing research is insufficient, leaving a space for the current study.

- `moves_2__counter_claiming`
  The author argues that prior research is WRONG or needs to be corrected.
  *Argumentative role:* "Something is WRONG" — explicit refutation.
  *Key signals:* "I argue that X is unsuited…", "contrary to…", explicit disagreement with a prior claim.

- `moves_2__indicating_a_gap`
  The author highlights that something has NOT BEEN DONE or is insufficiently studied.
  *Argumentative role:* "Something is MISSING" — points to an absence in the literature.
  *Key signals:* "No study has…", "little is known about…", "little attention has been paid to…", "X remains unexplored/understudied", "there is a lack of…", "to date, no studies have…".

- `moves_2__question_raising`
  The author highlights that something is UNCLEAR, ambiguous, or inconsistent in prior research.
  *Argumentative role:* "Something is UNCLEAR" — identifies uncertainty or contradiction.
  *Key signals:* rhetorical questions, "it is unclear whether…", "it remains to be seen…", "findings have been inconsistent…", "it is not yet established…".

- `moves_2__continuing_a_tradition`
  The author argues that existing research needs to be EXTENDED or built upon.
  *Argumentative role:* "Something needs to be ADDED" — calls for more research in an established direction.
  *Key signals:* "More research is needed on…", "it is important to further examine…", "following X's work…", "the present study extends…".

### Move 3 — Occupying the Niche
**This move answers: "How does this specific study fill the identified gap?"**
It presents the current paper's contribution: its aims, methods, findings, or structure.

- `move_3__outlining_purpose`
  The author states the aims, objectives, research questions, or hypotheses of THIS paper.
  *Argumentative role:* "THIS PAPER AIMS TO / ASKS…" — announces the study's goal.
  *Key signals:* "This study aims/seeks/investigates…", "The present paper examines…", "The research questions are…", "We hypothesize that…".

- `move_3__announcing_present_research`
  The author describes the methods, design, rationale for methodology, or data of THIS paper.
  *Argumentative role:* "THIS PAPER DOES IT BY…" — describes how the study was conducted.
  *Key signals:* "This study uses/employs/adopts…", "Data were collected from…", "The methodology involves…", descriptions of sample, instruments, procedures, rationale for method choice.

- `move_3__announcing_principal_findings`
  The author presents key results, conclusions, or implications of THIS paper.
  *Argumentative role:* "THIS PAPER FOUND / SUGGESTS…" — announces outcomes or potential contributions.
  *Key signals:* "The findings show/suggest/indicate…", "The results demonstrate…", modality markers pointing to implications (may, might, could, would) when describing what the study offers.

- `move_3__indicating_the_structure_of_research`
  The author explains how the remainder of the article is organised.
  *Argumentative role:* "THE PAPER IS STRUCTURED AS…" — roadmap for the reader.
  *Key signals:* "The paper is organised as follows…", "Section 2 presents…", "The remainder of this paper…".

---

## Decision procedure (apply per sentence)

1. Does this sentence argue the topic is **important or central**?
   → `move_1__claiming_centrality`

2. Does this sentence **define or describe** the topic with **no citation**?
   → `move_1__making_topic_generalization`

3. Does this sentence **report prior research** (has a citation)?
   → `move_1__reviewing_items_of_previous_research`
   *Exception:* if the citation is paired with a gap/problem statement → classify as a Move 2 step.

4. Does this sentence say something is **wrong** with prior work?
   → `moves_2__counter_claiming`

5. Does this sentence say something is **missing** from the literature?
   → `moves_2__indicating_a_gap`

6. Does this sentence raise something **unclear** or contradictory?
   → `moves_2__question_raising`

7. Does this sentence call for more research or explicitly build on prior work?
   → `moves_2__continuing_a_tradition`

8. Does this sentence state **THIS paper's** aims / RQs / hypotheses?
   → `move_3__outlining_purpose`

9. Does this sentence describe **THIS paper's** methods, data, or design?
   → `move_3__announcing_present_research`

10. Does this sentence present **THIS paper's** findings or implications?
    → `move_3__announcing_principal_findings`

11. Does this sentence describe how **THIS paper is organised**?
    → `move_3__indicating_the_structure_of_research`

---

## Segmentation rules

**Start a new segment** whenever applying the decision procedure above yields a DIFFERENT label than the previous sentence.

**Keep sentences in the same segment** when:
- They answer the **same question** in the decision procedure (same label).
- Subsequent sentences elaborate, exemplify, qualify, or contrast within the same step.
- Connectors like "However", "Furthermore", "In addition", "Meanwhile" do NOT change the rhetorical function (same label → same segment).
- A sentence introduced by "for instance" / "for example" extends the previous sentence's point.
- A sentence with anaphoric reference ("this", "such", "the former/latter", pronouns) continues the prior sentence's step.
- Frame markers ("first", "second", "then", "finally") structure a single communicative act → one segment.

**Special cases:**
- Moves can recur and interleave (e.g., M1 RIPR → M3 OP → M1 RIPR again is valid).
- A single sentence CAN be its own segment if it has a clearly distinct communicative function (e.g., a one-sentence gap statement or purpose statement).
- When sentences straddle two steps (e.g., a citation that also establishes centrality), use the decision priority above (CC > RIPR; Move 2 > Move 1 for citation+gap).
"""



# ---------------------------------------------------------------------------
# Manual-based prompts (use guidelines/Swales_CARS_Annotation_Manual.md)
# ---------------------------------------------------------------------------

_MANUAL_PATH = Path(__file__).parent.parent / "guidelines" / "Swales_CARS_Annotation_Manual.md"
_MANUAL_TEXT = _MANUAL_PATH.read_text()


_MANUAL_FRAMING_CLF = """You are an expert in academic discourse analysis, specialising in Swales' CARS (Create a Research Space) model for research article introductions.

## Task
Following the annotation manual below, split the given research article introduction into coherent argumentative segments AND assign each segment a Swales CARS label.

## Output Format
Return a JSON object with a "segments" array. Each element has "text" and "label":
{
  "segments": [
    {"text": "<verbatim text chunk from the input>", "label": "<label>"},
    ...
  ]
}

## Requirements
1. Cover the **entire** introduction — every word must appear in exactly one segment.
2. Each segment must contain at least one **complete sentence**. Never cut mid-sentence.
3. The "text" value must be **verbatim** text from the input, character-for-character.
4. Segments must appear in the **same sequential order** as in the text.
5. A segment ends and a new one begins whenever the **label would change** (i.e. whenever the communicative function changes).

## Label format
Use these exact label strings (one per segment):
- move_1__claiming_centrality
- move_1__making_topic_generalization
- move_1__reviewing_items_of_previous_research
- moves_2__counter_claiming
- moves_2__indicating_a_gap
- moves_2__question_raising
- moves_2__continuing_a_tradition
- move_3__outlining_purpose
- move_3__announcing_present_research
- move_3__announcing_principal_findings
- move_3__indicating_the_structure_of_research

---

# Annotation manual (authoritative source for labels and decisions)

"""

_MANUAL_FRAMING_SEG = """You are an expert in academic discourse analysis, specialising in the rhetorical structure of research article introductions.

## Task
Following the annotation manual below, split the given research article introduction into coherent argumentative segments. Each segment should correspond to a distinct **communicative step** as defined in the manual. You do NOT need to assign labels — only segment the text.

## Output Format
Return a JSON object with a "segments" array. Each element contains only the verbatim text of one segment:
{
  "segments": [
    {"text": "<verbatim text chunk from the input>"},
    ...
  ]
}

## Requirements
1. Cover the **entire** introduction — every word must appear in exactly one segment.
2. Each segment must contain at least one **complete sentence**. Never cut mid-sentence.
3. The "text" value must be **verbatim** text from the input, character-for-character.
4. Segments must appear in the **same sequential order** as in the text.
5. A segment ends and a new one begins whenever the **communicative function** would change. Ignore the manual's label-naming instructions — they are provided only for context on what counts as a communicative-function shift.

---

# Annotation manual (use to identify communicative-function shifts; ignore the label-naming part)

"""

SYSTEM_PROMPT_CLF_MANUAL = _MANUAL_FRAMING_CLF + _MANUAL_TEXT
SYSTEM_PROMPT_SEG_MANUAL = _MANUAL_FRAMING_SEG + _MANUAL_TEXT


# ---------------------------------------------------------------------------
# Numbered class scheme (v4): the model predicts an integer 1–11 per segment;
# run_experiment maps it back to the canonical CARS label string via this map.
# Order mirrors the manual's label listing (move 1 → 2 → 3).
# ---------------------------------------------------------------------------
CLASS_NUMBER_TO_LABEL = {
    1:  "move_1__claiming_centrality",
    2:  "move_1__making_topic_generalization",
    3:  "move_1__reviewing_items_of_previous_research",
    4:  "moves_2__counter_claiming",
    5:  "moves_2__indicating_a_gap",
    6:  "moves_2__question_raising",
    7:  "moves_2__continuing_a_tradition",
    8:  "move_3__outlining_purpose",
    9:  "move_3__announcing_present_research",
    10: "move_3__announcing_principal_findings",
    11: "move_3__indicating_the_structure_of_research",
}
# Inverse, keyed by the normalised (single-underscore) label so both gold
# ("move_1_…") and model ("move_1__…") spellings resolve.
LABEL_TO_CLASS_NUMBER = {
    lbl.replace("__", "_"): num for num, lbl in CLASS_NUMBER_TO_LABEL.items()
}


def label_to_number(label: str) -> int | None:
    """Map a CARS label string (single- or double-underscore) to its 1–11 number."""
    return LABEL_TO_CLASS_NUMBER.get((label or "").replace("__", "_"))


# Numbered label set shown to the model in the v4 prompt.
NUMBERED_LABEL_SET = """## Label set — predict the NUMBER (an integer 1–11)

For each segment, output the **integer** of its CARS class (not the name):

**Move 1 — Establishing a Territory**
1. claiming centrality
2. making a topic generalization
3. reviewing items of previous research

**Move 2 — Establishing a Niche**
4. counter-claiming
5. indicating a gap
6. question-raising
7. continuing a tradition

**Move 3 — Occupying the Niche**
8. outlining the purpose
9. announcing present research
10. announcing principal findings
11. indicating the structure of research

Return the integer only, e.g. `{"text": "...", "label": 3}`.
"""

# v4 framing: segmentation instructions FIRST, then the JSON output format
# (numeric label), then the numbered label set. The "Guiding principle"
# (communicative-function reminder) is appended at the BOTTOM by
# get_system_prompt, not the top.
_MANUAL_FRAMING_CLF_V4 = """You are an expert in academic discourse analysis, specialising in Swales' CARS (Create a Research Space) model for research article introductions.

## Task
Following the annotation manual below, (1) split the given research article introduction into coherent argumentative segments, then (2) assign each segment its CARS class number.

## Segmentation instructions (do this first)
1. Cover the **entire** introduction — every word must appear in exactly one segment.
2. Each segment must contain at least one **complete sentence**. Never cut mid-sentence.
3. The "text" value must be **verbatim** text from the input, character-for-character.
4. Segments must appear in the **same sequential order** as in the text.
5. A segment ends and a new one begins whenever the communicative function (and therefore the class number) would change.

## Output Format
Return a JSON object with a "segments" array. Each element has "text" and an integer "label":
{
  "segments": [
    {"text": "<verbatim text chunk from the input>", "label": <integer 1-11>},
    ...
  ]
}

""" + NUMBERED_LABEL_SET + """
---

# Annotation manual (authoritative source for the classes and decisions)

"""

SYSTEM_PROMPT_CLF_MANUAL_V4 = _MANUAL_FRAMING_CLF_V4 + _MANUAL_TEXT


# ---------------------------------------------------------------------------
# v5: move-first reasoning + a one-sentence rationale per segment (built on v4).
# ---------------------------------------------------------------------------
# Templated gold rationales for the few-shot demos — each names the MOVE first,
# then the single signal that fixes the sub-step. Keyed by class number.
RATIONALE_BY_NUMBER = {
    1:  "Move 1 (Establishing a Territory): claims the topic is important, central, or widely relevant.",
    2:  "Move 1 (Establishing a Territory): a general statement or definition of the topic with no citation.",
    3:  "Move 1 (Establishing a Territory): reports what prior research did or found, marked by in-text citation(s).",
    4:  "Move 2 (Establishing a Niche): refutes or corrects a specific prior claim.",
    5:  "Move 2 (Establishing a Niche): points to something missing or unstudied in the literature.",
    6:  "Move 2 (Establishing a Niche): raises an unclear, ambiguous, or contradictory point in prior work.",
    7:  "Move 2 (Establishing a Niche): calls to extend or build on an existing line of research.",
    8:  "Move 3 (Occupying the Niche): states this study's aims, research questions, or hypotheses.",
    9:  "Move 3 (Occupying the Niche): describes this study's own methods, design, or data.",
    10: "Move 3 (Occupying the Niche): announces this study's findings or implications.",
    11: "Move 3 (Occupying the Niche): describes how the article itself is organized.",
}

# Model-generated, example-specific rationales for the few-shot demos (produced
# by generate_fewshot_rationales.py). Keyed by task id -> list aligned to the
# task's gold segments. Falls back to RATIONALE_BY_NUMBER templates when absent.
_RATIONALES_PATH = Path(__file__).parent / "few_shot_rationales.json"
try:
    GENERATED_RATIONALES = json.loads(_RATIONALES_PATH.read_text())
except Exception:
    GENERATED_RATIONALES = {}


def rationale_for(task_id: int | None, index: int, number: int) -> str:
    """Example-specific rationale for segment `index` of `task_id`, else the
    generic per-class template for `number`."""
    entries = GENERATED_RATIONALES.get(str(task_id)) if task_id is not None else None
    if entries and index < len(entries):
        r = (entries[index] or {}).get("rationale", "").strip()
        if r:
            return r
    return RATIONALE_BY_NUMBER.get(number, "")


_MANUAL_FRAMING_CLF_V5 = """You are an expert in academic discourse analysis, specialising in Swales' CARS (Create a Research Space) model for research article introductions.

## Task
Following the annotation manual below: (1) split the introduction into coherent argumentative segments; (2) for each segment, first identify its **MOVE**, then its sub-step; (3) give a one-sentence rationale and output the class number.

## Segmentation instructions (do this first)
1. Cover the **entire** introduction — every word must appear in exactly one segment.
2. Each segment must contain at least one **complete sentence**. Never cut mid-sentence.
3. The "text" value must be **verbatim** text from the input, character-for-character.
4. Segments must appear in the **same sequential order** as in the text.
5. A segment ends and a new one begins whenever the communicative function (and therefore the class number) would change.

## Label in TWO steps — decide the MOVE first, then the sub-step
Before picking a number, first decide which of the three MOVES the segment performs:
- **Move 1 — Establishing a Territory:** background, importance, or prior research on the topic → classes **1–3**.
- **Move 2 — Establishing a Niche:** shows existing research is insufficient — a counter-claim, gap, open question, or a tradition to extend → classes **4–7**.
- **Move 3 — Occupying the Niche:** the present study itself — its purpose, methods, findings, or structure → classes **8–11**.
Only **after** you have fixed the move, choose the specific sub-step number from within that move's range. Committing to the move first prevents assigning a label from the wrong move.

## Output Format
Return a JSON object with a "segments" array. Each element has "text", a one-sentence "rationale", and an integer "label":
{
  "segments": [
    {"text": "<verbatim text chunk from the input>", "rationale": "<one sentence: name the Move (1/2/3), then the signal that fixes the sub-step>", "label": <integer 1-11>},
    ...
  ]
}
Write the "rationale" BEFORE deciding the "label": state the Move first, then the one signal that pins the sub-step (e.g. an in-text citation ⇒ reviewing prior research; an explicit aim ⇒ outlining the purpose). Keep it to a single sentence.

""" + NUMBERED_LABEL_SET + """
---

# Annotation manual (authoritative source for the classes and decisions)

"""

SYSTEM_PROMPT_CLF_MANUAL_V5 = _MANUAL_FRAMING_CLF_V5 + _MANUAL_TEXT


# v6: v5 (move-first + rationale) + targeted fixes from the v5 rationale-error
# analysis. Appended AFTER the manual + v3 clarifications, as final tie-breakers.
CLARIFICATIONS_CLF_V6_FIXES = """

---

# Final tie-breakers (v6 — apply these LAST; they override earlier guidance in any conflict)

These three fixes target the most common v5 errors, diagnosed from the model's own rationales.

## F1. One communicative FUNCTION per segment — segment by function, not by topic
Sentences that share a TOPIC but do different rhetorical things are DIFFERENT segments. **Use your own rationale as a split test:** if the rationale you would write joins two functions with "and" — e.g. "reviews prior work **and** stresses its importance", "describes the framework **and** announces the study", "reviews X **and** identifies a gap" — then that is TWO (or more) segments: split it so each segment's rationale names exactly ONE function. The opening of an introduction typically interleaves topic generalizations (class 2), centrality claims (class 1), and reviews (class 3) about the SAME topic — segment and label each of these separately; do not merge them into one block under a single label.

## F2. A described framework / approach / concept is Move 1, not Move 3
Describing a theoretical framework, concept, or analytical approach — especially when it carries citations — is **Move 1 reviewing/territory** (class 3, or class 2 if genuinely uncited), NOT "announcing the present research" (class 9). Reserve class 9 for the PRESENT study's OWN conduct, marked by first-person + study nouns ("we collected", "our participants", "the data were drawn from", "this study applies X to our corpus"). Phrasings like "The approach is…", "X is informed by…", "X can be conceptualized as…", "the framework draws on (citations)" describe the territory ⇒ class 3.

## F3. A cited deficiency or "move beyond" is still Move 1 (review), not Move 2
A contrast, limitation, inadequacy, or call to "extend / move beyond" that is DOCUMENTED with a citation (or that reports what the literature shows) stays **reviewing previous research (class 3)**. Only assign Move 2 (gap = class 5, continuing-a-tradition = class 7) when the insufficiency is asserted as the niche the PRESENT study fills — typically uncited and adjacent to the purpose statement. Surface signals alone ("however", "remains inadequate", "few syntheses", "move beyond", "underprepared") do NOT make a segment Move 2.
"""


CLARIFICATIONS_CLF_V7_FIXES = """

---

# Final tie-breakers (v7 — apply these LAST; they OVERRIDE all earlier guidance, including the citation rule in section A, in any conflict)

## A*. The citation is a STRONG CUE, not an absolute switch — the communicative FUNCTION decides the Move-1 sub-step
Earlier guidance said "a citation ⇒ reviewing (class 3)" and "no citation ⇒ never class 3". Treat those as defaults only: when the function and the citation point different ways, FUNCTION wins. Apply these carve-outs:
- **A*1 (cited importance is still Claiming Centrality, class 1).** If the segment's main act is an explicit VALUE / IMPORTANCE claim about the topic, label it **class 1** even when it carries a citation — e.g. "Academic writing is an integral part of tertiary education (cite)", "X is increasingly essential …", "X plays a crucial role …". A citation that merely backs an importance claim does not turn it into a review.
- **A*2 (uncited synthesis is still Reviewing, class 3).** A sentence that SUMMARIZES, CONCLUDES, or SYNTHESIZES prior research is **class 3** even with NO parenthetical citation — e.g. "Based on these findings, it can be concluded that …", "Evidently, …", "Taken together, the studies show …". The absence of a citation does NOT force class 2.
- **A*3 (cited concept-definition is Topic Generalization, class 2).** A GENERAL DEFINITION or description of what a concept IS — explaining the concept itself, not reporting what a specific study found — is **class 2** even if it carries a citation — e.g. "Meaning is graded along a scale of intensity (cite)", "Force and focus refer to …".
- **Within-Move-1 decision order:** explicit importance claim ⇒ class 1; explains WHAT a concept is ⇒ class 2; reports or synthesizes what research DID/FOUND ⇒ class 3.

## B*. A framework / approach is the PRESENT study (class 9) when it is THIS study's own analytical choice (refines F2)
Describing a framework, approach, or concept is **Move 1 reviewing (class 3)** when it is presented as general prior knowledge. BUT it is **Announcing Present Research (class 9)** when the segment ties the framework to what THIS paper does — first-person or study-deictic cues: "in this study", "we adopt / we use / our analysis", "studying X from a Y perspective is justified by …", "the present study applies Y to …", "Y provides us with …". When the sentence connects the framework to the present study's own method or its justification, choose class 9 even if it carries a citation.
"""


CLARIFICATIONS_CLF_V8_FIXES = """

---

# Final tie-breakers (v8 — apply these LAST; they OVERRIDE all earlier guidance, INCLUDING the manual's "a citation ⇒ reviewing" default)

The hardest decision is WITHIN Move 1 (classes 1 / 2 / 3). Decide it with the rules and worked contrasts below — do NOT fall back to "this sentence has a citation ⇒ class 3".

## M0. Decision ORDER within Move 1 — apply in this order, stop at the first match
**This SUPERSEDES the earlier section-A guidance ("when unsure between centrality and reviewing, choose reviewing"; "centrality is the marked, harder-to-earn label"). Do NOT prefer class 3 over class 1 for an importance claim just because it carries a citation.**
1. **Is the segment's MAIN point that the topic is important / central / essential / valuable / consequential / a growing concern?** → **class 1 (Claiming Centrality)** — decide this FIRST, and choose class 1 **even if the segment carries a citation**.
2. Otherwise, does it report or synthesize what specific prior research DID or FOUND? → **class 3 (Reviewing)**.
3. Otherwise (a general statement or definition, with no importance claim) → **class 2 (Topic Generalization)**.

## M1. Class 1 vs Class 2 vs Class 3 — three distinct functions
- **Class 1 — Claiming Centrality:** an EXPLICIT claim that the topic is important / central / essential / vital / widely consequential. This holds **even if the sentence carries a citation** — a citation that merely backs an importance claim does NOT turn it into class 3.
- **Class 2 — Making a Topic Generalization:** a plain general statement, fact, or definition about the topic that is NOT itself an importance claim. This is the DEFAULT for general statements.
- **Class 3 — Reviewing Items of Previous Research:** reports what specific prior research DID or FOUND ("X examined…", "studies have shown…", "(Smith, 2019) argued…"), or synthesizes prior findings.

## M2. Do NOT over-use Class 1 (centrality precision)
A general or definitional statement WITHOUT an explicit importance word is **class 2, not class 1** — even when it sounds broad or consequential. Assign class 1 only when you can point to an explicit importance marker (*important, vital, crucial, central, essential, key, plays a major role, growing concern, increasingly necessary*).

## M3. Worked contrasts — same topic, different class (learn the boundary from these)
- "Writing is increasingly **essential** for academic and professional success (Graham, 2019)." → **1** (explicit importance claim; the citation does NOT make it class 3).
- "Academic writing is an **integral part** of tertiary education (Rahman & Rahim, 2016)." → **1** (importance claim + citation = still class 1).
- "Writing is the process of producing text to convey meaning." → **2** (plain definition, no importance claim).
- "Speaking is one of the four language skills, alongside listening, reading, and writing." → **2** (general statement, not an importance claim).
- "Graham (2019) **found** that explicit instruction improves students' writing quality." → **3** (reports what a study found).
- "Based on these findings, it can be concluded that several factors affect writing development." → **3** (synthesizes prior research, even with no parenthetical citation).

## B*. A framework / approach tied to THIS study ⇒ Announcing Present Research (class 9) (kept from v7)
Describing a framework/approach is **class 3** when presented as general prior knowledge, but **class 9** when the segment ties it to what THIS paper does — first-person / study-deictic cues ("in this study", "we adopt / we use / our analysis", "studying X from a Y perspective is justified by…", "Y provides us with…"), even if it carries a citation.
"""


# A dedicated few-shot exemplar that demonstrates the CC/MTG/RR boundary in one
# short opening paragraph (constructed, not a corpus paper → no eval leakage).
# Shown in the v8 few-shot block to teach the boundary by example.
BOUNDARY_PRIMER_V8 = '''

## Example 0 (Move-1 boundary primer — Claiming Centrality vs Topic Generalization vs Reviewing)

**Introduction:**
Vocabulary knowledge is widely recognized as essential to second language proficiency. Vocabulary refers to the set of words a learner knows and can use in comprehension and production. Numerous studies have examined how vocabulary is acquired: Nation (2001) catalogued the dimensions of word knowledge, and Schmitt (2008) reviewed strategies for vocabulary instruction. Based on this body of work, it is generally accepted that both incidental and intentional learning contribute to vocabulary growth.

**Correct segmentation and labels:**
{
  "segments": [
    {"text": "Vocabulary knowledge is widely recognized as essential to second language proficiency.", "rationale": "Move 1 (Establishing a Territory): an explicit importance claim ('essential to … proficiency') ⇒ Claiming Centrality, even though it is a broad statement.", "label": 1},
    {"text": "Vocabulary refers to the set of words a learner knows and can use in comprehension and production.", "rationale": "Move 1 (Establishing a Territory): a plain definition of the concept, no importance claim and no citation ⇒ Making a Topic Generalization.", "label": 2},
    {"text": "Numerous studies have examined how vocabulary is acquired: Nation (2001) catalogued the dimensions of word knowledge, and Schmitt (2008) reviewed strategies for vocabulary instruction.", "rationale": "Move 1 (Establishing a Territory): reports what specific studies did/found (Nation 2001 catalogued…, Schmitt 2008 reviewed…) ⇒ Reviewing items of previous research.", "label": 3},
    {"text": "Based on this body of work, it is generally accepted that both incidental and intentional learning contribute to vocabulary growth.", "rationale": "Move 1 (Establishing a Territory): synthesizes the prior findings ('Based on this body of work, it is generally accepted…') ⇒ Reviewing, even without a parenthetical citation.", "label": 3}
  ]
}
'''


# ---------------------------------------------------------------------------
# Operational clarifications — prompt version v2  (2026-06-03)
# ---------------------------------------------------------------------------
# Appended AFTER the taxonomy / manual so it has the last word. These do not
# replace the manual; they are operational tie-breakers derived from the v1
# error analysis on the 15-paper subset of annotator 1 (run
# 2026-06-01_15-52-20_classification-fewshot-manual). They target the three
# changes requested after that analysis:
#   (1) CC-vs-RR/MTG over-prediction  — 18 RR→CC + 8 CC→RR + 5 MTG→RR errors
#       (~50% of all label errors); M1-CC precision was only 0.57.
#   (3) under-segmentation             — 10/15 papers under-segmented,
#       mean 7.7 AI segs vs gold 9.2 (boundary recall 0.72 vs annotator 1's 0.84).
#   (4) gap recall + OP/APR confusion  — 5 Gap→MTG and 5 OP→APR errors.

CLARIFICATIONS_CLF_V2 = """

---

# Operational clarifications (v2 — apply these as tie-breakers)

These clarifications refine how the manual above is applied in borderline cases. When the manual and a clarification below both bear on a decision, follow the clarification.

## A. Claiming Centrality (CC) vs Reviewing Previous Research (RIPR) vs Making Topic Generalizations (MTG)

This is the most common source of error, so apply the following carefully:

1. **A sentence that contains an in-text citation and reports what specific prior work studied, found, argued, or proposed is RIPR — even if the topic is described as popular, growing, or widely studied.** Phrases like "many studies have focused on X (refs)", "there has been growing interest in X (refs)", "X has been widely investigated (refs)" describe *research activity* and are **RIPR**, not Claiming Centrality.

2. **Reserve Claiming Centrality for an explicit evaluative claim that the topic itself is important, significant, central, vital, essential, necessary, or consequential** — a value judgement about the topic, not a report that research on it exists. Look for evaluative markers (*important, vital, crucial, essential, necessary, significant, plays a key role, has profound implications*). If you cannot point to such an explicit value claim, it is **not** CC.

3. **NB 2 of the manual ("if centrality and review overlap → CC") is narrow.** It applies only when an explicit importance claim and a citation genuinely co-occur **in the same sentence**. Do not let one CC sentence pull neighbouring pure-review sentences into CC — those remain RIPR.

4. **When you are unsure between CC and RIPR and the sentence has a citation reporting prior work, choose RIPR.** When unsure between CC and MTG and there is no citation, choose MTG. Centrality is the *marked*, harder-to-earn label.

5. Reminder (manual NB 1): an uncited definition or general description of the topic is **MTG**; the same content **with a citation** is **RIPR**.

## B. Segmentation granularity and move recurrence

1. **Do not collapse the whole literature review into a single RIPR segment.** Introductions in this corpus are typically segmented into roughly **8–12 steps**; the most common v1 error is producing too few segments by merging distinct functions.

2. **Moves recur and interleave** (the manual: "moves are not necessarily in order"). The review commonly alternates RIPR with CC, MTG, or a Move-2 gap, and a M1-RIPR step frequently **resumes after** a Move-2 or Move-3 sentence. Start a new segment **each time** the communicative function changes back — including returning to RIPR after a gap statement.

3. A single sentence is a valid segment when its function differs from its neighbours (e.g. a one-sentence centrality claim, gap statement, or purpose statement embedded in a run of review). Do not absorb it into the surrounding step.

4. This is *not* a licence to split a genuinely continuous same-function run: keep elaboration, exemplification, anaphora, and frame-marker lists within one segment, as the manual requires.

## C. Indicating a Gap (IG) and Outlining the Purpose (OP) vs Announcing Present Research (APR)

1. **A statement of insufficiency, scarcity, or absence is a gap (IG) even when it is embedded inside the literature review and even without the exact phrase "no study has".** Contrastive insufficiency markers after a review — "however, X has been largely overlooked / has received little attention", "few studies have examined", "X remains underexplored/underexamined", "our understanding of X is limited" — are **Move 2 (IG)**, not MTG or RIPR.

2. **OP vs APR turns on aims-vs-methods.** A sentence that states **what this paper sets out to do, ask, or test** — its aims, objectives, research questions, or hypotheses ("this study aims to…", "the purpose of this paper is…", "we ask whether…", "the research questions are…") — is **Outlining the Purpose (OP)**. A sentence that states **how the study is carried out** — methodology, design, data, sample, instruments, or the rationale for a method choice ("this study uses…", "data were collected from…", "we adopt a … design") — is **Announcing Present Research (APR)**. When a purpose sentence names the aim, label it **OP** even if it briefly mentions the method.
"""

CLARIFICATIONS_CLF_V3 = """

---

# Operational clarifications (v3 — apply these as tie-breakers)

These refine how the manual above is applied in borderline cases. When the manual and a clarification below both bear on a decision, follow the clarification.

## A. Move-1 sub-step is decided by the CITATION, not by tone

The single most common error is confusing Claiming Centrality (CC), Reviewing Previous Research (RIPR), and Making Topic Generalizations (MTG). Decide them by this hard, mechanical rule — **ignore how "important" or "definitional" the sentence sounds**:

1. **If the sentence contains an in-text citation — integral ("Smith (2019) found…") or parenthetical ("(Smith, 2019)") — it is RIPR.** This holds even when the cited sentence also reads as an importance claim or a general definition. A citation always wins: citation ⇒ RIPR, never CC, never MTG.
2. **Never assign RIPR to a sentence that has no citation.** An uncited general statement, definition, or description of the topic is **MTG**. Do not infer a hidden citation from technical or scholarly content — no citation marker means MTG.
3. **Reserve Claiming Centrality for an explicit, UNCITED value claim that the topic itself is important, vital, central, essential, necessary, or consequential.** Look for an explicit evaluative marker about the topic. If you cannot point to one, it is not CC.
4. A CC sentence does not pull neighbouring pure-review sentences into CC; those remain RIPR by rule A.1.

## B. Segmentation granularity and move recurrence

1. **Do not collapse the whole literature review into a single RIPR segment.** Intros in this corpus are typically segmented into roughly **8–12 steps**; the most common error is too few segments.
2. **Moves recur and interleave.** The review commonly alternates RIPR with CC, MTG, or a Move-2 step, and an RIPR step frequently **resumes after** a Move-2/Move-3 sentence. Start a new segment each time the function changes back.
3. A single sentence is a valid segment when its function differs from its neighbours (a one-sentence centrality, gap, or purpose statement embedded in a run of review).
4. This is *not* a licence to split a continuous same-function run: keep elaboration, exemplification, anaphora, and frame-marker lists within one segment.

## C. Move 2 — Continuing a Tradition (CT) vs Indicating a Gap (IG); Counter-Claiming is rare

1. **CT vs IG tie-breaker — decide by existence-of-research:**
   - If research on the topic **exists** and the sentence calls to **extend, build on, or continue** it — "this provided the impetus for the study", "more research is needed on…", "it is important to further examine…", "following X's work…", "there is a lot still to discover" — it is **Continuing a Tradition (CT)**.
   - Only when the sentence states a **pure absence** — "no study has…", "little is known about…", "X remains unexamined", "there is a lack of…" — is it **Indicating a Gap (IG)**.
   - When both readings are possible, prefer **CT** (the niche is being extended, not declared empty).
2. **Counter-Claiming is very rare — double-check before assigning it.** Use `moves_2__counter_claiming` only for an explicit refutation that prior work is *wrong* or *mistaken* ("contrary to X…", "X is unsuited…", direct disagreement with a specific prior claim). A mere contrast, limitation, or insufficiency is **not** a counter-claim — it is CT, IG, or (if cited) RIPR. If you are not certain it refutes a specific prior claim, do not label it counter-claiming.

## D. Outlining the Purpose (OP) vs Announcing Present Research (APR)

OP vs APR turns on **aims vs methods**. A sentence stating **what this paper sets out to do, ask, or test** — its aims, objectives, research questions, or hypotheses ("this study aims to…", "the purpose of this paper is…", "we ask whether…", "this paper provides an overview of…") — is **OP**, even if it briefly mentions a method. A sentence stating **how the study is carried out** — methodology, design, data, sample, instruments, or the rationale for a method choice — is **APR**.
"""

SYSTEM_PROMPT_CLF_MANUAL_V2 = SYSTEM_PROMPT_CLF_MANUAL + CLARIFICATIONS_CLF_V2
SYSTEM_PROMPT_CLASSIFICATION_V2 = SYSTEM_PROMPT_CLASSIFICATION + CLARIFICATIONS_CLF_V2



# ---------------------------------------------------------------------------
# Sentence-by-sentence classification prompts
# ---------------------------------------------------------------------------
# In this variant the model never sees the whole introduction at once. It is
# fed one TARGET sentence at a time, together with the immediately preceding and
# following sentence as context, and must assign a single CARS label to the
# TARGET sentence. Segments are reconstructed afterwards by merging consecutive
# sentences that share the same predicted label.

# Re-use the taxonomy + decision procedure already written for the full-intro
# classification prompt so the two variants stay in sync (only the framing and
# output format differ).
_TAX_START = SYSTEM_PROMPT_CLASSIFICATION.index("## Swales CARS Label Taxonomy")
_TAX_END = SYSTEM_PROMPT_CLASSIFICATION.index("## Segmentation rules")
CARS_TAXONOMY_BLOCK = SYSTEM_PROMPT_CLASSIFICATION[_TAX_START:_TAX_END].rstrip() + "\n"

# Canonical label list (double-underscore form used in the prompt taxonomy).
CARS_LABELS = [
    "move_1__claiming_centrality",
    "move_1__making_topic_generalization",
    "move_1__reviewing_items_of_previous_research",
    "moves_2__counter_claiming",
    "moves_2__indicating_a_gap",
    "moves_2__question_raising",
    "moves_2__continuing_a_tradition",
    "move_3__outlining_purpose",
    "move_3__announcing_present_research",
    "move_3__announcing_principal_findings",
    "move_3__indicating_the_structure_of_research",
]

_SENTENCE_TASK_FRAMING = """## Task
You are analysing a research article introduction **one sentence at a time**. For each request you receive:
- `[PREVIOUS]` — the sentence immediately before the target (context only; may be empty at the start of the introduction),
- `[TARGET]`   — the single sentence you must classify,
- `[NEXT]`     — the sentence immediately after the target (context only; may be empty at the end).

Assign exactly ONE Swales CARS label to the **TARGET** sentence, according to its **communicative function** in context. Use `[PREVIOUS]` and `[NEXT]` only to disambiguate the target's rhetorical role — for example to resolve anaphoric references ("this", "such", "the former"), continued reasoning, or a list that spans sentences. Do NOT classify the context sentences; classify only the target.

When the target merely continues the same communicative step as its neighbour (e.g. it elaborates, exemplifies, qualifies, or contrasts within the same rhetorical goal, or is joined by a connector such as "However", "Moreover", "Therefore"), it should receive the **same label** as that neighbouring step. A label changes only when the communicative function genuinely changes.

## Output Format
Return a JSON object with a single "label" field:
{"label": "<one label from the taxonomy below>"}

"""

SYSTEM_PROMPT_SENTENCE_CLF = (
    "You are an expert in academic discourse analysis, specialising in Swales' "
    "CARS (Create a Research Space) model for research article introductions.\n\n"
    + _SENTENCE_TASK_FRAMING
    + "---\n\n"
    + CARS_TAXONOMY_BLOCK
)


_MANUAL_FRAMING_SENTENCE_CLF = """You are an expert in academic discourse analysis, specialising in Swales' CARS (Create a Research Space) model for research article introductions.

## Task
You are analysing a research article introduction **one sentence at a time**, following the annotation manual below. For each request you receive:
- `[PREVIOUS]` — the sentence immediately before the target (context only; may be empty),
- `[TARGET]`   — the single sentence you must classify,
- `[NEXT]`     — the sentence immediately after the target (context only; may be empty).

Assign exactly ONE Swales CARS label to the **TARGET** sentence, according to its communicative function as defined in the manual. Use `[PREVIOUS]` and `[NEXT]` only to disambiguate the target's role (anaphora, continued reasoning, lists that span sentences). Classify only the target sentence.

When the target merely continues the same communicative step as its neighbour, it should receive the **same label** as that step; the label changes only when the communicative function genuinely changes.

## Output Format
Return a JSON object with a single "label" field:
{"label": "<one label from the list below>"}

## Label format
Use exactly one of these label strings:
- move_1__claiming_centrality
- move_1__making_topic_generalization
- move_1__reviewing_items_of_previous_research
- moves_2__counter_claiming
- moves_2__indicating_a_gap
- moves_2__question_raising
- moves_2__continuing_a_tradition
- move_3__outlining_purpose
- move_3__announcing_present_research
- move_3__announcing_principal_findings
- move_3__indicating_the_structure_of_research

---

# Annotation manual (authoritative source for labels and decisions)

"""

SYSTEM_PROMPT_SENTENCE_CLF_MANUAL = _MANUAL_FRAMING_SENTENCE_CLF + _MANUAL_TEXT


# User message template for one sentence (with its context window).
SENTENCE_USER_TEMPLATE = """Classify the TARGET sentence below. Return only the JSON object.

[PREVIOUS]: {prev}
[TARGET]: {target}
[NEXT]: {next}"""


# ---------------------------------------------------------------------------
# Few-shot prompt builders
# ---------------------------------------------------------------------------

import json


def _clean(text: str) -> str:
    """Collapse whitespace runs to single spaces (mirrors run_experiment._clean)."""
    return " ".join(text.split())


def _gold_segments_for_task(task: dict) -> list[dict]:
    """Extract (text, label) gold segments from a Label Studio task in order."""
    text = task["data"].get("clean_text") or _clean(task["data"]["text"])
    results = task["annotations"][0]["result"]
    segs = []
    for r in results:
        if r.get("type") != "labels":
            continue
        v = r["value"]
        s, e = v["start"], v["end"]
        # Re-map start/end into clean_text if needed
        raw = task["data"]["text"]
        clean_prefix = _clean(raw[:s])
        clean_seg = _clean(raw[s:e])
        cs = len(clean_prefix) + (1 if clean_prefix and not clean_prefix.endswith(" ") and clean_seg else 0)
        # Simpler: locate clean_seg directly in text
        idx = text.find(clean_seg)
        if idx == -1:
            continue
        label = v["labels"][0] if v.get("labels") else None
        # Normalize label: gold uses single underscore; prompt taxonomy uses double underscore.
        if label and "__" not in label:
            label = label.replace("move_1_", "move_1__", 1) \
                         .replace("moves_2_", "moves_2__", 1) \
                         .replace("move_3_", "move_3__", 1)
        segs.append({"start": idx, "end": idx + len(clean_seg), "text": clean_seg, "label": label})
    segs.sort(key=lambda s: s["start"])
    return segs


def _balanced_example_ids(large: list[int] | None = None,
                          average: list[int] | None = None,
                          fine: list[int] | None = None,
                          n: int | None = None,
                          exclude: int | None = None) -> list[int]:
    """Return a balanced list of few-shot IDs spanning all three granularity
    groups. By default picks ONE example from each group (large-span → average →
    fine-grained), so the prompt shows the full segmentation range in 3 examples.

    If `exclude` is given (the id of the task currently being annotated), it is
    never returned as a demo: within each group we prefer the first member that
    is not `exclude`, falling back to the other member of the same group. This
    prevents train/test leakage when an eval paper is itself a reserved demo
    (e.g. annotating task 34 swaps in 66, and vice versa).
    """
    large = list(large if large is not None else FEW_SHOT_LARGE_SPAN_IDS)
    average = list(average if average is not None else FEW_SHOT_AVERAGE_IDS)
    fine = list(fine if fine is not None else FEW_SHOT_FINE_GRAINED_IDS)
    groups = [large, average, fine]
    if exclude is not None:
        # Push the excluded id to the back of its group so it is only chosen if
        # the group has no other member.
        groups = [[x for x in g if x != exclude] + ([exclude] if exclude in g else [])
                  for g in groups]
    if n is None:
        # one from each group, ordered few → average → many segments
        return [g[0] for g in groups if g]
    # enforce >=1 from each group, distributing the remainder round-robin
    take = [1 if g else 0 for g in groups]
    remaining = n - sum(take)
    gi = 0
    while remaining > 0:
        g = groups[gi % len(groups)]
        if take[gi % len(groups)] < len(g):
            take[gi % len(groups)] += 1
            remaining -= 1
        gi += 1
        if gi > 100:
            break
    return [gid for g, k in zip(groups, take) for gid in g[:k]]


def _format_segments_seg_only(segs: list[dict]) -> str:
    payload = {"segments": [{"text": s["text"]} for s in segs]}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_segments_with_labels(segs: list[dict]) -> str:
    payload = {"segments": [{"text": s["text"], "label": s["label"]} for s in segs]}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_segments_with_numbers(segs: list[dict]) -> str:
    """Few-shot formatter for the v4 numbered scheme: emit the class NUMBER."""
    payload = {"segments": [{"text": s["text"], "label": label_to_number(s["label"])}
                            for s in segs]}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_segments_with_numbers_rationale(segs: list[dict],
                                             task_id: int | None = None) -> str:
    """Few-shot formatter for v5: number + a move-first rationale per segment.

    Prefers a model-generated, example-specific rationale (keyed by task_id +
    segment index); falls back to the generic per-class template.
    """
    out = []
    for i, s in enumerate(segs):
        num = label_to_number(s["label"])
        out.append({"text": s["text"],
                    "rationale": rationale_for(task_id, i, num),
                    "label": num})
    return json.dumps({"segments": out}, indent=2, ensure_ascii=False)


def build_few_shot_block(mode: str, all_tasks: list[dict],
                         ids: list[int] | None = None,
                         exclude_id: int | None = None,
                         numbered: bool = False,
                         rationale: bool = False) -> str:
    """Build the worked-examples block for the few-shot prompt.

    mode = "segmentation" -> examples include only text
    mode = "classification" -> examples include text + label
    numbered = True (v4/v5) -> classification examples emit the class NUMBER (1–11)
        instead of the label string, matching the v4/v5 output format.
    rationale = True (v5) -> each numbered example also carries a one-sentence
        gold rationale (move-first), matching the v5 output format.

    exclude_id: id of the task being annotated; if it is itself a reserved demo
    the other member of its granularity group is substituted (no self-leakage).
    """
    if ids is None:
        ids = _balanced_example_ids(exclude=exclude_id)
    by_id = {t["id"]: t for t in all_tasks}

    block = "\n\n---\n\n# Worked examples\n\n"
    block += ("Below are worked examples drawn from the corpus, spanning the full "
              "range of segmentation granularity: some introductions are best "
              "segmented into a few large segments (when many sentences share the "
              "same communicative function), most into a moderate number, and some "
              "into many smaller segments (when the function changes frequently). "
              "Use the **communicative function** of each sentence, not its length "
              "or position, to decide.\n\n")

    if mode == "classification":
        if numbered and rationale:
            formatter = _format_segments_with_numbers_rationale
        elif numbered:
            formatter = _format_segments_with_numbers
        else:
            formatter = _format_segments_with_labels
    else:
        formatter = _format_segments_seg_only

    for i, ex_id in enumerate(ids, 1):
        task = by_id.get(ex_id)
        if not task:
            continue
        text = task["data"].get("clean_text") or _clean(task["data"]["text"])
        segs = _gold_segments_for_task(task)
        block += f"## Example {i} (task {ex_id} — {len(segs)} segment(s))\n\n"
        block += f"**Introduction:**\n{text}\n\n"
        label = "Correct segmentation and labels" if mode == "classification" else "Correct segmentation"
        rendered = (formatter(segs, ex_id) if formatter is _format_segments_with_numbers_rationale
                    else formatter(segs))
        block += f"**{label}:**\n{rendered}\n\n"

    return block


# ---------------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt version registry
# ---------------------------------------------------------------------------
# Each prompt iteration gets a version id so runs are reproducible and we can
# track which prompt produced which results. The version only changes the
# *base* (rules/manual) content; --mode, --few-shot, --prompt-source compose on
# top of it exactly as before. PROMPT_VERSIONS below is the changelog.
#
#   v1 — baseline. The manual/builtin prompt exactly as first used; this is the
#        configuration behind run 2026-06-01_15-52-20_classification-fewshot-manual.
#   v2 — v1 + CLARIFICATIONS_CLF_V2 (classification only). Operational
#        tie-breakers for CC-vs-RR/MTG, segmentation granularity / move
#        recurrence, and gap recall + OP/APR — the three fixes from the v1
#        error analysis. Segmentation-only runs are unaffected (no labels).
PROMPT_VERSIONS = {
    "v1": {
        "date": "2026-05-28",
        "summary": "Baseline: manual/builtin prompt as first used (no operational clarifications).",
    },
    "v2": {
        "date": "2026-06-03",
        "summary": "v1 + operational clarifications for classification: CC-vs-RR/MTG "
                   "tie-breakers, segmentation granularity / move recurrence, gap recall "
                   "+ OP/APR disambiguation. Targets the top v1 confusions.",
    },
    "v3": {
        "date": "2026-06-10",
        "summary": "v1 + CLARIFICATIONS_CLF_V3. From the 10-paper held-out error "
                   "analysis: citation⇒RIPR / no-citation⇒not-RIPR as a hard mechanical "
                   "rule (also enforced by postprocess.apply_citation_rules), removed the "
                   "aggressive 'insufficiency-after-review=IG' rule, added an existence-of-"
                   "research CT-vs-IG tie-breaker and a 'counter-claiming is rare' caution. "
                   "Pair with citation post-processing in run_experiment.py.",
    },
    "v4": {
        "date": "2026-06-10",
        "summary": "v3 clarifications, restructured: segmentation instructions FIRST, the "
                   "model predicts a NUMBER (1–11) per segment (mapped back to the label by "
                   "run_experiment), and the 'Guiding principle' communicative-function "
                   "reminder is moved to the BOTTOM. Classification + manual source only.",
    },
    "v5": {
        "date": "2026-06-10",
        "summary": "v4 + move-first reasoning (identify Move 1/2/3, then the sub-step) to "
                   "cut cross-move errors, plus a one-sentence 'rationale' field the model "
                   "writes before each label; few-shot demos carry gold move-first "
                   "rationales. Classification + manual source only.",
    },
    "v6": {
        "date": "2026-06-10",
        "summary": "v5 (move-first + rationale, generated few-shot rationales) + three "
                   "tie-breaker fixes from the v5 rationale-error analysis: F1 one-function-"
                   "per-segment (split when the rationale says 'and') to stop topic-merging; "
                   "F2 described framework/approach ⇒ Move 1 (RR) not APR; F3 cited "
                   "deficiency/'move beyond' stays RR not Move 2. Classification + manual only.",
    },
    "v7": {
        "date": "2026-06-10",
        "summary": "v6 + Type-A and Type-B fixes from the v6 rationale-error analysis. "
                   "A*: citation is a CUE not a switch — cited importance⇒CC(1), uncited "
                   "synthesis⇒RR(3), cited concept-definition⇒MTG(2). B*: framework/approach "
                   "tied to THIS study (first-person/own-data) ⇒ APR(9), else RR(3). Best "
                   "paired with --no-citation-postproc (the post-processor would undo A*). "
                   "Classification + manual only.",
    },
    "v8": {
        "date": "2026-06-10",
        "summary": "v6 + B* + focused Move-1 fixes for the CC/MTG/RR gap (from the 32-paper "
                   "human comparison): M1 three-function rule, M2 centrality-precision (plain "
                   "general statement⇒MTG not CC), M3 worked CC/MTG/RR contrasts (teach the "
                   "boundary by example), strengthened cited-importance⇒CC overriding the "
                   "citation default. Drops v7's A*2/A*3 scatter. Run with --no-citation-postproc. "
                   "Classification + manual only.",
    },
}
LATEST_PROMPT_VERSION = "v3"


def get_system_prompt(mode: str, few_shot: bool, source: str,
                      all_tasks: list[dict] | None = None,
                      version: str = "v1",
                      exclude_task_id: int | None = None) -> str:
    """Assemble the system prompt for a given experiment configuration.

    mode: "segmentation" | "classification"
    few_shot: include worked corpus examples
    source: "builtin" | "manual"
    all_tasks: required when few_shot=True
    version: prompt version id (see PROMPT_VERSIONS). Default "v1" preserves the
        original baseline behaviour. "v2" appends CLARIFICATIONS_CLF_V2 to the
        classification base (no effect on segmentation-only runs).
    exclude_task_id: id of the task being annotated. When it is itself a reserved
        few-shot demo (e.g. 34 or 66), the other member of its granularity group
        is substituted so the model never sees the gold labels of the very paper
        it is annotating. No effect for non-demo tasks.
    """
    if mode not in ("segmentation", "classification"):
        raise ValueError(f"Unknown mode: {mode}")
    if source not in ("builtin", "manual"):
        raise ValueError(f"Unknown source: {source}")
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version!r}. "
                         f"Known: {sorted(PROMPT_VERSIONS)}")

    # v4–v8 are classification-only, manual-source restructurings (numbered
    # labels, segmentation-first, reminder at the bottom). v5+ add move-first
    # reasoning + a one-sentence rationale field; v6/v7/v8 add tie-breaker fixes.
    _NUMBERED_VERSIONS = ("v4", "v5", "v6", "v7", "v8")
    numbered = (version in _NUMBERED_VERSIONS and mode == "classification")
    rationale = (version in ("v5", "v6", "v7", "v8") and mode == "classification")
    if version in _NUMBERED_VERSIONS and (mode != "classification" or source != "manual"):
        raise ValueError(f"prompt version {version} supports only --mode classification "
                         "--prompt-source manual")

    if version in ("v5", "v6", "v7", "v8"):
        base = SYSTEM_PROMPT_CLF_MANUAL_V5
    elif version == "v4":
        base = SYSTEM_PROMPT_CLF_MANUAL_V4
    elif source == "builtin":
        base = SYSTEM_PROMPT_CLASSIFICATION if mode == "classification" else SYSTEM_PROMPT_BASE
    else:
        base = SYSTEM_PROMPT_CLF_MANUAL if mode == "classification" else SYSTEM_PROMPT_SEG_MANUAL

    # v2 → V2 clarifications; v3+ → V3 clarifications; v6/v7/v8 also append
    # tie-breaker fixes (after the manual + V3 block). v7 = V6 fixes + A*/B*;
    # v8 = V6 fixes + focused Move-1 fixes (+ B*).
    if mode == "classification":
        if version == "v2":
            base = base + CLARIFICATIONS_CLF_V2
        elif version in ("v3", "v4", "v5", "v6", "v7", "v8"):
            base = base + CLARIFICATIONS_CLF_V3
            if version in ("v6", "v7", "v8"):
                base = base + CLARIFICATIONS_CLF_V6_FIXES
            if version == "v7":
                base = base + CLARIFICATIONS_CLF_V7_FIXES
            if version == "v8":
                base = base + CLARIFICATIONS_CLF_V8_FIXES

    # The "Guiding principle" reminder goes at the top for v1–v3 and at the
    # BOTTOM for v4/v5 (after the manual, clarifications, and worked examples).
    prompt = base if numbered else (COMMUNICATIVE_FUNCTION_REMINDER + base)

    if few_shot:
        if all_tasks is None:
            raise ValueError("all_tasks is required when few_shot=True")
        if source == "builtin" and mode == "segmentation":
            prompt += "\n\n" + GUIDELINE_MINI_EXAMPLES
        prompt += build_few_shot_block(mode, all_tasks, exclude_id=exclude_task_id,
                                       numbered=numbered, rationale=rationale)
        # v8 adds a dedicated CC/MTG/RR boundary primer as a worked example.
        if version == "v8":
            prompt += BOUNDARY_PRIMER_V8

    if numbered:
        prompt += "\n\n---\n\n" + COMMUNICATIVE_FUNCTION_REMINDER

    return prompt


# ---------------------------------------------------------------------------
# Sentence-level few-shot + entry point
# ---------------------------------------------------------------------------

def _label_for_offset(gold_segs: list[dict], lo: int, hi: int) -> str | None:
    """Label of the gold segment that best overlaps the [lo, hi) span."""
    best_ov, best_lbl = 0, None
    for g in gold_segs:
        ov = max(0, min(hi, g["end"]) - max(lo, g["start"]))
        if ov > best_ov:
            best_ov, best_lbl = ov, g["label"]
    return best_lbl


def build_sentence_few_shot_block(all_tasks: list[dict], split_fn,
                                  ids: list[int] | None = None,
                                  max_per_label: int = 2) -> str:
    """Build a worked-examples block of single-sentence classifications.

    For each reserved few-shot task, the gold-annotated introduction is split
    into sentences (via `split_fn`, the same splitter the experiment uses); each
    sentence inherits the label of the gold segment it falls in. We then show a
    handful of `[PREVIOUS] / [TARGET] / [NEXT] -> {"label": ...}` examples,
    balanced so each label appears at most `max_per_label` times.

    split_fn(text) -> list[(sentence_text, start, end)].
    """
    if ids is None:
        ids = _balanced_example_ids()
    by_id = {t["id"]: t for t in all_tasks}

    # Collect candidate (prev, target, next, label) examples across the demos.
    candidates: list[tuple[str, str, str, str]] = []
    for ex_id in ids:
        task = by_id.get(ex_id)
        if not task:
            continue
        text = task["data"].get("clean_text") or _clean(task["data"]["text"])
        gold = _gold_segments_for_task(task)
        if not gold:
            continue
        sents = split_fn(text)
        labels = [_label_for_offset(gold, s, e) for (_, s, e) in sents]
        for i, ((stext, _, _), lbl) in enumerate(zip(sents, labels)):
            if not lbl:
                continue
            prev = sents[i - 1][0] if i > 0 else ""
            nxt = sents[i + 1][0] if i + 1 < len(sents) else ""
            candidates.append((prev, stext, nxt, lbl))

    # Balance: keep at most `max_per_label` examples per label, preserving order.
    seen: dict[str, int] = {}
    chosen: list[tuple[str, str, str, str]] = []
    for cand in candidates:
        lbl = cand[3]
        if seen.get(lbl, 0) >= max_per_label:
            continue
        seen[lbl] = seen.get(lbl, 0) + 1
        chosen.append(cand)

    block = "\n\n---\n\n# Worked examples (single-sentence classification)\n\n"
    block += ("Each example shows a target sentence in its context window and the "
              "correct label. Note how a sentence joined by a connector or an "
              "anaphoric reference keeps the label of the step it continues, while "
              "a genuine shift in communicative function changes the label.\n\n")
    for i, (prev, target, nxt, lbl) in enumerate(chosen, 1):
        block += f"## Example {i}\n"
        block += f"[PREVIOUS]: {prev}\n[TARGET]: {target}\n[NEXT]: {nxt}\n"
        block += json.dumps({"label": lbl}, ensure_ascii=False) + "\n\n"
    return block


def get_sentence_system_prompt(source: str, few_shot: bool,
                               all_tasks: list[dict] | None = None,
                               split_fn=None) -> str:
    """Assemble the system prompt for the sentence-by-sentence classifier.

    source:   "builtin" | "manual"
    few_shot: include worked single-sentence examples
    split_fn: sentence splitter (required when few_shot=True)
    """
    if source not in ("builtin", "manual"):
        raise ValueError(f"Unknown source: {source}")

    base = (SYSTEM_PROMPT_SENTENCE_CLF_MANUAL if source == "manual"
            else SYSTEM_PROMPT_SENTENCE_CLF)
    prompt = COMMUNICATIVE_FUNCTION_REMINDER + base

    if few_shot:
        if all_tasks is None or split_fn is None:
            raise ValueError("all_tasks and split_fn are required when few_shot=True")
        prompt += build_sentence_few_shot_block(all_tasks, split_fn)

    return prompt
