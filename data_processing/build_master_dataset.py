#!/usr/bin/env python3
"""Build the master paper-level dataset: one JSON file with every paper's
metadata, every introduction (original + AI-generated), and every annotation
(human + LLM) — the single input for all downstream statistics.

Output structure (data/master_dataset.json):

{
  "meta": { build info, sources, label scheme, coverage summary },
  "papers": [
    {
      "paper_id": 1, "paper_key": "001",
      "title": ..., "filename": ..., "nativeness": ..., "quartile": ...,
      "swales_topic": {          # is the paper ITSELF about Swales/rhetorical analysis?
        "is_swales_related": bool,       # any term hit (incl. broad rhetorical-analysis vocab)
        "mentions_swales_or_cars": bool, # strict: explicit Swales/CARS mention
        "matched_terms": [...], "matched_in": [...],  # where: title/abstract/original_intro/body
        "abstract_found": bool   # searched over the paper's FULL text
      },
      "intros": [
        {
          "intro_id": "001_original", "intro_type": "original",
          "model": null, "provider": null, "mode": null,
          "text": <canonical text — all annotation offsets refer to this>,
          "word_count": ..., "char_count": ...,
          "target_words": null, "length_ratio": null,
          "generation_run": null, "request_date": null, "n_cited_refs": null,
          "citation_usage": { see below },
          "annotations": {
            "expert":      { role "gold",   segments [...] },
            "annotator_1": { role "second", segments [...] },   # 16 papers
            "annotator_2": { role "second", segments [...] }    # 16 papers
            # LLM auto-labels can be merged with --llm-run name=run_dir
          }
        },
        { "intro_id": "001_gpt-5.5_basic", "intro_type": "generated",
          "model": "gpt-5.5", "provider": "openai", "mode": "basic",
          "text": ..., "word_count": ..., "target_words": ..., "length_ratio": ...,
          "generation_run": "2026-07-07_10-31-37_gpt-5.5_basic",
          "citation_usage": {
            "n_refs_given": 8,          # refs fed to the model (cited in the original intro)
            "n_refs_used": 6,           # of those, how many the text actually cites
            "used_ratio": 0.75,
            "n_citations_in_text": 14,  # distinct (author, year) citations extracted
            "n_citations_matched": 7,
            "n_citations_unmatched": 7, # citations to works NOT in the given list
            "used_ref_indices": [...],  # `index` of each used ref in references_json_with_citations
            "unmatched_citations": ["Kang, 2015", ...]
          },
          "annotations": {} },   # Step-8 auto-labels land here
        ... one entry per (model, mode) generation run ...
      ]
    }, ...
  ]
}

The original intro also carries `citation_usage`, computed against the same
reference set (the works cited in the original intro). Since that set is
extracted FROM the original intro, its used_ratio should be close to 1.0 —
it doubles as a validation of the citation extractor, and as the human
baseline for the "how many given works did the model use" comparison.

Every segment is normalized to:
  { "start", "end", "text" (exact slice of the intro's canonical text),
    "label" (canonical "move_N__step"), "label_raw" (as in the source),
    "move" ("move_1"|"move_2"|"move_3"), "step", "in_scheme" }

Label normalization: the sources disagree on separators ("move_1_x" vs
"move_1__x") and Move-2 spelling ("moves_2"); everything is mapped to the
canonical "move_{n}__{step}" form. Out-of-scheme labels (e.g. annotator 2's
"moves_2_make_a_counterclaim") are kept but flagged in_scheme=false.

The ChatGPT run's offsets refer to its own whitespace-collapsed copy of the
text; they are remapped onto the canonical tasks.json text via non-whitespace
character alignment, so ALL annotations of an intro share one offset space.

Usage (from repo root):
    python data_processing/build_master_dataset.py
    # options:
    #   --llm-run chatgpt=intro_analysis/results/<run_dir>   (repeatable)
    #   --intro-llm-run intro_analysis/results/<run_dir>     (repeatable;
    #       a Step-8 annotate_master_dataset.py run — auto-labels for ALL
    #       intros, original + generated, merged by intro_id)
    #   --output data/master_dataset.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EXCEL = REPO_ROOT / "data/raw_data/80-papers_corpus_nativeness_quartiles.xlsx"
DEFAULT_HUMAN_ANNOTATIONS = REPO_ROOT / "data/annotation/all_human_annotations.json"
DEFAULT_GENERATION_RESULTS = REPO_ROOT / "intro_generation/results"
DEFAULT_REFS_DIR = REPO_ROOT / "data/raw_data/references_json_with_citations"
DEFAULT_PROCESSED_TEXTS = REPO_ROOT / "data/raw_data/processed_texts"
# No LLM annotations by default (the ChatGPT auto-labels will be re-run).
# When ready, merge a run with:
#   --llm-run chatgpt=intro_analysis/results/<run_dir>
# e.g. the v8-FINAL run: 2026-06-10_17-44-18_classification-fewshot-manual-v8-m0primer
DEFAULT_LLM_RUNS: dict[str, Path] = {}
DEFAULT_OUTPUT = REPO_ROOT / "data/master_dataset.json"

# Terms that mark a paper as being ABOUT Swales-style rhetorical analysis.
# Case-insensitive, except "CARS" which must appear as an uppercase word.
SWALES_TERMS_CI = [
    "swales",
    "create a research space",
    "creating a research space",
    "rhetorical move",
    "rhetorical structure",
    "rhetorical analysis",
    "move analysis",
    "move structure",
    "genre analysis",
]
SWALES_TERMS_CS = ["CARS"]
# The strict subset: an explicit mention of Swales or his CARS model
# (vs the broader rhetorical-analysis vocabulary above).
SWALES_STRONG_TERMS = {"swales", "create a research space",
                       "creating a research space", "CARS"}

CANONICAL_SCHEME = {
    "move_1": [
        "claiming_centrality",
        "making_topic_generalization",
        "reviewing_items_of_previous_research",
    ],
    "move_2": [
        "continuing_a_tradition",
        "indicating_a_gap",
        "question_raising",
    ],
    "move_3": [
        "announcing_present_research",
        "announcing_principal_findings",
        "indicating_the_structure_of_research",
        "outlining_purpose",
    ],
}

ANNOTATOR_ROLES = {"expert": "gold", "annotator_1": "second", "annotator_2": "second"}

_LABEL_RE = re.compile(r"^moves?_(\d)_{1,2}(.+)$")


def normalize_label(raw: str) -> dict:
    m = _LABEL_RE.match(raw.strip())
    if not m:
        return {"label": raw, "move": None, "step": None, "in_scheme": False}
    move = f"move_{m.group(1)}"
    step = m.group(2)
    return {
        "label": f"{move}__{step}",
        "move": move,
        "step": step,
        "in_scheme": step in CANONICAL_SCHEME.get(move, []),
    }


def make_segment(start: int, end: int, canonical_text: str, raw_label: str) -> dict:
    seg = {"start": start, "end": end, "text": canonical_text[start:end]}
    seg.update(normalize_label(raw_label))
    seg["label_raw"] = raw_label
    return seg


# ---------------------------------------------------------------------------
# Offset remapping (run's whitespace-collapsed text -> canonical text)
# ---------------------------------------------------------------------------

def _nonws_positions(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if not ch.isspace()]


def remap_span(start: int, end: int, src_text: str, src_nonws: list[int],
               src_nonws_rank: list[int], dst_nonws: list[int]) -> tuple[int, int]:
    """Map [start, end) in src_text to the canonical text by aligning the
    k-th non-whitespace character of each. Returns a span trimmed to
    non-whitespace boundaries."""
    # rank of first non-ws char at/after `start`, and of last non-ws char before `end`
    first = src_nonws_rank[start]
    last = src_nonws_rank[end - 1] if not src_text[end - 1].isspace() else src_nonws_rank[end] - 1
    if first >= len(dst_nonws) or last < first:
        raise ValueError(f"cannot remap span ({start},{end})")
    return dst_nonws[first], dst_nonws[last] + 1


def build_nonws_rank(text: str) -> list[int]:
    """rank[i] = number of non-whitespace chars strictly before i (len = len(text)+1)."""
    rank = [0] * (len(text) + 1)
    n = 0
    for i, ch in enumerate(text):
        rank[i] = n
        if not ch.isspace():
            n += 1
    rank[len(text)] = n
    return rank


def collapse(text: str) -> str:
    return " ".join(text.split())


def _collapse_nows(text: str) -> str:
    """All whitespace removed — the unit of the non-ws rank alignment."""
    return "".join(text.split())


def _nonws_rank_map(src_c: str, dst_c: str) -> dict[int, int]:
    """Best-effort map from src collapsed-char index to dst collapsed-char index
    when the two texts are NOT identical after whitespace removal (e.g. an
    annotator's export carries a '[PILOT] ' prefix or a handful of divergent
    characters). Built from difflib matching blocks; indices inside non-matching
    stretches are absent and callers snap to the nearest mapped neighbour."""
    sm = difflib.SequenceMatcher(None, src_c, dst_c, autojunk=False)
    mp: dict[int, int] = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            mp[a + k] = b + k
    return mp


def remap_span_aligned(start: int, end: int, src_text: str,
                       src_rank: list[int], dst_nonws: list[int],
                       rank_map: dict[int, int] | None) -> tuple[int, int]:
    """Like remap_span, but tolerates small non-whitespace differences between
    the source and canonical texts via `rank_map` (None = texts identical after
    whitespace removal, use the direct k-th-non-ws-char alignment)."""
    first = src_rank[start]
    last = src_rank[end - 1] if not src_text[end - 1].isspace() else src_rank[end] - 1
    if rank_map is not None:
        f, l = first, last
        while f <= last and f not in rank_map:   # snap forward
            f += 1
        while l >= f and l not in rank_map:      # snap backward
            l -= 1
        if f > l:
            raise ValueError(f"span ({start},{end}) has no alignable characters")
        first, last = rank_map[f], rank_map[l]
    if first >= len(dst_nonws) or last < first or last >= len(dst_nonws):
        raise ValueError(f"cannot remap span ({start},{end})")
    return dst_nonws[first], dst_nonws[last] + 1


# ---------------------------------------------------------------------------
# Citation extraction & usage (generated intros vs the refs fed to the model)
# ---------------------------------------------------------------------------

# Narrative citations: "Hung (2017)", "Smith and Jones (2019)", "Lee et al.
# (2020a)", "Hyland (2001, 2008)", "Gibbons (2002: 16)", "Allan (2001, p. 148)"
_NARRATIVE_CIT_RE = re.compile(
    r"([A-Z][\w'’\-]+"                                   # first surname
    r"(?:(?:,\s*|\s+(?:and|&)\s+)[A-Z][\w'’\-]+)*"       # more surnames
    r"(?:['’]s)?(?:\s+et\s+al\.?)?)"                     # possessive / et al.
    r"\s*\(\s*(\d{4}[a-z]?(?:\s*,\s*\d{4}[a-z]?)*)"      # one or more years
    r"(?:\s*[,:;][^)]*)?\)",                             # optional page suffix
    re.UNICODE,
)
# Possessive citation without parentheses: "Gebhard's 2010 case study"
_POSSESSIVE_CIT_RE = re.compile(r"([A-Z][\w'’\-]+)['’]s\s+(\d{4})\b")
# Parenthetical groups (innermost) that contain a year
_PAREN_GROUP_RE = re.compile(r"\(([^()]*\b\d{4}[a-z]?\b[^()]*)\)")
_YEAR_RE = re.compile(r"\b(\d{4})[a-z]?\b")
_CIT_PREFIX_RE = re.compile(
    r"^(?:e\.?g\.?|cf\.?|i\.?e\.?|see(?:\s+also)?|also|and|in|as\s+in|reviewed\s+in|adapted\s+from|from)[\s,.:]+",
    re.I,
)


def _norm_name(text: str) -> str:
    """Lowercase, strip diacritics, keep letters/spaces only."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z]+", " ", text)
    return " ".join(text.lower().split())


def _norm_cit_authors(raw: str) -> str:
    raw = raw.strip(" ,&")
    prev = None
    while prev != raw:  # strip stacked prefixes ("see also e.g., ...")
        prev, raw = raw, _CIT_PREFIX_RE.sub("", raw)
    norm = _norm_name(re.sub(r"\bet\s+al\.?|['’]s\b", " ", raw))
    return re.sub(r"^(?:and|also|see|eg|e g|cf|ie|i e|in)\s+", "", norm)


def _paren_group_citations(group: str, citations: set) -> None:
    """One parenthetical group: items split on ';', each item segmented into
    author + year-cluster pairs, so 'Kidd & Christiansen, 2018, and Williams,
    2020' yields two citations and 'Jiang, 2002, 2004' yields two years for
    one author."""
    for item in group.split(";"):
        clusters = []  # [author_raw, [years]]
        prev_end = 0
        for m in _YEAR_RE.finditer(item):
            between = item[prev_end:m.start()]
            if clusters and re.fullmatch(r"[\s,]*(?:and\s+)?", between):
                clusters[-1][1].append(m.group(1))
            else:
                clusters.append([between, [m.group(1)]])
            prev_end = m.end()
        for author_raw, years in clusters:
            norm = _norm_cit_authors(author_raw)
            if not norm:
                continue  # bare "(2019)" — caught by the narrative pattern
            for y in years:
                citations.add((norm, y))


def extract_citations(text: str) -> set[tuple[str, str]]:
    """All distinct author–year citations in `text` as (normalized_authors, year)."""
    citations = set()
    for authors, years in _NARRATIVE_CIT_RE.findall(text):
        norm = _norm_cit_authors(authors)
        if norm:
            for y in _YEAR_RE.findall(years):
                citations.add((norm, y))
    for surname, year in _POSSESSIVE_CIT_RE.findall(text):
        norm = _norm_cit_authors(surname)
        if norm:
            citations.add((norm, year))
    # Parenthetical citations; peel nested parentheses layer by layer so a
    # group containing an inner "(2011)" still gets processed.
    work = text
    for _ in range(3):
        groups = _PAREN_GROUP_RE.findall(work)
        for g in groups:
            _paren_group_citations(g, citations)
        stripped = re.sub(r"\([^()]*\)", " ", work)
        if stripped == work:
            break
        work = stripped
    return citations


def load_cited_refs(refs_dir: Path) -> dict[int, list[dict]]:
    """paper_id -> the refs fed to the generator (cited in the original intro).

    Each ref is reduced to a list of identity `variants` (author string, year)
    a citation can be matched against. Two identity sources are needed because
    they often disagree: (a) the enriched METADATA (first-author name + year —
    what the generation prompt showed the model, but year/authors are sometimes
    a different edition or plain wrong), and (b) the original in-text
    `citation_text` of every introduction_citation_contexts entry (surnames +
    year exactly as the original intro cites the work)."""
    out = {}
    for f in sorted(refs_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        d = json.load(open(f))
        if "references" not in d:
            continue
        pid = int(d["paper_id"])
        cited = []
        for r in d["references"]:
            contexts = r.get("introduction_citation_contexts")
            if not contexts:
                continue
            variants = []
            authors = r.get("authors") or r.get("author_surnames") or []
            if isinstance(authors, str):
                authors = [authors]
            first = _norm_name(authors[0]) if authors else ""
            year = _YEAR_RE.search(str(r.get("year") or ""))
            if first:
                variants.append((first, year.group(1) if year else None))
            for ctx in contexts:
                cit = ctx.get("citation_text") or ""
                names = _norm_name(re.sub(r"\bet\s+al\b\.?|\d", " ", cit))
                if not names:
                    continue
                years = _YEAR_RE.findall(cit)
                variants.extend((names, y) for y in years or [None])
            cited.append({"index": r.get("index"), "variants": variants})
        out[pid] = cited
    return out


_NAME_STOPWORDS = {"et", "al", "and", "the", "of", "de", "van", "von"}


def _authors_overlap(variant_authors: str, cit_authors: str) -> bool:
    """A ref identity may be a bare surname ('abaeian'), a full name
    ('hsui ting hung'), or an org name ('flipped learning network'); citations
    use surnames. Match if any usable name token is shared."""
    if re.search(rf"\b{re.escape(variant_authors)}\b", cit_authors):
        return True
    tokens = [t for t in variant_authors.split()
              if len(t) >= 2 and t not in _NAME_STOPWORDS]
    return any(re.search(rf"\b{t}\b", cit_authors) for t in tokens)


def citation_usage(text: str, given_refs: list[dict]) -> dict:
    """Compare the citations in `text` against the refs given to the model."""
    citations = extract_citations(text)
    used_indices = []
    matched_citations = set()
    for ref in given_refs:
        hits = {
            c for c in citations
            if any((vy is None or vy == c[1]) and _authors_overlap(va, c[0])
                   for va, vy in ref["variants"])
        }
        if hits:
            used_indices.append(ref["index"])
            matched_citations |= hits
    unmatched = sorted(citations - matched_citations)
    n_given = len(given_refs)
    return {
        "n_refs_given": n_given,
        "n_refs_used": len(used_indices),
        "used_ratio": round(len(used_indices) / n_given, 3) if n_given else None,
        "n_citations_in_text": len(citations),
        "n_citations_matched": len(matched_citations),
        "n_citations_unmatched": len(unmatched),
        "used_ref_indices": used_indices,
        "unmatched_citations": [f"{a}, {y}" for a, y in unmatched],
    }


# ---------------------------------------------------------------------------
# Swales-topic flag (is the paper itself about Swales / rhetorical analysis?)
# ---------------------------------------------------------------------------

def extract_abstract(processed_text: str) -> str | None:
    """The paper's abstract from its processed text. Handles the three shapes
    in the corpus: a '## abstract' heading, an inline 'Abstract: ...'
    paragraph, and an inline 'Summary. ...' paragraph."""
    m = re.search(r"^##\s*abstract\s*$(.*?)(?=^##\s|\Z)", processed_text,
                  re.I | re.M | re.S)
    if m:
        return m.group(1).strip()
    for para in processed_text.split("\n\n"):
        p = para.strip()
        if re.match(r"^(abstract|summary)\s*[:.—-]", p, re.I):
            return re.sub(r"^(abstract|summary)\s*[:.—-]\s*", "", p, flags=re.I).strip()
    return None


def swales_topic(title: str, abstract: str | None, intro_text: str,
                 body_text: str) -> dict:
    """Search the paper's ENTIRE text — title, abstract, original introduction
    and full body (processed_texts: lit review, methods, results, discussion) —
    for Swales / rhetorical-analysis vocabulary. `mentions_swales_or_cars` is
    the strict flag (explicit Swales/CARS mention anywhere); `is_swales_related`
    additionally counts the broader rhetorical-analysis terms."""
    sources = {
        "title": title,
        "abstract": abstract or "",
        "original_intro": intro_text,
        "body": body_text,
    }
    matched_terms: list[str] = []
    matched_in: list[str] = []
    for where, haystack in sources.items():
        low = haystack.lower()
        found = [t for t in SWALES_TERMS_CI if t in low]
        found += [t for t in SWALES_TERMS_CS
                  if re.search(rf"\b{re.escape(t)}\b", haystack)]
        if found:
            matched_in.append(where)
            matched_terms += [t for t in found if t not in matched_terms]
    return {
        "is_swales_related": bool(matched_terms),
        "mentions_swales_or_cars": any(t in SWALES_STRONG_TERMS for t in matched_terms),
        "matched_terms": matched_terms,
        "matched_in": matched_in,
        "abstract_found": abstract is not None,
        "searched_in": "title + abstract + original intro + full body",
    }


def load_processed_texts(processed_dir: Path) -> dict[int, str]:
    """paper_id -> full processed text (abstract + body, intro excluded)."""
    out = {}
    for f in sorted(processed_dir.glob("*.txt")):
        m = re.match(r"^(\d{3})_", f.name)
        if m:
            out[int(m.group(1))] = f.read_text()
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_corpus_metadata(excel_path: Path) -> dict[int, dict]:
    import pandas as pd

    df = pd.read_excel(excel_path)
    papers = {}
    for _, row in df.iterrows():
        pid = int(row["paper_id"])
        papers[pid] = {
            "paper_id": pid,
            "paper_key": f"{pid:03d}",
            "title": str(row["title"]).strip(),
            "filename": str(row["filename"]).strip(),
            "nativeness": str(row["nativeness"]).strip(),
            "quartile": str(row["quartile"]).strip(),
        }
    return papers


def load_human_annotations(path: Path) -> dict[int, dict]:
    """paper_id -> {"text": canonical text, "annotators": {name: entry}}.

    Span offsets in the source file are relative to each ANNOTATOR'S OWN text
    (annotator 2's Label Studio export carries a '[PILOT] ' prefix; annotator 1's XML
    texts differ from the canonical text by a few characters), so every span
    is remapped onto the canonical text via non-whitespace alignment — the
    same mechanism used for the LLM annotation runs — and validated against
    the span text stored alongside it."""
    entries = json.load(open(path))
    out = {}
    for e in entries:
        pid = int(e["paper_task_id"])
        text = e["text"]
        dst_nonws = _nonws_positions(text)
        dst_c = _collapse_nows(text)
        annotators = {}
        for name, ann in e["annotators"].items():
            src = ann.get("text") or text
            src_rank = build_nonws_rank(src)
            src_c = _collapse_nows(src)
            rank_map = None if src_c == dst_c else _nonws_rank_map(src_c, dst_c)
            segments = []
            for r in ann["result"]:
                cs, ce = remap_span_aligned(r["start"], r["end"], src,
                                            src_rank, dst_nonws, rank_map)
                got, want = _collapse_nows(text[cs:ce]), _collapse_nows(r["text"])
                if got != want and difflib.SequenceMatcher(
                        None, got, want, autojunk=False).ratio() < 0.97:
                    raise ValueError(
                        f"paper {pid} / {name}: span ({r['start']},{r['end']}) "
                        f"failed to remap onto the canonical text")
                segments.append(make_segment(cs, ce, text, r["labels"][0]))
            segments.sort(key=lambda s: s["start"])
            annotators[name] = {
                "annotator": name,
                "annotator_type": "human",
                "role": ANNOTATOR_ROLES.get(name, "second"),
                "model": None,
                "source": str(path.relative_to(REPO_ROOT)),
                "num_segments": len(segments),
                "segments": segments,
            }
        out[pid] = {"text": text, "annotators": annotators}
    return out


def load_llm_annotation_run(run_dir: Path, canonical_texts: dict[int, str]) -> dict[int, dict]:
    """paper_id -> annotation entry, with spans remapped onto the canonical text."""
    run_meta = json.load(open(run_dir / "run_meta.json"))
    items = json.load(open(run_dir / "ai_segments.json"))
    out = {}
    for item in items:
        pid = int(item["task_id"])
        canon = canonical_texts.get(pid)
        if canon is None:
            print(f"  [warn] LLM run has task {pid} not in corpus — skipped")
            continue
        src = item["text"]
        if collapse(src) != collapse(canon):
            print(f"  [warn] paper {pid}: run text != canonical text — skipped")
            continue
        src_rank = build_nonws_rank(src)
        dst_nonws = _nonws_positions(canon)
        segments = []
        for seg in item["ai"]:
            cs, ce = remap_span(seg["start"], seg["end"], src, None, src_rank, dst_nonws)
            if collapse(canon[cs:ce]) != collapse(src[seg["start"]:seg["end"]]):
                raise ValueError(f"paper {pid}: span remap mismatch at ({seg['start']},{seg['end']})")
            new = make_segment(cs, ce, canon, seg["label"])
            if seg.get("rationale"):
                new["rationale"] = seg["rationale"]
            segments.append(new)
        segments.sort(key=lambda s: s["start"])
        out[pid] = {
            "annotator": "chatgpt",
            "annotator_type": "llm",
            "role": "auto",
            "model": run_meta.get("model"),
            "provider": run_meta.get("provider"),
            "prompt_version": run_meta.get("prompt_version"),
            "run_tag": run_meta.get("run_tag"),
            "source": str(run_dir.relative_to(REPO_ROOT)),
            "num_segments": len(segments),
            "segments": segments,
        }
    return out


def discover_generation_runs(results_dir: Path) -> list[Path]:
    """All run dirs with a generated_intros.json; if the same (model, mode)
    appears twice, keep the latest (dir names sort chronologically)."""
    runs = {}
    for d in sorted(p.parent for p in results_dir.glob("*/generated_intros.json")):
        meta = json.load(open(d / "run_meta.json"))
        runs[(meta["model"], meta["mode"])] = d  # later timestamp overwrites earlier
    return sorted(runs.values())


def load_generation_run(run_dir: Path) -> tuple[dict, dict[int, dict]]:
    meta = json.load(open(run_dir / "run_meta.json"))
    intros = {}
    for e in json.load(open(run_dir / "generated_intros.json")):
        pid = int(e["paper_id"])
        text = e["introduction"]
        model_slug = re.sub(r"[^A-Za-z0-9.-]+", "-", e["model"])
        intros[pid] = {
            "intro_id": f"{pid:03d}_{model_slug}_{e['mode']}",
            "intro_type": "generated",
            "model": e["model"],
            "provider": e["provider"],
            "mode": e["mode"],
            "text": text,
            "word_count": e.get("word_count", len(text.split())),
            "char_count": len(text),
            "target_words": e.get("target_words"),
            "length_ratio": e.get("length_ratio"),
            "n_cited_refs": e.get("n_cited_refs"),
            "request_date": e.get("request_date"),
            "generation_run": run_dir.name,
            "annotations": {},
        }
    return meta, intros


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> dict:
    print(f"Corpus metadata: {args.excel}")
    papers = load_corpus_metadata(args.excel)
    print(f"  {len(papers)} papers")

    print(f"Human annotations: {args.human_annotations}")
    human = load_human_annotations(args.human_annotations)
    missing_gold = [pid for pid in papers if pid not in human or "expert" not in human[pid]["annotators"]]
    if missing_gold:
        sys.exit(f"FATAL: papers without gold (expert) annotation: {missing_gold}")

    canonical_texts = {pid: human[pid]["text"] for pid in human}

    llm_annotations = {}
    for name, run_dir in args.llm_runs.items():
        print(f"LLM annotation run '{name}': {run_dir.name}")
        llm_annotations[name] = load_llm_annotation_run(run_dir, canonical_texts)
        print(f"  {len(llm_annotations[name])} papers annotated")

    print(f"Cited refs (given to the generator): {args.refs_dir}")
    cited_refs = load_cited_refs(args.refs_dir)
    print(f"  {len(cited_refs)} papers, {sum(len(v) for v in cited_refs.values())} cited refs")

    print(f"Processed texts (for the Swales-topic flag): {args.processed_texts}")
    bodies = load_processed_texts(args.processed_texts)
    abstracts = {pid: extract_abstract(t) for pid, t in bodies.items()}
    n_abs = sum(1 for v in abstracts.values() if v)
    print(f"  {len(bodies)} texts, {n_abs} abstracts found")

    gen_runs = discover_generation_runs(args.generation_results)
    print(f"Generation runs found: {len(gen_runs)}")
    generated = []  # list of (run_meta, {pid: intro})
    for d in gen_runs:
        meta, intros = load_generation_run(d)
        print(f"  {d.name}: {len(intros)} intros ({meta['model']} / {meta['mode']})")
        generated.append((meta, intros))

    out_papers = []
    for pid in sorted(papers):
        p = dict(papers[pid])
        p["swales_topic"] = swales_topic(p["title"], abstracts.get(pid),
                                         canonical_texts[pid], bodies.get(pid, ""))
        text = canonical_texts[pid]
        given = cited_refs.get(pid, [])
        original = {
            "intro_id": f"{pid:03d}_original",
            "intro_type": "original",
            "model": None,
            "provider": None,
            "mode": None,
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "target_words": None,
            "length_ratio": None,
            "n_cited_refs": len(given),
            "request_date": None,
            "generation_run": None,
            "citation_usage": citation_usage(text, given),
            "annotations": dict(human[pid]["annotators"]),
        }
        for name, per_paper in llm_annotations.items():
            if pid in per_paper:
                original["annotations"][name] = per_paper[pid]

        intros = [original]
        for _, per_paper in generated:
            if pid in per_paper:
                intro = per_paper[pid]
                intro["citation_usage"] = citation_usage(intro["text"], given)
                intros.append(intro)
            else:
                print(f"  [warn] paper {pid} missing from a generation run")
        p["intros"] = intros
        out_papers.append(p)

    n_by_annotator = {}
    for p in out_papers:
        for name in p["intros"][0]["annotations"]:
            n_by_annotator[name] = n_by_annotator.get(name, 0) + 1

    dataset = {
        "meta": {
            "built": datetime.now().isoformat(timespec="seconds"),
            "description": (
                "Master paper-level dataset: corpus metadata + original and "
                "AI-generated introductions + all annotations (Swales CARS)."
            ),
            "builder": "data_processing/build_master_dataset.py",
            "sources": {
                "corpus_metadata": str(args.excel.relative_to(REPO_ROOT)),
                "human_annotations": str(args.human_annotations.relative_to(REPO_ROOT)),
                "llm_annotation_runs": {n: str(d.relative_to(REPO_ROOT)) for n, d in args.llm_runs.items()},
                "generation_runs": [d.name for d in gen_runs],
                "cited_refs": str(args.refs_dir.relative_to(REPO_ROOT)),
                "abstracts": str(args.processed_texts.relative_to(REPO_ROOT)),
            },
            "swales_topic_terms": {"case_insensitive": SWALES_TERMS_CI, "case_sensitive": SWALES_TERMS_CS},
            "label_scheme": CANONICAL_SCHEME,
            "n_papers": len(out_papers),
            "n_intros": sum(len(p["intros"]) for p in out_papers),
            "original_intro_annotation_coverage": n_by_annotator,
            "notes": [
                "All annotation offsets refer to the intro's `text` field.",
                "Segment `label` is normalized to move_{n}__{step}; `label_raw` keeps the source spelling.",
                "`in_scheme` is false for labels outside the 10 canonical CARS steps.",
                "Generated intros have empty `annotations` until the Step-8 auto-labelling run; "
                "LLM annotations of the original intros are merged only when --llm-run is given.",
                "citation_usage compares author–year citations extracted from the intro text against "
                "the refs fed to the generator (refs cited in the original intro). On the original "
                "intro itself used_ratio should be ~1.0 (extractor sanity check / human baseline).",
                "swales_topic searches the paper's full text (title, abstract, original intro, body) for "
                "Swales-style rhetorical-analysis vocabulary; mentions_swales_or_cars is the strict flag "
                "(candidate confound: these authors know the CARS model).",
            ],
        },
        "papers": out_papers,
    }
    return dataset


def verify(dataset: dict) -> None:
    """Post-build invariants."""
    problems = []
    for p in dataset["papers"]:
        types = [i["intro_type"] for i in p["intros"]]
        if types.count("original") != 1:
            problems.append(f"paper {p['paper_id']}: {types.count('original')} original intros")
        for intro in p["intros"]:
            for name, ann in intro["annotations"].items():
                for s in ann["segments"]:
                    if intro["text"][s["start"]:s["end"]] != s["text"]:
                        problems.append(f"paper {p['paper_id']} {name}: segment text/offset mismatch")
                    if not s["in_scheme"]:
                        print(f"  [note] paper {p['paper_id']} {name}: out-of-scheme label '{s['label_raw']}'")
    if problems:
        sys.exit("FATAL:\n  " + "\n  ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    ap.add_argument("--human-annotations", type=Path, default=DEFAULT_HUMAN_ANNOTATIONS)
    ap.add_argument("--generation-results", type=Path, default=DEFAULT_GENERATION_RESULTS)
    ap.add_argument("--refs-dir", type=Path, default=DEFAULT_REFS_DIR)
    ap.add_argument("--processed-texts", type=Path, default=DEFAULT_PROCESSED_TEXTS)
    ap.add_argument(
        "--llm-run", action="append", default=None, metavar="NAME=DIR",
        help="LLM annotation run for the ORIGINAL intros, as name=run_dir "
             "(repeatable; default: none)",
    )
    ap.add_argument(
        "--intro-llm-run", action="append", default=[], type=Path, metavar="DIR",
        help="Step-8 annotate_master_dataset.py run dir with auto-labels for "
             "ALL intros (original + generated), merged by intro_id "
             "(repeatable; default: none)",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if args.llm_run is None:
        args.llm_runs = dict(DEFAULT_LLM_RUNS)
    else:
        args.llm_runs = {}
        for spec in args.llm_run:
            name, _, d = spec.partition("=")
            if not d:
                ap.error(f"--llm-run must be NAME=DIR, got '{spec}'")
            args.llm_runs[name] = Path(d).resolve()

    dataset = build(args)

    for run_dir in args.intro_llm_run:
        sys.path.insert(0, str(REPO_ROOT))
        from intro_analysis.annotate_master_dataset import merge_run_into_dataset
        summary = merge_run_into_dataset(dataset, run_dir.resolve())
        print(f"Intro-level LLM annotations from {run_dir.name}: "
              f"{summary['n_merged']} intros merged "
              f"({summary['n_dropped_segments']} unlocated segments dropped)")
        if summary["missing"]:
            print(f"  [warn] intros without annotation: {summary['missing']}")

    verify(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    size_mb = args.output.stat().st_size / 1e6
    m = dataset["meta"]
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")
    print(f"  papers: {m['n_papers']}  intros: {m['n_intros']}")
    print(f"  original-intro annotation coverage: {m['original_intro_annotation_coverage']}")

    swales_papers = [p for p in dataset["papers"] if p["swales_topic"]["is_swales_related"]]
    n_strong = sum(1 for p in swales_papers if p["swales_topic"]["mentions_swales_or_cars"])
    print(f"\nSwales-related papers: {len(swales_papers)}/{m['n_papers']} "
          f"({n_strong} with an explicit Swales/CARS mention, marked *)")
    for p in swales_papers:
        st = p["swales_topic"]
        star = "*" if st["mentions_swales_or_cars"] else " "
        print(f" {star}{p['paper_key']} [{', '.join(st['matched_terms'])}] "
              f"(in {'/'.join(st['matched_in'])}) {p['title'][:60]}")

    print("\nCitation usage (mean over papers):")
    print(f"  {'source':38s} {'given':>6s} {'used':>6s} {'ratio':>6s} {'extra':>6s}")
    rows = {}
    for p in dataset["papers"]:
        for i in p["intros"]:
            key = "ORIGINAL (human)" if i["intro_type"] == "original" else f"{i['model']} / {i['mode']}"
            rows.setdefault(key, []).append(i["citation_usage"])
    for key, cu in rows.items():
        n = len(cu)
        mean = lambda f: sum(f(c) for c in cu) / n
        print(f"  {key:38s} {mean(lambda c: c['n_refs_given']):6.1f} "
              f"{mean(lambda c: c['n_refs_used']):6.1f} "
              f"{mean(lambda c: c['n_refs_used'] / c['n_refs_given'] if c['n_refs_given'] else 0):6.2f} "
              f"{mean(lambda c: c['n_citations_unmatched']):6.1f}")


if __name__ == "__main__":
    main()
