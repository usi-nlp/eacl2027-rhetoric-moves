#!/usr/bin/env python3
"""
Add introduction citation contexts to reference JSONs.

For each reference, searches the paper's *introduction text* (taken from the
Excel corpus metadata) for in-text citations and records:

  "introduction_citation_contexts": [
    {
      "citation_text": "(Ahmad & Samad, 2018)",
      "context": "...full sentence containing the citation..."
    },
    ...
  ]
  "cited_in_introduction": true | false

The introduction text is the ground-truth annotated intro from the Excel file
(data/raw_data/80-papers_corpus_nativeness_quartiles.xlsx), which is the same
text used for Swales-move annotation.

Usage:
  python data_processing/add_citation_contexts.py \
      --refs-dir data/raw_data/references_json_merged \
      --excel   data/raw_data/80-papers_corpus_nativeness_quartiles.xlsx \
      [--output-dir data/raw_data/references_json_with_citations]
"""

import argparse
import difflib
import html
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Filename normalisation for fuzzy matching
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    """Lowercase, strip special chars, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[-_''’]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # remove the annotation suffix that appears in JSON filenames
    s = re.sub(r"\bswales moves\b", "", s).strip()
    return s


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Manual overrides for files where fuzzy matching fails (typos, truncated names, etc.)
# Keys are JSON filenames (basename); values are text filenames (basename).
MANUAL_MAPPING: dict[str, str] = {
    "A_Comparative_Study_of_Form-Focused_and_Communicative_Methods_of_Language_Teaching_in_ESP_Courses-Sw.json":
        "American_British_Q4-_A_comaparative.txt",
    "Cognitive_Predicator_of_child-Swales_Moves.json":
        "Language_Learning_Q1-Cognitive_Predictors_of_Child_Second_Language_Comprehension_and_Syntactic_Learning.txt",
    "DIFFICULTIES_IN_THE_ACQUISITION_AND_USE_OF_ENGLISH_ARTICLES_AMONGADULT_PASHTO_LEARNERS-Swales_Moves.json":
        "Issues_in_Language_Studies_Q4-Difficulties_in.txt",
    "Designing_and_Validating_a_Comprehensive_Questionnaire_to_Assess_Online-Swales_Moves.json":
        "International_Journal_of_Lang._Q2-Designing_and_Validating.txt",
    "Domain-General_Cognitive_Ability_PredictsBilingual_Childrens_Receptive_Vocabulary_inthe_Majority_Lan.json":
        "Language_Learning_Q1-Domain-General_Cognitive_Ability_Predicts.txt",
    "Establishing_a_zone_of_prioritized_curricularivity-Swales_Moves.json":
        "Topics_in_Linguistics_Q3-Establishing-a-zone-of-prioritized-exploring-a-critical-approach-to-negotiating-multimodal-discourses-in-EFL-textbooks.txt",
    "Foreign_langauge_classroom_anxiety-Swales_Moves.json":
        "International_Journal_of_Eng._Q2-_Foreign_language_classroom_anxiety_among_English_for_Specific_purposes_ESP_students.txt",
    "Hedging_functions_of-Swales_Moves.json":
        "Topics_in_Linguistics_Q3-Hedging-functions-of-extraposed-clauses-in-English-and-declarative-subject-clauses-in-Croatian-academic-writing.txt",
    "Overcoming_cultural_constraints_in_translating-Swales_Moves.json":
        "Language_Linguistics_Literature_Q2-Overcoming_Cultural_Constraints_in_Translating_English_Series_A_Case_Study_of_Subtitling_Family_Guy_into_Arabic.txt",
    "Proficiency_and_preference-Swales_Moves.json":
        "Modern_Language_Q1-Proficiency_and_PreferenceOrganization_in_Second_LanguageMandarin_Chinese_Refusals.txt",
    "SELF-MENTION_IN_RESEARCH_SECTIONS_OF_UNIVERSITY_WEBSITES_IN_ESTONIA_AND_THE_UNITED_KINGDOM-Swales_Mo.json":
        "British_American_Q4-_Self-Mention.txt",
    "Syntactic_ambiguty_of_nominal-Swales_Moves.json":
        "International_Journal_of_Eng._Q2-Syntactic_ambiguity_of_complex_nominal_groups_in_technical_English.txt",
    "Teacher_and_Supervisors_Negotatiating-Swales_Moves.json":
        "Modern_Language_Q1-Teachers_and_Supervisors_Negotiating_Identities_of_Experience.txt",
    "Teaching_English_oral_communication_for_Chinas_English_minor_undergraduates_Barriers_challenges_and_.json":
        "Asian_Journal_of_Q3-Teaching_English_oral_communication.txt",
    "Tourism_Discourse_Revisited_An_Analysis_of_Evaluative_Strategies_in_Tourist_Brochures_from_a_Systemi.json":
        "Tourism.txt",
    "Validation_of_C-Test_among_Afghan_Students_of_English_as_a_Foreign_Language-Swales_Moves.json":
        "International_Journal_of_Lang._Q2-Validation_of_C-Test.txt",
    "invesitgating_the_realtionship_between-Swales_Moves.json":
        "Issues_in_Language_Studies_Q4-Investigating_the_Relationship_between_Metalinguistic.txt",
}


def build_json_to_text_mapping(json_dir: str, text_dir: str) -> dict[str, str | None]:
    """Return {json_filename: text_filepath_or_None} for every JSON in json_dir."""
    # Build index of text files: normalised_title_part -> full path
    text_index: dict[str, Path] = {}
    for p in Path(text_dir).glob("*.txt"):
        stem = p.stem
        # Strip journal-prefix pattern like  "JournalName_Q2-PaperTitle"
        m = re.search(r"Q[1-4][-_](.+)", stem, re.IGNORECASE)
        title_part = m.group(1) if m else stem
        norm = _normalise(title_part)
        text_index[norm] = p

    mapping: dict[str, str | None] = {}
    for jf in sorted(Path(json_dir).glob("*.json")):
        # Apply manual override first
        if jf.name in MANUAL_MAPPING:
            override_name = MANUAL_MAPPING[jf.name]
            override_path = Path(text_dir) / override_name
            mapping[jf.name] = str(override_path) if override_path.exists() else None
            continue

        with open(jf) as fh:
            data = json.load(fh)
        paper_filename: str = data.get("paper_filename", jf.stem)
        norm_paper = _normalise(paper_filename)

        best_path: Path | None = None
        best_score = 0.0
        for norm_text, tpath in text_index.items():
            score = _token_jaccard(norm_paper, norm_text)
            if score > best_score:
                best_score = score
                best_path = tpath

        mapping[jf.name] = str(best_path) if best_score >= 0.35 else None

    return mapping


# ---------------------------------------------------------------------------
# Section detection in markdown text
# ---------------------------------------------------------------------------

def _build_section_index(text: str) -> list[tuple[int, str]]:
    """
    Return [(char_offset, section_heading), ...] sorted by offset.
    Detects markdown ATX headings (## / ###) and also plain numbered headings.
    """
    sections: list[tuple[int, str]] = [(-1, "Preamble")]
    for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        sections.append((m.start(), m.group(2).strip()))
    return sections


def section_at(pos: int, section_index: list[tuple[int, str]]) -> str:
    """Return the heading that was most recently opened before `pos`."""
    current = "Preamble"
    for offset, heading in section_index:
        if offset <= pos:
            current = heading
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Citation pattern building
# ---------------------------------------------------------------------------

def _extract_surnames_from_reference(reference_str: str) -> list[str]:
    """
    Pull the leading surnames from a reference string (APA or Chicago style).
    E.g.
      APA:     "Ahmad, N. K. & Samad, A. B. (2018)..." -> ["Ahmad", "Samad"]
      APA 3:   "Hung, B. P., Vien, T. & Vu, N. N. (2018)..." -> ["Hung", "Vien", "Vu"]
      Chicago: "Herzog, George. 1934. Speech-melody..." -> ["Herzog"]
    """
    # Cut at year: parenthesised APA "(2018" or bare period-delimited "1934."
    m_year = re.search(r"\(\d{4}|\.\s*\d{4}[.\s,;]", reference_str)
    before_year = reference_str[: m_year.start()] if m_year else reference_str[:80]

    # Split by "&" or "and" to separate the last author from the rest
    chunks = re.split(r"\s*(?:&|\band\b)\s*", before_year)
    surnames = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        if i == 0 or "," in chunk:
            # APA format: "Surname, Initials." possibly repeated with commas.
            # Extract every "Surname, Initial" occurrence within the chunk.
            apa_surnames = re.findall(
                r"([A-ZÁÉÍÓÚÀÂÄÔÜÇÑŞĞİĀ-ɏ][A-Za-záéíóúàâäôüçñşğıÁÉÍÓÚÀÂÄÔÜÇÑŞĞİĀ-ɏ\-]+),\s*[A-ZÀ-ɏ]", chunk)
            if apa_surnames:
                surnames.extend(apa_surnames)
            else:
                # Fallback: first capitalised word
                m = re.match(r"([A-ZÁÉÍÓÚÀÂÄÔÜÇÑŞĞİĀ-ɏ][A-Za-záéíóúàâäôüçñşğıÁÉÍÓÚÀÂÄÔÜÇÑŞĞİĀ-ɏ\-]+)", chunk)
                if m:
                    surnames.append(m.group(1))
        else:
            # Chicago "Firstname Surname" order — last capitalised word
            words = re.findall(r"[A-ZÁÉÍÓÚÀÂÄÔÜÇÑŞĞİĀ-ɏ][a-záéíóúàâäôüçñşğıÁÉÍÓÚÀÂÄÔÜÇÑŞĞİĀ-ɏ\-]+", chunk)
            if words:
                surnames.append(words[-1])
    return surnames[:3]


def _is_author_list_corrupted(authors: list[str]) -> bool:
    """Return True if the GROBID-parsed author list looks unreliable."""
    for name in authors:
        name = name.strip()
        # Suspiciously long "names", digit-containing, or URL fragments
        if len(name) > 40 or re.search(r"\d", name) or re.search(r"https?://|www\.", name):
            return True
    return False


def _clean_surname(surname: str) -> str | None:
    """Return the surname if it looks like a real surname, else None."""
    if not surname:
        return None
    # Must start with uppercase letter
    if not surname[0].isupper():
        return None
    # Must be at least 2 chars
    if len(surname) < 2:
        return None
    return surname


def _surnames_from_authors(authors: list[str]) -> list[str]:
    """
    Extract surnames from full-name strings.
    Handles "First Last", "F. Last", "Last, F.", "Last F." formats.
    Returns [] if the list looks corrupted.
    """
    if _is_author_list_corrupted(authors):
        return []
    result = []
    for name in authors:
        name = name.strip()
        if not name:
            continue
        parts = name.split()
        if len(parts) == 1:
            candidate = parts[0].rstrip(".")
        else:
            last = parts[-1].rstrip(".,")
            first = parts[0].rstrip(".,")
            # If last part is a single capital letter (initial like "F" or "F."), use first
            candidate = first if re.match(r"^[A-Z]\.?$", last) else last
        cleaned = _clean_surname(candidate)
        if cleaned:
            result.append(cleaned)
    return result[:3]


def build_citation_patterns(ref: dict) -> list[re.Pattern]:
    """
    Build a list of regex patterns that could represent this reference
    as an in-text citation.

    Covers:
      - (Smith, 2018) / (Smith 2018)
      - (Smith & Jones, 2018)
      - (Smith et al., 2018) / Smith et al. 2018
      - Smith (2018)
      - Smith and Jones (2018)
      - Smith et al. (2018)
      - bare "Smith 2018" inside compound lists like (Smith 2018; Jones 2019)
    """
    reference_str = ref.get("reference", "")
    authors_list = ref.get("authors") or []

    # --- Year: use JSON year, but also extract from reference string as fallback ---
    # Handles cases where GROBID/enrichment assigned the wrong year.
    year = str(ref.get("year", "")).strip()
    years: list[str] = []
    if year and re.match(r"\d{4}", year):
        years.append(year[:4])  # strip any trailing letter to start
    if reference_str:
        m_y = re.search(r"\((\d{4})[a-z]?\)", reference_str)   # APA: (2015)
        if not m_y:
            m_y = re.search(r"\.\s*(\d{4})[a-z]?[.\s,;]", reference_str)  # Chicago: . 2011.
        if m_y and m_y.group(1) not in years:
            years.append(m_y.group(1))
    if not years:
        return []

    # --- Surnames: combine from authors list AND reference string ---
    surnames_a = _surnames_from_authors(authors_list) if authors_list else []
    surnames_r = _extract_surnames_from_reference(reference_str) if reference_str else []
    # When both sources agree on the primary author, trust the reference string
    # (GROBID metadata can assign wrong co-authors from adjacent references)
    if surnames_r and surnames_a and surnames_r[0] == surnames_a[0]:
        surnames = surnames_r[:]
    elif len(surnames_r) >= len(surnames_a):
        surnames = surnames_r[:]
        for s in surnames_a:
            if s not in surnames:
                surnames.append(s)
    else:
        surnames = surnames_a[:]
        for s in surnames_r:
            if s not in surnames:
                surnames.append(s)
    surnames = [s for s in surnames if _clean_surname(s)][:3]
    if not surnames:
        return []

    def _esc(s):
        return re.escape(s)

    # Optional trailing page/para reference: ", p. 83", ", pp. 83-85", or ": 83"
    _page = r"(?:(?:,\s*pp?[.]?\s*|\s*:\s*)\d+(?:[–\-]\d+)?)?"
    # Lookahead for "end of citation" — semicolon, closing paren/bracket, or whitespace
    _end = r"(?=[,\s;)\]])"

    patterns = []
    # Build patterns for each candidate year (handles metadata year errors)
    for year in years:
        patterns.extend(_patterns_for_year(surnames, year, _page, _end))
    return list(dict.fromkeys(patterns))  # deduplicate preserving order


def _patterns_for_year(surnames: list[str], year: str,
                        _page: str, _end: str) -> list[re.Pattern]:
    """Build all citation regex patterns for a given surname list and year."""
    def _esc(s):
        return re.escape(s)

    patterns = []
    n = len(surnames)

    # Year separator: "Smith 2018" or "Smith, 2018" (inside compound lists)
    _ys = r"[,\s]+"

    # Optional in-paren prefix: "e.g., ", "see ", "cf. ", "adapted from ", etc.
    _pref = r"(?:(?:e\.g|cf|i\.e|see|adapted\s+from|for\s+\w+,\s*see)[.,]?\s+)?"

    # Optional trailing note before closing paren: ", for a review", ", for reviews"
    _tail = r"(?:,\s*for\s+[\w\s]{1,25})?"

    # Year-list: matches "2005, 2010" or "2017a, 2017b" — one entry is our target year
    _yl = rf"(?:\d{{4}}[a-z]?,\s*)*{year}[a-z]?(?:,\s*\d{{4}}[a-z]?)*"

    if n == 1:
        s = _esc(surnames[0])
        # (Smith, 2018) / (Smith 2018) / (e.g., Smith, 2018) / (Smith 2018: 5)
        patterns.append(re.compile(rf"\({_pref}{s},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        # Multiple years: (Smith, 2005, 2010) / (Smith, 2017a, 2017b)
        patterns.append(re.compile(rf"\({_pref}{s},?\s*{_yl}{_tail}\)", re.IGNORECASE))
        # Smith (2018)
        patterns.append(re.compile(rf"\b{s}\s+\({year}[a-z]?\)", re.IGNORECASE))
        # Bare: "Smith 2018" or "Smith, 2018" inside compound list
        patterns.append(re.compile(rf"\b{s}{_ys}{year}[a-z]?{_end}", re.IGNORECASE))

    elif n == 2:
        s1, s2 = _esc(surnames[0]), _esc(surnames[1])
        # (Smith & Jones, 2018) / (e.g., Smith & Jones, 2018)
        patterns.append(re.compile(
            rf"\({_pref}{s1}\s*(?:&|and)\s*{s2},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        # (Smith & Jones, 2005, 2010) — multiple years
        patterns.append(re.compile(
            rf"\({_pref}{s1}\s*(?:&|and)\s*{s2},?\s*{_yl}{_tail}\)", re.IGNORECASE))
        # (Smith, Jones 2018) — comma-only separator between authors (no &)
        patterns.append(re.compile(
            rf"\({_pref}{s1},\s+{s2},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        # Smith and Jones (2018) / Smith & Jones (2018)
        patterns.append(re.compile(
            rf"\b{s1}\s+(?:and|&)\s+{s2}\s+\({year}[a-z]?\)", re.IGNORECASE))
        # Smith, Jones (2018) — narrative comma-only form
        patterns.append(re.compile(
            rf"\b{s1},\s+{s2}\s+\({year}[a-z]?\)", re.IGNORECASE))
        # Bare: "Smith & Jones 2018" or "Smith & Jones, 2018"
        patterns.append(re.compile(
            rf"\b{s1}\s*(?:&|and)\s*{s2}{_ys}{year}[a-z]?{_end}", re.IGNORECASE))
        # et al. variants using first surname
        patterns.append(re.compile(
            rf"\({_pref}{s1}\s+et\s+al[.,]?\s*,?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        patterns.append(re.compile(
            rf"\b{s1}\s+et\s+al[.,]?\s+\({year}[a-z]?\)", re.IGNORECASE))
        patterns.append(re.compile(
            rf"\b{s1}\s+et\s+al[.,]?{_ys}{year}[a-z]?{_end}", re.IGNORECASE))

    else:  # 3+ authors
        s1 = _esc(surnames[0])
        s2 = _esc(surnames[1])
        s3 = _esc(surnames[2]) if len(surnames) > 2 else None

        # et al. forms
        patterns.append(re.compile(
            rf"\({_pref}{s1}\s+et\s+al[.,]?\s*,?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        patterns.append(re.compile(
            rf"\b{s1}\s+et\s+al[.,]?\s+\({year}[a-z]?\)", re.IGNORECASE))
        patterns.append(re.compile(
            rf"\b{s1}\s+et\s+al[.,]?{_ys}{year}[a-z]?{_end}", re.IGNORECASE))

        # Spelled-out 3-author forms
        if s3:
            _sep = r"[,\s]+"
            # Handles " & ", " and ", ", and ", ", & " between s2 and s3
            _conj = r"[,\s]*(?:&|and)\s*"
            patterns.append(re.compile(
                rf"\b{s1}{_sep}{s2}{_conj}{s3}\s+\({year}[a-z]?\)", re.IGNORECASE))
            patterns.append(re.compile(
                rf"\b{s1}{_sep}{s2}{_conj}{s3}{_ys}{year}[a-z]?{_end}", re.IGNORECASE))
            patterns.append(re.compile(
                rf"\({_pref}{s1}{_sep}{s2}{_conj}{s3},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
            # Comma-only between all three authors (no & or and before 3rd)
            patterns.append(re.compile(
                rf"\({_pref}{s1}{_sep}{s2}{_sep}{s3},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))

        # First two surnames only (fallback for 3+ authors)
        patterns.append(re.compile(
            rf"\({_pref}{s1}\s*(?:&|and)\s*{s2},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        patterns.append(re.compile(
            rf"\b{s1}\s+(?:and|&)\s+{s2}\s+\({year}[a-z]?\)", re.IGNORECASE))
        # Free-text 2-author fallback (catches multi-citation parens like "(S1 & S2, Y; ...")
        patterns.append(re.compile(
            rf"\b{s1}\s*(?:&|and)\s*{s2}{_ys}{year}[a-z]?{_end}", re.IGNORECASE))

    # Single-author fallback for ALL multi-author refs
    if n > 1:
        s1 = _esc(surnames[0])
        patterns.append(re.compile(rf"\({_pref}{s1},?\s*{year}[a-z]?{_page}{_tail}\)", re.IGNORECASE))
        # Multiple years: (Smith, 2005, 2010)
        patterns.append(re.compile(rf"\({_pref}{s1},?\s*{_yl}{_tail}\)", re.IGNORECASE))
        patterns.append(re.compile(rf"\b{s1}\s+\({year}[a-z]?\)", re.IGNORECASE))
        patterns.append(re.compile(rf"\b{s1}{_ys}{year}[a-z]?{_end}", re.IGNORECASE))

    return patterns


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _enclosing_sentence(text: str, match_start: int, match_end: int, window: int = 300) -> str:
    """
    Return the sentence (or a ±window char window) that contains the match.
    Tries to find sentence boundaries; falls back to a fixed window.
    """
    # Search backwards for a sentence start
    chunk_start = max(0, match_start - window)
    chunk_end = min(len(text), match_end + window)
    chunk = text[chunk_start:chunk_end]

    # find the match position within the chunk
    local_start = match_start - chunk_start
    local_end = match_end - chunk_start

    # try to narrow to sentence
    before = chunk[:local_start]
    after = chunk[local_end:]

    # sentence start: last newline or ". " before the match
    sent_start = max(
        before.rfind("\n"),
        before.rfind(". "),
        before.rfind("! "),
        before.rfind("? "),
    )
    sent_start = sent_start + 1 if sent_start >= 0 else 0  # skip the delimiter

    # sentence end: next ". " or newline after the match
    sent_end_candidates = [
        after.find(".\n"), after.find(". "), after.find("! "), after.find("? "), after.find("\n\n")
    ]
    sent_end_candidates = [x for x in sent_end_candidates if x >= 0]
    if sent_end_candidates:
        sent_end = local_end + min(sent_end_candidates) + 1
    else:
        sent_end = len(chunk)

    return chunk[sent_start:sent_end].strip()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def find_citation_contexts(ref: dict, text: str, section_index: list) -> list[dict]:
    """
    Search `text` for all in-text occurrences of `ref` and return a list of
    {citation_text, section, context} dicts.
    """
    patterns = build_citation_patterns(ref)
    if not patterns:
        return []

    results = []
    seen_positions: set[int] = set()

    for pat in patterns:
        for m in pat.finditer(text):
            # deduplicate overlapping matches
            if any(abs(m.start() - p) < 5 for p in seen_positions):
                continue
            seen_positions.add(m.start())

            citation_text = m.group(0)
            sec = section_at(m.start(), section_index)
            context = _enclosing_sentence(text, m.start(), m.end())
            results.append({
                "citation_text": citation_text,
                "section": sec,
                "context": context,
            })

    # Sort by position in text (approximate via context length — use section order instead)
    return results


def _norm_stem(s: str) -> str:
    """Normalize a filename stem for fuzzy matching: lowercase, non-alphanum→underscore,
    strip trailing _swales_moves, strip leading/trailing underscores."""
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = re.sub(r"_swales_moves$", "", s)
    s = s.strip("_")
    return s


def load_excel_intros(excel_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Return (exact_map, norm_map) from the corpus Excel.
    exact_map: {Excel filename (raw): intro} — for papers with paper_filename set
    norm_map:  {normalized stem: intro}      — for fuzzy fallback matching
                Keys include both the filename stem and the title stem.
    """
    try:
        import openpyxl
    except ImportError:
        print("openpyxl not installed — run: pip install openpyxl")
        sys.exit(1)

    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    fn_col = headers.index("filename")
    intro_col = headers.index("introduction")
    title_col = headers.index("title") if "title" in headers else None

    exact: dict[str, str] = {}
    norm: dict[str, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        filename = row[fn_col]
        intro = row[intro_col]
        if filename and intro:
            fn_str = str(filename).strip()
            intro_str = str(intro).strip()
            exact[fn_str] = intro_str
            norm[_norm_stem(fn_str)] = intro_str
            # Also index by title (often has correct spelling vs typo'd filename)
            if title_col is not None and row[title_col]:
                title_norm = _norm_stem(str(row[title_col]))
                if title_norm not in norm:
                    norm[title_norm] = intro_str
    return exact, norm


def _match_intro(paper_filename: str, json_stem: str,
                 exact: dict[str, str], norm: dict[str, str]) -> str | None:
    """Return intro text for this paper using exact then fuzzy matching."""
    # 1. Exact match on paper_filename field
    if paper_filename and paper_filename in exact:
        return exact[paper_filename]

    # 2. Normalise the JSON file stem, strip journal prefix (e.g. "language_learning_q1_")
    candidate = _norm_stem(json_stem)
    # Strip leading journal-quartile prefix: word(s)_q[1-4]_
    candidate = re.sub(r"^[a-z_]+_q[1-4][_-]", "", candidate).strip("_")

    if candidate in norm:
        return norm[candidate]

    # 3. Prefix or suffix containment: one name may be truncated/prefixed form of the other
    # Require at least 6 chars to avoid spurious short matches
    for key, text in norm.items():
        shorter = candidate if len(candidate) <= len(key) else key
        longer  = key if shorter is candidate else candidate
        if len(shorter) >= 6 and (longer.startswith(shorter) or longer.endswith(shorter)):
            return text

    # 4. Also try without stripping the journal prefix (catches edge cases)
    full_candidate = _norm_stem(json_stem)
    if full_candidate in norm:
        return norm[full_candidate]
    for key, text in norm.items():
        shorter = full_candidate if len(full_candidate) <= len(key) else key
        longer  = key if shorter is full_candidate else full_candidate
        if len(shorter) >= 12 and (longer.startswith(shorter) or longer.endswith(shorter)):
            return text

    # 5. Common-prefix length ≥ 30: handles word-concatenation like "StrategiesThe" vs "strategies_the"
    for cand in (candidate, full_candidate):
        for key, text in norm.items():
            shared = 0
            for a, b in zip(cand, key):
                if a != b:
                    break
                shared += 1
            if shared >= 30:
                return text

    # 6. Fuzzy prefix match: compare candidate against same-length prefix of each key.
    #    Catches typos like "comaparative" vs "comparative". Require candidate ≥ 8 chars.
    for cand in (candidate, full_candidate):
        if len(cand) >= 8:
            for key, text in norm.items():
                key_prefix = key[:len(cand)]
                ratio = difflib.SequenceMatcher(None, cand, key_prefix).ratio()
                if ratio >= 0.88:
                    return text

    return None


def process_file(json_path: str, intro_text: str | None, output_path: str) -> dict:
    """
    Search `intro_text` for citations of every reference in the JSON.
    Adds `introduction_citation_contexts` and `cited_in_introduction` to each reference.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    # Intro text comes from Excel — plain text, no HTML entities to worry about
    text = intro_text or ""
    # Build a trivial section index (no headings in the intro excerpt)
    section_index: list = [(-1, "Introduction")]

    total_refs = len(data.get("references", []))
    matched_refs = 0

    for ref in data.get("references", []):
        if text:
            contexts = find_citation_contexts(ref, text, section_index)
            # Strip the section field — it's always "Introduction" here
            contexts = [{"citation_text": c["citation_text"], "context": c["context"]}
                        for c in contexts]
        else:
            contexts = []

        ref["introduction_citation_contexts"] = contexts
        ref["cited_in_introduction"] = bool(contexts)
        if contexts:
            matched_refs += 1

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    return {"total": total_refs, "matched": matched_refs, "has_intro": bool(intro_text)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refs-dir", default="data/raw_data/references_json_merged",
                        help="Directory containing reference JSON files")
    parser.add_argument("--excel", default="data/raw_data/80-papers_corpus_nativeness_quartiles.xlsx",
                        help="Excel file with paper metadata including introduction text")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: in-place update of refs-dir)")
    args = parser.parse_args()

    refs_dir = args.refs_dir
    output_dir = args.output_dir or refs_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading introduction texts from {args.excel} …")
    exact_map, norm_map = load_excel_intros(args.excel)
    print(f"  Loaded {len(exact_map)} introductions")

    print(f"\nProcessing {refs_dir} …")
    total_refs = 0
    total_matched = 0
    no_intro = []

    for jf in sorted(os.listdir(refs_dir)):
        if not jf.endswith(".json"):
            continue
        # Skip cache/summary files that aren't actual papers
        if jf.startswith("_"):
            continue
        json_path = os.path.join(refs_dir, jf)
        out_path = os.path.join(output_dir, jf)

        with open(json_path) as fh:
            meta = json.load(fh)
        paper_filename = (meta.get("paper_filename") or "").strip()
        json_stem = jf[:-5]  # strip .json

        intro_text = _match_intro(paper_filename, json_stem, exact_map, norm_map)
        if intro_text is None:
            no_intro.append(jf)

        stats = process_file(json_path, intro_text, out_path)
        total_refs += stats["total"]
        total_matched += stats["matched"]

        intro_status = f"{len(intro_text)} chars" if intro_text else "NO INTRO"
        print(f"  {jf[:60]:<60}  refs={stats['total']:3d}  cited_in_intro={stats['matched']:3d}"
              f"  intro={intro_status}")

    print(f"\nDone. {total_matched}/{total_refs} references cited in introductions.")
    if no_intro:
        print(f"\nNo intro found for {len(no_intro)} files (paper_filename not in Excel):")
        for jf in no_intro:
            print(f"  {jf}")
    print(f"\nOutput written to: {output_dir}")


if __name__ == "__main__":
    main()
