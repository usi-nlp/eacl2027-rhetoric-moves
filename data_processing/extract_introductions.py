#!/usr/bin/env python3
"""
Prepare research paper texts for introduction generation.

Keeps: abstract + all body sections (literature review, methods, results, etc.)
Removes:
  - Everything before the abstract (title, authors, affiliations, journal info)
  - The introduction section (the target we want to generate)
  - Acknowledgement sections
  - References and everything after

Input:  data/raw_data/pdf_parsing_docling/*.txt
Output: data/raw_data/processed_texts/*.txt
"""

import re
import sys
from pathlib import Path

INPUT_DIR  = Path("data/raw_data/pdf_parsing_docling")
OUTPUT_DIR = Path("data/raw_data/processed_texts")

# Introduction section header
INTRO_RE = re.compile(
    r"^##\s+(?:(?:\d+|[IVX]+)\.?\s+)?introduction\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Any ## section header (used to detect section boundaries)
SECTION_RE = re.compile(r"^##\s+\S", re.MULTILINE)

# References section (stop here)
REFS_RE = re.compile(
    r"^##\s+(?:\d+\.?\s+)?references\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Sections to remove entirely
ACK_RE = re.compile(
    r"^##\s+(?:\d+\.?\s+)?acknowledg\w*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ABOUT_AUTHOR_RE = re.compile(
    r"^##\s+(?:\d+\.?\s+)?about\s+the\s+author\w*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Abstract markers — tried in order, first match wins
ABSTRACT_PATTERNS = [
    re.compile(r"^##\s+abstracts?\s*$",                         re.IGNORECASE | re.MULTILINE),
    re.compile(r"^##\s+[Aa]\s+[Rr]\s+[Tt]\s+[Ii]\s+[Cc]\s+[Ll]\s+[Ee]", re.MULTILINE),
    re.compile(r"^(?:ABSTRACT|Abstract)\s*[.:]",                re.MULTILINE),
    re.compile(r"^Summary\s*\.",                                 re.MULTILINE),
    re.compile(r"^Key\s*[Ww]ords?\s*[:\. ]",                   re.MULTILINE),
]

# Download watermarks injected mid-paragraph by Wiley/publisher PDFs
WATERMARK_RE = re.compile(
    r"\d{7,}, \d{4}, \d+, Downloaded from https?://\S+ by .*?\. "
    r"See the Terms and Conditions.*?rules of use; OA articles are gover",
    re.DOTALL,
)


def find_abstract_start(text: str) -> int | None:
    for pat in ABSTRACT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.start()
    return None


def remove_section(text: str, section_re: re.Pattern) -> str:
    """Remove every occurrence of a section matched by section_re (header + body)."""
    while True:
        m = section_re.search(text)
        if not m:
            break
        sec_start = m.start()
        # Find the next ## after this section header
        next_m = SECTION_RE.search(text, m.end())
        sec_end = next_m.start() if next_m else len(text)
        text = text[:sec_start] + text[sec_end:]
    return text


def clean(text: str) -> str:
    text = WATERMARK_RE.sub("", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"  +", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process(text: str) -> tuple[str | None, str]:
    """Return (processed_text, warning). processed_text is None on fatal error."""

    abstract_start = find_abstract_start(text)
    if abstract_start is None:
        return None, "no abstract marker found"

    intro_m = INTRO_RE.search(text)
    if not intro_m:
        return None, "no introduction section found"

    refs_m = REFS_RE.search(text)
    if not refs_m:
        return None, "no references section found"

    intro_start = intro_m.start()
    # Body begins at the next ## after the intro header
    next_sec = SECTION_RE.search(text, intro_m.end())
    if not next_sec:
        return None, "no section found after introduction"
    intro_end  = next_sec.start()
    refs_start = refs_m.start()

    # Part 1: from abstract up to (but not including) the introduction
    # Part 2: from the section after introduction up to references
    part1 = text[abstract_start:intro_start] if abstract_start < intro_start else ""
    part2 = text[intro_end:refs_start]

    combined = part1 + "\n\n" + part2 if part1.strip() else part2

    # Remove acknowledgement and about-the-author sections
    combined = remove_section(combined, ACK_RE)
    combined = remove_section(combined, ABOUT_AUTHOR_RE)

    return clean(combined), ""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {INPUT_DIR}")

    ok = skipped = 0
    for src in files:
        result, warning = process(src.read_text(encoding="utf-8"))
        if result is None:
            print(f"SKIP  {src.name}: {warning}")
            skipped += 1
            continue
        (OUTPUT_DIR / src.name).write_text(result, encoding="utf-8")
        ok += 1

    print(f"\nDone: {ok} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
