#!/usr/bin/env python3
"""Enrich reference metadata + abstracts in GROBID JSON outputs.

Resolution priority per reference:
1) DOI lookup (OpenAlex, Semantic Scholar, Crossref, ScienceDirect/Elsevier, DOI.org, Europe PMC, PubMed)
2) Title/year lookup (OpenAlex, Semantic Scholar, Crossref, Europe PMC, PubMed, arXiv, DBLP)
3) arXiv lookup (when arXiv id is detected)
4) Web search fallback: DuckDuckGo + page scraping (citation_*, DC, JSON-LD meta tags)
5) Local heuristic fallback (keeps existing fields where no confident hit)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.parse
import xml.etree.ElementTree as ET

import requests


def _load_dotenv(dotenv_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (without overwriting)."""
    try:
        with open(dotenv_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


_load_dotenv()


OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "https://export.arxiv.org/api/query"
DBLP_API = "https://dblp.org/search/publ/api"
BASE_API = "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
ELSEVIER_ABSTRACT_API = "https://api.elsevier.com/content/abstract"
SCOPUS_SEARCH_API = "https://api.elsevier.com/content/search/scopus"
DOI_ORG_BASE = "https://doi.org"
EUROPE_PMC_SEARCH_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_ESEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ERIC_SEARCH_API = "https://api.ies.ed.gov/eric/"
RESEARCHGATE_SEARCH = "https://www.researchgate.net/search/publication"
UNPAYWALL_API = "https://api.unpaywall.org/v2"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_title_key(text: str) -> str:
    value = normalize_ws(text).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_ws(value)


def cleanup_query_title(title: str) -> str:
    value = normalize_ws(title)
    if not value:
        return ""
    # Strip "(Vol. N)" / "(No. N)" / "(vol. N, ...)" noise anywhere in title.
    value = re.sub(r"\s*\(\s*[Vv]ol\.?\s*\d[^)]*\)", "", value).strip(" .,;")
    value = re.sub(r"\s*\(\s*[Nn]o\.?\s*\d[^)]*\)", "", value).strip(" .,;")
    parts = [part.strip(" .;,") for part in value.split(".") if part.strip(" .;,")]
    if len(parts) >= 2:
        tail = parts[-1].lower()
        if any(token in tail for token in _PUBLISHER_TOKENS | {"cambridge", "oxford"}):
            value = ". ".join(parts[:-1]).strip(" .;,")
    return value


_PUBLISHER_TOKENS = {
    "press", "publisher", "publishing", "university press",
    "routledge", "springer", "wiley", "elsevier",
    "palgrave", "macmillan", "blackwell", "taylor", "francis", "academic",
}


def _strip_venue_suffix(title: str) -> str:
    """Strip a journal/venue citation or publisher name that GROBID appended after '. '."""
    parts = [p.strip() for p in title.split(". ") if p.strip()]
    if len(parts) < 2:
        return title
    last = parts[-1]
    last_lower = last.lower()
    if (
        re.search(r",\s*\d+\s*[\(\[]\d+", last)              # "5(4), 17"
        or re.search(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b", last)  # page range "17-23"
        or re.search(r"\bvol\b|\bno\b|\bpp\b", last_lower)
        or re.search(r"^\(Unpublished|^\(In press", last)
        or re.search(r"University$|Institute$|College$", last)
        or any(token in last_lower for token in _PUBLISHER_TOKENS)
    ):
        return ". ".join(parts[:-1]).strip(" .,;")
    return title


def _title_query_variants(raw_title: str) -> list[str]:
    """Return de-duplicated title query candidates, from most complete to most focused."""
    seen: set[str] = set()
    result: list[str] = []

    def _add(s: str) -> None:
        s = s.strip(' .,;:"\'')
        if s and s not in seen and len(s) >= 10:
            seen.add(s)
            result.append(s)

    # 1. Publisher-cleaned title.
    full = cleanup_query_title(raw_title)
    _add(full)

    # 2. Venue-suffix stripped (removes journal/pages GROBID merged into title).
    stripped = _strip_venue_suffix(full)
    _add(stripped)

    # 3. Main title without subtitle — everything before ": ".
    if ": " in stripped:
        _add(stripped.split(": ")[0])

    # 4. First sentence (before ". ").  Always add it when meaningful (>= 5 words).
    #    Also add the second sentence as a separate candidate to handle book-chapter
    #    references where the first sentence is a short inline quote and the real
    #    chapter title is the second sentence (e.g. "It is probably the reason why.
    #    Hedging in BA and MA theses by German ESL students").
    dot_parts = [p.strip() for p in stripped.split(". ") if p.strip()]
    if dot_parts:
        first = dot_parts[0]
        if len(first.split()) >= 5:
            _add(first)
        if len(dot_parts) >= 2 and len(dot_parts[1].split()) >= 5:
            _add(dot_parts[1])

    return result


def similarity(a: str, b: str) -> float:
    left = normalize_title_key(a)
    right = normalize_title_key(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def token_set(text: str) -> set[str]:
    return {token for token in normalize_title_key(text).split() if len(token) > 2}


def token_overlap(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(1, len(left))


def contains_surname(reference_text: str, surname: str) -> bool:
    if not surname:
        return False
    pattern = r"\b" + re.escape(surname.lower()) + r"\b"
    return re.search(pattern, reference_text.lower()) is not None


def authors_to_surnames(authors: list[str]) -> list[str]:
    out: list[str] = []
    for author in authors:
        cleaned = ensure_str(author)
        if not cleaned:
            continue
        parts = cleaned.split()
        if parts:
            out.append(parts[-1])
    return out


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    return normalize_ws(str(value))


def ensure_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = ensure_str(item)
        if text:
            out.append(text)
    return out


def extract_year(text: str) -> str:
    match = re.search(r"\((\d{4}[a-z]?)\)", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{4}[a-z]?)\b", text)
    if match:
        return match.group(1)
    return ""


def strip_doi(doi: str) -> str:
    value = ensure_str(doi).lower()
    value = value.replace("https://doi.org/", "")
    value = value.replace("http://doi.org/", "")
    value = value.replace("doi:", "")
    return value.strip(" /")


def extract_doi(reference_text: str) -> str:
    match = re.search(r"(10\.\d{4,9}/[-._;()/:a-z0-9]+)", reference_text, flags=re.IGNORECASE)
    if not match:
        return ""
    doi = strip_doi(match.group(1))
    return doi.rstrip(".,;")


def extract_arxiv_id(reference_text: str) -> str:
    # Supports modern and legacy IDs.
    patterns = [
        r"arxiv[:\s]*([a-z\-]+\/\d{7}|\d{4}\.\d{4,5})(v\d+)?",
        r"\b([a-z\-]+\/\d{7}|\d{4}\.\d{4,5})(v\d+)?\b",
    ]
    text = reference_text.lower()
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return ensure_str(match.group(1))
    return ""


def parse_pages_from_crossref(item: dict[str, Any]) -> str:
    page = ensure_str(item.get("page", ""))
    if not page:
        return ""
    return page.replace("--", "-")


def parse_issued_year(item: dict[str, Any]) -> str:
    issued = item.get("issued") or item.get("published-print") or item.get("published-online") or {}
    parts = issued.get("date-parts") or []
    if parts and parts[0]:
        return ensure_str(parts[0][0])
    return ""


def parse_openalex_authorships(authorships: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    surnames: list[str] = []
    for item in authorships:
        author = item.get("author") or {}
        name = ensure_str(author.get("display_name", ""))
        if name:
            authors.append(name)
            name_parts = name.split()
            if name_parts:
                surnames.append(name_parts[-1])
    return authors, surnames


def parse_crossref_author_list(author_list: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    surnames: list[str] = []
    for author in author_list:
        given = ensure_str(author.get("given", ""))
        family = ensure_str(author.get("family", ""))
        if given and family:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)
        elif given:
            authors.append(given)
        if family:
            surnames.append(family)
    return authors, surnames


def parse_semantic_scholar_authors(author_list: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    surnames: list[str] = []
    for author in author_list:
        name = ensure_str(author.get("name", ""))
        if not name:
            continue
        authors.append(name)
        parts = name.split()
        if parts:
            surnames.append(parts[-1])
    return authors, surnames


def jats_to_text(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return normalize_ws(no_tags)


def inverted_index_to_text(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, pos_list in index.items():
        for pos in pos_list:
            positions.append((pos, token))
    if not positions:
        return ""
    positions.sort(key=lambda x: x[0])
    return normalize_ws(" ".join(token for _, token in positions))


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    return str(value)


@dataclass
class CandidateMetadata:
    title: str
    authors: list[str]
    author_surnames: list[str]
    year: str
    venue: str
    volume: str
    issue: str
    pages: str
    doi: str
    arxiv_id: str
    abstract: str
    source: str
    score: float
    url: str = ""  # landing page URL — used to scrape abstract when APIs have none


def heuristic_metadata(reference_text: str, current: dict[str, Any]) -> CandidateMetadata:
    ref = normalize_ws(reference_text)
    year = extract_year(ref) or ensure_str(current.get("year", ""))
    title = ensure_str(current.get("title", ""))
    if title:
        # Remove journal/venue name that GROBID may have merged after ". " in the title.
        title = _strip_venue_suffix(title)
    if not title:
        # Flexible year-pattern extraction covering common reference formats:
        #   APA standard:   Author (2012). Title
        #   No period:      Author (2015) Title  /  Author (2011), Title
        #   Year range:     Author (2002-2007). Title
        #   Bare year:      Harris, Z., 1959. Title  /  Author. 2001. Title
        m = re.search(r"\(\d{4}(?:[-–]\d{2,4})?[a-z]?\)[.,]?\s+(.+)", ref) or re.search(
            r"[,\.]\s*\d{4}[a-z]?\s*[.,]\s+(.+)", ref
        )
        if m:
            chunk = m.group(1)
            stop_markers = [" In ", " In: ", " Retrieved from ", " DOI", " http", " https"]
            end = len(chunk)
            for marker in stop_markers:
                pos = chunk.find(marker)
                if pos != -1:
                    end = min(end, pos)
            candidate = chunk[:end].strip(" .;,")
            if candidate:
                title = candidate

    authors = ensure_str_list(current.get("authors", []))
    surnames = ensure_str_list(current.get("author_surnames", []))
    if not authors or not surnames:
        lead = ref.split("(", 1)[0].strip(" .,")
        lead = re.sub(r"\b(et al\.)$", "", lead, flags=re.IGNORECASE).strip()
        if lead:
            chunks = [part.strip() for part in re.split(r"\s*&\s*|\s+and\s+|,\s*(?=[A-Z])", lead) if part.strip()]
            parsed_authors: list[str] = []
            parsed_surnames: list[str] = []
            for chunk in chunks:
                surname = chunk.split(",", 1)[0].strip()
                # Skip standalone initials (e.g., "T."), but do not strip leading letters from real surnames.
                if re.fullmatch(r"[A-Z]\.?", surname):
                    continue
                if surname:
                    parsed_authors.append(surname)
                    parsed_surnames.append(surname)
            if parsed_authors:
                authors = parsed_authors
            if parsed_surnames:
                surnames = parsed_surnames

    return CandidateMetadata(
        title=title,
        authors=authors,
        author_surnames=surnames,
        year=year,
        venue=ensure_str(current.get("venue", "")),
        volume=ensure_str(current.get("volume", "")),
        issue=ensure_str(current.get("issue", "")),
        pages=ensure_str(current.get("pages", "")),
        doi=strip_doi(ensure_str(current.get("doi", ""))),
        arxiv_id=ensure_str(current.get("arxiv_id", "")),
        abstract=ensure_str(current.get("abstract", "")),
        source="heuristic",
        score=0.2,
    )


class MetaTagParser(HTMLParser):
    """Extract <meta> tags and JSON-LD blocks from an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, list[str]] = {}
        self.json_ld_blocks: list[str] = []
        self._in_json_ld = False
        self._json_ld_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            name = d.get("name") or d.get("property") or ""
            content = d.get("content") or ""
            if name and content:
                self.meta.setdefault(name.lower(), []).append(content)
        elif tag == "script" and "ld+json" in d.get("type", "").lower():
            self._in_json_ld = True
            self._json_ld_buf = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_ld_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_json_ld:
            self._in_json_ld = False
            self.json_ld_blocks.append("".join(self._json_ld_buf))

    def first(self, *keys: str) -> str:
        for key in keys:
            vals = self.meta.get(key.lower())
            if vals:
                return normalize_ws(vals[0])
        return ""

    def all_values(self, *keys: str) -> list[str]:
        out: list[str] = []
        for key in keys:
            out.extend(normalize_ws(v) for v in (self.meta.get(key.lower()) or []) if normalize_ws(v))
        return out


class MetadataResolver:
    def __init__(
        self,
        timeout_seconds: int,
        sleep_seconds: float,
        mailto: str | None,
        elsevier_api_key: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = sleep_seconds
        self.mailto = mailto
        self.elsevier_api_key = elsevier_api_key or ""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "arg-analysis-reference-enricher/1.0",
            }
        )

    def _sleep(self) -> None:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _openalex_params(self) -> dict[str, str]:
        if not self.mailto:
            return {}
        return {"mailto": self.mailto}

    def resolve(self, reference_text: str, current: dict[str, Any]) -> CandidateMetadata:
        heuristic = heuristic_metadata(reference_text, current)
        base_title = heuristic.title or ensure_str(current.get("title", ""))
        base_year = heuristic.year or ensure_str(current.get("year", ""))
        current_doi = strip_doi(ensure_str(current.get("doi", "")))
        extracted_doi = extract_doi(reference_text)
        doi = extracted_doi or current_doi
        arxiv_id = ensure_str(current.get("arxiv_id", "")) or extract_arxiv_id(reference_text)

        candidates: list[CandidateMetadata] = []
        if doi:
            oa_by_doi = self._openalex_by_doi(doi)
            if oa_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, oa_by_doi))
            ss_by_doi = self._semantic_scholar_by_doi(doi)
            if ss_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, ss_by_doi))
            cr_by_doi = self._crossref_by_doi(doi)
            if cr_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, cr_by_doi))
            sd_by_doi = self._sciencedirect_by_doi(doi)
            if sd_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, sd_by_doi))
            doi_org_by_doi = self._doi_org_by_doi(doi)
            if doi_org_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, doi_org_by_doi))
            epmc_by_doi = self._europe_pmc_by_doi(doi)
            if epmc_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, epmc_by_doi))
            pubmed_by_doi = self._pubmed_by_doi(doi)
            if pubmed_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, pubmed_by_doi))
            eric_by_doi = self._eric_by_doi(doi)
            if eric_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, eric_by_doi))
            scopus_by_doi = self._scopus_by_doi(doi)
            if scopus_by_doi:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, scopus_by_doi))

        if arxiv_id:
            arxiv_hit = self._arxiv_by_id(arxiv_id)
            if arxiv_hit:
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, arxiv_hit))

        # Try multiple title query variants (full → venue-stripped → subtitle-less →
        # first-sentence).  Stop as soon as a variant produces a confident match so
        # we don't make unnecessary API calls for well-matched entries.
        query_variants = _title_query_variants(base_title) if base_title else []
        tried_queries: set[str] = set()
        for qvar in query_variants:
            if qvar in tried_queries:
                continue
            tried_queries.add(qvar)
            for candidate in self._openalex_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._semantic_scholar_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._crossref_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._europe_pmc_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._pubmed_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._arxiv_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._dblp_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._academia_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._eric_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            for candidate in self._scopus_by_title(qvar):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            # Early exit: full title already matched well — no need for shorter variants.
            if candidates and max(c.score for c in candidates) >= 0.65:
                break

        # Web search fallback: search DuckDuckGo and scrape academic meta tags from
        # result pages.  Runs in two situations:
        #   (a) No API returned a confident match (score < 0.55) — web may find the
        #       paper in a small/regional journal not indexed by the major APIs.
        #   (b) A confident match was found but it has no abstract — the web page
        #       for that paper often carries the abstract even when APIs don't.
        best_api_score = max((c.score for c in candidates), default=0.0)
        best_api_abstract = any(ensure_str(c.abstract) for c in candidates if c.score >= 0.55)
        if base_title and (best_api_score < 0.55 or not best_api_abstract):
            for candidate in self._web_by_title(base_title, year=base_year, authors=heuristic.authors):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))
            # ResearchGate — best-effort only; often blocked
            for candidate in self._researchgate_by_title(base_title):
                candidates.append(self._score_candidate(reference_text, base_title, base_year, heuristic.authors, candidate))

        if not candidates:
            return heuristic

        best = max(candidates, key=lambda c: c.score)
        # If the top metadata hit has no abstract, try to borrow one from another
        # candidate — but ONLY when both candidates clearly refer to the same paper:
        #   (a) identical non-empty DOI, or
        #   (b) near-identical titles (≥ 0.90) AND the same publication year.
        # This prevents stitching an abstract from a different paper onto the winning
        # metadata (e.g. two papers on the same broad topic with similar titles).
        if not ensure_str(best.abstract):
            for candidate in candidates:
                if not ensure_str(candidate.abstract):
                    continue
                best_doi = ensure_str(best.doi)
                cand_doi = ensure_str(candidate.doi)
                same_doi = bool(best_doi and cand_doi and best_doi == cand_doi)
                same_paper_by_title = (
                    similarity(best.title, candidate.title) >= 0.90
                    and bool(best.year and candidate.year and best.year == candidate.year)
                )
                if same_doi or same_paper_by_title:
                    best.abstract = candidate.abstract
                    if best.source != candidate.source:
                        best.source = f"{best.source}+{candidate.source}_abstract"
                    break

        # If still no abstract, try scraping the landing page URLs from all candidates
        # that match the best paper (same DOI or near-identical title+year).  This
        # handles cases where the API (e.g. OpenAlex) has metadata but no abstract,
        # yet the paper's own journal page does carry the abstract in meta tags.
        if not ensure_str(best.abstract):
            best_doi = ensure_str(best.doi)
            for candidate in candidates:
                url = ensure_str(candidate.url)
                if not url:
                    continue
                cand_doi = ensure_str(candidate.doi)
                same_doi = bool(best_doi and cand_doi and best_doi == cand_doi)
                same_paper = same_doi or (
                    similarity(best.title, candidate.title) >= 0.88
                    and bool(best.year and candidate.year and best.year == candidate.year)
                )
                if not same_paper:
                    continue
                # Convert known PDF-style Dialnet URLs to their HTML landing pages
                url = re.sub(
                    r"dialnet\.unirioja\.es/descarga/articulo/(\d+)\.pdf",
                    r"dialnet.unirioja.es/servlet/articulo?codigo=\1",
                    url,
                )
                scraped = self._scrape_academic_page(url)
                abstract = ensure_str(scraped.abstract) if scraped else ""
                # Reject page-description text masquerading as abstracts:
                #   - too short (< 80 chars) to be a real abstract
                #   - starts with a phrase like "Información del artículo" or
                #     "Article information" (repository catalogue boilerplate)
                #   - is mostly a repetition of the paper title (sim ≥ 0.75)
                if abstract and len(abstract) >= 80:
                    boilerplate_prefixes = (
                        "información del artículo", "article information",
                        "informação do artigo", "informations de l'article",
                    )
                    if not any(abstract.lower().startswith(p) for p in boilerplate_prefixes):
                        if similarity(abstract, best.title) < 0.75:
                            best.abstract = abstract
                            if best.source != "web_scrape":
                                best.source = f"{best.source}+web_scrape_abstract"
                            break

        # Last resort: if we have a good DOI match but still no abstract, ask
        # Unpaywall for open-access copies of the paper.  OA repository pages
        # (institutional repos, author homepages, preprint servers) freely serve
        # the abstract in citation_* meta tags without paywalls or bot-blocking.
        if not ensure_str(best.abstract) and best.score >= 0.55:
            best_doi = ensure_str(best.doi)
            if best_doi:
                for oa_url in self._unpaywall_oa_urls(best_doi):
                    scraped = self._scrape_academic_page(oa_url)
                    abstract = ensure_str(scraped.abstract) if scraped else ""
                    if abstract and len(abstract) >= 80:
                        boilerplate_prefixes = (
                            "información del artículo", "article information",
                            "informação do artigo", "informations de l'article",
                        )
                        if not any(abstract.lower().startswith(p) for p in boilerplate_prefixes):
                            if similarity(abstract, best.title) < 0.75:
                                best.abstract = abstract
                                best.source = f"{best.source}+unpaywall_abstract"
                                break

        if best.score < 0.55:
            return heuristic
        return best

    def _score_candidate(
        self,
        reference_text: str,
        query_title: str,
        query_year: str,
        query_authors: list[str],
        candidate: CandidateMetadata,
    ) -> CandidateMetadata:
        title_score = similarity(query_title, candidate.title)
        title_in_reference = max(similarity(reference_text, candidate.title), token_overlap(candidate.title, reference_text))

        author_match_count = 0
        candidate_surnames = candidate.author_surnames or authors_to_surnames(candidate.authors)
        if candidate_surnames:
            for surname in candidate_surnames:
                if contains_surname(reference_text, surname):
                    author_match_count += 1
        author_in_reference = author_match_count > 0

        venue_overlap_score = token_overlap(candidate.venue, reference_text) if candidate.venue else 0.0
        # Require high overlap so that common domain words ("International", "English",
        # "Lingüística") shared between the candidate venue and the reference text for
        # unrelated reasons do not falsely satisfy the venue check.
        venue_in_reference = venue_overlap_score >= 0.70 if candidate.venue else False

        year_bonus = 0.0
        if query_year and candidate.year and str(query_year) == str(candidate.year):
            year_bonus = 0.08

        author_bonus = 0.0
        query_surnames = authors_to_surnames(query_authors)
        if query_surnames and candidate_surnames:
            q = {normalize_title_key(name) for name in query_surnames if name}
            c = {normalize_title_key(name) for name in candidate_surnames if name}
            if q and c:
                overlap = len(q.intersection(c)) / max(1, len(q))
                author_bonus = 0.15 * min(1.0, overlap)

        source_base = {
            "openalex_doi": 0.78,
            "semantic_scholar_doi": 0.76,
            "crossref_doi": 0.74,
            "sciencedirect_doi": 0.73,
            "doi_org_doi": 0.70,
            "europepmc_doi": 0.69,
            "pubmed_doi": 0.69,
            "arxiv_id": 0.72,
            "openalex_title": 0.56,
            "semantic_scholar_title": 0.54,
            "crossref_title": 0.53,
            "europepmc_title": 0.52,
            "pubmed_title": 0.52,
            "arxiv_title": 0.51,
            "dblp_title": 0.50,
            "academia_title": 0.45,
            "base_title": 0.55,
            "web_scrape": 0.54,
            "eric_doi": 0.72,
            "eric_title": 0.54,
            "scopus_doi": 0.75,
            "scopus_title": 0.55,
            "researchgate_scrape": 0.53,
            "heuristic": 0.20,
        }.get(candidate.source, 0.45)

        candidate.score = min(
            1.0,
            source_base + (0.22 * title_score) + (0.22 * title_in_reference) + year_bonus + author_bonus,
        )

        strict_title_sources = {
            "openalex_title",
            "semantic_scholar_title",
            "crossref_title",
            "europepmc_title",
            "pubmed_title",
            "arxiv_title",
            "dblp_title",
            "academia_title",
            "base_title",
            "web_scrape",
            "eric_title",
            "scopus_title",
            "researchgate_scrape",
        }
        if candidate.source in strict_title_sources:
            # Title-only retrieval is noisy, but some references (books/chapters) do not cleanly expose venue/authors.
            if title_in_reference < 0.55:
                candidate.score = min(candidate.score, 0.35)
            elif not author_in_reference:
                # Venue-based corroboration is unreliable: shared domain vocabulary
                # ("International", "English", "Lingüística") can make an unrelated
                # venue appear to match the reference text.  Require that at least one
                # candidate author surname actually appears in the reference string.
                # This correctly rejects candidates where titles are lexically similar
                # but the paper is completely different.
                candidate.score = min(candidate.score, 0.50)
        elif candidate.source.endswith("_doi") or candidate.source == "arxiv_id":
            # DOI/arXiv-id lookups are stronger, but still guard against obvious mismatches.
            if title_in_reference < 0.35:
                candidate.score = min(candidate.score, 0.50)
            # Year mismatch is a strong signal that GROBID extracted a wrong DOI.
            # A reference always cites the paper's own publication year, so a gap
            # of more than 3 years means we almost certainly resolved the wrong paper.
            if query_year and candidate.year:
                qy = _year_int(str(query_year))
                cy = _year_int(str(candidate.year))
                if qy and cy and abs(qy - cy) > 3:
                    candidate.score = min(candidate.score, 0.52)
        return candidate

    def _openalex_by_doi(self, doi: str) -> CandidateMetadata | None:
        url = f"{OPENALEX_BASE}/works/https://doi.org/{urllib.parse.quote(doi, safe='')}"
        try:
            resp = self.session.get(url, params=self._openalex_params(), timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return None
            item = resp.json()
        except requests.RequestException:
            self._sleep()
            return None
        candidate = self._candidate_from_openalex(item, source="openalex_doi")
        self._sleep()
        return candidate

    def _openalex_by_title(self, title: str) -> list[CandidateMetadata]:
        params: dict[str, str] = {
            "search": title,
            "per-page": "5",
            "select": "display_name,authorships,publication_year,primary_location,locations,biblio,doi,abstract_inverted_index",
        }
        params.update(self._openalex_params())
        try:
            resp = self.session.get(f"{OPENALEX_BASE}/works", params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            results = (resp.json().get("results") or [])
        except requests.RequestException:
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for item in results:
            candidate = self._candidate_from_openalex(item, source="openalex_title")
            if candidate:
                out.append(candidate)
        self._sleep()
        return out

    def _candidate_from_openalex(self, item: dict[str, Any], source: str) -> CandidateMetadata | None:
        title = ensure_str(item.get("display_name", ""))
        if not title:
            return None
        authors, surnames = parse_openalex_authorships(item.get("authorships") or [])
        biblio = item.get("biblio") or {}
        venue = ensure_str(((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""))
        abstract = inverted_index_to_text(item.get("abstract_inverted_index"))
        doi = strip_doi(ensure_str(item.get("doi", "")))

        # Collect all HTML landing page URLs (prefer non-PDF) so we can scrape an
        # abstract when OpenAlex's own record has none.
        landing_url = ""
        all_locations = [item.get("primary_location")] + list(item.get("locations") or [])
        for loc in all_locations:
            if not loc:
                continue
            url = ensure_str(loc.get("landing_page_url") or "")
            if url and not url.lower().endswith(".pdf"):
                landing_url = url
                break
        if not landing_url:
            # Fallback: accept a PDF-based URL — we'll try to find its HTML sibling
            for loc in all_locations:
                if not loc:
                    continue
                url = ensure_str(loc.get("landing_page_url") or loc.get("pdf_url") or "")
                if url:
                    landing_url = url
                    break

        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=ensure_str(item.get("publication_year", "")),
            venue=venue,
            volume=ensure_str(biblio.get("volume", "")),
            issue=ensure_str(biblio.get("issue", "")),
            pages=ensure_str(biblio.get("first_page", "")),
            doi=doi,
            arxiv_id="",
            abstract=abstract,
            source=source,
            score=0.0,
            url=landing_url,
        )

    def _semantic_scholar_by_doi(self, doi: str) -> CandidateMetadata | None:
        fields = "title,authors,year,venue,externalIds,abstract"
        url = f"{SEMANTIC_SCHOLAR_BASE}/paper/DOI:{urllib.parse.quote(doi, safe='')}"
        try:
            resp = self.session.get(url, params={"fields": fields}, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return None
            item = resp.json()
        except requests.RequestException:
            self._sleep()
            return None
        candidate = self._candidate_from_semantic_scholar(item, source="semantic_scholar_doi")
        self._sleep()
        return candidate

    def _semantic_scholar_by_title(self, title: str) -> list[CandidateMetadata]:
        fields = "title,authors,year,venue,externalIds,abstract"
        params = {
            "query": title,
            "limit": "5",
            "fields": fields,
        }
        try:
            resp = self.session.get(f"{SEMANTIC_SCHOLAR_BASE}/paper/search", params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            data = resp.json().get("data") or []
        except requests.RequestException:
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for item in data:
            candidate = self._candidate_from_semantic_scholar(item, source="semantic_scholar_title")
            if candidate:
                out.append(candidate)
        self._sleep()
        return out

    def _candidate_from_semantic_scholar(self, item: dict[str, Any], source: str) -> CandidateMetadata | None:
        title = ensure_str(item.get("title", ""))
        if not title:
            return None
        authors, surnames = parse_semantic_scholar_authors(item.get("authors") or [])
        external_ids = item.get("externalIds") or {}
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=ensure_str(item.get("year", "")),
            venue=ensure_str(item.get("venue", "")),
            volume="",
            issue="",
            pages="",
            doi=strip_doi(ensure_str(external_ids.get("DOI", ""))),
            arxiv_id=ensure_str(external_ids.get("ArXiv", "")),
            abstract=ensure_str(item.get("abstract", "")),
            source=source,
            score=0.0,
        )

    def _crossref_by_doi(self, doi: str) -> CandidateMetadata | None:
        try:
            resp = self.session.get(f"{CROSSREF_BASE}/works/{urllib.parse.quote(doi, safe='')}", timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return None
            item = (resp.json().get("message") or {})
        except requests.RequestException:
            self._sleep()
            return None
        candidate = self._candidate_from_crossref(item, source="crossref_doi")
        self._sleep()
        return candidate

    def _crossref_by_title(self, title: str) -> list[CandidateMetadata]:
        params = {
            "query.bibliographic": title,
            "rows": "5",
        }
        try:
            resp = self.session.get(f"{CROSSREF_BASE}/works", params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            items = (resp.json().get("message") or {}).get("items") or []
        except requests.RequestException:
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for item in items:
            candidate = self._candidate_from_crossref(item, source="crossref_title")
            if candidate:
                out.append(candidate)
        self._sleep()
        return out

    def _sciencedirect_by_doi(self, doi: str) -> CandidateMetadata | None:
        if not self.elsevier_api_key:
            return None  # API key required — skip silently
        url = f"{ELSEVIER_ABSTRACT_API}/doi/{urllib.parse.quote(doi, safe='')}"
        headers = {"X-ELS-APIKey": self.elsevier_api_key, "Accept": "application/json"}
        payload: dict[str, Any] = {}
        try:
            # META_ABS includes dc:description (abstract) but requires institutional
            # network access.  Fall back to no-view if the server rejects it (401/403),
            # which at minimum gives us full bibliographic metadata from anywhere.
            for view_param in ({"view": "META_ABS"}, {}):
                resp = self.session.get(
                    url,
                    params={**view_param, "httpAccept": "application/json"},
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code == 200:
                    payload = resp.json().get("abstracts-retrieval-response") or {}
                    break
                if resp.status_code not in (401, 403):
                    # Unexpected error (404, 5xx…) — give up entirely
                    self._sleep()
                    return None
                # 401/403 → try next (less privileged) view
        except requests.RequestException:
            self._sleep()
            return None
        if not payload:
            self._sleep()
            return None

        core = payload.get("coredata") or {}
        title = ensure_str(core.get("dc:title", ""))
        if not title:
            self._sleep()
            return None

        # dc:creator can be a dict {"author": [...]} or a list of strings depending
        # on the API view.  Handle both shapes.
        authors: list[str] = []
        surnames: list[str] = []
        creator_raw = core.get("dc:creator", {})
        if isinstance(creator_raw, dict):
            author_list = creator_raw.get("author") or []
        elif isinstance(creator_raw, list):
            author_list = creator_raw
        else:
            author_list = []
        for author in author_list:
            if isinstance(author, dict):
                given = ensure_str(author.get("ce:given-name", ""))
                # Surname can be in "ce:surname" or nested under "preferred-name"
                surname = ensure_str(
                    author.get("ce:surname")
                    or (author.get("preferred-name") or {}).get("ce:surname")
                    or ""
                )
                full = normalize_ws(f"{given} {surname}") if (given or surname) else ""
            else:
                full = ensure_str(author)
                surname = full.split()[-1] if full.split() else ""
            if full:
                authors.append(full)
            if surname:
                surnames.append(surname)

        abstract = ensure_str(core.get("dc:description", ""))
        if not abstract:
            abstract = normalize_ws(flatten_text((payload.get("item") or {}).get("bibrecord", {}).get("head", {}).get("abstracts", "")))

        venue = ensure_str(core.get("prism:publicationName", ""))
        year = ensure_str(core.get("prism:coverDate", ""))[:4]
        self._sleep()
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=year,
            venue=venue,
            volume=ensure_str(core.get("prism:volume", "")),
            issue=ensure_str(core.get("prism:issueIdentifier", "")),
            pages=ensure_str(core.get("prism:pageRange", "")),
            doi=strip_doi(ensure_str(core.get("prism:doi", ""))) or doi,
            arxiv_id="",
            abstract=abstract,
            source="sciencedirect_doi",
            score=0.0,
        )

    # Fields requested from the Scopus Search API for every query.
    _SCOPUS_FIELDS = (
        "dc:title,dc:description,dc:creator,prism:publicationName,"
        "prism:volume,prism:issueIdentifier,prism:pageRange,prism:coverDate,prism:doi"
    )

    def _candidate_from_scopus_entry(self, entry: dict[str, Any], source: str) -> CandidateMetadata | None:
        title = ensure_str(entry.get("dc:title", ""))
        if not title:
            return None
        # dc:creator comes as a simple "Surname I." string in search results
        creator_str = ensure_str(entry.get("dc:creator", ""))
        authors = [creator_str] if creator_str else []
        # Extract surname: everything before the first space in "Abdi R."
        surnames = [creator_str.split()[0]] if creator_str and creator_str.split() else []
        year = ensure_str(entry.get("prism:coverDate", ""))[:4]
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=year,
            venue=ensure_str(entry.get("prism:publicationName", "")),
            volume=ensure_str(entry.get("prism:volume", "")),
            issue=ensure_str(entry.get("prism:issueIdentifier", "")),
            pages=ensure_str(entry.get("prism:pageRange", "")),
            doi=strip_doi(ensure_str(entry.get("prism:doi", ""))),
            arxiv_id="",
            abstract=ensure_str(entry.get("dc:description", "")),
            source=source,
            score=0.0,
        )

    def _scopus_by_doi(self, doi: str) -> CandidateMetadata | None:
        """Look up a paper in Scopus by DOI via the Scopus Search API.

        Covers all Elsevier and many other publisher journals.  Returns abstract
        when the subscription permits it; always returns full bibliographic metadata.
        Requires self.elsevier_api_key.
        """
        if not self.elsevier_api_key:
            return None
        try:
            resp = self.session.get(
                SCOPUS_SEARCH_API,
                params={
                    "query": f"DOI({doi})",
                    "field": self._SCOPUS_FIELDS,
                    "count": "1",
                    "httpAccept": "application/json",
                },
                headers={"X-ELS-APIKey": self.elsevier_api_key},
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                self._sleep()
                return None
            entries = (resp.json().get("search-results") or {}).get("entry") or []
        except (requests.RequestException, ValueError):
            self._sleep()
            return None
        candidate = self._candidate_from_scopus_entry(entries[0], source="scopus_doi") if entries else None
        self._sleep()
        return candidate

    def _scopus_by_title(self, title: str) -> list[CandidateMetadata]:
        """Search Scopus by title.  Returns up to 5 candidates."""
        if not self.elsevier_api_key:
            return []
        query_title = title.replace('"', ' ')
        try:
            resp = self.session.get(
                SCOPUS_SEARCH_API,
                params={
                    "query": f'TITLE("{query_title}")',
                    "field": self._SCOPUS_FIELDS,
                    "count": "5",
                    "httpAccept": "application/json",
                },
                headers={"X-ELS-APIKey": self.elsevier_api_key},
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                self._sleep()
                return []
            entries = (resp.json().get("search-results") or {}).get("entry") or []
        except (requests.RequestException, ValueError):
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for entry in entries:
            candidate = self._candidate_from_scopus_entry(entry, source="scopus_title")
            if candidate:
                out.append(candidate)
        self._sleep()
        return out

    def _doi_org_by_doi(self, doi: str) -> CandidateMetadata | None:
        try:
            resp = self.session.get(
                f"{DOI_ORG_BASE}/{urllib.parse.quote(doi, safe='')}",
                headers={"Accept": "application/vnd.citationstyles.csl+json"},
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                self._sleep()
                return None
            item = resp.json()
        except requests.RequestException:
            self._sleep()
            return None

        title = ensure_str(item.get("title", ""))
        if not title:
            self._sleep()
            return None
        authors: list[str] = []
        surnames: list[str] = []
        for author in item.get("author") or []:
            given = ensure_str(author.get("given", ""))
            family = ensure_str(author.get("family", ""))
            full = normalize_ws(f"{given} {family}")
            if full:
                authors.append(full)
            if family:
                surnames.append(family)

        container = item.get("container-title")
        venue = ensure_str(container[0] if isinstance(container, list) and container else container or "")
        issued = (item.get("issued") or {}).get("date-parts") or []
        year = ensure_str(issued[0][0]) if issued and issued[0] else ""
        abstract = jats_to_text(ensure_str(item.get("abstract", "")))
        self._sleep()
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=year,
            venue=venue,
            volume=ensure_str(item.get("volume", "")),
            issue=ensure_str(item.get("issue", "")),
            pages=ensure_str(item.get("page", "")),
            doi=strip_doi(ensure_str(item.get("DOI", ""))) or doi,
            arxiv_id="",
            abstract=abstract,
            source="doi_org_doi",
            score=0.0,
        )

    def _candidate_from_europe_pmc(self, item: dict[str, Any], source: str) -> CandidateMetadata | None:
        title = ensure_str(item.get("title", ""))
        if not title:
            return None
        author_string = ensure_str(item.get("authorString", ""))
        authors = [normalize_ws(chunk) for chunk in author_string.split(",") if normalize_ws(chunk)]
        surnames = [name.split()[-1] for name in authors if name.split()]
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=ensure_str(item.get("pubYear", "")),
            venue=ensure_str(item.get("journalTitle", "")),
            volume=ensure_str(item.get("journalVolume", "")),
            issue=ensure_str(item.get("issue", "")),
            pages=ensure_str(item.get("pageInfo", "")),
            doi=strip_doi(ensure_str(item.get("doi", ""))),
            arxiv_id="",
            abstract=ensure_str(item.get("abstractText", "")),
            source=source,
            score=0.0,
        )

    def _europe_pmc_by_doi(self, doi: str) -> CandidateMetadata | None:
        params = {"query": f"DOI:{doi}", "format": "json", "pageSize": "1", "resultType": "core"}
        try:
            resp = self.session.get(EUROPE_PMC_SEARCH_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return None
            results = ((resp.json().get("resultList") or {}).get("result") or [])
        except requests.RequestException:
            self._sleep()
            return None
        candidate = self._candidate_from_europe_pmc(results[0], source="europepmc_doi") if results else None
        self._sleep()
        return candidate

    def _europe_pmc_by_title(self, title: str) -> list[CandidateMetadata]:
        params = {"query": f'TITLE:"{title}"', "format": "json", "pageSize": "5", "resultType": "core"}
        try:
            resp = self.session.get(EUROPE_PMC_SEARCH_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            results = ((resp.json().get("resultList") or {}).get("result") or [])
        except requests.RequestException:
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for item in results:
            candidate = self._candidate_from_europe_pmc(item, source="europepmc_title")
            if candidate:
                out.append(candidate)
        self._sleep()
        return out

    def _pubmed_fetch_ids(self, term: str, max_results: int) -> list[str]:
        params = {"db": "pubmed", "retmode": "json", "retmax": str(max_results), "term": term}
        try:
            resp = self.session.get(PUBMED_ESEARCH_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            id_list = ((resp.json().get("esearchresult") or {}).get("idlist") or [])
        except requests.RequestException:
            self._sleep()
            return []
        self._sleep()
        return [ensure_str(item) for item in id_list if ensure_str(item)]

    def _pubmed_fetch_candidates(self, pmids: list[str], source: str) -> list[CandidateMetadata]:
        if not pmids:
            return []
        params = {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)}
        try:
            resp = self.session.get(PUBMED_EFETCH_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            root = ET.fromstring(resp.text)
        except (requests.RequestException, ET.ParseError):
            self._sleep()
            return []

        out: list[CandidateMetadata] = []
        for article in root.findall(".//PubmedArticle"):
            article_title = ensure_str("".join((article.find(".//ArticleTitle") or ET.Element("x")).itertext()))
            if not article_title:
                continue
            abstract_chunks: list[str] = []
            for abstract_node in article.findall(".//Abstract/AbstractText"):
                text = ensure_str("".join(abstract_node.itertext()))
                if text:
                    abstract_chunks.append(text)
            authors: list[str] = []
            surnames: list[str] = []
            for author in article.findall(".//AuthorList/Author"):
                fore = ensure_str(author.findtext("ForeName", default=""))
                last = ensure_str(author.findtext("LastName", default=""))
                full = normalize_ws(f"{fore} {last}")
                if full:
                    authors.append(full)
                if last:
                    surnames.append(last)
            year = (
                ensure_str(article.findtext(".//PubDate/Year", default=""))
                or ensure_str(article.findtext(".//ArticleDate/Year", default=""))
            )
            doi = ""
            for aid in article.findall(".//ArticleIdList/ArticleId"):
                if ensure_str(aid.attrib.get("IdType", "")).lower() == "doi":
                    doi = strip_doi(ensure_str("".join(aid.itertext())))
                    if doi:
                        break
            out.append(
                CandidateMetadata(
                    title=article_title,
                    authors=authors,
                    author_surnames=surnames,
                    year=year,
                    venue=ensure_str(article.findtext(".//Journal/Title", default="")),
                    volume=ensure_str(article.findtext(".//JournalIssue/Volume", default="")),
                    issue=ensure_str(article.findtext(".//JournalIssue/Issue", default="")),
                    pages=ensure_str(article.findtext(".//Pagination/MedlinePgn", default="")),
                    doi=doi,
                    arxiv_id="",
                    abstract=normalize_ws(" ".join(abstract_chunks)),
                    source=source,
                    score=0.0,
                )
            )
        self._sleep()
        return out

    def _pubmed_by_doi(self, doi: str) -> CandidateMetadata | None:
        ids = self._pubmed_fetch_ids(term=f'"{doi}"[AID] OR "{doi}"[DOI]', max_results=3)
        candidates = self._pubmed_fetch_candidates(ids, source="pubmed_doi")
        return candidates[0] if candidates else None

    def _pubmed_by_title(self, title: str) -> list[CandidateMetadata]:
        ids = self._pubmed_fetch_ids(term=f'"{title}"[Title]', max_results=5)
        return self._pubmed_fetch_candidates(ids, source="pubmed_title")

    def _arxiv_by_title(self, title: str) -> list[CandidateMetadata]:
        params = {"search_query": f'ti:"{title}"', "start": 0, "max_results": 5}
        try:
            resp = self.session.get(ARXIV_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            root = ET.fromstring(resp.text)
        except (requests.RequestException, ET.ParseError):
            self._sleep()
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        out: list[CandidateMetadata] = []
        for entry in root.findall("atom:entry", ns):
            candidate_title = ensure_str(entry.findtext("atom:title", "", ns))
            if not candidate_title:
                continue
            summary = ensure_str(entry.findtext("atom:summary", "", ns))
            published = ensure_str(entry.findtext("atom:published", "", ns))
            year = published[:4] if len(published) >= 4 else ""
            author_names: list[str] = []
            surnames: list[str] = []
            for author in entry.findall("atom:author", ns):
                name = ensure_str(author.findtext("atom:name", "", ns))
                if not name:
                    continue
                author_names.append(name)
                parts = name.split()
                if parts:
                    surnames.append(parts[-1])
            out.append(
                CandidateMetadata(
                    title=candidate_title,
                    authors=author_names,
                    author_surnames=surnames,
                    year=year,
                    venue="arXiv",
                    volume="",
                    issue="",
                    pages="",
                    doi="",
                    arxiv_id="",
                    abstract=summary,
                    source="arxiv_title",
                    score=0.0,
                )
            )
        self._sleep()
        return out

    def _candidate_from_crossref(self, item: dict[str, Any], source: str) -> CandidateMetadata | None:
        titles = item.get("title") or []
        title = ensure_str(titles[0] if titles else "")
        if not title:
            return None
        authors, surnames = parse_crossref_author_list(item.get("author") or [])
        container = item.get("container-title") or []
        venue = ensure_str(container[0] if container else "")
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=parse_issued_year(item),
            venue=venue,
            volume=ensure_str(item.get("volume", "")),
            issue=ensure_str(item.get("issue", "")),
            pages=parse_pages_from_crossref(item),
            doi=strip_doi(ensure_str(item.get("DOI", ""))),
            arxiv_id="",
            abstract=jats_to_text(ensure_str(item.get("abstract", ""))),
            source=source,
            score=0.0,
        )

    def _dblp_by_title(self, title: str) -> list[CandidateMetadata]:
        params = {
            "q": title,
            "h": "5",
            "format": "json",
        }
        try:
            resp = self.session.get(DBLP_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            hits = (((resp.json().get("result") or {}).get("hits") or {}).get("hit") or [])
        except requests.RequestException:
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for hit in hits:
            info = hit.get("info") or {}
            authors_raw = ((info.get("authors") or {}).get("author") or [])
            if isinstance(authors_raw, str):
                author_list = [authors_raw]
            else:
                author_list = [ensure_str(a) for a in authors_raw if ensure_str(a)]
            surnames = [a.split()[-1] for a in author_list if a]
            candidate_title = ensure_str(info.get("title", ""))
            if not candidate_title:
                continue
            venue = ensure_str(info.get("venue", ""))
            out.append(
                CandidateMetadata(
                    title=candidate_title,
                    authors=author_list,
                    author_surnames=surnames,
                    year=ensure_str(info.get("year", "")),
                    venue=venue,
                    volume="",
                    issue="",
                    pages=ensure_str(info.get("pages", "")),
                    doi=strip_doi(ensure_str(info.get("doi", ""))),
                    arxiv_id="",
                    abstract="",
                    source="dblp_title",
                    score=0.0,
                )
            )
        self._sleep()
        return out

    def _academia_by_title(self, title: str) -> list[CandidateMetadata]:
        params = {"q": title}
        try:
            resp = self.session.get("https://www.academia.edu/search", params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            html = resp.text
        except requests.RequestException:
            self._sleep()
            return []

        links = re.findall(
            r'<a[^>]+href="(https?://(?:www\.)?academia\.edu/[^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        out: list[CandidateMetadata] = []
        seen_titles: set[str] = set()
        for _, anchor_text in links:
            candidate_title = normalize_ws(re.sub(r"<[^>]+>", " ", anchor_text))
            if len(candidate_title) < 20:
                continue
            key = normalize_title_key(candidate_title)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            out.append(
                CandidateMetadata(
                    title=candidate_title,
                    authors=[],
                    author_surnames=[],
                    year="",
                    venue="Academia.edu",
                    volume="",
                    issue="",
                    pages="",
                    doi="",
                    arxiv_id="",
                    abstract="",
                    source="academia_title",
                    score=0.0,
                )
            )
            if len(out) >= 5:
                break
        self._sleep()
        return out

    def _base_by_title(self, title: str) -> list[CandidateMetadata]:
        """Query the BASE (Bielefeld Academic Search Engine) API.

        BASE indexes 250M+ documents including small/regional OA journals not
        covered by OpenAlex or Crossref.  No API key required; free for research.
        """
        params = {
            "func": "PerformSearch",
            "query": f'dctitle:"{title}"',
            "hits": "5",
            "offset": "0",
            "format": "json",
            "userinfo": self.mailto or "arg-analysis-enricher",
        }
        try:
            resp = self.session.get(BASE_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            data = resp.json()
        except (requests.RequestException, ValueError):
            self._sleep()
            return []

        docs = (data.get("response") or {}).get("docs") or []
        out: list[CandidateMetadata] = []
        for doc in docs:
            candidate_title = ensure_str(
                (doc.get("dctitle") or [""])[0]
                if isinstance(doc.get("dctitle"), list)
                else doc.get("dctitle", "")
            )
            if not candidate_title:
                continue

            # Authors
            authors_raw = doc.get("dcauthor") or []
            if isinstance(authors_raw, str):
                authors_raw = [authors_raw]
            authors: list[str] = [ensure_str(a) for a in authors_raw if ensure_str(a)]
            surnames: list[str] = [a.split()[-1] for a in authors if a.split()]

            # Abstract (dcdescription is a list)
            desc_raw = doc.get("dcdescription") or []
            if isinstance(desc_raw, str):
                desc_raw = [desc_raw]
            abstract = normalize_ws(" ".join(ensure_str(d) for d in desc_raw if ensure_str(d)))

            # Year
            date_raw = ensure_str(
                (doc.get("dcdate") or [""])[0]
                if isinstance(doc.get("dcdate"), list)
                else doc.get("dcdate", "")
            )
            year = date_raw[:4] if date_raw else ""

            # Venue / source journal
            source_raw = doc.get("dcsource") or []
            if isinstance(source_raw, str):
                source_raw = [source_raw]
            venue = ensure_str(source_raw[0]) if source_raw else ""

            # DOI
            identifiers = doc.get("dcidentifier") or []
            if isinstance(identifiers, str):
                identifiers = [identifiers]
            doi = ""
            for ident in identifiers:
                stripped = strip_doi(ensure_str(ident))
                if stripped.startswith("10."):
                    doi = stripped
                    break

            out.append(CandidateMetadata(
                title=candidate_title,
                authors=authors,
                author_surnames=surnames,
                year=year,
                venue=venue,
                volume="",
                issue="",
                pages="",
                doi=doi,
                arxiv_id="",
                abstract=abstract,
                source="base_title",
                score=0.0,
            ))
        self._sleep()
        return out

    def _candidate_from_eric(self, doc: dict[str, Any], source: str) -> CandidateMetadata | None:
        title = ensure_str(doc.get("title", ""))
        if not title:
            return None
        # ERIC author format is usually "Last, First Middle" — surname is the part before the comma
        authors_raw = doc.get("author") or []
        if isinstance(authors_raw, str):
            authors_raw = [authors_raw]
        authors: list[str] = [ensure_str(a) for a in authors_raw if ensure_str(a)]
        surnames: list[str] = [a.split(",")[0].strip() for a in authors if a]

        # DOI/identifier list may contain the DOI alongside ERIC IDs like "EJ123456"
        identifiers = doc.get("identifier") or []
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        doi = ""
        for ident in identifiers:
            stripped = strip_doi(ensure_str(ident))
            if stripped.startswith("10."):
                doi = stripped
                break

        year_raw = doc.get("publicationdateyear") or doc.get("publicationDateYear") or ""
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=ensure_str(year_raw)[:4],
            venue=ensure_str(doc.get("source", "")),
            volume="",
            issue="",
            pages="",
            doi=doi,
            arxiv_id="",
            abstract=ensure_str(doc.get("description", "")),
            source=source,
            score=0.0,
        )

    def _eric_by_doi(self, doi: str) -> CandidateMetadata | None:
        """Look up a paper in the ERIC database by DOI.

        ERIC (Education Resources Information Center) is the US Department of
        Education's free academic database, excellent for ELT/applied-linguistics
        papers that may not appear in OpenAlex or Crossref.
        """
        params = {
            "search": f'identifier:"{doi}"',
            "fields": "title,author,description,publicationdateyear,source,issn,identifier",
            "format": "json",
            "rows": "1",
        }
        try:
            resp = self.session.get(ERIC_SEARCH_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return None
            docs = (resp.json().get("response") or {}).get("docs") or []
        except (requests.RequestException, ValueError):
            self._sleep()
            return None
        candidate = self._candidate_from_eric(docs[0], source="eric_doi") if docs else None
        self._sleep()
        return candidate

    def _eric_by_title(self, title: str) -> list[CandidateMetadata]:
        """Search ERIC by title. Returns up to 5 candidates."""
        params = {
            "search": f'title:"{title}"',
            "fields": "title,author,description,publicationdateyear,source,issn,identifier",
            "format": "json",
            "rows": "5",
        }
        try:
            resp = self.session.get(ERIC_SEARCH_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return []
            docs = (resp.json().get("response") or {}).get("docs") or []
        except (requests.RequestException, ValueError):
            self._sleep()
            return []
        out: list[CandidateMetadata] = []
        for doc in docs:
            candidate = self._candidate_from_eric(doc, source="eric_title")
            if candidate:
                out.append(candidate)
        self._sleep()
        return out

    def _researchgate_by_title(self, title: str) -> list[CandidateMetadata]:
        """Attempt to find a paper on ResearchGate by scraping its search page.

        ResearchGate aggressively blocks automated requests, so this often
        returns an empty list.  When it does work, the publication pages carry
        standard citation_* meta tags that _scrape_academic_page can parse.
        """
        try:
            resp = self.session.get(
                RESEARCHGATE_SEARCH,
                params={"q": title},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.researchgate.net/",
                },
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                self._sleep()
                return []
            html = resp.text
        except requests.RequestException:
            self._sleep()
            return []

        # Extract publication page URLs from search results
        pub_urls: list[str] = []
        for m in re.finditer(r'href="(/publication/\d+[^"]*)"', html):
            url = "https://www.researchgate.net" + m.group(1).split("?")[0]
            if url not in pub_urls:
                pub_urls.append(url)
            if len(pub_urls) >= 5:
                break

        out: list[CandidateMetadata] = []
        for url in pub_urls:
            scraped = self._scrape_academic_page(url)
            if scraped:
                scraped.source = "researchgate_scrape"
                out.append(scraped)
        self._sleep()
        return out

    def _unpaywall_oa_urls(self, doi: str) -> list[str]:
        """Return open-access landing-page URLs for a DOI via the Unpaywall API.

        Unpaywall indexes OA copies of 50M+ papers (institutional repositories,
        author homepages, preprint servers).  These pages are freely scrapeable
        and typically expose the abstract in standard citation_* meta tags.
        No API key required; only a contact email is needed.
        """
        if not doi:
            return []
        email = self.mailto or "arg-analysis-enricher@example.com"
        try:
            resp = self.session.get(
                f"{UNPAYWALL_API}/{urllib.parse.quote(doi, safe='')}",
                params={"email": email},
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                self._sleep()
                return []
            data = resp.json()
        except (requests.RequestException, ValueError):
            self._sleep()
            return []

        urls: list[str] = []
        seen: set[str] = set()

        def _add_url(u: str) -> None:
            u = ensure_str(u)
            if u and u not in seen and not u.lower().endswith(".pdf"):
                seen.add(u)
                urls.append(u)

        # Prefer the best OA location, then fall back to all OA locations
        best = data.get("best_oa_location") or {}
        _add_url(best.get("url_for_landing_page") or "")
        for loc in data.get("oa_locations") or []:
            _add_url(loc.get("url_for_landing_page") or "")

        self._sleep()
        return urls

    def _google_scholar_urls(self, query: str, n: int = 5) -> list[str]:
        """Search Google Scholar and return up to n article page URLs.

        Scholar is used instead of general search engines because it ranks
        academic paper pages first and its HTML is stable enough to parse.
        Result page URLs are extracted from <h3 class="gs_rt"> anchor tags;
        PDF links are skipped in favour of HTML landing pages.
        """
        try:
            resp = self.session.get(
                "https://scholar.google.com/scholar",
                params={"q": query, "hl": "en", "num": str(min(n * 2, 10))},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=self.timeout_seconds,
            )
            # 429 = rate-limited; anything other than 200 means no results
            if resp.status_code != 200:
                self._sleep()
                return []
            html = resp.text
        except requests.RequestException:
            self._sleep()
            return []

        # Scholar result links live inside <h3 class="gs_rt"><a href="...">
        urls: list[str] = []
        for m in re.finditer(r'<h3[^>]*class="gs_rt"[^>]*>.*?<a[^>]+href="([^"]+)"', html, re.DOTALL):
            url = m.group(1)
            # Prefer HTML landing pages over PDF files
            if url.lower().endswith(".pdf"):
                continue
            if url.startswith("http") and "google.com" not in url and url not in urls:
                urls.append(url)
            if len(urls) >= n:
                break

        self._sleep()
        return urls

    # Browser-like headers that pass basic bot-detection on publisher sites
    # (ScienceDirect, Springer, Wiley, etc.).
    _SCRAPE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def _scrape_academic_page(self, url: str) -> CandidateMetadata | None:
        """Fetch a web page and extract paper metadata from standard academic meta tags.

        Uses realistic browser headers so that publisher sites (ScienceDirect,
        Springer, Wiley) return the full HTML with citation_* meta tags rather
        than a bot-detection redirect.

        For ScienceDirect URLs, also tries the /science/article/abs/ variant
        which serves the abstract page rather than the full-text PDF gate.
        """
        # ScienceDirect: the /article/pii/ URL often redirects to a paywall;
        # /article/abs/pii/ serves the abstract page with citation meta tags.
        url = re.sub(
            r"(sciencedirect\.com/science/article)/pii/",
            r"\1/abs/pii/",
            url,
        )
        try:
            resp = self.session.get(
                url,
                headers=self._SCRAPE_HEADERS,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            if resp.status_code != 200:
                self._sleep()
                return None
            html = resp.text
        except requests.RequestException:
            self._sleep()
            return None

        parser = MetaTagParser()
        try:
            parser.feed(html)
        except Exception:
            self._sleep()
            return None

        # ── Title ──────────────────────────────────────────────────────────────
        title = parser.first(
            "citation_title",
            "dc.title", "DC.Title",
            "og:title", "twitter:title",
        )
        # ── Abstract ───────────────────────────────────────────────────────────
        abstract = jats_to_text(parser.first(
            "citation_abstract",
            "dc.description", "DC.Description",
            "description",
            "og:description",
        ))
        # ── Authors ────────────────────────────────────────────────────────────
        authors_raw = parser.all_values("citation_author", "dc.creator", "DC.Creator")
        # ── Bibliographic ──────────────────────────────────────────────────────
        year_raw = parser.first("citation_date", "citation_publication_date", "dc.date", "DC.Date")
        year = year_raw[:4] if year_raw else ""
        doi_raw = strip_doi(parser.first("citation_doi", "dc.identifier", "DC.Identifier"))
        doi = doi_raw if doi_raw.startswith("10.") else ""
        journal = parser.first("citation_journal_title", "citation_conference_title", "dc.source", "DC.Source")
        volume = parser.first("citation_volume")
        issue = parser.first("citation_issue")
        first_page = parser.first("citation_firstpage")
        last_page = parser.first("citation_lastpage")
        pages = f"{first_page}-{last_page}" if first_page and last_page else first_page

        # ── JSON-LD fallback ──────────────────────────────────────────────────
        if not title:
            for blob in parser.json_ld_blocks:
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, list):
                        obj = obj[0] if obj else {}
                    if not isinstance(obj, dict):
                        continue
                    t = ensure_str(obj.get("name") or obj.get("headline") or "")
                    if t:
                        title = t
                        abstract = abstract or jats_to_text(ensure_str(
                            obj.get("description") or obj.get("abstract") or ""
                        ))
                        if not year:
                            year = ensure_str(obj.get("datePublished") or "")[:4]
                        break
                except (json.JSONDecodeError, AttributeError):
                    continue

        if not title:
            self._sleep()
            return None

        authors: list[str] = []
        surnames: list[str] = []
        for name in authors_raw:
            if not name:
                continue
            authors.append(name)
            parts = name.split()
            if parts:
                surnames.append(parts[-1])

        self._sleep()
        return CandidateMetadata(
            title=title,
            authors=authors,
            author_surnames=surnames,
            year=year,
            venue=journal,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
            arxiv_id="",
            abstract=abstract,
            source="web_scrape",
            score=0.0,
        )

    def _web_by_title(self, title: str, year: str = "", authors: list[str] | None = None) -> list[CandidateMetadata]:
        """Find a paper using BASE academic search engine + page scraping as a fallback.

        Strategy:
          1. Query BASE (Bielefeld Academic Search Engine) with the exact title —
             covers 250M+ OA documents including small/regional journals.
          2. If BASE finds nothing, try a DuckDuckGo search and scrape the result
             pages for academic meta tags (citation_*, DC, JSON-LD).
        """
        # ── Step 1: BASE API ─────────────────────────────────────────────────
        base_candidates = self._base_by_title(title)
        if base_candidates:
            return base_candidates

        # ── Step 2: Google Scholar search + page scraping ───────────────────
        # Build query variants from most specific to most general, stopping
        # as soon as we get result URLs.
        keywords = sorted(
            [w for w in normalize_title_key(title).split() if len(w) > 3],
            key=len, reverse=True,
        )[:7]
        keyword_str = " ".join(keywords)
        surname = ""
        if authors:
            parts = (authors[0] or "").split()
            if parts:
                surname = parts[-1]

        urls: list[str] = []
        for q in ([f'"{title}"'] + ([f"{surname} {keyword_str}"] if surname else []) + [keyword_str]):
            urls = self._google_scholar_urls(q)
            if urls:
                break

        skip_domains = ("google.", "bing.", "yahoo.", "facebook.", "twitter.",
                        "reddit.", "wikipedia.", "amazon.", "youtube.")
        out: list[CandidateMetadata] = []
        seen: set[str] = set()
        for url in urls:
            if any(d in url.lower() for d in skip_domains) or url in seen:
                continue
            seen.add(url)
            candidate = self._scrape_academic_page(url)
            if candidate:
                out.append(candidate)
        return out

    def _arxiv_by_id(self, arxiv_id: str) -> CandidateMetadata | None:
        params = {"id_list": arxiv_id, "max_results": "1"}
        try:
            resp = self.session.get(ARXIV_API, params=params, timeout=self.timeout_seconds)
            if resp.status_code != 200:
                self._sleep()
                return None
            xml_text = resp.text
        except requests.RequestException:
            self._sleep()
            return None
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            self._sleep()
            return None
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            self._sleep()
            return None
        title = ensure_str(entry.findtext("atom:title", "", ns))
        if not title:
            self._sleep()
            return None
        summary = ensure_str(entry.findtext("atom:summary", "", ns))
        published = ensure_str(entry.findtext("atom:published", "", ns))
        year = published[:4] if len(published) >= 4 else ""
        author_names: list[str] = []
        surnames: list[str] = []
        for author in entry.findall("atom:author", ns):
            name = ensure_str(author.findtext("atom:name", "", ns))
            if name:
                author_names.append(name)
                parts = name.split()
                if parts:
                    surnames.append(parts[-1])
        self._sleep()
        return CandidateMetadata(
            title=title,
            authors=author_names,
            author_surnames=surnames,
            year=year,
            venue="arXiv",
            volume="",
            issue="",
            pages="",
            doi="",
            arxiv_id=arxiv_id,
            abstract=summary,
            source="arxiv_id",
            score=0.0,
        )


def _year_int(year_str: str) -> int:
    """Parse a 4-digit year from a string; return 0 if unparseable."""
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", year_str)
    return int(m.group(1)) if m else 0


def is_suspect_entry(ref_obj: dict[str, Any], reference_text: str) -> bool:
    """Return True if the current enriched metadata looks wrong and should be re-resolved.

    Criteria (any one is enough to flag):
    - source is "heuristic" (always a low-quality placeholder)
    - metadata_match_score < 0.55 (resolver was not confident)
    - resolved title has very low overlap with the raw reference string (< 0.25)
    - enriched year differs by more than 3 years from the year stated in the raw
      reference text (strong signal of a wrong-paper DOI hit)
    - for non-DOI sources: none of the resolved author surnames appear in the
      reference text (strong signal of a wrong-paper match)
    """
    source = ensure_str(ref_obj.get("metadata_source", ""))
    if not source:
        return False  # not yet enriched — nothing to update

    if source == "heuristic":
        return True

    score = float(ref_obj.get("metadata_match_score", 0.0) or 0.0)
    if score < 0.55:
        return True

    resolved_title = ensure_str(ref_obj.get("title", ""))
    if resolved_title:
        title_in_ref = max(
            similarity(reference_text, resolved_title),
            token_overlap(resolved_title, reference_text),
        )
        if title_in_ref < 0.25:
            return True

    # Year mismatch: a reference always cites the paper's own publication year.
    # If GROBID pulled a wrong DOI, the resolved year often differs markedly.
    # For DOI-based sources the threshold is tighter (> 3 years) because the
    # DOI should point to the exact paper.  For title-based sources we allow
    # a wider gap (> 6 years) since a title search may legitimately surface a
    # different edition, conference version, or revised reprint.
    enriched_year = _year_int(ensure_str(ref_obj.get("year", "")))
    ref_year = _year_int(extract_year(reference_text))
    if enriched_year and ref_year:
        gap = abs(enriched_year - ref_year)
        _doi_based = {
            "openalex_doi", "semantic_scholar_doi", "crossref_doi",
            "sciencedirect_doi", "doi_org_doi", "europepmc_doi", "pubmed_doi",
            "eric_doi", "scopus_doi",
        }
        year_threshold = 3 if source.split("+")[0] in _doi_based else 6
        if gap > year_threshold:
            return True

    _doi_sources = {
        "openalex_doi", "semantic_scholar_doi", "crossref_doi",
        "sciencedirect_doi", "doi_org_doi", "europepmc_doi", "pubmed_doi",
        "eric_doi", "scopus_doi",
    }
    # Composite sources like "openalex_title+crossref_doi_abstract" — use the
    # base source (before "+") to decide whether this was a DOI-based hit.
    base_source = source.split("+")[0]
    if base_source not in _doi_sources:
        resolved_surnames = ensure_str_list(ref_obj.get("author_surnames", []))
        if not resolved_surnames:
            resolved_surnames = authors_to_surnames(ensure_str_list(ref_obj.get("authors", [])))
        if resolved_surnames and not any(contains_surname(reference_text, s) for s in resolved_surnames):
            return True

    return False


def update_fields(reference_obj: dict[str, Any], resolved: CandidateMetadata, overwrite_nonempty: bool) -> list[str]:
    """Apply resolved metadata to reference_obj. Returns list of field names that changed."""
    changed: list[str] = []
    field_map: dict[str, Any] = {
        "title": resolved.title,
        "authors": resolved.authors,
        "year": resolved.year,
        "venue": resolved.venue,
        "volume": resolved.volume,
        "issue": resolved.issue,
        "pages": resolved.pages,
        "doi": resolved.doi,
        "arxiv_id": resolved.arxiv_id,
        "abstract": resolved.abstract,
        "metadata_source": resolved.source,
        "metadata_match_score": round(resolved.score, 4),
    }
    for field, new_val in field_map.items():
        old_val = reference_obj.get(field)
        if not overwrite_nonempty and old_val not in (None, "", []):
            continue
        if old_val != new_val and new_val not in (None, "", []):
            reference_obj[field] = new_val
            changed.append(field)
    return changed


def iter_reference_json_files(refs_dir: Path) -> list[Path]:
    return [path for path in sorted(refs_dir.glob("*.json")) if not path.name.startswith("_")]


def cache_key(reference_obj: dict[str, Any], reference_text: str) -> str:
    doi = strip_doi(ensure_str(reference_obj.get("doi", "")))
    if not doi:
        doi = extract_doi(reference_text)
    if doi:
        return f"doi:{doi}"
    return f"ref:{normalize_title_key(reference_text)[:260]}"


def serialize_candidate(candidate: CandidateMetadata) -> dict[str, Any]:
    return {
        "title": candidate.title,
        "authors": candidate.authors,
        "author_surnames": candidate.author_surnames,
        "year": candidate.year,
        "venue": candidate.venue,
        "volume": candidate.volume,
        "issue": candidate.issue,
        "pages": candidate.pages,
        "doi": candidate.doi,
        "arxiv_id": candidate.arxiv_id,
        "abstract": candidate.abstract,
        "source": candidate.source,
        "score": candidate.score,
        "url": candidate.url,
    }


def deserialize_candidate(data: dict[str, Any]) -> CandidateMetadata:
    return CandidateMetadata(
        title=ensure_str(data.get("title", "")),
        authors=ensure_str_list(data.get("authors", [])),
        author_surnames=ensure_str_list(data.get("author_surnames", [])),
        year=ensure_str(data.get("year", "")),
        venue=ensure_str(data.get("venue", "")),
        volume=ensure_str(data.get("volume", "")),
        issue=ensure_str(data.get("issue", "")),
        pages=ensure_str(data.get("pages", "")),
        doi=strip_doi(ensure_str(data.get("doi", ""))),
        arxiv_id=ensure_str(data.get("arxiv_id", "")),
        abstract=ensure_str(data.get("abstract", "")),
        source=ensure_str(data.get("source", "")),
        score=float(data.get("score", 0.0) or 0.0),
        url=ensure_str(data.get("url", "")),
    )


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _short(text: str, width: int = 70) -> str:
    """Truncate text for display."""
    text = text.strip()
    return text if len(text) <= width else text[:width - 1] + "…"


def _suspect_reason(ref_obj: dict[str, Any], reference_text: str) -> str:
    """Human-readable explanation of why an entry was flagged as suspect."""
    source = ensure_str(ref_obj.get("metadata_source", ""))
    score = float(ref_obj.get("metadata_match_score", 0.0) or 0.0)
    if source == "heuristic":
        return "source=heuristic"
    if score < 0.55:
        return f"score={score:.2f}<0.55"
    resolved_title = ensure_str(ref_obj.get("title", ""))
    if resolved_title:
        title_in_ref = max(
            similarity(reference_text, resolved_title),
            token_overlap(resolved_title, reference_text),
        )
        if title_in_ref < 0.25:
            return f"title_overlap={title_in_ref:.2f}<0.25"
    enriched_year = _year_int(ensure_str(ref_obj.get("year", "")))
    ref_year = _year_int(extract_year(reference_text))
    if enriched_year and ref_year:
        gap = abs(enriched_year - ref_year)
        _doi_based_r = {
            "openalex_doi", "semantic_scholar_doi", "crossref_doi",
            "sciencedirect_doi", "doi_org_doi", "europepmc_doi", "pubmed_doi",
            "eric_doi",
        }
        year_threshold_r = 3 if source.split("+")[0] in _doi_based_r else 6
        if gap > year_threshold_r:
            return f"year_mismatch(enriched={enriched_year}, ref_text={ref_year}, gap={gap})"
    _doi_sources = {
        "openalex_doi", "semantic_scholar_doi", "crossref_doi",
        "sciencedirect_doi", "doi_org_doi", "europepmc_doi", "pubmed_doi",
        "eric_doi", "scopus_doi",
    }
    base_source = source.split("+")[0]
    if base_source not in _doi_sources:
        resolved_surnames = ensure_str_list(ref_obj.get("author_surnames", []))
        if not resolved_surnames:
            resolved_surnames = authors_to_surnames(ensure_str_list(ref_obj.get("authors", [])))
        if resolved_surnames and not any(contains_surname(reference_text, s) for s in resolved_surnames):
            return f"no_author_match (resolved={resolved_surnames[:2]})"
    return "unknown"


def process_file(
    path: Path,
    resolver: MetadataResolver,
    cache: dict[str, dict[str, Any]],
    overwrite_nonempty: bool,
    max_refs_per_file: int,
    update_mode: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    refs = payload.get("references")
    if not isinstance(refs, list):
        print(f"  SKIP {path.name}: no references list")
        return {"file": path.name, "status": "skipped_no_references", "references": 0, "processed": 0, "updated": 0, "looked_up": 0, "suspect": 0}

    processed = 0
    updated = 0
    looked_up = 0
    suspect = 0
    with_abstract = 0
    n_refs = len(refs)

    for ref_obj in refs:
        if not isinstance(ref_obj, dict):
            continue
        raw_ref = ensure_str(ref_obj.get("reference", ""))
        if not raw_ref:
            continue
        if max_refs_per_file > 0 and processed >= max_refs_per_file:
            break

        ref_idx = ref_obj.get("index", processed + 1)
        ref_label = f"[ref {ref_idx:>3}/{n_refs}]"

        key = cache_key(ref_obj, raw_ref)
        cached = cache.get(key)
        must_refresh = False
        force_overwrite = overwrite_nonempty
        is_suspect = False

        # --update mode: if the existing enriched data looks wrong, invalidate
        # the cache entry and force a fresh lookup + field overwrite.
        if update_mode and ensure_str(ref_obj.get("metadata_source", "")):
            if is_suspect_entry(ref_obj, raw_ref):
                reason = _suspect_reason(ref_obj, raw_ref)
                old_title = _short(ensure_str(ref_obj.get("title", raw_ref)), 60)
                print(f"  {ref_label} SUSPECT ({reason}): {old_title!r}")
                cache.pop(key, None)
                cached = None
                must_refresh = True
                force_overwrite = True
                is_suspect = True
                suspect += 1

        if cached:
            cached_source = ensure_str(cached.get("source", ""))
            cached_abstract = ensure_str(cached.get("abstract", ""))
            existing_source = ensure_str(ref_obj.get("metadata_source", ""))
            existing_abstract = ensure_str(ref_obj.get("abstract", ""))
            if cached_source == "heuristic":
                must_refresh = True
            if not cached_abstract and (existing_source or existing_abstract == ""):
                must_refresh = True

        if cached and not must_refresh:
            resolved = deserialize_candidate(cached)
            if verbose:
                abs_flag = " [has abstract]" if ensure_str(resolved.abstract) else ""
                print(f"  {ref_label} cached  src={resolved.source}  score={resolved.score:.2f}{abs_flag}  {_short(resolved.title, 55)!r}")
        else:
            resolved = resolver.resolve(raw_ref, ref_obj)
            cache[key] = serialize_candidate(resolved)
            looked_up += 1
            abs_flag = " [abstract found]" if ensure_str(resolved.abstract) else " [no abstract]"
            print(f"  {ref_label} looked up → src={resolved.source}  score={resolved.score:.2f}{abs_flag}")
            print(f"              title: {_short(resolved.title, 65)!r}")

        # Heuristic entries are explicitly low-confidence placeholders, so always
        # overwrite their fields regardless of the --overwrite-nonempty flag.
        is_heuristic = ensure_str(ref_obj.get("metadata_source", "")) == "heuristic"
        changed_fields = update_fields(ref_obj, resolved, overwrite_nonempty=force_overwrite or is_heuristic)
        if changed_fields:
            updated += 1
            # Always log what changed for suspect entries or fresh lookups;
            # for verbose mode log cache-hit changes too.
            if is_suspect or must_refresh or verbose:
                # Highlight the most informative changed fields
                important = [f for f in changed_fields if f in ("title", "authors", "year", "abstract", "doi", "metadata_source", "metadata_match_score")]
                other = [f for f in changed_fields if f not in important]
                parts = important + (["...+" + str(len(other)) + " more"] if other else [])
                print(f"              changed: {', '.join(parts)}")
        elif is_suspect or must_refresh:
            print(f"              no fields changed (resolved score={resolved.score:.2f}, threshold not met)")

        if ensure_str(ref_obj.get("abstract", "")):
            with_abstract += 1
        processed += 1

    payload["llm_metadata_enrichment"] = {
        "updated_at_unix": int(time.time()),
        "method": "openalex_semantic_scholar_crossref_sciencedirect_scopus_doi_org_europepmc_pubmed_eric_dblp_arxiv_researchgate_unpaywall_web_heuristic",
        "references_with_abstract": with_abstract,
        "reference_count": n_refs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    suspect_part = f"  suspect={suspect}" if update_mode else ""
    print(
        f"  → done: processed={processed}/{n_refs}  looked_up={looked_up}"
        f"  updated={updated}  with_abstract={with_abstract}/{n_refs}{suspect_part}"
    )

    return {
        "file": path.name,
        "status": "ok",
        "references": n_refs,
        "processed": processed,
        "updated": updated,
        "looked_up": looked_up,
        "with_abstract": with_abstract,
        "suspect": suspect,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs-dir", default="data/raw_data/references_json_grobid", help="Directory containing per-paper reference JSON files")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="HTTP timeout per lookup")
    parser.add_argument("--sleep-seconds", type=float, default=0.08, help="Delay between API requests")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files to process (0 = all)")
    parser.add_argument("--max-refs-per-file", type=int, default=0, help="Limit references processed per file (0 = all)")
    parser.add_argument("--overwrite-nonempty", action="store_true", help="Also overwrite existing non-empty metadata fields")
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Re-check existing enriched data and re-resolve entries that look wrong. "
            "An entry is flagged as suspect when: its metadata_match_score < 0.55, "
            "the resolved title has very low overlap with the raw reference string, "
            "or (for non-DOI sources) none of the resolved author surnames appear in "
            "the reference text. Suspect entries have their cache invalidated and are "
            "re-resolved with field overwrite enabled."
        ),
    )
    parser.add_argument("--cache-path", default="", help="Path to cache JSON (default: refs-dir/_metadata_cache.json)")
    parser.add_argument("--summary-path", default="", help="Path to summary JSON (default: refs-dir/_llm_metadata_summary.json)")
    parser.add_argument("--mailto", default="", help="Optional contact email for OpenAlex polite pool")
    parser.add_argument(
        "--elsevier-api-key",
        default=os.environ.get("ELSEVIER_API_KEY", ""),
        help="Elsevier/ScienceDirect API key (default: ELSEVIER_API_KEY env var / .env file)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Log every reference including cache hits, not just lookups and changes")
    args = parser.parse_args()

    refs_dir = Path(args.refs_dir)
    if not refs_dir.exists() or not refs_dir.is_dir():
        raise SystemExit(f"Invalid refs directory: {refs_dir}")

    cache_path = Path(args.cache_path) if args.cache_path else (refs_dir / "_metadata_cache.json")
    summary_path = Path(args.summary_path) if args.summary_path else (refs_dir / "_llm_metadata_summary.json")
    cache = load_cache(cache_path)

    files = iter_reference_json_files(refs_dir)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No JSON files found in {refs_dir}")

    resolver = MetadataResolver(
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        mailto=(args.mailto or "").strip() or None,
        elsevier_api_key=(args.elsevier_api_key or "").strip() or None,
    )

    summaries: list[dict[str, Any]] = []
    total_refs = 0
    total_processed = 0
    total_updated = 0
    total_looked_up = 0
    total_with_abstract = 0
    total_suspect = 0

    mode_label = " [UPDATE MODE]" if args.update else ""
    print(f"Processing {len(files)} file(s) from {refs_dir}{mode_label}\n")

    for idx, path in enumerate(files, start=1):
        print(f"{'='*60}")
        print(f"[{idx}/{len(files)}] {path.name}")
        result = process_file(
            path=path,
            resolver=resolver,
            cache=cache,
            overwrite_nonempty=args.overwrite_nonempty,
            max_refs_per_file=args.max_refs_per_file,
            update_mode=args.update,
            verbose=args.verbose,
        )
        summaries.append(result)
        total_refs += int(result.get("references", 0) or 0)
        total_processed += int(result.get("processed", 0) or 0)
        total_updated += int(result.get("updated", 0) or 0)
        total_looked_up += int(result.get("looked_up", 0) or 0)
        total_with_abstract += int(result.get("with_abstract", 0) or 0)
        total_suspect += int(result.get("suspect", 0) or 0)

    save_cache(cache_path, cache)
    summary = {
        "refs_dir": str(refs_dir),
        "files_processed": len(files),
        "total_references": total_refs,
        "references_processed": total_processed,
        "references_updated": total_updated,
        "lookups_performed": total_looked_up,
        "references_with_abstract": total_with_abstract,
        "suspect_re_resolved": total_suspect,
        "updated_at_unix": int(time.time()),
        "cache_path": str(cache_path),
        "files": summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"  files processed : {len(files)}")
    print(f"  total refs      : {total_refs}")
    print(f"  looked up       : {total_looked_up}")
    print(f"  updated         : {total_updated}")
    print(f"  with abstract   : {total_with_abstract}")
    if args.update:
        print(f"  suspect fixed   : {total_suspect}")
    print(f"  cache           : {cache_path}")
    print(f"  summary         : {summary_path}")


if __name__ == "__main__":
    main()
