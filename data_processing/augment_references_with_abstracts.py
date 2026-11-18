#!/usr/bin/env python3
"""Augment extracted reference JSON files with abstracts.

Primary source order:
1) OpenAlex (DOI lookup, then title search)
2) Crossref (DOI lookup, then bibliographic query)

The script updates each reference object in-place with fields:
- abstract
- abstract_source
- abstract_match_score
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any
from difflib import SequenceMatcher

import requests


OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(doi: str) -> str:
    value = normalize_ws(doi).lower()
    value = value.replace("https://doi.org/", "").replace("http://doi.org/", "")
    value = value.replace("doi:", "")
    return value.strip(" /")


def title_key(title: str) -> str:
    value = normalize_ws(title).lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return normalize_ws(value)


def jats_to_text(value: str) -> str:
    # Crossref abstracts may contain JATS XML fragments.
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


@dataclass
class AbstractHit:
    abstract: str
    source: str
    score: float


def best_title_similarity(query_title: str, candidate_title: str) -> float:
    q = title_key(query_title)
    c = title_key(candidate_title)
    if not q or not c:
        return 0.0
    return SequenceMatcher(a=q, b=c).ratio()


class AbstractResolver:
    def __init__(self, timeout: int, sleep_seconds: float, mailto: str | None = None) -> None:
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.mailto = mailto
        self.session = requests.Session()

    def _sleep(self) -> None:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _openalex_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def resolve(self, doi: str, title: str, year: str) -> AbstractHit | None:
        doi_norm = normalize_doi(doi) if doi else ""

        if doi_norm:
            hit = self._openalex_by_doi(doi_norm)
            if hit:
                return hit

        if title:
            hit = self._openalex_by_title(title, year)
            if hit:
                return hit

        if doi_norm:
            hit = self._crossref_by_doi(doi_norm)
            if hit:
                return hit

        if title:
            hit = self._crossref_by_title(title, year)
            if hit:
                return hit

        return None

    def _openalex_by_doi(self, doi: str) -> AbstractHit | None:
        url = f"{OPENALEX_BASE}/works/https://doi.org/{doi}"
        try:
            resp = self.session.get(url, params=self._openalex_params(), timeout=self.timeout)
            if resp.status_code != 200:
                self._sleep()
                return None
            data = resp.json()
        except requests.RequestException:
            self._sleep()
            return None

        abstract = inverted_index_to_text(data.get("abstract_inverted_index"))
        self._sleep()
        if not abstract:
            return None
        return AbstractHit(abstract=abstract, source="openalex_doi", score=1.0)

    def _openalex_by_title(self, title: str, year: str) -> AbstractHit | None:
        params = {
            "search": title,
            "per-page": "5",
            "select": "display_name,abstract_inverted_index,publication_year,doi",
        }
        params.update(self._openalex_params())
        try:
            resp = self.session.get(f"{OPENALEX_BASE}/works", params=params, timeout=self.timeout)
            if resp.status_code != 200:
                self._sleep()
                return None
            data = resp.json()
        except requests.RequestException:
            self._sleep()
            return None

        results = data.get("results") or []
        best: AbstractHit | None = None
        for result in results:
            abstract = inverted_index_to_text(result.get("abstract_inverted_index"))
            if not abstract:
                continue
            candidate_title = result.get("display_name") or ""
            sim = best_title_similarity(title, candidate_title)
            year_bonus = 0.0
            if year and str(result.get("publication_year") or "") == str(year):
                year_bonus = 0.1
            score = min(1.0, sim + year_bonus)
            if best is None or score > best.score:
                best = AbstractHit(abstract=abstract, source="openalex_title", score=score)

        self._sleep()
        if best and best.score >= 0.55:
            return best
        return None

    def _crossref_by_doi(self, doi: str) -> AbstractHit | None:
        try:
            resp = self.session.get(f"{CROSSREF_BASE}/works/{doi}", timeout=self.timeout)
            if resp.status_code != 200:
                self._sleep()
                return None
            data = resp.json().get("message") or {}
        except requests.RequestException:
            self._sleep()
            return None

        abstract = jats_to_text(data.get("abstract") or "")
        self._sleep()
        if not abstract:
            return None
        return AbstractHit(abstract=abstract, source="crossref_doi", score=1.0)

    def _crossref_by_title(self, title: str, year: str) -> AbstractHit | None:
        params = {
            "query.bibliographic": title,
            "rows": "5",
        }
        try:
            resp = self.session.get(f"{CROSSREF_BASE}/works", params=params, timeout=self.timeout)
            if resp.status_code != 200:
                self._sleep()
                return None
            items = (resp.json().get("message") or {}).get("items") or []
        except requests.RequestException:
            self._sleep()
            return None

        best: AbstractHit | None = None
        for item in items:
            abstract = jats_to_text(item.get("abstract") or "")
            if not abstract:
                continue
            titles = item.get("title") or []
            candidate_title = titles[0] if titles else ""
            sim = best_title_similarity(title, candidate_title)
            pub_year = ""
            parts = (((item.get("issued") or {}).get("date-parts") or [[""]])[0])
            if parts and parts[0]:
                pub_year = str(parts[0])
            year_bonus = 0.1 if year and pub_year == str(year) else 0.0
            score = min(1.0, sim + year_bonus)
            if best is None or score > best.score:
                best = AbstractHit(abstract=abstract, source="crossref_title", score=score)

        self._sleep()
        if best and best.score >= 0.55:
            return best
        return None


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key(doi: str, title: str, year: str) -> str:
    doi_norm = normalize_doi(doi) if doi else ""
    if doi_norm:
        return f"doi:{doi_norm}"
    return f"title:{title_key(title)}|year:{str(year or '').strip()}"


def process_file(
    path: Path,
    resolver: AbstractResolver,
    cache: dict[str, dict[str, Any]],
    overwrite_abstracts: bool,
    max_refs: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    refs = payload.get("references") or []

    looked_up = 0
    added = 0
    skipped_existing = 0
    misses = 0

    for i, ref in enumerate(refs, start=1):
        if max_refs > 0 and i > max_refs:
            break

        if ref.get("abstract") and not overwrite_abstracts:
            skipped_existing += 1
            continue

        doi = str(ref.get("doi") or "")
        title = str(ref.get("title") or "")
        if not title:
            # GROBID can leave title empty. Use a prefix of raw reference text as backup query text.
            raw = normalize_ws(str(ref.get("reference") or ""))
            title = raw[:220]
        year = str(ref.get("year") or "")

        key = cache_key(doi=doi, title=title, year=year)
        cached = cache.get(key)
        if cached is None:
            looked_up += 1
            hit = resolver.resolve(doi=doi, title=title, year=year)
            if hit:
                cached = {
                    "abstract": hit.abstract,
                    "abstract_source": hit.source,
                    "abstract_match_score": round(hit.score, 4),
                }
            else:
                cached = {
                    "abstract": "",
                    "abstract_source": "",
                    "abstract_match_score": 0.0,
                }
            cache[key] = cached

        abstract = str(cached.get("abstract") or "")
        if abstract:
            ref["abstract"] = abstract
            ref["abstract_source"] = str(cached.get("abstract_source") or "")
            ref["abstract_match_score"] = float(cached.get("abstract_match_score") or 0.0)
            added += 1
        else:
            if overwrite_abstracts:
                ref["abstract"] = ""
                ref["abstract_source"] = ""
                ref["abstract_match_score"] = 0.0
            misses += 1

    payload["abstract_enrichment"] = {
        "sources": ["openalex", "crossref"],
        "updated_at_unix": int(time.time()),
        "references_with_abstract": sum(1 for r in refs if r.get("abstract")),
        "reference_count": len(refs),
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "file": path.name,
        "reference_count": len(refs),
        "looked_up": looked_up,
        "added": added,
        "misses": misses,
        "skipped_existing": skipped_existing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refs-dir",
        default="data/raw_data/references_json_grobid",
        help="Directory containing per-paper reference JSON files",
    )
    parser.add_argument(
        "--cache-path",
        default="",
        help="Path to cache JSON (defaults to refs-dir/_abstract_cache.json)",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Path to summary JSON (defaults to refs-dir/_abstract_summary.json)",
    )
    parser.add_argument("--overwrite-abstracts", action="store_true", help="Re-query references that already have abstracts")
    parser.add_argument("--max-papers", type=int, default=0, help="Limit number of paper JSON files (0 = all)")
    parser.add_argument("--max-refs-per-paper", type=int, default=0, help="Limit refs processed per paper (0 = all)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Sleep between outbound API requests")
    parser.add_argument("--mailto", default="", help="Optional contact email sent to OpenAlex")
    args = parser.parse_args()

    refs_dir = Path(args.refs_dir)
    if not refs_dir.exists():
        raise SystemExit(f"References directory not found: {refs_dir}")

    cache_path = Path(args.cache_path) if args.cache_path else (refs_dir / "_abstract_cache.json")
    summary_path = Path(args.summary_path) if args.summary_path else (refs_dir / "_abstract_summary.json")

    files = sorted(p for p in refs_dir.glob("*.json") if not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"No per-paper JSON files found in: {refs_dir}")

    if args.max_papers > 0:
        files = files[: args.max_papers]

    cache = load_cache(cache_path)
    resolver = AbstractResolver(
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        mailto=(args.mailto or "").strip() or None,
    )

    per_file: list[dict[str, Any]] = []
    totals = {
        "papers": 0,
        "looked_up": 0,
        "added": 0,
        "misses": 0,
        "skipped_existing": 0,
        "reference_count": 0,
    }

    for idx, path in enumerate(files, start=1):
        stats = process_file(
            path=path,
            resolver=resolver,
            cache=cache,
            overwrite_abstracts=args.overwrite_abstracts,
            max_refs=args.max_refs_per_paper,
        )
        per_file.append(stats)
        totals["papers"] += 1
        totals["looked_up"] += stats["looked_up"]
        totals["added"] += stats["added"]
        totals["misses"] += stats["misses"]
        totals["skipped_existing"] += stats["skipped_existing"]
        totals["reference_count"] += stats["reference_count"]
        print(
            f"[{idx}/{len(files)}] {path.name}: added={stats['added']} misses={stats['misses']} "
            f"lookups={stats['looked_up']}"
        )

    save_cache(cache_path, cache)

    summary_payload = {
        "refs_dir": str(refs_dir),
        "cache_path": str(cache_path),
        "papers_processed": totals["papers"],
        "total_references": totals["reference_count"],
        "lookups_performed": totals["looked_up"],
        "abstracts_added": totals["added"],
        "misses": totals["misses"],
        "skipped_existing": totals["skipped_existing"],
        "files": per_file,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(f"Summary: {summary_path}")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    main()
