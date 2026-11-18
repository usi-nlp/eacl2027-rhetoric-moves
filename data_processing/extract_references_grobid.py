#!/usr/bin/env python3
"""Extract bibliographic references from PDFs using the GROBID REST API.

This is an alternative extraction method to the existing docling-based pipeline.
It calls GROBID's /api/processReferences endpoint and converts TEI XML output
into per-paper JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any
import xml.etree.ElementTree as ET

import requests


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "paper"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_text(node: ET.Element | None, xpath: str) -> str:
    if node is None:
        return ""
    found = node.find(xpath, TEI_NS)
    if found is None:
        return ""
    return normalize_ws("".join(found.itertext()))


def find_all_texts(node: ET.Element | None, xpath: str) -> list[str]:
    if node is None:
        return []
    values: list[str] = []
    for item in node.findall(xpath, TEI_NS):
        text = normalize_ws("".join(item.itertext()))
        if text:
            values.append(text)
    return values


def parse_tei_references(tei_xml: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    bibl_nodes = root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS)
    if not bibl_nodes and root.tag.endswith("biblStruct"):
        bibl_nodes = [root]

    references: list[dict[str, Any]] = []
    for idx, bibl in enumerate(bibl_nodes, start=1):
        title = find_text(
            bibl,
            "./tei:analytic/tei:title[@level='a']"
            "|./tei:analytic/tei:title"
            "|./tei:monogr/tei:title[@level='j']"
            "|./tei:monogr/tei:title",
        )
        raw_ref = find_text(bibl, ".//tei:note[@type='raw_reference']")
        doi = find_text(bibl, ".//tei:idno[@type='DOI']")
        arxiv_id = find_text(bibl, ".//tei:idno[@type='arXiv']")
        year = find_text(bibl, ".//tei:imprint/tei:date[@when]")
        if not year:
            year = find_text(bibl, ".//tei:imprint/tei:date")

        author_surnames = find_all_texts(bibl, ".//tei:author/tei:persName/tei:surname")
        authors = find_all_texts(bibl, ".//tei:author/tei:persName")

        journal_or_venue = find_text(bibl, ".//tei:monogr/tei:title")
        volume = find_text(bibl, ".//tei:imprint/tei:biblScope[@unit='volume']")
        issue = find_text(bibl, ".//tei:imprint/tei:biblScope[@unit='issue']")
        pages = find_text(bibl, ".//tei:imprint/tei:biblScope[@unit='page']")

        entry = {
            "index": idx,
            "reference": raw_ref or normalize_ws("".join(bibl.itertext())),
            "title": title,
            "authors": authors,
            "author_surnames": author_surnames,
            "year": year,
            "venue": journal_or_venue,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "doi": doi,
            "arxiv_id": arxiv_id,
        }
        references.append(entry)

    return references


def call_grobid_process_references(
    pdf_path: Path,
    base_url: str,
    timeout_seconds: int,
    include_raw_citations: bool,
    consolidate_citations: int,
    max_retries: int,
    retry_wait_seconds: float,
) -> tuple[int, str]:
    endpoint = f"{base_url.rstrip('/')}/api/processReferences"
    form_data = {
        "includeRawCitations": "1" if include_raw_citations else "0",
        "consolidateCitations": str(consolidate_citations),
    }
    headers = {"Accept": "application/xml"}

    for attempt in range(1, max_retries + 1):
        with pdf_path.open("rb") as handle:
            files = {"input": (pdf_path.name, handle, "application/pdf")}
            try:
                resp = requests.post(
                    endpoint,
                    data=form_data,
                    files=files,
                    headers=headers,
                    timeout=timeout_seconds,
                )
            except requests.RequestException:
                if attempt < max_retries:
                    time.sleep(retry_wait_seconds)
                    continue
                raise

        if resp.status_code == 503 and attempt < max_retries:
            time.sleep(retry_wait_seconds)
            continue

        return resp.status_code, resp.text

    return 503, ""


def process_pdf(
    pdf_path: Path,
    refs_dir: Path,
    base_url: str,
    timeout_seconds: int,
    include_raw_citations: bool,
    consolidate_citations: int,
    max_retries: int,
    retry_wait_seconds: float,
    overwrite: bool,
    save_tei: bool,
) -> dict[str, Any]:
    stem = sanitize_name(pdf_path.stem)
    json_out = refs_dir / f"{stem}.json"
    tei_out = refs_dir / f"{stem}.tei.xml"

    if json_out.exists() and not overwrite:
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        return {
            "paper_file": payload.get("paper_file", pdf_path.name),
            "reference_count": payload.get("reference_count", 0),
            "status": "skipped_existing",
            "error": None,
        }

    status_code, tei_xml = call_grobid_process_references(
        pdf_path=pdf_path,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        include_raw_citations=include_raw_citations,
        consolidate_citations=consolidate_citations,
        max_retries=max_retries,
        retry_wait_seconds=retry_wait_seconds,
    )

    if status_code != 200:
        return {
            "paper_file": pdf_path.name,
            "reference_count": 0,
            "status": "failed",
            "error": f"grobid_status={status_code}",
        }

    references = parse_tei_references(tei_xml)
    payload: dict[str, Any] = {
        "paper_file": pdf_path.name,
        "paper_stem": stem,
        "method": "grobid_processReferences",
        "grobid_base_url": base_url,
        "grobid_options": {
            "includeRawCitations": int(include_raw_citations),
            "consolidateCitations": int(consolidate_citations),
        },
        "reference_count": len(references),
        "references": references,
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_tei:
        tei_out.write_text(tei_xml, encoding="utf-8")

    return {
        "paper_file": pdf_path.name,
        "reference_count": len(references),
        "status": "ok",
        "error": None,
    }


def check_grobid_alive(base_url: str, timeout_seconds: int) -> bool:
    url = f"{base_url.rstrip('/')}/api/isalive"
    try:
        resp = requests.get(url, timeout=timeout_seconds)
        return resp.status_code == 200 and resp.text.strip().lower() == "true"
    except requests.RequestException:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default="data/raw_data/raw_pdfs", help="Directory with input PDFs")
    parser.add_argument(
        "--refs-dir",
        default="data/raw_data/references_json_grobid",
        help="Directory for per-paper GROBID reference JSON files",
    )
    parser.add_argument("--max-papers", type=int, default=0, help="Limit number of PDFs (0 = all)")
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL for GROBID service, e.g. http://localhost:8070",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="HTTP timeout for each PDF request",
    )
    parser.add_argument(
        "--include-raw-citations",
        action="store_true",
        help="Request raw citation strings from GROBID",
    )
    parser.add_argument(
        "--consolidate-citations",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="GROBID consolidateCitations parameter (0, 1, or 2)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries for transient failures (e.g., HTTP 503)",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=5.0,
        help="Sleep duration between retries",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON outputs")
    parser.add_argument(
        "--save-tei",
        action="store_true",
        help="Also save raw TEI XML responses to refs-dir",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    refs_dir = Path(args.refs_dir)
    refs_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    if args.max_papers > 0:
        pdf_files = pdf_files[: args.max_papers]

    if not check_grobid_alive(args.grobid_url, timeout_seconds=10):
        raise SystemExit(
            "GROBID service is not alive. Start GROBID first, e.g.:\n"
            "docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-crf"
        )

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, pdf in enumerate(pdf_files, start=1):
        try:
            result = process_pdf(
                pdf_path=pdf,
                refs_dir=refs_dir,
                base_url=args.grobid_url,
                timeout_seconds=args.timeout_seconds,
                include_raw_citations=args.include_raw_citations,
                consolidate_citations=args.consolidate_citations,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
                overwrite=args.overwrite,
                save_tei=args.save_tei,
            )
            summary.append(
                {
                    "paper_file": result["paper_file"],
                    "reference_count": result["reference_count"],
                    "status": result["status"],
                }
            )
            print(
                f"[{idx}/{len(pdf_files)}] {pdf.name} -> refs={result['reference_count']} "
                f"status={result['status']}"
            )
            if result["error"]:
                failures.append({"paper_file": pdf.name, "error": result["error"]})
        except Exception as exc:  # noqa: BLE001
            failures.append({"paper_file": pdf.name, "error": str(exc)})
            print(f"[{idx}/{len(pdf_files)}] FAILED: {pdf.name}: {exc}")

    summary_path = refs_dir / "_summary.json"
    summary_payload = {
        "processed_papers": len(summary),
        "failed_papers": len(failures),
        "method": "grobid_processReferences",
        "grobid_url": args.grobid_url,
        "summary": summary,
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(f"Summary: {summary_path}")
    print(f"References JSON dir: {refs_dir}")


if __name__ == "__main__":
    main()
