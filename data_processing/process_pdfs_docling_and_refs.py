#!/usr/bin/env python3
"""Process PDFs into text with Docling and build reference JSON with arXiv abstracts.

Pipeline steps for each PDF:
1) Extract full text (Markdown/plain text) with Docling.
2) Find and split the references section into entries.
3) Query arXiv API for each reference title and attach abstract when found.
4) Save per-paper text and per-paper JSON outputs.
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
from docling.document_converter import DocumentConverter


ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_CACHE: dict[str, dict] = {}


def _get_accelerator_device(device_name: str, accelerator_device_cls: Any) -> Any:
    name = device_name.strip().upper()
    if name in {"CUDA", "GPU"} and hasattr(accelerator_device_cls, "CUDA"):
        return accelerator_device_cls.CUDA
    if name == "MPS" and hasattr(accelerator_device_cls, "MPS"):
        return accelerator_device_cls.MPS
    if name == "CPU" and hasattr(accelerator_device_cls, "CPU"):
        return accelerator_device_cls.CPU
    if hasattr(accelerator_device_cls, "AUTO"):
        return accelerator_device_cls.AUTO
    return None


def build_docling_converter(
    use_gpu: bool,
    device: str,
    ocr_backend: str,
    ocr_batch_size: int,
    layout_batch_size: int,
    page_batch_size: int,
) -> DocumentConverter:
    if not use_gpu:
        print("Docling accelerator: CPU (manual override)")
        return DocumentConverter()

    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
            ThreadedPdfPipelineOptions,
        )
        from docling.document_converter import PdfFormatOption
    except Exception as exc:  # noqa: BLE001
        print(f"Docling GPU setup unavailable ({exc}); falling back to default converter")
        return DocumentConverter()

    try:
        from docling.datamodel.base_models import InputFormat
    except Exception:  # noqa: BLE001
        InputFormat = None

    pipeline_options: Any
    try:
        pipeline_options = ThreadedPdfPipelineOptions()
    except Exception:  # noqa: BLE001
        pipeline_options = PdfPipelineOptions()

    selected_device = _get_accelerator_device(device, AcceleratorDevice)
    if selected_device is not None:
        pipeline_options.accelerator_options = AcceleratorOptions(device=selected_device)

    if ocr_backend:
        try:
            pipeline_options.ocr_options = RapidOcrOptions(backend=ocr_backend)
        except Exception:  # noqa: BLE001
            pass

    # Improve throughput by increasing batch sizes when supported by this docling version.
    for attr_name, attr_val in (
        ("ocr_batch_size", ocr_batch_size),
        ("layout_batch_size", layout_batch_size),
        ("page_batch_size", page_batch_size),
    ):
        if hasattr(pipeline_options, attr_name):
            setattr(pipeline_options, attr_name, attr_val)

    format_key = InputFormat.PDF if InputFormat is not None and hasattr(InputFormat, "PDF") else "pdf"
    try:
        converter = DocumentConverter(
            format_options={format_key: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        print(
            f"Docling accelerator: {device.upper()} | ocr_backend={ocr_backend} | "
            f"ocr_batch={ocr_batch_size} layout_batch={layout_batch_size} page_batch={page_batch_size}"
        )
        return converter
    except Exception as exc:  # noqa: BLE001
        print(f"Docling GPU converter init failed ({exc}); falling back to default converter")
        return DocumentConverter()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "paper"


def extract_text_with_docling(converter: DocumentConverter, pdf_path: Path) -> str:
    result = converter.convert(str(pdf_path))

    # Prefer markdown export if available; fallback to plain text if needed.
    if hasattr(result.document, "export_to_markdown"):
        text = result.document.export_to_markdown()
        if text and text.strip():
            return text

    if hasattr(result.document, "export_to_text"):
        text = result.document.export_to_text()
        if text and text.strip():
            return text

    return str(result.document)


def find_references_block(full_text: str) -> str:
    text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    heading = re.search(
        r"(?im)^\s{0,3}(?:#{1,6}\s*)?(references|bibliography|works cited)\s*:?[ \t]*$",
        text,
    )
    if not heading:
        return ""
    return text[heading.end() :].strip()


def split_reference_entries(ref_block: str) -> list[str]:
    if not ref_block:
        return []

    lines = [ln.strip() for ln in ref_block.splitlines()]
    lines = [ln for ln in lines if ln]

    start_pat = re.compile(
        r"^(\[\d+\]|\d+\.|[A-Z][A-Za-z'`\-]+,\s*[A-Z]|.+\(\d{4}[a-z]?\))"
    )

    entries: list[str] = []
    current: list[str] = []

    for line in lines:
        is_new = bool(start_pat.match(line))
        if is_new and current:
            entries.append(normalize_ws(" ".join(current)))
            current = [line]
        else:
            current.append(line)

    if current:
        entries.append(normalize_ws(" ".join(current)))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in entries:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def guess_title_from_reference(reference: str) -> str:
    ref = normalize_ws(reference)
    ref = re.sub(r"^(\[\d+\]|\d+\.)\s*", "", ref)

    quoted = re.search(r"[\"“](.+?)[\"”]", ref)
    if quoted:
        return normalize_ws(quoted.group(1))

    after_year = re.search(r"\(\d{4}[a-z]?\)\.?\s*(.+)", ref)
    if after_year:
        chunk = after_year.group(1)
        # Use the first sentence-like chunk as an approximate title.
        cand = chunk.split(". ")[0].strip(" .;")
        if len(cand) >= 10:
            return normalize_ws(cand)

    parts = [p.strip(" .;") for p in ref.split(". ") if p.strip()]
    if len(parts) >= 2:
        # Often [authors, title, venue, ...]
        if len(parts[1]) >= 10:
            return normalize_ws(parts[1])
        return normalize_ws(parts[0])

    return ref


def arxiv_query(title: str, timeout: int = 30) -> dict:
    key = normalize_ws(title).lower()
    if key in ARXIV_CACHE:
        return ARXIV_CACHE[key]

    def _fetch(search_query: str) -> dict:
        try:
            resp = requests.get(
                ARXIV_API_URL,
                params={"search_query": search_query, "start": 0, "max_results": 1},
                timeout=timeout,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception:  # noqa: BLE001
            return {"found": False}

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return {"found": False}

        arxiv_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

        return {
            "found": True,
            "arxiv_title": normalize_ws(arxiv_title),
            "abstract": normalize_ws(abstract),
            "arxiv_url": url,
            "published": published,
        }

    exact = _fetch(f'ti:"{title}"')
    if exact.get("found"):
        ARXIV_CACHE[key] = exact
        return exact

    broad = _fetch(f'all:"{title}"')
    result = broad if broad.get("found") else {"found": False}
    ARXIV_CACHE[key] = result
    return result


def process_pdf(
    pdf_path: Path,
    converter: DocumentConverter,
    text_dir: Path,
    refs_dir: Path,
    overwrite: bool,
    arxiv_sleep_seconds: float,
    text_only: bool,
) -> dict:
    stem = sanitize_name(pdf_path.stem)
    text_out = text_dir / f"{stem}.txt"
    json_out = refs_dir / f"{stem}.json"

    if text_out.exists() and not overwrite:
        full_text = text_out.read_text(encoding="utf-8")
    else:
        full_text = extract_text_with_docling(converter, pdf_path)
        text_out.write_text(full_text, encoding="utf-8")

    if text_only:
        return {
            "paper_file": pdf_path.name,
            "paper_stem": stem,
            "text_file": text_out.name,
            "reference_count": 0,
            "arxiv_match_count": 0,
            "references": [],
        }

    references_block = find_references_block(full_text)
    references = split_reference_entries(references_block)

    ref_items: list[dict] = []
    for idx, ref in enumerate(references, start=1):
        guessed_title = guess_title_from_reference(ref)
        arxiv_data = arxiv_query(guessed_title)
        ref_items.append(
            {
                "index": idx,
                "reference": ref,
                "guessed_title": guessed_title,
                "arxiv_found": bool(arxiv_data.get("found", False)),
                "arxiv": arxiv_data if arxiv_data.get("found") else None,
            }
        )
        if arxiv_sleep_seconds > 0:
            time.sleep(arxiv_sleep_seconds)

    payload = {
        "paper_file": pdf_path.name,
        "paper_stem": stem,
        "text_file": text_out.name,
        "reference_count": len(ref_items),
        "arxiv_match_count": sum(1 for r in ref_items if r["arxiv_found"]),
        "references": ref_items,
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default="data/raw_data/raw_pdfs", help="Directory with input PDFs")
    parser.add_argument("--text-dir", default="data/raw_data/pdf_parsing_docling", help="Directory for extracted text")
    parser.add_argument(
        "--refs-dir",
        default="data/raw_data/references_json",
        help="Directory for per-paper reference JSON files",
    )
    parser.add_argument("--max-papers", type=int, default=0, help="Limit number of PDFs (0 = all)")
    parser.add_argument(
        "--docling-cpu-only",
        action="store_true",
        help="Disable GPU setup and run docling with default CPU settings",
    )
    parser.add_argument(
        "--docling-device",
        default="cuda",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Docling accelerator device when GPU setup is enabled",
    )
    parser.add_argument(
        "--docling-ocr-backend",
        default="torch",
        help="RapidOCR backend. 'torch' is required for known OCR GPU support",
    )
    parser.add_argument(
        "--docling-ocr-batch-size",
        type=int,
        default=16,
        help="OCR batch size for threaded PDF pipeline",
    )
    parser.add_argument(
        "--docling-layout-batch-size",
        type=int,
        default=16,
        help="Layout model batch size for threaded PDF pipeline",
    )
    parser.add_argument(
        "--docling-page-batch-size",
        type=int,
        default=16,
        help="Page batch size for GPU batch inference when supported",
    )
    parser.add_argument(
        "--arxiv-sleep-seconds",
        type=float,
        default=0.15,
        help="Sleep between arXiv requests to reduce throttling",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing text and JSON outputs")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only extract text files with Docling; skip references and arXiv lookups",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    text_dir = Path(args.text_dir)
    refs_dir = Path(args.refs_dir)

    text_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    if args.max_papers > 0:
        pdf_files = pdf_files[: args.max_papers]

    converter = build_docling_converter(
        use_gpu=not args.docling_cpu_only,
        device=args.docling_device,
        ocr_backend=args.docling_ocr_backend,
        ocr_batch_size=args.docling_ocr_batch_size,
        layout_batch_size=args.docling_layout_batch_size,
        page_batch_size=args.docling_page_batch_size,
    )

    summary: list[dict] = []
    failures: list[dict] = []

    for i, pdf in enumerate(pdf_files, start=1):
        try:
            result = process_pdf(
                pdf_path=pdf,
                converter=converter,
                text_dir=text_dir,
                refs_dir=refs_dir,
                overwrite=args.overwrite,
                arxiv_sleep_seconds=args.arxiv_sleep_seconds,
                text_only=args.text_only,
            )
            summary.append(
                {
                    "paper_file": result["paper_file"],
                    "reference_count": result["reference_count"],
                    "arxiv_match_count": result["arxiv_match_count"],
                }
            )
            if args.text_only:
                print(f"[{i}/{len(pdf_files)}] {pdf.name} -> text extracted")
            else:
                print(
                    f"[{i}/{len(pdf_files)}] {pdf.name} -> refs={result['reference_count']}, arxiv_matches={result['arxiv_match_count']}"
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"paper_file": pdf.name, "error": str(exc)})
            print(f"[{i}/{len(pdf_files)}] FAILED: {pdf.name}: {exc}")

    summary_path = (text_dir / "_text_summary.json") if args.text_only else (refs_dir / "_summary.json")
    summary_payload = {
        "processed_papers": len(summary),
        "failed_papers": len(failures),
        "text_only": bool(args.text_only),
        "summary": summary,
        "failures": failures,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(f"Summary: {summary_path}")
    print(f"Text output dir: {text_dir}")
    print(f"References JSON dir: {refs_dir}")


if __name__ == "__main__":
    main()
