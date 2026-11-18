#!/usr/bin/env python3
"""
Step 7 — AI introduction generation.

For each selected paper this script assembles two inputs and asks an LLM to
write the paper's Introduction:

  * the paper's content  — the body text in
    `data/raw_data/processed_texts/{id}_*.txt` (abstract + literature review +
    methodology + results + discussion; the original introduction is NOT in
    this file, so there is no leakage of the text we later compare against).
  * the cited sources    — the abstracts of the works the original authors
    cited in their introduction, read from
    `data/raw_data/references_json_with_citations/{id}_*.json` (every reference
    whose `introduction_citation_contexts` is non-empty).

Two modes:
  * --mode basic        write the introduction directly.
  * --mode cars_guided  also give the model Swales' CARS move/step definitions
                        and have it first PLAN the rhetorical moves, then WRITE.

The model is selectable across three back-ends and reuses the cached multi-provider
`LLMClient`:
  * a closed GPT model (`--model gpt-5.5`, provider inferred),
  * a closed Claude model (`--model claude-opus-4-8`),
  * an open-weight model behind an OpenAI-compatible endpoint
    (`--provider openai_compatible --model <hf-name>`), served either by a local
    vLLM server or a hosted provider — the privacy-friendly condition (no
    closed model sees the inputs). Set OPENAI_COMPAT_BASE_URL
    (and OPENAI_COMPAT_API_KEY for hosted) or pass --base-url.

Cost: for the large GPT/Claude runs use `--batch` — both providers run batches
asynchronously at 50% of the synchronous price. The run submits one batch and
exits; retrieve it later with `--fetch-batch <run_dir>`.

Usage:
    # 5 papers, basic mode, default model (gpt-5.5)
    python intro_generation/generate_intros.py --n 5 --mode basic

    # CARS-guided, Claude, specific papers
    python intro_generation/generate_intros.py --mode cars_guided \
        --model claude-opus-4-8 --paper-ids 1 4 8

    # Open-weight model via a local vLLM / hosted OpenAI-compatible endpoint
    OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1 \
    python intro_generation/generate_intros.py --n 10 --mode basic \
        --provider openai_compatible --model Qwen/Qwen3-32B

    # Submit all 76 papers as a 50%-off batch (GPT or Claude), then fetch later
    python intro_generation/generate_intros.py --paper-ids $(seq 1 76) \
        --model gpt-5.5 --mode basic --batch
    python intro_generation/generate_intros.py \
        --fetch-batch intro_generation/results/2026-..._gpt-5.5_basic

Output: a timestamped run folder under intro_generation/results/ containing
`generated_intros.json` (one record per paper), a `texts/` folder with the raw
introduction of each paper for easy reading, and `run_meta.json`. A `--batch`
submission instead writes `batch_state.json` (batch id + per-paper map) for
`--fetch-batch` to complete.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
from dotenv import load_dotenv

# Allow running as a script from the repo root and reuse the shared LLM client.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intro_analysis.llm_client import LLMClient, infer_provider, parse_json
from intro_generation.prompts import (
    GENERATION_MODES,
    USER_PROMPT_TEMPLATE,
    get_system_prompt,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

PROCESSED_DIR = REPO_ROOT / "data" / "raw_data" / "processed_texts"
REFS_DIR = REPO_ROOT / "data" / "raw_data" / "references_json_with_citations"
EXCEL_PATH = REPO_ROOT / "data" / "raw_data" / "80-papers_corpus_nativeness_quartiles.xlsx"

_ID_RE = re.compile(r"^(\d{3})")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _index_by_id(directory: Path, suffix: str) -> dict[str, Path]:
    """Map a 3-digit paper id (parsed from the filename prefix) to its file."""
    out: dict[str, Path] = {}
    for p in sorted(directory.glob(f"*{suffix}")):
        m = _ID_RE.match(p.name)
        if m:
            out.setdefault(m.group(1), p)
    return out


def load_gold_intro_lengths() -> dict[str, int]:
    """Map 3-digit paper id → word count of the original (gold) introduction.

    Read from the corpus Excel `introduction` column — the same human intro the
    AI output is later compared against — so each AI intro can be targeted to its
    own paper's length (lengths span 156–1815 words, median ~592)."""
    if not EXCEL_PATH.exists():
        return {}
    wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True, data_only=True)
    ws = wb.active
    headers = [c.value for c in next(ws.iter_rows(max_row=1))]
    id_col = headers.index("paper_id")
    intro_col = headers.index("introduction")
    out: dict[str, int] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid, intro = row[id_col], row[intro_col]
        if pid is not None and intro:
            out[f"{int(pid):03d}"] = len(str(intro).split())
    wb.close()
    return out


def length_instruction(target_words: int | None, tolerance: float) -> str:
    """Soft per-paper length guidance for the user prompt."""
    if not target_words:
        return ("Aim for a typical research-article introduction length for this "
                "field (about 400–800 words).")
    lo = round(target_words * (1 - tolerance))
    hi = round(target_words * (1 + tolerance))
    return (f"Aim for approximately {target_words} words (ideally between {lo} and "
            f"{hi} words) — this matches the length of the original article's "
            f"introduction. Do not substantially exceed {hi} words.")


def clean_abstract(text: str | None) -> str:
    """Strip HTML entities/tags and collapse whitespace in an abstract field."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _format_authors(ref: dict) -> str:
    authors = ref.get("authors") or ref.get("author_surnames") or []
    if isinstance(authors, str):
        authors = [authors]
    authors = [a for a in authors if a]
    if not authors:
        return "Unknown author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} & {authors[1]}"
    return f"{authors[0]} et al."


def format_references(cited_refs: list[dict]) -> str:
    """Render every work cited in the original intro for the prompt.

    All cited works are included regardless of whether an abstract was found:
    each is shown with its authors, year, title and venue, and the abstract is
    appended only when available. Works without an abstract are still listed
    (title + authors) so the model can infer what they are from the title — per
    the agreed design, we surface the full cited set, not just abstract-bearing
    works."""
    if not cited_refs:
        return "(No external works were cited in the original introduction.)"
    blocks = []
    for i, ref in enumerate(cited_refs, 1):
        author = _format_authors(ref)
        year = ref.get("year") or "n.d."
        title = (ref.get("title") or "").strip() or "[title unavailable]"
        venue = (ref.get("venue") or "").strip()
        header = f"[{i}] {author} ({year}). {title}."
        if venue:
            header += f" {venue}."
        abstract = clean_abstract(ref.get("abstract"))
        body = f"    Abstract: {abstract}" if abstract else "    (No abstract available — title only.)"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def load_paper(paper_id: str, processed: dict[str, Path],
               refs: dict[str, Path], max_body_chars: int,
               target_words: int | None = None) -> dict | None:
    """Assemble one paper's generation inputs, or None if assets are missing."""
    if paper_id not in processed or paper_id not in refs:
        return None
    body = processed[paper_id].read_text().strip()
    truncated = False
    if max_body_chars and len(body) > max_body_chars:
        body = body[:max_body_chars]
        truncated = True

    refs_data = json.loads(refs[paper_id].read_text())
    cited = [r for r in refs_data.get("references", [])
             if r.get("introduction_citation_contexts")]

    return {
        "paper_id": paper_id,
        "title": refs_data.get("paper_title") or processed[paper_id].stem,
        "nativeness": refs_data.get("paper_nativeness"),
        "quartile": refs_data.get("paper_quartile"),
        "processed_file": processed[paper_id].name,
        "refs_file": refs[paper_id].name,
        "body": body,
        "body_truncated": truncated,
        "cited_refs": cited,
        "target_words": target_words,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompts(paper: dict, mode: str, length_tolerance: float) -> tuple[str, str]:
    """Return the (system, user) prompt pair for one paper — shared by the sync
    and batch paths so both send identical inputs."""
    system_prompt = get_system_prompt(mode)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=paper["title"],
        body=paper["body"],
        references=format_references(paper["cited_refs"]),
        length_instruction=length_instruction(paper["target_words"], length_tolerance),
    )
    return system_prompt, user_prompt


def paper_meta(paper: dict) -> dict:
    """The per-paper fields needed to assemble a result record without the body
    text — small enough to persist in a batch's state file."""
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "nativeness": paper["nativeness"],
        "quartile": paper["quartile"],
        "n_cited_refs": len(paper["cited_refs"]),
        "body_truncated": paper["body_truncated"],
        "target_words": paper["target_words"],
    }


def assemble_record(meta: dict, mode: str, model: str, provider: str,
                    introduction: str, usage: dict | None,
                    request_date: str | None = None) -> dict:
    introduction = (introduction or "").strip()
    tgt = meta.get("target_words")
    word_count = len(introduction.split())
    return {
        "paper_id": meta["paper_id"],
        "title": meta["title"],
        "nativeness": meta["nativeness"],
        "quartile": meta["quartile"],
        "mode": mode,
        "model": model,
        "provider": provider,
        # Date the request was made to the model (submission date for batches) —
        # provenance for reproducibility, since model behaviour drifts over time.
        "request_date": request_date,
        "n_cited_refs": meta["n_cited_refs"],
        "body_truncated": meta["body_truncated"],
        "target_words": tgt,
        "introduction": introduction,
        "word_count": word_count,
        "length_ratio": round(word_count / tgt, 2) if tgt else None,
        "usage": usage,
    }


def generate_one(client: LLMClient, paper: dict, mode: str,
                 length_tolerance: float, request_date: str | None = None) -> dict:
    system_prompt, user_prompt = build_prompts(paper, mode, length_tolerance)
    tgt = paper["target_words"]
    print(f"  → {client.model} [{client.provider}/{client.reasoning_effort}] "
          f"paper {paper['paper_id']} — {len(paper['cited_refs'])} cited refs, "
          f"{len(paper['body'])} body chars, target≈{tgt or 'n/a'} words…", flush=True)
    raw = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    # cars_guided plans the moves internally; only the prose is returned.
    # Robust extraction: tolerate malformed JSON (unescaped quotes) so a stray
    # quote in the prose doesn't drop the whole paper.
    introduction = _parse_introduction(raw)
    return assemble_record(paper_meta(paper), mode, client.model, client.provider,
                           introduction, client.last_usage, request_date)


def write_results(results: list[dict], run_dir: Path) -> None:
    """Write generated_intros.json + per-paper texts/ for a finished run."""
    texts_dir = run_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        (texts_dir / f"{res['paper_id']}.txt").write_text(res["introduction"] + "\n")
    with open(run_dir / "generated_intros.json", "w") as f:
        json.dump(results, f, indent=2)


def _parse_introduction(text: str) -> str:
    """Extract the introduction from a model reply, tolerating malformed JSON.

    Open models (and occasionally the hosted ones) emit `{"introduction": "..."}`
    with UNescaped double-quotes or newlines inside the value — common on
    quote-heavy academic prose — which breaks strict JSON parsing (e.g.
    "Expecting ',' delimiter"). We fall back to extracting the value by regex so
    a stray inner quote no longer loses the whole paper."""
    if not text:
        return ""
    # 1. Strict JSON first (parse_json also strips ```json fences).
    try:
        obj = parse_json(text)
        if isinstance(obj, dict) and obj.get("introduction"):
            return str(obj["introduction"]).strip()
    except Exception:  # noqa: BLE001 — fall through to lenient extraction
        pass
    # 2. Lenient: grab everything between "introduction":" and the final quote,
    #    keeping inner unescaped quotes. Greedy so we stop at the LAST quote.
    m = re.search(r'"introduction"\s*:\s*"(.*)"\s*}?\s*$', text.strip(), re.DOTALL)
    if m:
        val = m.group(1)
        for a, b in (('\\"', '"'), ('\\n', '\n'), ('\\t', '\t'), ('\\\\', '\\')):
            val = val.replace(a, b)
        return val.strip()
    # 3. Last resort: strip any leading JSON scaffolding and return the prose.
    cleaned = re.sub(r'^\s*\{?\s*"introduction"\s*:\s*"?', '', text.strip())
    return re.sub(r'"?\s*\}?\s*$', '', cleaned).strip()


def submit_batch_run(client: LLMClient, papers: list[dict], mode: str,
                     length_tolerance: float, run_dir: Path, base_meta: dict) -> None:
    """Submit all papers as one 50%-off async batch and persist the state."""
    requests, custom_id_map = [], {}
    for paper in papers:
        cid = f"paper-{paper['paper_id']}"
        system_prompt, user_prompt = build_prompts(paper, mode, length_tolerance)
        requests.append({"custom_id": cid, "system": system_prompt, "user": user_prompt})
        custom_id_map[cid] = paper_meta(paper)

    print(f"Submitting {len(requests)} requests as a {client.provider} batch "
          f"(≈50% cheaper, async)…", flush=True)
    batch_id = client.submit_batch(requests)

    run_dir.mkdir(parents=True, exist_ok=True)
    state = {**base_meta, "batch_id": batch_id, "n_submitted": len(requests),
             "custom_id_map": custom_id_map}
    with open(run_dir / "batch_state.json", "w") as f:
        json.dump(state, f, indent=2)
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({**base_meta, "batch_id": batch_id,
                   "system_prompt_sample": get_system_prompt(mode)}, f, indent=2)

    print(f"\nSubmitted batch {batch_id} ({client.provider}).")
    print(f"State saved → {run_dir / 'batch_state.json'}")
    print("Batches usually finish within an hour (max 24h). Retrieve with:\n"
          f"  python intro_generation/generate_intros.py --fetch-batch {run_dir}")


def fetch_batch_run(run_dir: Path) -> None:
    """Retrieve a previously submitted batch and write its results."""
    state_path = run_dir / "batch_state.json"
    if not state_path.exists():
        sys.exit(f"No batch_state.json in {run_dir} — was this a --batch run?")
    state = json.loads(state_path.read_text())
    provider, model, batch_id = state["provider"], state["model"], state["batch_id"]

    client = LLMClient(model=model, provider=provider,
                       reasoning_effort=state.get("reasoning_effort", "low"),
                       use_cache=state.get("prompt_caching", True),
                       key_suffix=state.get("key_suffix"))
    status = client.poll_batch(batch_id)
    print(f"Batch {batch_id} [{provider}] status={status['status']}  {status}")
    if not status["done"]:
        print("Not finished yet — re-run --fetch-batch later.")
        return
    if status["status"] not in ("completed", "ended"):
        print(f"⚠ batch ended abnormally ({status['status']}).", file=sys.stderr)

    fetched = client.fetch_batch(batch_id)
    results, errors = [], []
    for cid, meta in state["custom_id_map"].items():
        r = fetched.get(cid)
        if not r or "text" not in r:
            errors.append((meta["paper_id"], (r or {}).get("error", "missing")))
            continue
        results.append(assemble_record(
            meta, state["mode"], model, provider,
            _parse_introduction(r["text"]), r.get("usage"),
            state.get("request_date")))

    results.sort(key=lambda x: x["paper_id"])
    write_results(results, run_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(state)
    meta.pop("custom_id_map", None)
    meta.update({"n_generated": len(results),
                 "paper_ids": [r["paper_id"] for r in results]})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFetched {len(results)} introduction(s) → {run_dir / 'generated_intros.json'}")
    if errors:
        print(f"⚠ {len(errors)} request(s) failed: {errors}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate AI research-article introductions (Step 7).")
    parser.add_argument("--mode", default="basic", choices=GENERATION_MODES,
                        help="basic: write directly. cars_guided: plan CARS moves, then write.")
    parser.add_argument("--model", default="gpt-5.5",
                        help="Model name. Provider inferred: gpt-*/o* → OpenAI, "
                             "claude-* → Anthropic. For an open-weight model pass "
                             "--provider openai_compatible and the served model name.")
    parser.add_argument("--provider", default="auto",
                        choices=["auto", "openai", "anthropic", "openai_compatible", "google"],
                        help="LLM back-end. 'auto' infers it from --model. Use "
                             "openai_compatible for an open-weight model on a local "
                             "vLLM or hosted OpenAI-compatible endpoint. (google/Gemini "
                             "is retained but no longer used for generation.)")
    parser.add_argument("--base-url", default=None,
                        help="Base URL for the openai_compatible provider (overrides "
                             "OPENAI_COMPAT_BASE_URL). E.g. http://localhost:8000/v1.")
    parser.add_argument("--key-suffix", default=None,
                        help="Select a key set: prefer {PROVIDER}_API_KEY_<SUFFIX> "
                             "(e.g. --key-suffix LAB → ANTHROPIC_API_KEY_LAB), "
                             "falling back to the plain name. Defaults to $LLM_KEY_SUFFIX.")
    parser.add_argument("--batch", action="store_true",
                        help="Submit the run as a 50%%-off async batch (OpenAI/Anthropic "
                             "only) and exit. Retrieve later with --fetch-batch.")
    parser.add_argument("--fetch-batch", default=None, metavar="RUN_DIR",
                        help="Retrieve a previously submitted --batch run from RUN_DIR "
                             "(reads its batch_state.json) and write the results.")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of intros to generate (ignored if --paper-ids/--all given).")
    parser.add_argument("--all", action="store_true",
                        help="Generate for EVERY available paper (all with both assets). "
                             "Use this instead of --paper-ids $(seq …): the corpus ids are "
                             "001–080 with gaps at 019/024/032/033, so a numeric range is wrong.")
    parser.add_argument("--paper-ids", nargs="+", type=int,
                        help="Specific paper ids to generate for (overrides --n/--seed).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed for the --n papers.")
    parser.add_argument("--reasoning-effort", default="low",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort (OpenAI/Gemini); Anthropic: extended thinking. Default low.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable prompt caching of the static system prompt.")
    parser.add_argument("--max-body-chars", type=int, default=0,
                        help="Truncate each paper body to this many chars. Default 0 = NO limit "
                             "(feed the full paper). Both models have 256k-token context, so full "
                             "bodies (longest ~29k tokens) fit easily. Set a positive value only to "
                             "cap prompt size/cost.")
    parser.add_argument("--length-tolerance", type=float, default=0.20,
                        help="Allowed +/- fraction around each paper's original intro length "
                             "(target band shown to the model). Default 0.20 (±20%%).")
    parser.add_argument("--results-dir", default="intro_generation/results",
                        help="Base dir; each run creates a timestamped subfolder here.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of papers to generate in parallel (synchronous runs). "
                             ">1 fires concurrent requests at the endpoint — vLLM batches them "
                             "via continuous batching, cutting a full run from hours to minutes. "
                             "Each worker uses its own client. Default 1 (sequential).")
    args = parser.parse_args()

    # --- Retrieve a previously submitted batch (no model selection needed) ---
    if args.fetch_batch:
        fetch_batch_run(Path(args.fetch_batch))
        return

    provider = infer_provider(args.model) if args.provider == "auto" else args.provider
    client = LLMClient(model=args.model, provider=provider,
                       reasoning_effort=args.reasoning_effort,
                       use_cache=not args.no_cache, base_url=args.base_url,
                       key_suffix=args.key_suffix)

    if args.batch and not client.supports_batch:
        sys.exit(f"--batch is only available for openai/anthropic, not {provider!r}. "
                 f"Run the open-weight model synchronously — vLLM batches internally.")

    processed = _index_by_id(PROCESSED_DIR, ".txt")
    refs = _index_by_id(REFS_DIR, ".json")
    gold_lengths = load_gold_intro_lengths()
    available = sorted(set(processed) & set(refs))
    if not available:
        sys.exit("No papers have both a processed_texts file and a references JSON.")

    if args.all:
        selected_ids = available
    elif args.paper_ids:
        wanted = [f"{pid:03d}" for pid in args.paper_ids]
        selected_ids = [pid for pid in wanted if pid in available]
        missing = [pid for pid in wanted if pid not in available]
        if missing:
            print(f"⚠ skipping ids with missing assets: {missing}", file=sys.stderr)
        if not selected_ids:
            sys.exit(f"None of {args.paper_ids} have both assets.")
    else:
        random.seed(args.seed)
        selected_ids = sorted(random.sample(available, min(args.n, len(available))))

    print(f"Provider={provider}  model={args.model}  mode={args.mode}  "
          f"cache={not args.no_cache}  effort={args.reasoning_effort}  "
          f"batch={args.batch}")
    print(f"Length targeting: per-paper original intro ±{args.length_tolerance:.0%} "
          f"({len(gold_lengths)} gold lengths loaded)")
    print(f"{'Submitting' if args.batch else 'Generating'} {len(selected_ids)} "
          f"intro(s): {selected_ids}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = f"{args.model}_{args.mode}".replace("/", "-")
    run_dir = Path(args.results_dir) / f"{timestamp}_{run_tag}"
    print(f"Run folder: {run_dir}")

    base_meta = {
        "timestamp": timestamp,
        "request_date": timestamp[:10],  # YYYY-MM-DD the request was made/submitted
        "mode": args.mode,
        "model": args.model,
        "provider": provider,
        "prompt_caching": not args.no_cache,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "key_suffix": client.key_suffix or None,
        "key_source": client.key_source,  # actual env var the key was read from
        "max_body_chars": args.max_body_chars,
        "length_tolerance": args.length_tolerance,
        "run_tag": run_tag,
    }

    papers = []
    for pid in selected_ids:
        paper = load_paper(pid, processed, refs, args.max_body_chars,
                           target_words=gold_lengths.get(pid))
        if paper is None:
            print(f"  ✖ paper {pid}: assets missing, skipped", file=sys.stderr)
            continue
        papers.append(paper)

    # --- Async batch submission (50% off) -----------------------------------
    if args.batch:
        submit_batch_run(client, papers, args.mode, args.length_tolerance,
                         run_dir, base_meta)
        return

    # --- Synchronous generation ---------------------------------------------
    req_date = base_meta["request_date"]
    results = []
    if args.concurrency > 1:
        # Fire concurrent requests; vLLM (and hosted APIs) batch them server-side.
        # Each worker thread gets its OWN LLMClient so the per-call `last_usage`
        # attribute never races between threads.
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tl = threading.local()

        def _worker(paper):
            if not hasattr(tl, "client"):
                tl.client = LLMClient(model=args.model, provider=provider,
                                      reasoning_effort=args.reasoning_effort,
                                      use_cache=not args.no_cache, base_url=args.base_url,
                                      key_suffix=args.key_suffix)
            return generate_one(tl.client, paper, args.mode, args.length_tolerance, req_date)

        print(f"Concurrency: {args.concurrency} parallel request(s)")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(_worker, p): p for p in papers}
            for fut in as_completed(futures):
                pid = futures[fut]["paper_id"]
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001 — keep going on per-paper failure
                    print(f"  ✖ paper {pid} failed: {e}", file=sys.stderr)
    else:
        for paper in papers:
            try:
                results.append(generate_one(client, paper, args.mode, args.length_tolerance,
                                            req_date))
            except Exception as e:  # noqa: BLE001 — keep going on per-paper failure
                print(f"  ✖ paper {paper['paper_id']} failed: {e}", file=sys.stderr)

    results.sort(key=lambda r: r["paper_id"])  # completion order → paper order
    write_results(results, run_dir)
    meta = {**base_meta, "n_generated": len(results),
            "paper_ids": [r["paper_id"] for r in results],
            "system_prompt_sample": get_system_prompt(args.mode)}
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated {len(results)} introduction(s) → {run_dir / 'generated_intros.json'}")


if __name__ == "__main__":
    main()
