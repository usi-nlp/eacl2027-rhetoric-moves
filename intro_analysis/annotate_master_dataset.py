#!/usr/bin/env python3
"""
Step 8 — CARS auto-annotation of EVERY introduction in the master dataset
(original + all generated) with the frontrunner classifier, via the OpenAI
Batch API (50% off, async), merged back into data/master_dataset.json.

Classifier configuration is LOCKED to the frontrunner (see PROMPT_VERSIONS in
prompts.py): --mode classification --prompt-source manual --few-shot
--prompt-version v8, gpt-5.5, reasoning effort low, NO citation
post-processing (v8 complies with the citation rule on its own; the
post-processor would undo the v8 cited-importance⇒CC override).

The model annotates a whitespace-collapsed copy of each intro (same as
run_experiment.py); at merge time spans are remapped onto the intro's
canonical `text` via non-whitespace character alignment so all annotations
of an intro share one offset space (same mechanism as build_master_dataset).

Intros whose text is a raw JSON artifact (the 9 known-broken Qwen3.5-27B
cars_guided outputs, see regen_qwen_cars_guided.sbatch) are skipped until
regenerated.

Usage (from repo root):
    # 1. Submit the batch
    python intro_analysis/annotate_master_dataset.py --submit --key-suffix LAB
    # 2. Check on it
    python intro_analysis/annotate_master_dataset.py --status <run_dir>
    # 3. Retrieve + parse when finished (writes master_annotations.json)
    python intro_analysis/annotate_master_dataset.py --fetch <run_dir>
    # 4. Merge into data/master_dataset.json (in place)
    python intro_analysis/annotate_master_dataset.py --merge <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from intro_analysis.prompts import get_system_prompt, USER_PROMPT_TEMPLATE
from intro_analysis.llm_client import LLMClient, parse_json
from intro_analysis.postprocess import map_numeric_labels
from intro_analysis.run_experiment import load_tasks
from data_processing.build_master_dataset import (
    collapse, build_nonws_rank, _nonws_positions, remap_span, make_segment)

load_dotenv(REPO_ROOT / ".env")

DEFAULT_DATASET = REPO_ROOT / "data/master_dataset.json"
DEFAULT_TASKS = REPO_ROOT / "data/annotation/human_intros/tasks.json"
DEFAULT_RESULTS = REPO_ROOT / "intro_analysis/results"

# Frontrunner classifier config — locked (prompt version v8, 2026-06-10).
CONFIG = {
    "model": "gpt-5.5",
    "provider": "openai",
    "reasoning_effort": "low",
    "mode": "classification",
    "few_shot": True,
    "prompt_source": "manual",
    "prompt_version": "v8",
    "citation_postproc": False,
    "run_tag": "classification-fewshot-manual-v8",
}

ANNOTATOR_NAME = "chatgpt"


def is_broken_text(text: str) -> bool:
    """True for the known generation-failure shape: the intro slot holds the
    model's raw JSON plan instead of prose (9 Qwen3.5-27B cars_guided outputs)."""
    t = text.lstrip()
    return t.startswith("{") or '"internal_plan"' in t


# ---------------------------------------------------------------------------
# Span resolution (model output -> offsets in the collapsed text)
# ---------------------------------------------------------------------------

_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", " ": " ",
})


def _soft_find(haystack: str, needle: str, start: int) -> tuple[int, int] | None:
    """Punctuation-tolerant search: normalize curly quotes/dashes/nbsp on both
    sides, find in normalized space (1:1 char mapping, offsets carry over)."""
    h = haystack.translate(_PUNCT_MAP)
    n = needle.translate(_PUNCT_MAP)
    idx = h.find(n, start)
    if idx == -1:
        idx = h.find(n)
    if idx == -1:
        return None
    return idx, idx + len(n)


def _anchor_find(text: str, excerpt: str, start: int,
                 anchor: int = 50) -> tuple[int, int] | None:
    """Last-resort recovery for excerpts the model rewrote mid-segment (e.g.
    dropped a period): locate the excerpt's head and tail independently and
    take the enclosing span, provided its length stays close to the excerpt's
    (guards against matching two unrelated points in the text)."""
    if len(excerpt) < 2 * anchor:
        return None
    head = _soft_find(text, excerpt[:anchor], start)
    if head is None:
        return None
    tail = _soft_find(text, excerpt[-anchor:], head[0])
    if tail is None:
        return None
    lo, hi = head[0], tail[1]
    if not (0.8 * len(excerpt) <= hi - lo <= 1.25 * len(excerpt)):
        return None
    return lo, hi


def resolve_spans(text: str, segments: list[dict]) -> list[dict]:
    """Find character offsets for each returned segment by searching for its
    (near-)verbatim text, forward from the previous match to respect order.
    Exact match first, then stripped, then punctuation-normalized, then
    head/tail anchors (flagged match_soft)."""
    resolved = []
    search_from = 0
    for seg in segments:
        excerpt = (seg.get("text") or "").strip()
        if not excerpt:
            resolved.append({**seg, "start": None, "end": None, "match_ok": False})
            continue
        soft = False
        idx = text.find(excerpt, search_from)
        if idx == -1:
            idx = text.find(excerpt)
        if idx != -1:
            lo, hi = idx, idx + len(excerpt)
        else:
            span = (_soft_find(text, excerpt, search_from)
                    or _anchor_find(text, excerpt, search_from))
            if span is None:
                resolved.append({**seg, "start": None, "end": None, "match_ok": False})
                continue
            lo, hi = span
            soft = True
        out = {**seg, "text": text[lo:hi], "start": lo, "end": hi, "match_ok": True}
        if soft:
            out["match_soft"] = True
        resolved.append(out)
        search_from = hi
    return resolved


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def iter_intros(dataset: dict):
    for paper in dataset["papers"]:
        for intro in paper["intros"]:
            yield paper, intro


def build_requests(dataset: dict, tasks: list[dict],
                   include_broken: bool = False,
                   only_missing: bool = False) -> tuple[list[dict], dict, list[str]]:
    """Return (requests, {intro_id: collapsed_text}, skipped_intro_ids).

    One system prompt per paper: the paper's own task id is excluded from the
    few-shot demo pool (papers 16/17/28/34/40/66 are reserved demos — each gets
    the other member of its granularity group), and ALL of a paper's intros
    (original + generated) share that prompt.

    only_missing: annotate only intros without an existing `chatgpt`
    annotation (incremental run, e.g. after regenerating broken intros).
    """
    requests, texts, skipped = [], {}, []
    prompt_cache: dict[int, str] = {}
    for paper, intro in iter_intros(dataset):
        pid = paper["paper_id"]
        if only_missing and ANNOTATOR_NAME in intro["annotations"]:
            continue
        if not include_broken and is_broken_text(intro["text"]):
            skipped.append(intro["intro_id"])
            continue
        if pid not in prompt_cache:
            prompt_cache[pid] = get_system_prompt(
                mode=CONFIG["mode"], few_shot=CONFIG["few_shot"],
                source=CONFIG["prompt_source"], all_tasks=tasks,
                version=CONFIG["prompt_version"], exclude_task_id=pid)
        clean = collapse(intro["text"])
        texts[intro["intro_id"]] = clean
        requests.append({
            "custom_id": intro["intro_id"],
            "system": prompt_cache[pid],
            "user": USER_PROMPT_TEMPLATE.format(text=clean),
        })
    return requests, texts, skipped


def submit(args: argparse.Namespace) -> None:
    dataset = json.load(open(args.dataset))
    tasks = load_tasks(str(args.tasks))
    requests, texts, skipped = build_requests(dataset, tasks,
                                              include_broken=args.include_broken,
                                              only_missing=args.only_missing)
    if not requests:
        sys.exit("Nothing to annotate.")
    print(f"{len(requests)} intros to annotate "
          f"({len(skipped)} skipped as broken: {skipped})")

    if args.dry_run:
        n_chars = sum(len(r["system"]) + len(r["user"]) for r in requests)
        print(f"[dry run] total prompt chars: {n_chars:,} "
              f"(≈{n_chars // 4:,} tokens); no batch submitted.")
        return

    client = LLMClient(model=CONFIG["model"], provider=CONFIG["provider"],
                       reasoning_effort=CONFIG["reasoning_effort"],
                       max_output_tokens=args.max_output_tokens,
                       key_suffix=args.key_suffix)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.results_dir / f"{timestamp}_master-dataset-annotation-v8"
    run_dir.mkdir(parents=True, exist_ok=True)

    shards = ([requests] if not args.shard_size else
              [requests[i:i + args.shard_size]
               for i in range(0, len(requests), args.shard_size)])
    batches = []
    for i, shard in enumerate(shards, 1):
        batch_id = client.submit_batch(shard)
        batches.append({"batch_id": batch_id,
                        "custom_ids": [r["custom_id"] for r in shard]})
        print(f"Submitted shard {i}/{len(shards)}: {batch_id} "
              f"({len(shard)} requests)")

    state = {
        **CONFIG,
        "dataset": str(args.dataset),
        "timestamp": timestamp,
        "key_source": client.key_source,
        "max_output_tokens": args.max_output_tokens,
        "n_submitted": len(requests),
        "skipped_broken": skipped,
        "batches": batches,
        "texts": texts,   # collapsed text per intro — the offset space of the run
    }
    with open(run_dir / "batch_state.json", "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({**CONFIG, "timestamp": timestamp,
                   "n_submitted": len(requests), "skipped_broken": skipped,
                   "batches": [b["batch_id"] for b in batches],
                   "prompt_sample": {"custom_id": requests[0]["custom_id"],
                                     "instructions": requests[0]["system"],
                                     "input": requests[0]["user"]}},
                  f, ensure_ascii=False, indent=1)
    print(f"\nState saved → {run_dir / 'batch_state.json'}")
    print(f"Fetch when finished with:\n"
          f"  python intro_analysis/annotate_master_dataset.py --fetch {run_dir}")


# ---------------------------------------------------------------------------
# Status / fetch
# ---------------------------------------------------------------------------

def _load_state(run_dir: Path) -> dict:
    path = run_dir / "batch_state.json"
    if not path.exists():
        sys.exit(f"No batch_state.json in {run_dir}")
    return json.load(open(path))


def _client_from_state(state: dict, key_suffix: str | None) -> LLMClient:
    return LLMClient(model=state["model"], provider=state["provider"],
                     reasoning_effort=state["reasoning_effort"],
                     max_output_tokens=state.get("max_output_tokens", 16384),
                     key_suffix=key_suffix or state.get("key_source", "")
                     .replace("OPENAI_API_KEY", "").strip("_") or None)


def status(run_dir: Path, key_suffix: str | None) -> bool:
    """Print status of every shard; return True when all are done."""
    state = _load_state(run_dir)
    client = _client_from_state(state, key_suffix)
    all_done = True
    for b in state["batches"]:
        s = client.poll_batch(b["batch_id"])
        print(f"{b['batch_id']}: {s}")
        all_done &= s["done"]
    return all_done


def fetch(run_dir: Path, key_suffix: str | None) -> None:
    state = _load_state(run_dir)
    client = _client_from_state(state, key_suffix)

    results: dict = {}
    for b in state["batches"]:
        s = client.poll_batch(b["batch_id"])
        print(f"{b['batch_id']}: {s}")
        if not s["done"]:
            sys.exit("Not finished yet — re-run --fetch later.")
        results.update(client.fetch_batch(b["batch_id"]))

    with open(run_dir / "raw_responses.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    annotations, failures = {}, {}
    n_segments = n_unlocated = 0
    usage_in = usage_cached = usage_out = 0
    for intro_id, res in results.items():
        if "error" in res:
            failures[intro_id] = {"stage": "api", "error": str(res["error"])}
            continue
        u = res.get("usage") or {}
        usage_in += u.get("prompt_tokens", u.get("input", 0)) or 0
        det = u.get("prompt_tokens_details") or {}
        usage_cached += (det.get("cached_tokens", 0) if isinstance(det, dict)
                         else 0) or u.get("cached_input", 0) or 0
        usage_out += u.get("completion_tokens", u.get("output", 0)) or 0
        try:
            raw = parse_json(res["text"]).get("segments", [])
        except Exception as e:
            failures[intro_id] = {"stage": "parse", "error": str(e)}
            continue
        clean = state["texts"][intro_id]
        segments = resolve_spans(clean, raw)
        map_numeric_labels(segments)
        bad = [s for s in segments if not s.get("match_ok")]
        if bad:
            n_unlocated += len(bad)
            failures[intro_id] = {
                "stage": "locate", "n_unlocated": len(bad),
                "excerpts": [(s.get("text") or "")[:80] for s in bad]}
        n_segments += len(segments) - len(bad)
        annotations[intro_id] = {"segments": segments}

    out = {
        "meta": {**{k: state[k] for k in CONFIG}, "timestamp": state["timestamp"],
                 "fetched": datetime.now().isoformat(timespec="seconds"),
                 "n_intros": len(annotations), "n_segments": n_segments,
                 "n_failures": len(failures), "n_unlocated_segments": n_unlocated,
                 "usage": {"input": usage_in, "cached_input": usage_cached,
                           "output": usage_out}},
        "annotations": annotations,
    }
    with open(run_dir / "master_annotations.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if failures:
        with open(run_dir / "failures.json", "w") as f:
            json.dump(failures, f, ensure_ascii=False, indent=1)

    print(f"\nParsed {len(annotations)}/{len(results)} intros "
          f"({n_segments} segments; {n_unlocated} unlocated, "
          f"{len(failures)} intros with issues)")
    print(f"Usage: {usage_in:,} in ({usage_cached:,} cached) / {usage_out:,} out")
    print(f"Wrote {run_dir / 'master_annotations.json'}")
    print(f"Merge with:\n"
          f"  python intro_analysis/annotate_master_dataset.py --merge {run_dir}")


# ---------------------------------------------------------------------------
# Merge into the master dataset
# ---------------------------------------------------------------------------

def merge_run_into_dataset(dataset: dict, run_dir: Path) -> dict:
    """Attach the run's annotations to every matching intro (in place).
    Returns a small summary dict. Importable by build_master_dataset so a
    dataset rebuild can re-merge this run."""
    state = _load_state(run_dir)
    parsed = json.load(open(run_dir / "master_annotations.json"))
    ann_by_id = parsed["annotations"]
    try:
        source = str(run_dir.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(run_dir)

    n_merged = n_dropped = 0
    missing = []
    for paper, intro in iter_intros(dataset):
        entry = ann_by_id.get(intro["intro_id"])
        if entry is None:
            # Only flag intros that will END UP without an annotation: broken
            # ones and, on incremental runs, already-annotated ones are fine.
            if (ANNOTATOR_NAME not in intro["annotations"]
                    and intro["intro_id"] not in state.get("skipped_broken", [])):
                missing.append(intro["intro_id"])
            continue
        canon = intro["text"]
        src = state["texts"][intro["intro_id"]]
        if collapse(src) != collapse(canon):
            missing.append(intro["intro_id"] + " (text drift)")
            continue
        src_rank = build_nonws_rank(src)
        dst_nonws = _nonws_positions(canon)
        segments = []
        for seg in entry["segments"]:
            if not seg.get("match_ok"):
                n_dropped += 1
                continue
            cs, ce = remap_span(seg["start"], seg["end"], src, None,
                                src_rank, dst_nonws)
            if collapse(canon[cs:ce]) != collapse(src[seg["start"]:seg["end"]]):
                raise ValueError(f"{intro['intro_id']}: span remap mismatch "
                                 f"at ({seg['start']},{seg['end']})")
            new = make_segment(cs, ce, canon, str(seg["label"]))
            if seg.get("rationale"):
                new["rationale"] = seg["rationale"]
            segments.append(new)
        segments.sort(key=lambda s: s["start"])
        intro["annotations"][ANNOTATOR_NAME] = {
            "annotator": ANNOTATOR_NAME,
            "annotator_type": "llm",
            "role": "auto",
            "model": state["model"],
            "provider": state["provider"],
            "prompt_version": state["prompt_version"],
            "run_tag": state["run_tag"],
            "reasoning_effort": state["reasoning_effort"],
            "source": source,
            "num_segments": len(segments),
            "segments": segments,
        }
        n_merged += 1

    # Accumulate run provenance (a full run + later incremental runs).
    runs = dataset["meta"]["sources"]["llm_annotation_runs"].get(ANNOTATOR_NAME)
    runs = [runs] if isinstance(runs, str) else (runs or [])
    if source not in runs:
        runs.append(source)
    dataset["meta"]["sources"]["llm_annotation_runs"][ANNOTATOR_NAME] = runs
    note = (f"{ANNOTATOR_NAME} auto-labels ({state['run_tag']}, {state['model']}) "
            f"from {source} merged onto {n_merged} intros on "
            f"{datetime.now().isoformat(timespec='seconds')}; "
            f"broken intros skipped: {state.get('skipped_broken', [])}")
    dataset["meta"]["notes"] = [n for n in dataset["meta"]["notes"]
                                if not (f"{ANNOTATOR_NAME} auto-labels" in n
                                        and f"from {source} " in n)] + [note]
    return {"n_merged": n_merged, "n_dropped_segments": n_dropped,
            "missing": missing}


def merge(run_dir: Path, dataset_path: Path) -> None:
    dataset = json.load(open(dataset_path))
    summary = merge_run_into_dataset(dataset, run_dir)

    # Invariant check (same as build_master_dataset.verify): every merged
    # segment's text is the exact slice of its intro's canonical text.
    for paper, intro in iter_intros(dataset):
        ann = intro["annotations"].get(ANNOTATOR_NAME)
        if not ann:
            continue
        for s in ann["segments"]:
            assert intro["text"][s["start"]:s["end"]] == s["text"], \
                f"{intro['intro_id']}: segment text/offset mismatch"

    with open(dataset_path, "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)
    print(f"Merged {summary['n_merged']} intro annotations into {dataset_path} "
          f"({summary['n_dropped_segments']} unlocated segments dropped)")
    if summary["missing"]:
        print(f"⚠ intros without annotation ({len(summary['missing'])}): "
              f"{summary['missing']}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--submit", action="store_true",
                        help="Build all requests and submit the OpenAI batch")
    action.add_argument("--status", metavar="RUN_DIR",
                        help="Poll a submitted run's batch status")
    action.add_argument("--fetch", metavar="RUN_DIR",
                        help="Retrieve a finished batch and parse the annotations")
    action.add_argument("--merge", metavar="RUN_DIR",
                        help="Merge a fetched run into the master dataset (in place)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS,
                    help="human_intros tasks.json (few-shot demo source)")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--key-suffix", default=None,
                    help="API key-set selector, e.g. LAB → OPENAI_API_KEY_LAB")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--shard-size", type=int, default=0,
                    help="Split the batch into shards of N requests (0 = one batch)")
    ap.add_argument("--include-broken", action="store_true",
                    help="Also annotate intros whose text is a raw JSON artifact")
    ap.add_argument("--only-missing", action="store_true",
                    help="With --submit: only annotate intros without an existing "
                         "chatgpt annotation (incremental run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --submit: build requests and report size, don't submit")
    args = ap.parse_args()

    if args.submit:
        submit(args)
    elif args.status:
        status(Path(args.status), args.key_suffix)
    elif args.fetch:
        fetch(Path(args.fetch), args.key_suffix)
    elif args.merge:
        merge(Path(args.merge), args.dataset)


if __name__ == "__main__":
    main()
