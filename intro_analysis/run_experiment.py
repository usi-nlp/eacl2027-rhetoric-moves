#!/usr/bin/env python3
"""
Intro Analysis Experiment — segmentation and/or Swales CARS classification.

Samples N introductions from human_intros/tasks.json, calls GPT to segment
(and optionally label) each introduction, then stores results alongside the
gold annotations for evaluation by evaluate_and_visualize.py.

Usage:
    # Segmentation only, no few-shot
    python intro_analysis/run_experiment.py --n 5 --mode segmentation

    # Segmentation + classification, with few-shot examples
    python intro_analysis/run_experiment.py --n 5 --mode classification --few-shot

    # Use the annotation manual as the prompt source
    python intro_analysis/run_experiment.py --n 5 --mode classification --prompt-source manual

The experiment is the cartesian product of:
    --mode          {segmentation, classification}
    --few-shot      flag
    --prompt-source {builtin, manual}

Few-shot reserved task IDs (excluded from the sampled pool whenever --few-shot
is set): see intro_analysis.prompts.FEW_SHOT_EXAMPLE_IDS. The set is split into
three granularity groups (large-span / average / fine-grained); the prompt shows
one example from each, so the model sees the full segmentation range.
"""

import argparse
import json
import openpyxl
import random
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intro_analysis.prompts import (
    FEW_SHOT_EXAMPLE_IDS,
    PROMPT_VERSIONS,
    USER_PROMPT_TEMPLATE,
    get_system_prompt,
)
from intro_analysis.llm_client import LLMClient, infer_provider, parse_json
from intro_analysis.postprocess import apply_citation_rules, map_numeric_labels

load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Helpers (text loading, GPT call, span resolution)
# ---------------------------------------------------------------------------

def _load_excel_intros(excel_path: str) -> dict[str, str]:
    """Return {paper_id: introduction_text} from the corpus Excel file."""
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active
    headers = [c.value for c in next(ws.iter_rows(max_row=1))]
    id_col = headers.index("paper_id")
    intro_col = headers.index("introduction")
    result = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid = row[id_col]
        intro = row[intro_col]
        if pid is not None and intro is not None:
            result[str(int(pid))] = str(intro)
    wb.close()
    return result


def _clean(text: str) -> str:
    """Collapse all whitespace runs to single spaces (strip PDF line-break noise)."""
    return " ".join(text.split())


def load_tasks(path: str, excel_path: str | None = None) -> list[dict]:
    with open(path) as f:
        tasks = json.load(f)
    for task in tasks:
        # Strip all newlines — collapses PDF line-breaks into single spaces so the
        # paragraph structure is not revealed to the model as a segmentation hint.
        task["data"]["clean_text"] = _clean(task["data"]["text"])
    return tasks


def extract_gold(task: dict) -> list[dict]:
    """Return gold annotations re-mapped into clean_text character space.

    Searches sequentially (each span searched after the previous one ends) to
    prevent overlapping matches caused by repeated phrases in the text.
    """
    clean = task["data"]["clean_text"]
    segments = []
    search_from = 0
    for ann in task["annotations"]:
        for r in sorted(ann["result"], key=lambda r: r["value"]["start"]):
            v = r["value"]
            norm = _clean(v["text"])
            idx = clean.find(norm, search_from)
            if idx == -1:
                # Fuzzy fallback: try from beginning in case earlier segment displaced us
                idx = clean.find(norm)
            if idx == -1:
                # Last resort: match on first 60 chars
                idx = clean.find(norm[:60], search_from)
                if idx == -1:
                    idx = clean.find(norm[:60])
            if idx == -1:
                continue  # genuinely unlocatable — skip
            end = idx + len(norm)
            segments.append({
                "start": idx,
                "end": end,
                "text": clean[idx:end],
                "label": v["labels"][0],
            })
            search_from = end
    return sorted(segments, key=lambda s: s["start"])


def call_llm(client: LLMClient, text: str,
             system_prompt: str | None = None) -> list[dict]:
    """Call the model (any provider) and return the raw segments list."""
    if system_prompt is None:
        from intro_analysis.prompts import SYSTEM_PROMPT_BASE
        system_prompt = SYSTEM_PROMPT_BASE
    output = client.generate(
        system_prompt=system_prompt,
        user_prompt=USER_PROMPT_TEMPLATE.format(text=text),
    )
    raw = parse_json(output)
    return raw.get("segments", [])


def resolve_spans(text: str, segments: list[dict]) -> list[dict]:
    """
    Find character offsets for each AI segment by searching for verbatim text.
    Searches forward from the previous match to respect order.
    """
    resolved = []
    search_from = 0
    for seg in segments:
        excerpt = seg.get("text", "")
        idx = text.find(excerpt, search_from)
        if idx == -1:
            stripped = excerpt.strip()
            idx = text.find(stripped, search_from)
            if idx == -1:
                resolved.append({**seg, "start": None, "end": None, "match_ok": False})
                continue
            excerpt = stripped
        resolved.append({
            **seg,
            "start": idx,
            "end": idx + len(excerpt),
            "match_ok": True,
        })
        search_from = idx + len(excerpt)
    return resolved


def process_task(client: LLMClient, task: dict,
                 system_prompt: str | None = None,
                 citation_postproc: bool = True) -> dict:
    text = task["data"]["clean_text"]
    gold = extract_gold(task)

    print(f"  → Calling {client.model} [{client.provider}/{client.reasoning_effort}] "
          f"for task {task['id']} ({len(text)} chars)…", flush=True)
    raw_segments = call_llm(client, text, system_prompt)
    ai_segments = resolve_spans(text, raw_segments)

    failed = [s for s in ai_segments if not s.get("match_ok")]
    if failed:
        print(f"    ⚠ {len(failed)} segment(s) could not be located in the text",
              file=sys.stderr)

    # v4 numbered scheme: map integer labels (1–11) back to canonical CARS
    # label strings before any label-based post-processing. No-op for v1–v3
    # (labels are already strings).
    n_mapped = map_numeric_labels(ai_segments)
    if n_mapped:
        print(f"    ↳ mapped {n_mapped} numeric label(s) → CARS strings", flush=True)

    # Citation surface-rule post-processing (Move 1 only): citation⇒RIPR,
    # no-citation+RIPR⇒MTG. Enforces the gold annotator's mechanical rule.
    if citation_postproc:
        n_relabeled = apply_citation_rules(ai_segments)
        if n_relabeled:
            print(f"    ↳ citation post-proc relabeled {n_relabeled} Move-1 segment(s)",
                  flush=True)

    return {
        "task_id": task["id"],
        "source_file": task["data"]["source_file"],
        "text": text,
        "gold": gold,
        "ai": ai_segments,
        "model": client.model,
        "provider": client.provider,
    }



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_tag(mode: str, few_shot: bool, source: str, version: str = "v1") -> str:
    tag = "classification" if mode == "classification" else "segmentation"
    if few_shot:
        tag += "-fewshot"
    if source == "manual":
        tag += "-manual"
    if version != "v1":
        tag += f"-{version}"
    return tag


def main():
    parser = argparse.ArgumentParser(description="Run GPT intro segmentation/classification experiment")
    parser.add_argument(
        "--tasks",
        default="data/annotation/human_intros/tasks.json",
        help="Path to tasks.json",
    )
    parser.add_argument(
        "--excel",
        default="data/raw_data/80-papers_corpus_nativeness_quartiles.xlsx",
        help="Path to corpus Excel file with clean introduction texts",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-5.5",
                        help="Model name. Provider is inferred: gpt-*/o* → OpenAI, "
                             "claude-* → Anthropic, gemini-* → Google.")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "openai", "anthropic", "google"],
        help="LLM back-end. 'auto' infers it from --model (default).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt caching of the (large, static) system prompt. "
             "Caching is on by default and substantially cheaper on repeated calls.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort (OpenAI/Gemini); for Anthropic toggles extended "
             "thinking (low = off). Default: low.",
    )
    parser.add_argument(
        "--mode",
        default="segmentation",
        choices=["segmentation", "classification"],
        help="segmentation: split text only. classification: split + assign CARS labels.",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Include worked corpus examples in the prompt. Reserved few-shot task IDs "
             "are excluded from the sampled pool so the model never sees its own demos.",
    )
    parser.add_argument(
        "--no-citation-postproc",
        action="store_true",
        help="Disable the citation surface-rule post-processing (Move-1 only: "
             "citation⇒RIPR, uncited RIPR⇒MTG). On by default for classification.",
    )
    parser.add_argument(
        "--prompt-source",
        default="builtin",
        choices=["builtin", "manual"],
        help="builtin: rules embedded in this repo. manual: use guidelines/Swales_CARS_Annotation_Manual.md.",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=sorted(PROMPT_VERSIONS),
        help="Prompt version (see PROMPT_VERSIONS in prompts.py). v1=baseline; "
             "v2 adds operational clarifications for classification.",
    )
    parser.add_argument(
        "--results-dir",
        default="intro_analysis/results",
        help="Base directory; each run creates a timestamped subfolder here",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        type=int,
        help="Specific task IDs to run (overrides --n and --seed)",
    )
    args = parser.parse_args()

    provider = infer_provider(args.model) if args.provider == "auto" else args.provider
    client = LLMClient(
        model=args.model,
        provider=provider,
        reasoning_effort=args.reasoning_effort,
        use_cache=not args.no_cache,
    )
    all_tasks = load_tasks(args.tasks, excel_path=args.excel)

    def build_prompt_for(task_id: int | None) -> str:
        """System prompt for one task. When few-shot, exclude the task's own id
        from the demo set so a paper that is itself a reserved demo (34/66) gets
        the other member of its granularity group instead of its own gold."""
        return get_system_prompt(
            mode=args.mode,
            few_shot=args.few_shot,
            source=args.prompt_source,
            all_tasks=all_tasks,
            version=args.prompt_version,
            exclude_task_id=task_id,
        )

    exclude_ids = set(FEW_SHOT_EXAMPLE_IDS) if args.few_shot else set()

    run_tag = _run_tag(args.mode, args.few_shot, args.prompt_source, args.prompt_version)
    print(f"Provider={provider}  model={args.model}  cache={not args.no_cache}")
    print(f"Mode={args.mode}  few_shot={args.few_shot}  source={args.prompt_source}  "
          f"version={args.prompt_version}  ({run_tag})")
    if exclude_ids:
        print(f"Few-shot reserved IDs (excluded from sampling): {sorted(exclude_ids)}")

    if args.task_ids:
        id_set = set(args.task_ids)
        # Explicit --task-ids are run as requested, INCLUDING reserved demo ids:
        # any paper that is also a demo gets the other member of its group as a
        # few-shot example (see build_prompt_for), so there is no self-leakage.
        selected = [t for t in all_tasks if t["id"] in id_set]
        if not selected:
            sys.exit(f"No tasks found with IDs {args.task_ids}")
        demos_in_eval = sorted(id_set & exclude_ids)
        if demos_in_eval:
            print(f"Eval set includes reserved demo id(s) {demos_in_eval}; "
                  f"each will be annotated with the other member of its few-shot group swapped in.")
    else:
        random.seed(args.seed)
        # Shuffle the full task list so the same seed always produces the same
        # ordering regardless of variant; then take the first N that aren't reserved.
        shuffled = random.sample(all_tasks, len(all_tasks))
        selected = [t for t in shuffled if t["id"] not in exclude_ids][: args.n]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.results_dir) / f"{timestamp}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run folder: {run_dir}")
    print(f"Running on {len(selected)} task(s) with {args.model}…")

    # Citation post-processing is a classification-only step (segmentation runs
    # carry no labels to relabel).
    citation_postproc = (args.mode == "classification") and not args.no_citation_postproc
    print(f"Citation post-processing: {citation_postproc}")

    results = []
    for task in selected:
        try:
            system_prompt = build_prompt_for(task["id"])
            result = process_task(client, task, system_prompt,
                                  citation_postproc=citation_postproc)
            results.append(result)
        except Exception as e:
            print(f"  ✖ Task {task['id']} failed: {e}", file=sys.stderr)

    out_path = run_dir / "ai_segments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sample_task = selected[0] if selected else {"id": None, "data": {"clean_text": ""}}
    sample_prompt = build_prompt_for(sample_task["id"]) if selected else ""
    meta = {
        "model": args.model,
        "provider": provider,
        "prompt_caching": not args.no_cache,
        "mode": args.mode,
        "few_shot": args.few_shot,
        "citation_postproc": citation_postproc,
        "prompt_source": args.prompt_source,
        "prompt_version": args.prompt_version,
        "run_tag": run_tag,
        "reasoning_effort": args.reasoning_effort,
        "n_tasks": len(results),
        "task_ids": [r["task_id"] for r in results],
        "few_shot_reserved_ids": sorted(exclude_ids),
        "timestamp": timestamp,
        "prompt_sample": {
            "task_id": sample_task["id"],
            "instructions": sample_prompt,
            "input": USER_PROMPT_TEMPLATE.format(text=sample_task["data"].get("clean_text", "")),
        },
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
