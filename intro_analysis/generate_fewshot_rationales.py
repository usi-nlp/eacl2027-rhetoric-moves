#!/usr/bin/env python3
"""
Generate example-specific few-shot rationales (one per gold segment).

For each reserved few-shot demo task, we give a model the task description, the
full annotation manual, the introduction, and the GOLD segmentation + labels,
and ask it to write a one-sentence, move-first rationale justifying each gold
label. The rationales are saved to `intro_analysis/few_shot_rationales.json`
keyed by task id (in gold-segment order) and are then used verbatim in the v5
few-shot block (replacing the generic per-class templates).

    python intro_analysis/generate_fewshot_rationales.py            # all demo ids
    python intro_analysis/generate_fewshot_rationales.py --ids 17 16 34
    python intro_analysis/generate_fewshot_rationales.py --model gpt-5.5 --reasoning-effort medium
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intro_analysis.prompts import (
    FEW_SHOT_EXAMPLE_IDS, _gold_segments_for_task, _clean,
    CLASS_NUMBER_TO_LABEL, label_to_number, _MANUAL_TEXT, NUMBERED_LABEL_SET,
)
from intro_analysis.llm_client import LLMClient, infer_provider, parse_json

load_dotenv(Path(__file__).parent.parent / ".env")

OUT = Path(__file__).parent / "few_shot_rationales.json"

# Readable class name per number (from the numbered label set), for the prompt.
_NUM_NAME = {
    1: "claiming centrality", 2: "making a topic generalization",
    3: "reviewing items of previous research", 4: "counter-claiming",
    5: "indicating a gap", 6: "question-raising", 7: "continuing a tradition",
    8: "outlining the purpose", 9: "announcing present research",
    10: "announcing principal findings", 11: "indicating the structure of research",
}

SYSTEM_PROMPT = """You are an expert annotator of Swales' CARS (Create a Research Space) model who is writing teaching rationales for a set of GOLD-annotated examples.

You will be given a research-article introduction that has ALREADY been segmented and labelled by an expert (the labels are correct and final). For EACH segment, write a single-sentence rationale that justifies its gold label. Each rationale must:
- name the MOVE first (Move 1 Establishing a Territory / Move 2 Establishing a Niche / Move 3 Occupying the Niche),
- then state the ONE decisive signal in that specific segment that fixes the sub-step (quote or paraphrase the cue — e.g. the in-text citation, the importance adjective, the gap phrase, the stated aim),
- be specific to this segment's wording (not a generic definition),
- be exactly one sentence.

Use the annotation manual below as the authoritative source for what each label means.

""" + NUMBERED_LABEL_SET + """
---

# Annotation manual

""" + _MANUAL_TEXT


def build_user_prompt(intro: str, segs: list[dict]) -> str:
    lines = [f"Introduction:\n{intro}\n", "Gold segments (in order):"]
    for i, s in enumerate(segs, 1):
        num = label_to_number(s["label"])
        lines.append(f"\n[{i}] label {num} ({_NUM_NAME.get(num,'?')}):\n{s['text']}")
    lines.append(
        f"\n\nReturn ONLY JSON: {{\"rationales\": [ ... ]}} with exactly {len(segs)} "
        "one-sentence rationales, in the same order as the segments above."
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="+", type=int, default=sorted(FEW_SHOT_EXAMPLE_IDS))
    ap.add_argument("--tasks", default="data/annotation/human_intros/tasks.json")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    a = ap.parse_args()

    client = LLMClient(model=a.model, provider=infer_provider(a.model),
                       reasoning_effort=a.reasoning_effort, use_cache=True)
    tasks = {t["id"]: t for t in json.load(open(a.tasks))}

    store = json.load(open(OUT)) if OUT.exists() else {}
    for tid in a.ids:
        task = tasks.get(tid)
        if not task:
            print(f"  ⚠ task {tid} not found", file=sys.stderr)
            continue
        intro = task["data"].get("clean_text") or _clean(task["data"]["text"])
        segs = _gold_segments_for_task(task)
        print(f"→ task {tid}: generating {len(segs)} rationale(s)…", flush=True)
        out = client.generate(system_prompt=SYSTEM_PROMPT,
                              user_prompt=build_user_prompt(intro, segs))
        rats = parse_json(out).get("rationales", [])
        if len(rats) != len(segs):
            print(f"  ⚠ task {tid}: got {len(rats)} rationales for {len(segs)} segments "
                  "— storing what aligns, padding the rest", file=sys.stderr)
        entries = []
        for i, s in enumerate(segs):
            entries.append({
                "label": s["label"],
                "number": label_to_number(s["label"]),
                "rationale": rats[i].strip() if i < len(rats) else "",
                "text_preview": s["text"][:80],
            })
        store[str(tid)] = entries

    OUT.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    n = sum(len(v) for v in store.values())
    print(f"\nWrote {n} rationale(s) across {len(store)} task(s) → {OUT}")


if __name__ == "__main__":
    main()
