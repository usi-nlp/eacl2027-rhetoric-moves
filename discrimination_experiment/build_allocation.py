#!/usr/bin/env python3
"""Build the item allocation for the human/LLM AI-vs-original discrimination experiment.

Design (60 papers freshly sampled from the full 76-paper pool, stratified by
quartile):
- Pool: 60 papers sampled from all 76 (Q1 12/16, Q2-Q4 16/20 each).
- Each paper contributes 2 items: its original intro + one claude-opus-4-8 generated intro.
- 2x2 assignment at the PAPER level (both of a paper's items share the cell):
    generation mode (basic / cars_guided) x label condition (labeled / unlabeled)
  15 papers per cell, stratified by quartile (Q1: 3/cell, Q2-Q4: 4/cell) and,
  within quartile, balanced on nativeness.
- Single 120-item presentation order (2026-07-17: the annotator works through one
  list at their own pace — no fixed sessions). Internally the order is built from
  4 balance blocks of 30 (each 15 originals + 15 generated, a paper's two items
  never in the same block), then reshuffled-within-block until every paper's two
  items are at least --min-pair-gap positions apart (same-topic intros must not
  appear close together, or the pairing becomes visible).
- Judgment: 5-point scale (1 definitely human ... 5 definitely AI).

Output: allocation.json (sealed truth — do NOT share with judges) and
judge_manifest.json (blinded item codes + session/position + label condition only).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations
from pathlib import Path

# Papers sampled per quartile (sums to 60; multiples of 4 keep the 2x2 cells
# quartile-balanced: Q1 3/cell, Q2-Q4 4/cell — corpus has Q1:16, Q2-Q4:20 each).
QUARTILE_SAMPLE = {"Q1": 12, "Q2": 16, "Q3": 16, "Q4": 16}

CELLS = [
    ("basic", "labeled"),
    ("basic", "unlabeled"),
    ("cars_guided", "labeled"),
    ("cars_guided", "unlabeled"),
]

SCALE = {
    "1": "definitely human",
    "2": "mostly human",
    "3": "can't decide",
    "4": "mostly AI",
    "5": "definitely AI",
}


def load_pool(dataset_path, model):
    with open(dataset_path) as f:
        data = json.load(f)
    pool = []
    for paper in data["papers"]:
        original = next(i for i in paper["intros"] if i["intro_type"] == "original")
        generated = {
            i["mode"]: i
            for i in paper["intros"]
            if i["intro_type"] == "generated" and i["model"] == model
        }
        if "basic" not in generated or "cars_guided" not in generated:
            raise SystemExit(f"paper {paper['paper_key']}: missing {model} intros")
        pool.append(
            {
                "paper_key": paper["paper_key"],
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "quartile": paper["quartile"],
                "nativeness": paper["nativeness"],
                "original": original,
                "generated": generated,
            }
        )
    if len(pool) != 76:
        raise SystemExit(f"expected 76 pool papers, got {len(pool)}")
    return pool


def sample_pool(pool, rng):
    """Sample the experiment papers from the full pool, stratified by quartile
    and, within quartile, proportionally by nativeness."""
    by_quartile = defaultdict(list)
    for paper in pool:
        by_quartile[paper["quartile"]].append(paper)
    sampled = []
    for quartile, target in sorted(QUARTILE_SAMPLE.items()):
        papers = by_quartile[quartile]
        if len(papers) < target:
            raise SystemExit(f"{quartile}: need {target} papers, pool has {len(papers)}")
        natives = sorted((p for p in papers if p["nativeness"] == "native"), key=lambda p: p["paper_key"])
        non_natives = sorted((p for p in papers if p["nativeness"] != "native"), key=lambda p: p["paper_key"])
        rng.shuffle(natives)
        rng.shuffle(non_natives)
        n_native = min(len(natives), round(target * len(natives) / len(papers)))
        sampled.extend(natives[:n_native] + non_natives[: target - n_native])
    if len(sampled) != sum(QUARTILE_SAMPLE.values()):
        raise SystemExit(f"sampled {len(sampled)} papers, expected {sum(QUARTILE_SAMPLE.values())}")
    return sampled


def assign_cells(pool, rng):
    """Deal papers into the 4 cells, stratified by quartile, balanced on nativeness."""
    by_quartile = defaultdict(list)
    for paper in pool:
        by_quartile[paper["quartile"]].append(paper)
    assignment = {}
    for quartile, papers in sorted(by_quartile.items()):
        natives = [p for p in papers if p["nativeness"] == "native"]
        non_natives = [p for p in papers if p["nativeness"] != "native"]
        rng.shuffle(natives)
        rng.shuffle(non_natives)
        ordered = natives + non_natives  # round-robin dealing spreads each group
        cell_order = CELLS[:]
        rng.shuffle(cell_order)
        for i, paper in enumerate(ordered):
            assignment[paper["paper_key"]] = cell_order[i % 4]
    counts = Counter(assignment.values())
    assert all(counts[c] == 15 for c in CELLS), counts
    return assignment


def assign_blocks(pool, rng, n_blocks=4):
    """Give each paper an ordered (block_original, block_generated) pair.

    Using each of the 12 ordered pairs of distinct blocks exactly 5 times
    guarantees every balance block gets exactly 15 originals + 15 generated items
    (which keeps originals/generated evenly spread over the presentation order).
    """
    pairs = list(permutations(range(1, n_blocks + 1), 2)) * 5
    rng.shuffle(pairs)
    keys = [p["paper_key"] for p in pool]
    rng.shuffle(keys)
    return dict(zip(keys, pairs))


def order_items(items, rng, min_gap, max_attempts=200_000):
    """Global presentation order: shuffle within each balance block, concatenate,
    retry until every paper's two items are >= min_gap positions apart."""
    by_block = defaultdict(list)
    for item in items:
        by_block[item["block"]].append(item)
    for attempt in range(1, max_attempts + 1):
        ordered = []
        for block in sorted(by_block):
            block_items = by_block[block][:]
            rng.shuffle(block_items)
            ordered.extend(block_items)
        seen, ok = {}, True
        for pos, item in enumerate(ordered, start=1):
            key = item["paper_key"]
            if key in seen and pos - seen[key] < min_gap:
                ok = False
                break
            seen[key] = pos
        if ok:
            return ordered, attempt
    raise SystemExit(f"could not satisfy min pair gap {min_gap} in {max_attempts} attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/master_dataset.json")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--min-pair-gap", type=int, default=20,
                        help="minimum positions between a paper's two items")
    parser.add_argument("--out-dir", default="discrimination_experiment")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pool = sample_pool(load_pool(args.dataset, args.model), rng)
    cells = assign_cells(pool, rng)
    blocks = assign_blocks(pool, rng)

    items = []
    for paper in pool:
        mode, label_condition = cells[paper["paper_key"]]
        block_original, block_generated = blocks[paper["paper_key"]]
        for intro, intro_type, block in (
            (paper["original"], "original", block_original),
            (paper["generated"][mode], "generated", block_generated),
        ):
            items.append(
                {
                    "paper_key": paper["paper_key"],
                    "intro_id": intro["intro_id"],
                    "intro_type": intro_type,
                    "mode": mode,
                    "label_condition": label_condition,
                    "block": block,
                    "quartile": paper["quartile"],
                    "nativeness": paper["nativeness"],
                    "word_count": int(intro["word_count"]),
                }
            )

    # One global presentation order; blinded item codes follow it.
    ordered, attempts = order_items(items, rng, args.min_pair_gap)
    for position, item in enumerate(ordered, start=1):
        item["position"] = position
        item["item_code"] = f"item_{position:03d}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "dataset": args.dataset,
        "generator_model": args.model,
        "pool_size": 76,
        "quartile_sample": QUARTILE_SAMPLE,
        "sampled_paper_ids": sorted(int(p["paper_id"]) for p in pool),
        "design": (
            "2x2 paper-level: mode (basic/cars_guided) x labels (labeled/unlabeled), "
            "15 papers/cell; single 120-item presentation order (4 internal balance "
            "blocks of 30, each paper's two items >= min_pair_gap apart)"
        ),
        "min_pair_gap": args.min_pair_gap,
        "response_scale": SCALE,
    }
    with open(out_dir / "allocation.json", "w") as f:
        json.dump({"meta": meta, "items": ordered}, f, indent=1)

    judge_fields = ("item_code", "position", "label_condition")
    manifest = [{k: item[k] for k in judge_fields} for item in ordered]
    with open(out_dir / "judge_manifest.json", "w") as f:
        json.dump({"meta": {k: meta[k] for k in ("built", "response_scale")}, "items": manifest}, f, indent=1)

    # Summary
    print(f"{len(ordered)} items, one presentation order (ordering attempts: {attempts}) -> {out_dir}/allocation.json")
    print("\ncell x quartile (papers):")
    cell_quart = Counter((cells[p["paper_key"]], p["quartile"]) for p in pool)
    for cell in CELLS:
        row = "  ".join(f"{q}:{cell_quart[(cell, q)]}" for q in ("Q1", "Q2", "Q3", "Q4"))
        print(f"  {cell[0]:>12}/{cell[1]:<9} {row}")
    print("\ncell x nativeness (papers):")
    cell_nat = Counter((cells[p["paper_key"]], p["nativeness"]) for p in pool)
    for cell in CELLS:
        print(f"  {cell[0]:>12}/{cell[1]:<9} native:{cell_nat[(cell, 'native')]:>2}  non-native:{cell_nat[(cell, 'non-native')]:>2}")
    print("\nbalance-block composition (items):")
    by_block = defaultdict(list)
    for item in ordered:
        by_block[item["block"]].append(item)
    for block in sorted(by_block):
        block_items = by_block[block]
        types = Counter(i["intro_type"] for i in block_items)
        labels = Counter(i["label_condition"] for i in block_items)
        print(
            f"  block {block}: {len(block_items)} items | "
            f"original:{types['original']} generated:{types['generated']} | "
            f"labeled:{labels['labeled']} unlabeled:{labels['unlabeled']}"
        )
    positions = defaultdict(list)
    for item in ordered:
        positions[item["paper_key"]].append(item["position"])
    gaps = [max(p) - min(p) for p in positions.values()]
    print(f"\npair gap: min {min(gaps)} (required >= {args.min_pair_gap}), median {sorted(gaps)[len(gaps)//2]}, max {max(gaps)}")


if __name__ == "__main__":
    main()
