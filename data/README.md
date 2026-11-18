# Master dataset

`master_dataset.json` is the single input of every analysis in the paper. It
holds, for each of the 76 papers of the corpus, the original introduction, the
eight generated introductions (4 models x 2 prompt settings) and every CARS
annotation of those texts.

| | |
|---|---|
| Papers | 76 applied-linguistics research articles (Q1: 16, Q2-Q4: 20 each) |
| Introductions | 684 = 76 original + 608 generated |
| Generators | `gpt-5.5`, `claude-opus-4-8`, `google/gemma-4-31B-it`, `Qwen/Qwen3.5-27B` |
| Prompt settings | `basic`, `cars_guided` |
| Annotated segments | 8,880 automatic (all 684 introductions) + expert gold on the 76 originals + second-annotator labels on 32 originals |
| Size | 12 MB, UTF-8 JSON |

## Loading

```python
import json

data = json.load(open("data/master_dataset.json"))
paper = data["papers"][0]
original = next(i for i in paper["intros"] if i["intro_type"] == "original")
claude_cars = next(i for i in paper["intros"]
                   if i["model"] == "claude-opus-4-8" and i["mode"] == "cars_guided")

for seg in claude_cars["annotations"]["chatgpt"]["segments"]:
    print(seg["move"], seg["step"], "|", seg["text"][:60])
```

For the statistics of the paper, load through
`comparative_analysis/move_data.py::load_intros`, which applies the analysis
convention of the paper: consecutive segments with the same step label are
merged into one unit before anything is counted.

## Structure

```text
{
  "meta":   { build info, label scheme, coverage, provenance notes },
  "papers": [
    {
      "paper_id": 1,            # 1-80; ids 19, 24, 32 and 33 are the four excluded papers
      "paper_key": "001",
      "title": "...",
      "filename": "...",        # name of the source annotation file, without extension
      "nativeness": "native" | "non-native",   # academic nativeness, from the source corpus
      "quartile": "Q1" | "Q2" | "Q3" | "Q4",   # journal prestige quartile
      "swales_topic": { ... },  # does the paper itself discuss Swales / CARS?
      "intros": [ ... 9 introductions ... ]
    }
  ]
}
```

### Introduction fields

| Field | Description |
|---|---|
| `intro_id` | `{paper_key}_original` or `{paper_key}_{model}_{mode}`, e.g. `001_gpt-5.5_basic` |
| `intro_type` | `original` or `generated` |
| `model`, `provider`, `mode` | Generator, API provider and prompt setting (`null` for originals). The two open-weight models were served locally with vLLM, hence the provider `openai_compatible` |
| `text` | The introduction. All annotation offsets refer to this string |
| `word_count`, `char_count` | Length of `text` |
| `target_words`, `length_ratio` | Length target given to the generator (the original's word count) and achieved length divided by target |
| `n_cited_refs` | Number of works cited in the original introduction, which are the works given to the generator |
| `request_date`, `generation_run` | When and in which run the text was generated |
| `citation_usage` | How many of the given works the text cites, and which author-year citations fall outside the given list |
| `annotations` | One entry per annotator, see below |

### Annotations

| Key | Who | Coverage |
|---|---|---|
| `expert` | Expert gold annotation (role `gold`), from the source corpus | 76 originals |
| `annotator_1`, `annotator_2` | Second human annotators (role `second`), used as the human baseline when validating the automatic annotator | 16 originals each, no overlap |
| `chatgpt` | The automatic annotator of the paper (role `auto`): `gpt-5.5`, prompt version `v8`, low reasoning effort | all 684 introductions |

Every annotation is a list of `segments` in reading order:

| Field | Description |
|---|---|
| `start`, `end` | Character offsets into the introduction's `text` |
| `text` | The segment text |
| `label` | `move_{n}__{step}`, e.g. `move_1__reviewing_items_of_previous_research` |
| `move`, `step` | The two halves of `label` |
| `label_raw` | The label as spelled in the source file |
| `in_scheme` | `false` for the five segments labelled with the unused eleventh step (Move 2 counter-claiming), which the analyses exclude |
| `rationale` | Automatic annotator only: the one-sentence justification the model wrote before choosing the label |

The label scheme has 3 moves and 10 steps:

| Move | Steps |
|---|---|
| `move_1` Establishing a territory | `claiming_centrality`, `making_topic_generalization`, `reviewing_items_of_previous_research` |
| `move_2` Establishing a niche | `continuing_a_tradition`, `indicating_a_gap`, `question_raising` |
| `move_3` Occupying the niche | `announcing_present_research`, `announcing_principal_findings`, `indicating_the_structure_of_research`, `outlining_purpose` |

## Provenance and notes

- The 76 original introductions and their expert gold annotations come from
  the corpus of D'Agostino, Sheikh Asadi and Musi
  ([LREC 2026](https://doi.org/10.63317/3cbozhs8zty6)). That corpus has 80
  papers. The four whose full text is only available behind a paywall were
  dropped.
- Nine `Qwen/Qwen3.5-27B` CARS-guided generations failed in the first run (the
  model returned its plan instead of the introduction) and were regenerated on
  2026-07-10. `meta.notes` lists the paper ids.
- `source`, `generation_run` and the paths under `meta.sources` name the
  internal files and runs each part was built from. They are kept as provenance
  and point to material that is not part of this repository (article PDFs,
  parsed full texts, raw API runs).
- The gold annotations can be written back out as a Label Studio task file with
  `python data_processing/export_gold_tasks.py`. The annotator scripts expect
  that file.

## Licence

The annotations, the generated introductions and the human-study data are
released under [CC BY-NC-SA 4.0](../LICENSE), like the rest of this repository. The original introductions
are excerpts of published articles. They remain the property of their authors
and publishers and are included for research purposes only. The full articles
are not redistributed. If you are a rights holder and want a text removed,
please open an issue.
