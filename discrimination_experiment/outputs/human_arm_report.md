# Discrimination experiment - human arm results

One expert judge, 120 items (60 papers x [own original + claude-opus-4-8 generation]), 2x2 paper-level design: generation mode (basic / cars_guided) x CARS move labels shown (labeled / unlabeled), 15 papers per cell. Rating scale 1 = sure human ... 5 = sure AI.

Source export: `labelstudio_export.json`; key: `allocation.json` (seed 20260716). All 120 rendered item texts were verified against the intro the key names.

**F1 is for the AI class** (positive = the introduction is generated). Signal-detection quantities (d', criterion, AUC), false-alarm rates and bootstrap confidence intervals are computed for every split and stored in `human_arm_results.json`; this report keeps to counts, accuracy and F1.

## 1. Headline

| quantity | value |
|---|---|
| items | 120 (60 papers) |
| correct | 108 / 120 |
| accuracy | 90.0% |
| **F1 (AI class)** | **0.900** |
| precision (AI) | 0.900 |
| recall (AI) | 0.900 |

Accuracy vs. chance: binomial p = 1.78e-20.

Confusion matrix:

| | said AI | said human |
|---|---|---|
| **generated** | 54 | 6 |
| **original** | 6 | 54 |

The judge called 60 of the 120 items AI - exactly the true base rate, so the errors are not a guessing bias in either direction.


## 2. Response profile

| rating | n |
|---|---|
| sure human | 2 |
| most likely human | 58 |
| can't decide | 0 |
| most likely AI | 57 |
| sure AI | 3 |

"Can't decide" was used **0 times** - the judge committed on every item. The extreme anchors were used 5 times; 115 ratings were the hedged 2/4, so the 5-point scale behaved as a forced binary with a confidence tag.

Rating distribution by group:

| group | 1 sure human | 2 most likely human | 3 can't decide | 4 most likely AI | 5 sure AI |
|---|---|---|---|---|---|
| original · unlabeled | 1 | 26 | 0 | 2 | 1 |
| original · labeled | 1 | 26 | 0 | 3 | 0 |
| generated · unlabeled | 0 | 3 | 0 | 25 | 2 |
| generated · labeled | 0 | 3 | 0 | 27 | 0 |

Decision basis (multi-select, required):

| basis | n items |
|---|---|
| argumentation_structure | 118 |
| writing_style | 15 |

Combinations: argumentation_structure (105), argumentation_structure+writing_style (13), writing_style (2). "Just a guess" was never selected. Free-text comments: 1.


## 3. RQ1 - do CARS move labels help?

Each paper contributes two items - its own original and its generation - so an item count is always twice the paper count, half original and half generated.

| condition | papers | items | originals | generated | correct | accuracy | F1 (AI) |
|---|---|---|---|---|---|---|---|
| labeled | 30 | 60 | 30 | 30 | 54/60 | 90.0% | 0.900 |
| unlabeled | 30 | 60 | 30 | 30 | 54/60 | 90.0% | 0.900 |

Fisher p = 1. **Showing the move annotation changed nothing measurable.**

The two conditions match cell for cell (both 27/30 generated caught and 3/30 originals wrongly flagged). That is coincidence, not duplicated data: the conditions cover disjoint papers, and every one of the 120 rendered item texts was checked against the intro the sealed key names before any statistic was computed.

A null at this sample size is 'no detectable effect at 60 items per condition', not a proven zero - the interval around the difference (in the JSON) still admits a moderate effect.


## 4. Generated vs. original items

| item type | items | correctly identified | rate | mean rating |
|---|---|---|---|---|
| generated | 60 | 54/60 | 90.0% | 3.83 |
| original | 60 | 54/60 | 90.0% | 2.18 |

Perfectly symmetric (Fisher p = 1): the judge is no better at spotting machines than at recognising humans. Rating separation Mann-Whitney p = 2.54e-17.


## 5. RQ2 - does CARS-guided generation evade detection?

| generation mode | generated intros | caught | rate | F1 (AI, whole cell) | accuracy (whole cell) |
|---|---|---|---|---|---|
| basic | 30 | 24/30 | 80.0% | 0.828 | 83.3% |
| cars_guided | 30 | 30/30 | 100.0% | 0.968 | 96.7% |

Fisher p = 0.0237. The originals paired with each mode were misjudged at the same rate (Fisher p = 0.671), so the difference is the generation mode, not the papers that landed in each cell.

**The effect runs opposite to the pre-registered RQ2: CARS-guided generations were *easier* to spot, not harder.**

Why, measured on these same 120 introductions. Only the generated rows have a generation mode; the originals carry none, and appear as the baseline for the papers each mode was applied to:

| group | intros | exactly Move 1-2-3 | returns to an earlier move | distinct move sequences | effective structures (Vendi) | repertoire | Move-2 share |
|---|---|---|---|---|---|---|---|
| originals of the basic papers | 30 | 20.0% | 56.7% | 16 | 12.98 | 5.13 | 0.069 |
| **generated · basic** | 30 | 13.3% | 76.7% | 16 | 13.36 | 6.17 | 0.136 |
| originals of the cars_guided papers | 30 | 10.0% | 73.3% | 19 | 16.17 | 5.30 | 0.147 |
| **generated · cars_guided** | 30 | 63.3% | 36.7% | 6 | 2.95 | 6.53 | 0.237 |

Among the generated items only, 63.3% of cars_guided intros are exactly the canonical funnel vs 13.3% of basic (Fisher p = 0.000139), and only 36.7% return to an earlier move vs 76.7% (p = 0.00376). The 30 cars_guided intros use just 6 distinct move sequences against 16 for basic and 29 across the 60 originals - human introductions are nearly all structurally unique, and only 15.0% of them are the textbook funnel.

So the guided prompt does not make the output more human, it makes it more *uniform*: the generator writes the model answer instead of an introduction, and that regularity is the tell. The two modes sit on different papers, so compare generated-with-generated; the originals' rows are the paper-set baseline. The two originals rows also differ from each other (Move-2 share 0.069 vs 0.147), which is ordinary paper-set variation - lead with the generated-vs-generated comparison, not the raw guided-minus-basic difference.


## 6. Design cells

The 2x2 cell is a property of the **paper**, not of the introduction: `basic` / `cars_guided` names the mode the paper's generation was written in, and the paper's own original sits in the same cell as its paired control. There is no such thing as a "basic original".

15 papers per cell, each contributing its original **and** its generation, so a cell holds 30 items of which 15 are generated. The 60 generated introductions are therefore 30 basic (15 labeled + 15 unlabeled) and 30 cars_guided - the split section 5 tests.

| paper cell | papers | items | originals | generated | correct | accuracy | F1 (AI) |
|---|---|---|---|---|---|---|---|
| basic · labeled | 15 | 30 | 15 | 15 | 25/30 | 83.3% | 0.828 |
| basic · unlabeled | 15 | 30 | 15 | 15 | 25/30 | 83.3% | 0.828 |
| cars_guided · labeled | 15 | 30 | 15 | 15 | 29/30 | 96.7% | 0.968 |
| cars_guided · unlabeled | 15 | 30 | 15 | 15 | 29/30 | 96.7% | 0.968 |
| **total** | 60 | 120 | 60 | 60 | 108/120 | 90.0% | 0.900 |

Regression models (cluster-robust by paper where estimable; full output in the JSON):

- `logit_said_ai ~ is_ai * labeled`: is_ai = +4.39 (p = 8.4e-06), labeled = +0.00 (p = 1), is_ai_x_labeled = +0.00 (p = 1)
- `ordinal_rating ~ is_ai * labeled`: is_ai = +4.46 (p = 4.71e-08), labeled = -0.03 (p = 0.969), is_ai_x_labeled = -0.40 (p = 0.699)
- `logit_said_ai ~ guided * labeled (generated only)`: complete separation (ConvergenceWarning): every cars_guided item was called AI, so the logistic model is undefined -- see the exact (Fisher) test in section 5
- `ordinal_rating ~ guided * labeled (generated only)`: guided = +0.76 (p = 0.536), labeled = -0.88 (p = 0.405), guided_x_labeled = +0.88 (p = 0.583)

## 7. Overall insights

- **An informed expert separates the two sources reliably but not perfectly**: 108/120 correct, F1 0.900, and the errors are balanced (6 generated intros passed as human, 6 real ones were flagged as machine). Even the structurally closest generator is not indistinguishable to someone who knows the CARS model - but roughly one introduction in ten fools them.
- **CARS guidance backfires.** Guided generations were caught 100.0% of the time against 80.0% for the plain prompt (p = 0.0237). Giving the model the rhetorical schema does not make it write like a human, it makes it write the schema: 63.3% of guided intros are exactly Move 1-2-3, and 30 of them use only 6 distinct move sequences where 60 human introductions use 29.
- **Uniformity is the tell, not fidelity.** Structural similarity to the paper's *own* original was unrelated to detection; what gave the generations away was looking like each other. Human introductions are almost all structurally unique.
- **Showing the move annotation did not help** (90.0% vs 90.0%, identical cell for cell), and the reason is visible in the basis data: the judge already reasoned from move structure on 118/120 items, including the unlabeled ones. The labels were redundant for an expert, not uninformative in principle - and the design has 60 items per condition, so a moderate effect could hide.
- **The judge really did use structure, but structure is not all of it.** Move-2 (niche) share, sub-step repertoire and gap count are the features that correlate with the ratings, and the errors sit exactly where those cues misfire. Yet a detector built on those features alone reaches 67.5% against the judge's 90.0%.
- **The characteristic failure is calling a real human introduction machine-written**, and it concentrates on native-authored papers (75.0% vs 93.8% accuracy, p = 0.0141). Fluent, conventionally-structured human prose reads as AI. Twelve native papers only, so this is a lead rather than a result.
- **The task was easy and fast**: median 47 s per item, "can't decide" never used, and generated items judged *faster* than originals. Whatever the cue is, it is available on a first read.

## 8. Suggested framing for the paper

**Lead with the reversal, not the accuracy.** "An expert detects our AI introductions 90% of the time" is a weak, expected result and invites the objection that a different generator or a naive reader would score differently. The finding that carries the paper is that *prompting with the rhetorical model made the output easier to detect*. That is counter-intuitive, it is actionable for anyone building CARS-aware generators, and it is the claim the corpus analysis independently supports.

**Suggested claim ladder** (each step licensed by a different piece of evidence, so the argument does not rest on the single discrimination result):

1. Corpus level (already in Results): CARS guidance straightens move ordering past the human rate and widens the sub-step repertoire.
2. Item level (section 5 here): within the 120 judged introductions, guidance collapses 30 texts onto a handful of move sequences while human introductions are nearly all unique.
3. Reader level (section 5 here): that uniformity is detectable - guided intros were caught more often than plain ones by an expert reader.
4. Mechanism (`human_arm_results.json`, `structure` and `similarity` blocks): the features that predict the judge's ratings are the same ones that differ between sources, and similarity to a specific original does not matter.

**Where each result belongs.**

- *Results*: one compact table - condition x (n, accuracy, F1) - plus the structure table from section 5. Keep d'/AUC out of the paper; they are in the JSON if a reviewer asks.
- *Results*: the within-paper pairing result (49/60 papers ordered correctly, `paired` block in the JSON) is the cleanest single number, because it controls for topic.
- *Analysis/Discussion*: the RQ1 null, framed as redundancy - an expert who already reads move structure gains nothing from being shown it. Note this also validates the automatic annotation as *consistent with* expert reading rather than as a crutch.
- *Discussion/Limitations*: the native-author false alarms, as a caution about structure-based AI detection penalising fluent conventional writing. This is the most quotable secondary finding and connects to the paper's applied-linguistics audience.
- *Limitations*: one judge, one generator, informed condition. State plainly that this is an expert upper bound and that the LLM-judge arm supplies the scale.

**Claims to avoid.** Do not write "AI introductions are detectable" without the qualifiers - it is one informed expert, one generator chosen for being the hardest case, and the reader knew the corpus-level tendencies. Do not read the RQ1 null as "move annotations are useless"; the honest statement is "no detectable benefit for an expert at this sample size". Do not present guided-minus-basic as a clean experimental contrast without noting it is between papers, and cite the null on the paired originals (p = 0.671) as the control.

**A title-level sentence you could use:** *prompting a language model with the rhetorical schema of a genre makes its output more schema-conformant than the genre itself, and therefore easier to identify as machine-written.*


## 9. Caveats

- **One judge.** Every rate is a single expert's; there is no inter-rater term and the per-cell n is 30 items (15 papers). Main effects are well estimated, the three-way interaction is not.
- **The judge was informed.** The guidelines carried the corpus-level human-vs-AI move findings, and those were computed on a corpus that includes these items. This arm measures an *informed* expert by design, so it is an upper bound on structure-aware human detection, not a naive-reader baseline.
- **One generator.** claude-opus-4-8 was selected as the structurally closest model, so these rates are the hardest case among the four generators, not an average over them.
- **Mode is between papers.** basic and cars_guided were assigned to different papers, so the RQ2 contrast carries paper variance; the paired-originals check is the control.
- **Reporting.** This report shows counts, accuracy and F1 only. d', criterion, AUC, false-alarm rates, bootstrap CIs and the full regression output are in `human_arm_results.json`.
