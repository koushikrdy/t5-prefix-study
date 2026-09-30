# Does the task prefix actually do anything?

Tiparthi Koushik Reddy (1863216), Universität Trier — NLP seminar.

A controlled ablation of one design decision in **T5** (Raffel et al., 2020,
*Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer*): the task prefix prepended to every input.

```
cola sentence: The boy smiled.        ->  acceptable
sst2 sentence: a masterful film .     ->  positive
mrpc sentence1: ... sentence2: ...    ->  equivalent
```

T5's premise is that one model, one loss and one decoding procedure can serve
every task, and the prefix is what tells the model which task it is looking at.
The paper reports only that the exact *wording* of the prefix mattered little.
This repository asks the sharper question — not *which* prefix, but whether it
needs to carry task identity **at all** — and finds that the honest answer
depends on how you measure it.

---

## Table of contents

1. [Findings](#findings)
2. [Research questions and hypotheses](#research-questions-and-hypotheses)
3. [Experimental design](#experimental-design)
4. [Training setup](#training-setup)
5. [Evaluation and metrics](#evaluation-and-metrics)
6. [Results](#results)
7. [Repository layout](#repository-layout)
8. [Reproducing](#reproducing)
9. [Tests](#tests)
10. [Limitations](#limitations)

---

## Findings

| | verdict | evidence |
|---|---|---|
| **H1** prefix is inert single-task | **supported** | SST-2 and MRPC move by ≤ 0.043 across all four conditions |
| **H2** prefix becomes load-bearing multi-task | **rejected** | trained with *no* prefix: SST-2 0.89 vs 0.88, MRPC 0.80 vs 0.78 |
| **H3** disjoint labels leak task identity | **rejected, and inverted** | closing the label channel did not raise the prefix's value — it exposed that disjoint labels had been *overstating* it |

**The headline.** The prefix is not a task router. It is load-bearing only
where the input itself is ambiguous, and the standard way of measuring it —
disjoint verbalizers scored by accuracy — systematically overstates what it
does. Swap the prefix at inference and every off-diagonal cell reads
0.00 under disjoint labels but
0.47 at worst under collided ones. Those zeros
were never the model collapsing; they were the verbalizer scoring any answer
from another task's label space as wrong.

---

## Research questions and hypotheses

**RQ1** In single-task fine-tuning, does replacing the correct prefix with
another task's prefix, an empty prefix, or a nonsense token change anything?

**RQ2** In multi-task fine-tuning, where one model serves three tasks at once,
does the prefix become load-bearing?

**RQ3** If the prefix matters less than T5's design implies, *why*? How much
task identity leaks through the **label space** instead?

RQ3 is the part that is not a re-run. T5's verbalizers are disjoint —
`acceptable/unacceptable`, `positive/negative`, `equivalent/not_equivalent` — so
a multi-task model can infer the task from the output side without reading the
prefix. Any measurement that leaves this channel open is measuring the prefix
and the verbalizer together.

---

## Experimental design

### Tasks — chosen for structure, not coverage

| task | source | input shape |
|---|---|---|
| CoLA | Warstadt et al. (2019) | one sentence |
| SST-2 | Socher et al. (2013) | one sentence |
| MRPC | Dolan & Brockett (2005) | two sentences |

CoLA and SST-2 have **identical input shapes** — a single sentence behind a
`sentence:` marker — so once the task token is removed they are literally the
same string. MRPC has two fields and stays identifiable by shape. That
asymmetry is a built-in control: if the prefix only matters where the input is
ambiguous, CoLA should break and MRPC should not.

### Prefix conditions

Only the leading task token changes. T5's field markers (`sentence:`,
`sentence1:`, `sentence2:`) always stay, so the ablation removes task identity
and not input structure.

| condition | CoLA input | discriminative? | truthful? |
|---|---|---|---|
| `correct` | `cola sentence: ...` | yes | yes |
| `wrong` | `sst2 sentence: ...` | yes | no |
| `empty` | `sentence: ...` | no | — |
| `nonsense` | `zorbex sentence: ...` | no | — |

`wrong` uses a deterministic rotation (cola → sst2 → mrpc → cola). In the
multi-task regime it still gives every task a *unique* token, which is what
separates "the prefix's meaning matters" from "the prefix just needs to be
distinct".

### Regimes

| | regime | tasks | labels | conditions | runs/seed |
|---|---|---|---|---|---|
| **A** | single-task | one at a time | disjoint | 4 | 12 |
| **B** | multi-task | all three | disjoint | 4 | 4 |
| **C** | multi-task | all three | collided (`yes`/`no`) | correct, empty | 2 |

Regime C re-verbalizes every task onto `yes`/`no`, closing the label channel so
that whatever the prefix is worth has to show up there.

### Inference-time prefix swap

Every `correct`-trained model is additionally evaluated with each task token
substituted in at test time (`cola`, `sst2`, `mrpc`, none, `zorbex`). Training
is unchanged, so this isolates what the prefix does at inference from what it
does during learning.

---

## Training setup

| | |
|---|---|
| model | `t5-small`, 60.5M parameters, released checkpoint |
| optimiser | AdamW, no weight decay |
| learning rate | 1e-3, linear warmup (6% of steps) then linear decay |
| gradient clipping | 1.0 (L2 norm) |
| batch size | 32 |
| max source length | 64 tokens |
| max target length | 6 tokens |
| per-task exposure | 3000 examples × 2 epochs = 6000 |
| steps | 188 per single-task run, 563 per multi-task run |
| precision | fp32, CPU |
| seeds | 1 (see [Limitations](#limitations)) |

**Per-task exposure is held constant across regimes.** A single-task run and a
multi-task run each see every one of their training tasks the same number of
times, so a multi-task run simply costs three times the optimizer steps.
Holding examples-per-task fixed rather than total-steps fixed is what keeps
"how much of this task did the model see" comparable, which is the quantity the
single-vs-multi comparison depends on.

**Why 3000 × 2 and lr 1e-3.** An earlier budget of 1600 × 2 is only 100
optimizer steps and left CoLA predicting its majority class outright — accuracy
0.695 against a 0.691 baseline, MCC ≈ 0. A probe at 4000 × 3 reached MCC 0.299;
3000 × 2 at the higher learning rate sits close to that at a third of the
compute. Measuring a prefix effect on a task the model has not learned measures
nothing.

The released checkpoint means the prefixes were **already seen during T5's
multi-task pre-training**, so the `correct` condition starts with an advantage
a from-scratch model would not have. This works *against* the finding that the
prefix is usually inert, which makes that finding safer rather than weaker.

Cost: 18 runs, ~193 minutes of wall-clock training on 2 CPU cores.

---

## Evaluation and metrics

Predictions come from **free greedy generation**, not from scoring the two
label strings against each other. Constrained scoring would make it impossible
for the model to answer the wrong question — precisely the failure the prefix
is supposed to prevent.

| metric | definition |
|---|---|
| **accuracy** | exact string match against the gold verbalizer. An off-label generation counts as wrong; it is never snapped to the nearer label. |
| **off-label rate** | share of generations that are not one of the target task's two labels. |
| **misroute rate** | share that are off-label for this task but *are* a label of another task — the model answered a different question. Defined only under disjoint labels. |
| **MCC** | Matthews correlation, reported for CoLA. |

**CoLA is scored by MCC, not accuracy.** Its validation set is 69% one class,
so a model that answers "acceptable" to everything scores 0.69 accuracy with
zero signal. GLUE scores CoLA by MCC and so does this repository: evaluation
carries a per-task *primary* metric, and the paired prefix-cost comparison is
computed on it. `tests/test_rendering.py` pins this behaviour.

Every cell records the majority-class baseline alongside the score.
Evaluation uses 400 held-out validation examples per task, sampled with a fixed
seed.

---

## Results

Single seed. Primary metric: MCC for CoLA, accuracy for SST-2 and MRPC.

### A — single-task, disjoint labels

| task | correct | wrong | none | nonsense |
|---|---|---|---|---|
| CoLA (MCC) | 0.240 | 0.000 | 0.228 | 0.180 |
| SST-2 | 0.887 | 0.870 | 0.882 | 0.877 |
| MRPC | 0.760 | 0.790 | 0.802 | 0.745 |

SST-2 and MRPC are flat. CoLA is the exception, and in an informative
direction: a *wrong but real* prefix drives MCC to
0.00, while a meaningless one barely costs anything
(0.18 vs 0.24 correct).
`sst2` is already bound to sentiment from pre-training and CoLA has to unlearn
it; `zorbex` is a blank slate.

### B — multi-task, disjoint labels

| task | correct | wrong | none | nonsense |
|---|---|---|---|---|
| CoLA (MCC) | 0.105 | 0.105 | 0.000 | 0.072 |
| SST-2 | 0.880 | 0.892 | 0.892 | 0.887 |
| MRPC | 0.780 | 0.777 | 0.800 | 0.782 |

H2 predicted a collapse without a prefix. It does not happen for SST-2 or MRPC.
Only CoLA — the task whose input shape collides with SST-2 — needs the prefix,
and without it 99% of its outputs land in
another task's label space.

### C — multi-task, collided labels (`yes`/`no`)

| task | correct | none |
|---|---|---|
| CoLA (MCC) | 0.093 | 0.105 |
| SST-2 | 0.860 | 0.840 |
| MRPC | 0.730 | 0.745 |

Closing the label channel was supposed to make the prefix matter more. It did
not move at all — CoLA even improves slightly. H3 is rejected.

### Inference-time prefix swap

Accuracy when the token is swapped at test time on the multi-task model:

| evaluated on | `cola` | `sst2` | `mrpc` | none | `zorbex` |
|---|---|---|---|---|---|
| **disjoint labels** |
| CoLA | 0.69 | 0.00 | 0.00 | 0.01 | 0.00 |
| SST-2 | 0.00 | 0.88 | 0.11 | 0.86 | 0.73 |
| MRPC | 0.00 | 0.68 | 0.78 | 0.79 | 0.79 |
| **collided labels** |
| CoLA | 0.68 | 0.47 | 0.48 | 0.57 | 0.56 |
| SST-2 | 0.54 | 0.86 | 0.84 | 0.82 | 0.81 |
| MRPC | 0.70 | 0.68 | 0.73 | 0.72 | 0.72 |

This table is the contribution. The same model and the same swap look
catastrophic under disjoint labels and merely degraded under collided ones. The
difference is not the model — it is the scoring.

---

## Repository layout

```
t5-prefix-study/
├── README.md                    this file
├── SUBMISSION.md                poster submission requirements and checklist
├── requirements.txt
├── .gitignore                   excludes t5_assets/, .venv/, caches
│
├── run_experiment.py            the pipeline: builds the run grid, trains,
│                                evaluates, appends to results/runs.jsonl
│
├── src/
│   ├── __init__.py
│   ├── config.py                tasks, verbalizers, prefix conditions,
│   │                            the run grid, TrainConfig
│   ├── data.py                  GLUE CSV -> text-to-text under a condition
│   ├── train.py                 fine-tuning loop (AdamW, warmup + decay)
│   ├── evaluate.py              greedy decoding; accuracy, off-label,
│   │                            misroute, MCC, per-task primary metric
│   ├── analysis.py              runs.jsonl -> aggregated tables
│   └── report.py                tables + figures (screen and poster variants)
│
├── scripts/
│   └── fetch_t5_assets.py       downloads t5-small and the three GLUE tasks
│
├── tests/
│   ├── test_rendering.py        input construction, metrics, grid shape
│   └── test_pipeline_wiring.py  train/decode loop on a toy T5, no checkpoint
│
├── appendix/
│   └── references.md            the reference list in plain text; the typeset
│                                version is poster/appendix_references.pdf
│
├── t5_assets/                   (gitignored) checkpoint + GLUE CSVs
│   ├── t5-small/
│   ├── cola_train.csv, cola_validation.csv
│   ├── sst2_train.csv, sst2_validation.csv
│   └── mrpc_train.csv, mrpc_validation.csv
│
├── results/
│   ├── runs.jsonl               one JSON object per run — the source of truth
│   ├── run.log                  stdout of the run, including per-step losses
│   ├── summary.json             every headline number, machine-readable
│   ├── eval_long.csv            one row per (run x task)
│   ├── swap_long.csv            one row per (run x task x inference token)
│   ├── table_single.csv         aggregated regime A
│   ├── table_multi_distinct.csv aggregated regime B
│   ├── table_multi_collided.csv aggregated regime C
│   └── prefix_cost_*.csv        per-condition deltas from `correct`, by seed
│
├── figures/
│   ├── accuracy_panels.png      per-task results, each on its GLUE metric
│   ├── swap_grid.png            prefix swap under both label schemes
│   ├── prefix_cost.png          paired deltas from the correct prefix
│   ├── misroute.png             share of answers to a different question
│   └── poster/                  column-width variants, larger type
│
└── poster/
    ├── poster.html              A1 portrait source (594 x 841 mm)
    ├── 1863216_poster.pdf       rendered poster
    ├── appendix.html            reference list source
    ├── appendix_references.pdf  rendered appendix
    ├── copy.md                  poster copy in plain text
    └── finalise_submission.py   merges the signed declaration, builds the ZIP
```

`results/runs.jsonl` is the source of truth. Every table, figure and number on
the poster is derived from it by `python -m src.report`; nothing is transcribed
by hand.

---

## Reproducing

`huggingface.co` may be unreachable from your environment, so the fetch step is
separate from the experiment.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. assets (~250 MB) — needs huggingface.co
.venv/bin/python scripts/fetch_t5_assets.py

# 2. tests — no checkpoint required
.venv/bin/python -m pytest tests/ -q

# 3. the grid (~3 h per seed on 2 CPU cores)
.venv/bin/python run_experiment.py --seeds 0 1 2

# 4. tables and figures
.venv/bin/python -m src.report
```

### Useful flags

| flag | effect |
|---|---|
| `--smoke` | one tiny run, checks the wiring end to end in under a minute |
| `--only single-cola` | substring filter on `run_id` |
| `--seeds 0 1 2` | which seeds to run (outermost loop, so seed 0 completes first) |
| `--max-minutes N` | stop cleanly after N minutes; rerun to resume |
| `--no-swap` | skip the inference-time swap grids |
| `--train-per-task`, `--epochs`, `--lr`, `--batch-size`, `--eval-n` | override the training budget |

`run_experiment.py` appends one JSON object per run to `results/runs.jsonl` and
skips runs already present, so it is safe to interrupt and restart.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 18 tests
```

`test_rendering.py` needs nothing but the source. It pins the claims the design
rests on: that `empty` removes the task token and not the field markers, that
CoLA and SST-2 are byte-identical without the token while MRPC is not, that the
label spaces are disjoint under one scheme and shared under the other, that an
off-label generation is never snapped onto a label, and that a majority-class
predictor scores 0.70 accuracy and exactly 0.00 MCC on CoLA.

`test_pipeline_wiring.py` builds a randomly-initialised T5 of a few thousand
parameters over a toy character vocabulary and pushes real `Example` objects
through `finetune()` and `generate()`. It asserts nothing about accuracy — only
that the loop steps, the loss is finite, decoding returns one string per
example, and two identical runs produce identical loss.

---

## Limitations

- **One seed.** Every number here is a single draw. The CoLA MCC of
  0.00 under a wrong prefix is a degenerate-prediction
  signature that replication should confirm before it is believed. The grid is
  written for three seeds and resumes with `--seeds 0 1 2`.
- **One model size.** `t5-small`, 60.5M parameters. T5's own results are
  strongly scale-dependent and nothing here should be read as holding at 3B
  or 11B.
- **Three tasks, all binary classification.** T5's prefix also has to separate
  translation, summarization and regression; three binary classifiers are an
  easier disambiguation problem than the real mixture.
- **Fine-tuning, not pre-training.** The prefixes were already seen during T5's
  multi-task pre-training.
- **Accuracy is not the only thing a prefix could buy.** Sample efficiency,
  robustness under distribution shift and calibration are untouched here.

---

## References

Full list in `appendix/references.md` and in `poster/appendix_references.pdf`.
The primary source is Raffel, C. et al. (2020), *Exploring the Limits of
Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21(140), 1–67.
