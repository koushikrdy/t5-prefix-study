"""Tasks, prefix conditions and training regimes for the prefix ablation.

The experiment asks a narrow question about one T5 design decision: the task
prefix that is prepended to every input ("cola sentence: ..."). Raffel et al.
report that the exact wording of the prefix did not matter much; they do not
report what happens when it carries no task information at all, and they do not
separate the single-task case from the multi-task case.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
# `body` is everything after the task token. It always keeps T5's own field
# markers ("sentence:", "sentence1:", "sentence2:") so that removing the task
# token removes *only* task identity and not the structure of the input. The
# body of CoLA and SST-2 is identical by construction, which is the point:
# with the task token gone, nothing in the input distinguishes them.

@dataclass(frozen=True)
class Task:
    name: str
    task_token: str            # the leading token T5 uses, e.g. "cola"
    fields: List[str]          # csv columns used
    markers: List[str]         # field markers, aligned with `fields`
    labels_distinct: List[str]  # verbalizer under the distinct label scheme
    labels_collided: List[str]  # verbalizer under the collided label scheme
    metric: str                 # "accuracy" or "mcc_and_accuracy"


TASKS: Dict[str, Task] = {
    "cola": Task(
        name="cola",
        task_token="cola",
        fields=["text_a"],
        markers=["sentence:"],
        labels_distinct=["unacceptable", "acceptable"],   # index = GLUE label
        labels_collided=["no", "yes"],
        metric="mcc_and_accuracy",
    ),
    "sst2": Task(
        name="sst2",
        task_token="sst2",
        fields=["text_a"],
        markers=["sentence:"],
        labels_distinct=["negative", "positive"],
        labels_collided=["no", "yes"],
        metric="accuracy",
    ),
    "mrpc": Task(
        name="mrpc",
        task_token="mrpc",
        fields=["text_a", "text_b"],
        markers=["sentence1:", "sentence2:"],
        labels_distinct=["not_equivalent", "equivalent"],
        labels_collided=["no", "yes"],
        metric="accuracy",
    ),
}

TASK_ORDER = ["cola", "sst2", "mrpc"]


# --------------------------------------------------------------------------
# Prefix conditions
# --------------------------------------------------------------------------
# Each condition is a rule for choosing the leading task token given the task
# an example actually belongs to.

NONSENSE_TOKEN = "zorbex"
SHARED_TOKEN = "task"

# deterministic rotation used by the "wrong" condition
WRONG_MAP = {"cola": "sst2", "sst2": "mrpc", "mrpc": "cola"}


def prefix_token(condition: str, task_name: str) -> str:
    """Return the leading token for `task_name` under `condition` ('' = none)."""
    if condition == "correct":
        return TASKS[task_name].task_token
    if condition == "wrong":
        return TASKS[WRONG_MAP[task_name]].task_token
    if condition == "empty":
        return ""
    if condition == "nonsense":
        return NONSENSE_TOKEN
    if condition == "shared":
        return SHARED_TOKEN
    raise ValueError(f"unknown prefix condition: {condition!r}")


CONDITIONS_SINGLE = ["correct", "wrong", "empty", "nonsense"]
# In the multi-task regime the conditions split 2x2 on whether the prefix is
# *discriminative* (a different token per task) and whether it is *truthful*:
#   correct   discriminative + truthful
#   wrong     discriminative + misleading   <- isolates meaning from distinctness
#   empty     not discriminative
#   nonsense  not discriminative (one shared token for every task)
# "shared" was dropped: a single "task" token for every example is
# non-discriminative in exactly the way "nonsense" already is, so it buys a
# run's worth of compute and no new contrast.
CONDITIONS_MULTI = ["correct", "wrong", "empty", "nonsense"]

# tokens that can appear at inference time in the prefix-swap grid
SWAP_TOKENS = ["cola", "sst2", "mrpc", "", NONSENSE_TOKEN]


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
@dataclass
class RunSpec:
    run_id: str
    regime: str                  # "single" | "multi"
    train_tasks: List[str]
    condition: str
    label_scheme: str            # "distinct" | "collided"
    seed: int
    eval_tasks: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.eval_tasks:
            self.eval_tasks = list(self.train_tasks)


def build_run_grid(seeds: List[int]) -> List[RunSpec]:
    """The full set of training runs.

    Block A  single-task, distinct labels   3 tasks x 4 conditions
    Block B  multi-task,  distinct labels   5 conditions
    Block C  multi-task,  collided labels   3 conditions

    Block C exists because the distinct verbalizers are themselves a channel
    for task identity: a model can pick the right label space from the output
    side without ever reading the prefix. Collapsing every task onto yes/no
    closes that channel, so whatever the prefix is worth has to show up there.
    """
    runs: List[RunSpec] = []
    for seed in seeds:
        for task in TASK_ORDER:
            for cond in CONDITIONS_SINGLE:
                runs.append(RunSpec(
                    run_id=f"single-{task}-{cond}-distinct-s{seed}",
                    regime="single", train_tasks=[task], condition=cond,
                    label_scheme="distinct", seed=seed))
        for cond in CONDITIONS_MULTI:
            runs.append(RunSpec(
                run_id=f"multi-all-{cond}-distinct-s{seed}",
                regime="multi", train_tasks=list(TASK_ORDER), condition=cond,
                label_scheme="distinct", seed=seed))
        for cond in ["correct", "empty"]:
            runs.append(RunSpec(
                run_id=f"multi-all-{cond}-collided-s{seed}",
                regime="multi", train_tasks=list(TASK_ORDER), condition=cond,
                label_scheme="collided", seed=seed))
    return runs


@dataclass
class TrainConfig:
    model_dir: str = "t5_assets/t5-small"
    data_dir: str = "t5_assets"
    # per-task exposure is held equal across regimes: a single-task run and a
    # multi-task run both see each of their training tasks this many times.
    # 1600x2 left CoLA at its majority class (accuracy 0.695 vs a 0.691
    # baseline, MCC ~ 0) - 100 optimizer steps is simply not enough for an
    # acceptability task. A probe at 4000x3 reached MCC 0.299; 3000x2 with a
    # higher learning rate sits close to that at a third of the compute.
    train_per_task: int = 3000
    epochs: int = 2
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 1e-3
    max_source_len: int = 64
    max_target_len: int = 6
    eval_n: int = 400
    warmup_frac: float = 0.06
