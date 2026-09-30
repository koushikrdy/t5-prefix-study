"""Turn the GLUE CSVs into T5's text-to-text format under a prefix condition."""
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import TASKS, Task, prefix_token


@dataclass
class Example:
    task: str
    source: str
    target: str
    label: int


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _body(task: Task, row: Dict[str, str]) -> str:
    parts = []
    for fld, marker in zip(task.fields, task.markers):
        parts.append(f"{marker} {row[fld].strip()}")
    return " ".join(parts)


def render_source(task_name: str, row: Dict[str, str], condition: str,
                  override_token: Optional[str] = None) -> str:
    """Build the model input.

    The task token is the only thing the condition changes. Field markers stay,
    so 'empty' removes task identity rather than input structure.
    """
    task = TASKS[task_name]
    token = override_token if override_token is not None else prefix_token(condition, task_name)
    body = _body(task, row)
    return f"{token} {body}".strip() if token else body


def verbalizer(task_name: str, label_scheme: str) -> List[str]:
    task = TASKS[task_name]
    return task.labels_distinct if label_scheme == "distinct" else task.labels_collided


def load_split(data_dir: str, task_name: str, split: str) -> List[Dict[str, str]]:
    path = os.path.join(data_dir, f"{task_name}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing {path} - run scripts/fetch_t5_assets.py on a machine that "
            f"can reach huggingface.co")
    return _read_csv(path)


def build_examples(data_dir: str, task_name: str, split: str, condition: str,
                   label_scheme: str, n: Optional[int], seed: int,
                   override_token: Optional[str] = None) -> List[Example]:
    rows = load_split(data_dir, task_name, split)
    if n is not None and n < len(rows):
        rng = random.Random(seed * 1000 + hash(task_name) % 997)
        rows = rng.sample(rows, n)
    verb = verbalizer(task_name, label_scheme)
    out = []
    for r in rows:
        lab = int(r["label"])
        out.append(Example(
            task=task_name,
            source=render_source(task_name, r, condition, override_token),
            target=verb[lab],
            label=lab,
        ))
    return out


def build_training_set(data_dir: str, tasks: List[str], condition: str,
                       label_scheme: str, per_task: int, epochs: int,
                       seed: int) -> List[Example]:
    """Per-task exposure is held constant across regimes.

    A single-task run and a multi-task run both see each of their training
    tasks `per_task * epochs` times, so a multi-task run simply costs more
    steps. That keeps 'how much of this task did the model see' fixed, which is
    the quantity the comparison depends on.
    """
    pool: List[Example] = []
    for t in tasks:
        base = build_examples(data_dir, t, "train", condition, label_scheme,
                              per_task, seed)
        for _ in range(epochs):
            pool.extend(base)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool
