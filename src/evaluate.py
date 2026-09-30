"""Greedy decoding plus the three things we measure on each generation."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .config import TASKS, TASK_ORDER
from .data import Example, build_examples, verbalizer


def _matthews(tp: int, tn: int, fp: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(num / den) if den > 0 else 0.0


@torch.no_grad()
def generate(model, tokenizer, examples: List[Example], batch_size: int,
             max_source_len: int, max_target_len: int) -> List[str]:
    model.eval()
    outs: List[str] = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        enc = tokenizer([e.source for e in chunk], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_source_len)
        gen = model.generate(**enc, max_new_tokens=max_target_len,
                             num_beams=1, do_sample=False)
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return [o.strip().lower() for o in outs]


def score(examples: List[Example], preds: List[str], task_name: str,
          label_scheme: str) -> Dict[str, float]:
    """accuracy, off-label rate, misroute rate, and MCC where GLUE uses it.

    off_label  the generation is not one of this task's two labels. An
               off-label generation counts as wrong for accuracy; it is never
               silently mapped onto the nearer label, because doing so would
               hide exactly the failure the prefix is supposed to prevent.
    misroute   the generation is off-label for this task but *is* a label of
               one of the other tasks, i.e. the model answered a different
               question. Only defined under the distinct label scheme, where
               the label spaces are disjoint.
    """
    own = [v.lower() for v in verbalizer(task_name, label_scheme)]
    foreign = set()
    if label_scheme == "distinct":
        for t in TASK_ORDER:
            if t != task_name:
                foreign.update(v.lower() for v in TASKS[t].labels_distinct)
        foreign -= set(own)

    correct = off = misroute = 0
    tp = tn = fp = fn = 0
    for ex, p in zip(examples, preds):
        gold = ex.target.lower()
        if p not in own:
            off += 1
            if p in foreign:
                misroute += 1
            pred_label = -1
        else:
            pred_label = own.index(p)
            if p == gold:
                correct += 1
        if pred_label == 1 and ex.label == 1:
            tp += 1
        elif pred_label == 0 and ex.label == 0:
            tn += 1
        elif pred_label == 1 and ex.label == 0:
            fp += 1
        elif pred_label == 0 and ex.label == 1:
            fn += 1
        # off-label predictions enter MCC as neither; they depress it via n

    n = len(examples)
    out = {
        "n": float(n),
        "accuracy": correct / n if n else 0.0,
        "off_label_rate": off / n if n else 0.0,
        "misroute_rate": misroute / n if n else 0.0,
        "majority_baseline": _majority(examples),
    }
    if TASKS[task_name].metric == "mcc_and_accuracy":
        out["mcc"] = _matthews(tp, tn, fp, fn)
        # GLUE scores CoLA by MCC. Accuracy there is close to unreadable: a
        # model that predicts "acceptable" for everything scores ~0.69.
        out["primary"] = out["mcc"]
        out["primary_metric"] = "mcc"
        out["primary_floor"] = 0.0          # MCC is 0 at chance
    else:
        out["primary"] = out["accuracy"]
        out["primary_metric"] = "accuracy"
        out["primary_floor"] = out["majority_baseline"]
    return out


def _majority(examples: List[Example]) -> float:
    if not examples:
        return 0.0
    ones = sum(1 for e in examples if e.label == 1)
    return max(ones, len(examples) - ones) / len(examples)


def evaluate_task(model, tokenizer, data_dir: str, task_name: str,
                  condition: str, label_scheme: str, n: int, seed: int,
                  batch_size: int, max_source_len: int, max_target_len: int,
                  override_token: Optional[str] = None) -> Dict[str, float]:
    exs = build_examples(data_dir, task_name, "validation", condition,
                         label_scheme, n, seed, override_token=override_token)
    preds = generate(model, tokenizer, exs, batch_size, max_source_len,
                     max_target_len)
    res = score(exs, preds, task_name, label_scheme)
    res["_preds_sample"] = preds[:5]
    return res
