"""Aggregate results/runs.jsonl into the tables the poster needs."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from .config import TASK_ORDER, CONDITIONS_SINGLE, CONDITIONS_MULTI


def load_runs(path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for task, e in r["eval"].items():
                rows.append({
                    "run_id": r["run_id"], "regime": r["regime"],
                    "condition": r["condition"], "label_scheme": r["label_scheme"],
                    "seed": r["seed"], "task": task,
                    "accuracy": e["accuracy"], "off_label_rate": e["off_label_rate"],
                    "misroute_rate": e["misroute_rate"],
                    "majority": e["majority_baseline"],
                    "mcc": e.get("mcc", float("nan")),
                    "primary": e.get("primary", e["accuracy"]),
                    "primary_metric": e.get("primary_metric", "accuracy"),
                    "primary_floor": e.get("primary_floor",
                                           e["majority_baseline"]),
                    "steps": r["steps"], "train_seconds": r["train_seconds"],
                    "final_loss": r["final_loss"],
                })
    return pd.DataFrame(rows)


def load_swap(path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for task, grid in r.get("swap", {}).items():
                for tok, e in grid.items():
                    rows.append({
                        "run_id": r["run_id"], "regime": r["regime"],
                        "label_scheme": r["label_scheme"], "seed": r["seed"],
                        "task": task, "inference_token": tok,
                        "accuracy": e["accuracy"],
                        "off_label_rate": e["off_label_rate"],
                        "misroute_rate": e["misroute_rate"],
                    })
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    g = df.groupby(keys).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        primary_mean=("primary", "mean"), primary_sd=("primary", "std"),
        off_mean=("off_label_rate", "mean"), misroute_mean=("misroute_rate", "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    return g.round(4)


def table_single(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.regime == "single") & (df.label_scheme == "distinct")]
    t = _agg(d, ["task", "condition"])
    t["condition"] = pd.Categorical(t["condition"], CONDITIONS_SINGLE, ordered=True)
    t["task"] = pd.Categorical(t["task"], TASK_ORDER, ordered=True)
    return t.sort_values(["task", "condition"])


def table_multi(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    d = df[(df.regime == "multi") & (df.label_scheme == scheme)]
    t = _agg(d, ["task", "condition"])
    order = [c for c in CONDITIONS_MULTI if c in set(t.condition)]
    t["condition"] = pd.Categorical(t["condition"], order, ordered=True)
    t["task"] = pd.Categorical(t["task"], TASK_ORDER, ordered=True)
    return t.sort_values(["task", "condition"])


def prefix_cost(df: pd.DataFrame, regime: str, scheme: str) -> pd.DataFrame:
    """Accuracy under each condition minus accuracy under 'correct',
    per task and seed, then averaged. Paired by seed, so the seed-to-seed
    variance of the baseline does not leak into the difference."""
    d = df[(df.regime == regime) & (df.label_scheme == scheme)]
    base = d[d.condition == "correct"].set_index(["task", "seed"])["primary"]
    rows = []
    for _, r in d.iterrows():
        b = base.get((r["task"], r["seed"]))
        if b is None:
            continue
        rows.append({"task": r["task"], "condition": r["condition"],
                     "seed": r["seed"], "delta": r["primary"] - b,
                     "metric": r["primary_metric"]})
    out = pd.DataFrame(rows)
    g = out.groupby(["task", "condition"]).agg(
        delta_mean=("delta", "mean"), delta_sd=("delta", "std"),
        n=("delta", "size")).reset_index()
    return g.round(4)


def write_all(runs_path: str, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df = load_runs(runs_path)
    sw = load_swap(runs_path)
    paths = {}
    for name, frame in [
        ("eval_long", df),
        ("table_single", table_single(df)),
        ("table_multi_distinct", table_multi(df, "distinct")),
        ("table_multi_collided", table_multi(df, "collided")),
        ("prefix_cost_single", prefix_cost(df, "single", "distinct")),
        ("prefix_cost_multi_distinct", prefix_cost(df, "multi", "distinct")),
        ("prefix_cost_multi_collided", prefix_cost(df, "multi", "collided")),
        ("swap_long", sw),
    ]:
        p = os.path.join(out_dir, f"{name}.csv")
        frame.to_csv(p, index=False)
        paths[name] = p
    return paths
