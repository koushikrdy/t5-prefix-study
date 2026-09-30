#!/usr/bin/env python3
"""Prefix ablation on t5-small.

  python run_experiment.py --seeds 0 1
  python run_experiment.py --smoke          # 1 tiny run, to check the wiring
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import (TASK_ORDER, SWAP_TOKENS, RunSpec, TrainConfig,
                        build_run_grid)
from src.data import build_training_set
from src.evaluate import evaluate_task
from src.train import finetune, set_seed

RESULTS = "results"


def _fresh_model(cfg: TrainConfig, seed: int):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_dir, legacy=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_dir)
    return tok, mdl


def run_one(spec: RunSpec, cfg: TrainConfig, do_swap: bool) -> Dict:
    print(f"\n=== {spec.run_id} ===", flush=True)
    tok, mdl = _fresh_model(cfg, spec.seed)
    train = build_training_set(cfg.data_dir, spec.train_tasks, spec.condition,
                               spec.label_scheme, cfg.train_per_task,
                               cfg.epochs, spec.seed)
    print(f"    train examples {len(train)}  ex: {train[0].source[:90]!r} -> "
          f"{train[0].target!r}", flush=True)
    loss, steps, secs = finetune(
        mdl, tok, train, batch_size=cfg.batch_size, lr=cfg.lr,
        max_source_len=cfg.max_source_len, max_target_len=cfg.max_target_len,
        seed=spec.seed, warmup_frac=cfg.warmup_frac)

    rec: Dict = {
        "run_id": spec.run_id, "regime": spec.regime,
        "train_tasks": spec.train_tasks, "condition": spec.condition,
        "label_scheme": spec.label_scheme, "seed": spec.seed,
        "train_examples": len(train), "steps": steps,
        "train_seconds": round(secs, 1), "final_loss": round(loss, 4),
        "eval": {}, "swap": {},
    }

    # Evaluation uses the same prefix condition the model was trained under:
    # the question is what the model can do when deployed the way it was built.
    for t in spec.eval_tasks:
        rec["eval"][t] = evaluate_task(
            mdl, tok, cfg.data_dir, t, spec.condition, spec.label_scheme,
            cfg.eval_n, spec.seed, cfg.eval_batch_size, cfg.max_source_len,
            cfg.max_target_len)
        e = rec["eval"][t]
        print(f"    {t}: acc {e['accuracy']:.3f}  off-label {e['off_label_rate']:.3f}"
              f"  misroute {e['misroute_rate']:.3f}", flush=True)

    # Prefix swap at inference: only informative for a model that was trained
    # with correct prefixes, so we run it there and nowhere else.
    if do_swap and spec.condition == "correct":
        for t in spec.eval_tasks:
            rec["swap"][t] = {}
            for tokname in SWAP_TOKENS:
                key = tokname if tokname else "(none)"
                rec["swap"][t][key] = evaluate_task(
                    mdl, tok, cfg.data_dir, t, spec.condition,
                    spec.label_scheme, cfg.eval_n, spec.seed,
                    cfg.eval_batch_size, cfg.max_source_len,
                    cfg.max_target_len, override_token=tokname)
            print(f"    swap[{t}]: " + "  ".join(
                f"{k}={v['accuracy']:.3f}" for k, v in rec["swap"][t].items()),
                flush=True)

    del mdl
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data-dir", default="t5_assets")
    ap.add_argument("--model-dir", default="t5_assets/t5-small")
    ap.add_argument("--train-per-task", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(RESULTS, "runs.jsonl"))
    ap.add_argument("--only", default=None, help="substring filter on run_id")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # Background jobs here do not survive a container restart, so the pipeline
    # is driven in timeboxed chunks instead: each invocation stops cleanly
    # before the wall-clock budget and the next one resumes from runs.jsonl.
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))

    cfg = TrainConfig(model_dir=args.model_dir, data_dir=args.data_dir,
                      train_per_task=args.train_per_task, epochs=args.epochs,
                      eval_n=args.eval_n, batch_size=args.batch_size, lr=args.lr)
    if args.smoke:
        cfg.train_per_task, cfg.epochs, cfg.eval_n = 64, 1, 32
        specs = [RunSpec(run_id="smoke-multi-correct", regime="multi",
                         train_tasks=list(TASK_ORDER), condition="correct",
                         label_scheme="distinct", seed=0)]
    else:
        specs = build_run_grid(args.seeds)
        if args.only:
            specs = [s for s in specs if args.only in s.run_id]

    os.makedirs(RESULTS, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["run_id"])

    print(f"{len(specs)} runs, {len(done)} already done", flush=True)
    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        if spec.run_id in done:
            print(f"[{i}/{len(specs)}] skip {spec.run_id}", flush=True)
            continue
        print(f"[{i}/{len(specs)}] {spec.run_id}  "
              f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
        rec = run_one(spec, cfg, do_swap=not args.no_swap)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if args.max_minutes and (time.time() - t0) / 60 >= args.max_minutes:
            left = sum(1 for s2 in specs[i:] if s2.run_id not in done)
            print(f"\nbudget reached after {(time.time()-t0)/60:.1f} min; "
                  f"{left} runs still to do - rerun to resume", flush=True)
            return
    print(f"\nall done in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
