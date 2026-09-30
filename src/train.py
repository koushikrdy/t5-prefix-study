"""Fine-tune t5-small on a text-to-text mixture. Plain loop, no Trainer."""
from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import Example


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class _Seq2SeqDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def _collate(tokenizer, max_source_len: int, max_target_len: int):
    def fn(batch: List[Example]):
        enc = tokenizer([b.source for b in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_source_len)
        lab = tokenizer([b.target for b in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_target_len)
        labels = lab["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    return fn


def finetune(model, tokenizer, examples: List[Example], *, batch_size: int,
             lr: float, max_source_len: int, max_target_len: int, seed: int,
             warmup_frac: float, log_every: int = 50) -> Tuple[float, int, float]:
    """Returns (final mean loss over the last 10% of steps, n_steps, seconds)."""
    set_seed(seed)
    loader = DataLoader(
        _Seq2SeqDataset(examples), batch_size=batch_size, shuffle=False,
        collate_fn=_collate(tokenizer, max_source_len, max_target_len))
    total = len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    warmup = max(1, int(total * warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    losses: List[float] = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            tail = losses[-log_every:]
            print(f"      step {step+1}/{total}  loss {sum(tail)/len(tail):.4f}"
                  f"  {(time.time()-t0)/(step+1):.2f}s/step", flush=True)
    tail_n = max(1, total // 10)
    return float(sum(losses[-tail_n:]) / tail_n), total, time.time() - t0
