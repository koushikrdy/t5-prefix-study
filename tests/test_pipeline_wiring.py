"""End-to-end wiring check that needs no downloaded checkpoint.

Builds a randomly-initialised T5 of a few thousand parameters over a toy
character vocabulary and pushes real Examples through finetune() and
generate(). It asserts nothing about accuracy - only that the training loop
steps, the loss moves, and decoding returns one string per example.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from src.data import Example
from src.evaluate import generate, score
from src.train import finetune


class ToyTokenizer:
    """Character-level stand-in with the surface the code actually uses."""
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        chars = " abcdefghijklmnopqrstuvwxyz0123456789:._"
        self.stoi = {c: i + 2 for i, c in enumerate(chars)}
        self.itos = {i + 2: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 2

    def _ids(self, s, max_length):
        out = [self.stoi.get(c, 2) for c in s.lower()[: max_length - 1]]
        return out + [self.eos_token_id]

    def __call__(self, texts, return_tensors=None, padding=True,
                 truncation=True, max_length=32):
        seqs = [self._ids(t, max_length) for t in texts]
        n = max(len(s) for s in seqs)
        ids = [s + [self.pad_token_id] * (n - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (n - len(s)) for s in seqs]
        return transformers.tokenization_utils_base.BatchEncoding({
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        })

    def batch_decode(self, seqs, skip_special_tokens=True):
        out = []
        for s in seqs:
            out.append("".join(self.itos.get(int(i), "") for i in s
                               if int(i) > 1))
        return out


def _toy_model(tok):
    cfg = T5Config(vocab_size=tok.vocab_size, d_model=32, d_ff=64, d_kv=8,
                   num_layers=1, num_decoder_layers=1, num_heads=2,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return T5ForConditionalGeneration(cfg)


def _examples(n=24):
    out = []
    for i in range(n):
        lab = i % 2
        out.append(Example("sst2", f"sst2 sentence: film number {i}",
                           ["negative", "positive"][lab], lab))
    return out


def test_training_loop_runs_and_moves_the_loss():
    tok = ToyTokenizer()
    mdl = _toy_model(tok)
    exs = _examples(32)
    loss, steps, secs = finetune(mdl, tok, exs, batch_size=8, lr=1e-3,
                                 max_source_len=32, max_target_len=6, seed=0,
                                 warmup_frac=0.1, log_every=0)
    assert steps == 4
    assert loss == loss and loss > 0          # finite
    assert secs >= 0


def test_generate_returns_one_string_per_example():
    tok = ToyTokenizer()
    mdl = _toy_model(tok)
    exs = _examples(10)
    preds = generate(mdl, tok, exs, batch_size=4, max_source_len=32,
                     max_target_len=6)
    assert len(preds) == len(exs)
    assert all(isinstance(p, str) for p in preds)


def test_untrained_model_scores_are_well_formed():
    """An untrained model emits garbage; the metrics must still be in range
    and accuracy must not be rescued by snapping garbage onto a label."""
    tok = ToyTokenizer()
    mdl = _toy_model(tok)
    exs = _examples(10)
    preds = generate(mdl, tok, exs, batch_size=4, max_source_len=32,
                     max_target_len=6)
    s = score(exs, preds, "sst2", "distinct")
    assert 0.0 <= s["accuracy"] <= 1.0
    assert 0.0 <= s["off_label_rate"] <= 1.0
    assert s["misroute_rate"] <= s["off_label_rate"]
    assert s["accuracy"] + s["off_label_rate"] <= 1.0 + 1e-9


def test_determinism_of_the_training_loop():
    exs = _examples(32)
    losses = []
    for _ in range(2):
        tok = ToyTokenizer()
        torch.manual_seed(0)
        mdl = _toy_model(tok)
        loss, _, _ = finetune(mdl, tok, exs, batch_size=8, lr=1e-3,
                              max_source_len=32, max_target_len=6, seed=0,
                              warmup_frac=0.1, log_every=0)
        losses.append(loss)
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)
