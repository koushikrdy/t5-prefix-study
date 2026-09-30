"""Tests that do not need the checkpoint: input construction and scoring."""
import os, sys, csv, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.config import TASKS, prefix_token, WRONG_MAP, build_run_grid
from src.data import render_source, verbalizer, build_examples, Example
from src.evaluate import score


ROW1 = {"text_a": "The boy smiled.", "text_b": "", "label": "1"}
ROW2 = {"text_a": "A man is playing.", "text_b": "A guy plays.", "label": "1"}


def test_correct_prefix_matches_t5_format():
    assert render_source("cola", ROW1, "correct") == "cola sentence: The boy smiled."
    assert render_source("sst2", ROW1, "correct") == "sst2 sentence: The boy smiled."
    assert render_source("mrpc", ROW2, "correct") == (
        "mrpc sentence1: A man is playing. sentence2: A guy plays.")


def test_empty_removes_only_the_task_token():
    """Field markers must survive, or we would be ablating two things at once."""
    assert render_source("cola", ROW1, "empty") == "sentence: The boy smiled."
    assert render_source("mrpc", ROW2, "empty") == (
        "sentence1: A man is playing. sentence2: A guy plays.")


def test_cola_and_sst2_are_indistinguishable_without_the_token():
    """The core structural fact the experiment leans on."""
    assert render_source("cola", ROW1, "empty") == render_source("sst2", ROW1, "empty")
    assert render_source("cola", ROW1, "correct") != render_source("sst2", ROW1, "correct")


def test_mrpc_stays_distinguishable_without_the_token():
    """MRPC has two fields, so its shape identifies it even with no prefix.
    Any prefix effect on MRPC therefore has a different explanation."""
    assert render_source("mrpc", ROW2, "empty") != render_source("cola", ROW1, "empty")


def test_wrong_condition_uses_another_real_task_token():
    for t, other in WRONG_MAP.items():
        assert prefix_token("wrong", t) == TASKS[other].task_token
        assert prefix_token("wrong", t) != TASKS[t].task_token


def test_nonsense_token_is_not_a_task_token():
    real = {TASKS[t].task_token for t in TASKS}
    assert prefix_token("nonsense", "cola") not in real


def test_label_spaces_are_disjoint_under_distinct_and_shared_under_collided():
    d = [set(v.lower() for v in verbalizer(t, "distinct")) for t in TASKS]
    assert d[0].isdisjoint(d[1]) and d[1].isdisjoint(d[2]) and d[0].isdisjoint(d[2])
    c = [set(verbalizer(t, "collided")) for t in TASKS]
    assert c[0] == c[1] == c[2]


def test_cola_primary_metric_is_mcc_not_accuracy():
    """CoLA validation is ~69% one class, so a majority-class predictor scores
    ~0.69 accuracy with zero real signal. GLUE scores it by MCC; so do we."""
    exs = [Example("cola", "x", "acceptable", 1)] * 7 + \
          [Example("cola", "x", "unacceptable", 0)] * 3
    always_acceptable = ["acceptable"] * 10
    s = score(exs, always_acceptable, "cola", "distinct")
    assert s["accuracy"] == pytest.approx(0.7)
    assert s["mcc"] == pytest.approx(0.0)
    assert s["primary_metric"] == "mcc"
    assert s["primary"] == pytest.approx(0.0)


def test_sst2_primary_metric_is_accuracy():
    exs = [Example("sst2", "x", "positive", 1)]
    s = score(exs, ["positive"], "sst2", "distinct")
    assert s["primary_metric"] == "accuracy"
    assert s["primary"] == pytest.approx(1.0)


def test_misroute_counts_only_foreign_labels():
    exs = [Example("cola", "x", "acceptable", 1)] * 4
    preds = ["acceptable", "positive", "unacceptable", "banana"]
    s = score(exs, preds, "cola", "distinct")
    assert s["accuracy"] == pytest.approx(0.25)
    assert s["off_label_rate"] == pytest.approx(0.5)     # positive + banana
    assert s["misroute_rate"] == pytest.approx(0.25)     # positive only


def test_off_label_is_never_mapped_onto_a_label():
    exs = [Example("sst2", "x", "positive", 1)] * 2
    s = score(exs, ["positive!", "  POSITIVE"], "sst2", "distinct")
    # trailing punctuation is a genuine miss; case and space are normalised
    # upstream in generate(), so here only the exact string counts
    assert s["accuracy"] == pytest.approx(0.0)


def test_collided_scheme_has_no_misroutes_by_construction():
    exs = [Example("cola", "x", "yes", 1)]
    s = score(exs, ["no"], "cola", "collided")
    assert s["misroute_rate"] == 0.0


def test_run_grid_shape():
    runs = build_run_grid([0])
    assert len(runs) == 3 * 4 + 4 + 2
    ids = {r.run_id for r in runs}
    assert len(ids) == len(runs)
    assert sum(r.regime == "single" for r in runs) == 12


def test_per_task_exposure_is_equal_across_regimes(tmp_path):
    d = tmp_path
    for t, cols in [("cola", 1), ("sst2", 1), ("mrpc", 2)]:
        for split in ("train", "validation"):
            with open(d / f"{t}_{split}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["text_a", "text_b", "label"])
                for i in range(50):
                    w.writerow([f"s{i}", f"t{i}" if cols == 2 else "", i % 2])
    from src.data import build_training_set
    single = build_training_set(str(d), ["cola"], "correct", "distinct", 10, 2, 0)
    multi = build_training_set(str(d), ["cola", "sst2", "mrpc"], "correct",
                               "distinct", 10, 2, 0)
    assert len(single) == 20
    assert sum(1 for e in multi if e.task == "cola") == 20
    assert len(multi) == 60
