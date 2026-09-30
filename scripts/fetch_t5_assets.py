#!/usr/bin/env python3
"""Download t5-small and the three GLUE tasks into ./t5_assets/.

Run on a machine that can reach huggingface.co:

    python3 -m pip install -U "transformers>=4.40" datasets sentencepiece
    python3 scripts/fetch_t5_assets.py
"""
import csv, json, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "t5_assets")
os.makedirs(OUT, exist_ok=True)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
mdir = os.path.join(OUT, "t5-small")
AutoTokenizer.from_pretrained("t5-small").save_pretrained(mdir)
AutoModelForSeq2SeqLM.from_pretrained("t5-small").save_pretrained(
    mdir, safe_serialization=True)
print("model ->", mdir)

from datasets import load_dataset
SPEC = {"cola": ("sentence", None), "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2")}
# datasets v5 requires a namespaced repo id; the bare "glue" is rejected.
CANDIDATES = ["nyu-mll/glue", "glue"]
for task, (ka, kb) in SPEC.items():
    ds = None
    for repo in CANDIDATES:
        try:
            ds = load_dataset(repo, task)
            break
        except Exception:  # noqa: BLE001
            continue
    if ds is None:
        raise SystemExit(f"could not load GLUE/{task} from {CANDIDATES}")
    for split in ("train", "validation"):
        path = os.path.join(OUT, f"{task}_{split}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["text_a", "text_b", "label"])
            for r in ds[split]:
                w.writerow([r[ka], r[kb] if kb else "", r["label"]])
        print(f"  {task}/{split}: {len(ds[split])}")
json.dump({"ok": True}, open(os.path.join(OUT, "MANIFEST.json"), "w"))
