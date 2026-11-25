"""
Download GSM8K, split it into train/val/test, save as .jsonl files,
and push the split to a private Hugging Face dataset repo.

Run this ONCE, locally. No GPU needed.
Kaggle and RunPod should download the resulting split from the HF Hub
repo instead of re-running this script, so every environment uses the
exact same files.
"""

import os
import json

from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

# --- Config -----------------------------------------------------------

# GSM8K is MIT-licensed (Cobbe et al., 2021) — redistributing a
# modified/split version is permitted; keeping this note as attribution.
SOURCE_DATASET = "openai/gsm8k"
SOURCE_CONFIG = "main"  # "main" = standard config (vs. "socratic")

# Fraction of the official GSM8K "train" split carved out for validation.
# The official "test" split is left completely untouched — it's our one
# shared benchmark set across every checkpoint (base, SFT, GRPO, distilled).
VAL_FRACTION = 0.1
SPLIT_SEED = 42  # fixed seed so the split is reproducible if ever re-run

# Local folder where the .jsonl files get written before upload.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "splits")

# Your own private HF dataset repo — Kaggle/RunPod will pull from here.
HF_REPO_ID = "sardakartik/gsm8k-split"


def save_jsonl(hf_split, filepath):
    """Write a Hugging Face Dataset split to disk as one JSON object per line."""
    with open(filepath, "w", encoding="utf-8") as f:
        for example in hf_split:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(hf_split)} examples to {filepath}")


def main():
    # Load HF_TOKEN from .env — never hardcode the token in code.
    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]

    # --- 1. Download the official GSM8K dataset ------------------------
    # Returns a DatasetDict with "train" (~7,473 examples) and
    # "test" (~1,319 examples). Each example has a "question" and an
    # "answer" (full reasoning steps, final number after a "####" marker).
    dataset = load_dataset(SOURCE_DATASET, SOURCE_CONFIG)

    # --- 2. Carve a validation set out of the official train split -----
    # Validation is only used internally during training (checkpoint
    # selection, early stopping) — it never appears in final reported
    # numbers, so exact reproducibility here matters less than for test.
    split = dataset["train"].train_test_split(test_size=VAL_FRACTION, seed=SPLIT_SEED)
    train_split = split["train"]
    val_split = split["test"]  # naming artifact of HF's API — this is OUR val set

    # Official test set, used as-is — NEVER touched, re-split, or trained on.
    test_split = dataset["test"]

    # --- 3. Save each split locally as .jsonl ---------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_jsonl(train_split, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val_split, os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test_split, os.path.join(OUTPUT_DIR, "test.jsonl"))

    # --- 4. Create (or reuse) your private HF dataset repo --------------
    create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        private=True,
        token=hf_token,
        exist_ok=True,  # safe to re-run this script without erroring
    )

    # --- 5. Upload the three split files ---------------------------------
    # This becomes the single source of truth: Kaggle and RunPod both pull
    # from here instead of re-deriving the split independently.
    api = HfApi(token=hf_token)
    for filename in ("train.jsonl", "val.jsonl", "test.jsonl"):
        api.upload_file(
            path_or_fileobj=os.path.join(OUTPUT_DIR, filename),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"Uploaded {filename} to {HF_REPO_ID}")

    print("Done. Dataset split is available at:")
    print(f"  https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()