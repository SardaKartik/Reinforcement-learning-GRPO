"""
One-off sanity check: confirms the GSM8K split we pushed to our private
Hugging Face dataset repo can actually be loaded back correctly — the
same way Kaggle/RunPod will load it later.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

HF_REPO_ID = "sardakartik/gsm8k-split"


def main():
    # Log in using the token from .env so we can access our PRIVATE repo.
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    # Load the split back from the Hub, exactly like Kaggle/RunPod will.
    splits = load_dataset(HF_REPO_ID)

    print(splits)
    print()
    print("Example from train split:")
    print(splits["train"][0])
    print()
    print(
        f"Counts -> train: {len(splits['train'])}, "
        f"validation: {len(splits['validation'])}, "
        f"test: {len(splits['test'])}"
    )


if __name__ == "__main__":
    main()