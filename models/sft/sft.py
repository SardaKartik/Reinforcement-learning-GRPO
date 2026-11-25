"""
SFT (Supervised Fine-Tuning) training script for the base model on GSM8K.

Trains Qwen2.5-0.5B-Instruct with LoRA on GSM8K's human-written
chain-of-thought solutions, using trl.SFTTrainer. Produces the checkpoint
used for benchmark(1) (see docs/execution-plan.md) and the starting point
for GRPO (Step 3).

See models/sft/sft_training_guide.md for the dataset, loss function, and
LoRA details behind this script.

This file is written to be copy-pasted into a Kaggle notebook, one
"# Cell N" block per notebook cell, built and tested incrementally
(cell by cell) rather than run all at once.

Suggested Kaggle notebook name: "gsm8k-sft-training"
"""

# ============================================================
# Cell 1 — Kaggle setup: install packages not already pre-installed
# ============================================================
# Kaggle already has torch, transformers, datasets, and huggingface_hub
# pre-installed — only trl (training loop) and peft (LoRA) need adding.
#
# NOTE: Kaggle's pre-installed torchao (0.10.0) is older than what peft's
# LoRA injection code requires (>=0.16.0) to safely run its internal
# backend-availability check. Without upgrading it, instantiating
# SFTTrainer(..., peft_config=...) raises:
#   ImportError: Found an incompatible version of torchao...
# Upgrading torchao here fixes it, even though we never use it directly.
# !pip install -q trl peft python-dotenv -U torchao

# ============================================================
# Cell 2 — Imports
# ============================================================
# Kaggle's "GPU T4 x2" gives this process TWO visible GPUs. SFTTrainer
# refuses to auto-split work across them via nn.DataParallel (it's not
# properly supported) and raises a ValueError instead. Our 0.5B model
# doesn't need two GPUs anyway (one T4's 16GB is already more than
# enough), so we just restrict visibility to GPU 0 and ignore the second.
# This MUST be set before torch/CUDA initializes — hence right at the
# top, before any other imports.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ============================================================
# Cell 3 — Hugging Face login (via Kaggle Secrets)
# ============================================================
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN)

# ============================================================
# Cell 4 — Config
# ============================================================
HF_DATASET_REPO = "sardakartik/gsm8k-split"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# v2 run (matched prompt, gentler hyperparameters). A NEW repo name on
# purpose: the v1 checkpoints (...-sft-epoch2 / ...-sft-epoch3), which the
# diagnostic results in sft_result.md were measured on, stay untouched.
OUTPUT_REPO = "sardakartik/qwen2.5-0.5b-gsm8k-sft-v2"

# The instruction appended to every training question. It MUST stay
# identical, character for character, to the "instruction" prompt in
# eval/accuracy.py's build_prompt(). v1 trained on the bare question but
# was evaluated WITH this instruction, and that train/eval mismatch cost
# the SFT model about 10 points (see sft_result.md).
INSTRUCTION = (
    "Solve this step by step, then give the final answer "
    "on its own line in the format: #### <number>"
)

MAX_SEQ_LENGTH = 512
LEARNING_RATE = 5e-5  # v1 used 1e-4
NUM_EPOCHS = 2        # v1 used 3 (validation loss turned upward during epoch 3)

# ============================================================
# Cell 5 — Load the dataset
# ============================================================
# Pulls our pre-split GSM8K train/validation/test from the private HF
# dataset repo (same repo used by eval/accuracy.py), so every phase of
# the project trains and evaluates on identical, fixed data.
dataset = load_dataset(HF_DATASET_REPO)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]  # NOT "val" — see data/execution-plan.md note

print(f"Train examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")
print(train_dataset[0])

# ============================================================
# Cell 6 — Load tokenizer and base model
# ============================================================
# Loads the actual model we're fine-tuning (Qwen2.5-0.5B-Instruct) and
# its tokenizer. The LoRA adapters (Cell 8+) get attached to this model
# afterward — the base weights loaded here stay frozen throughout.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

print(f"Tokenizer loaded: {tokenizer.name_or_path}")
print(f"Model loaded: {model.config.name_or_path}")
print(f"Model device: {model.device}")

# ============================================================
# Cell 7 — Format each example into TRL's "prompt-completion" structure,
# print one to manually verify it looks correct
# ============================================================
# NOTE (version-specific correction): TRL's older API used a single
# merged "text" string plus a separate DataCollatorForCompletionOnlyLM
# to mask the prompt portion of the loss. That collator class has since
# been REMOVED from trl (confirmed via its own GitHub issue tracker).
# The current, correct approach: give SFTTrainer separate "prompt" and
# "completion" fields (each a list of chat messages); it applies the
# chat template and masks the prompt from the loss AUTOMATICALLY —
# completion_only_loss=True is the default for this dataset shape, so we
# don't even need to configure masking ourselves anymore.
# Refs:
#   https://github.com/huggingface/trl/issues/3827
#   https://huggingface.co/docs/trl/sft_trainer
#
# NOTE: no explicit "system" message is added here, but Qwen2.5's own
# chat_template.jinja (part of its tokenizer config on HF, not something
# we wrote) auto-inserts a default system prompt ("You are Qwen, created
# by Alibaba Cloud. You are a helpful assistant.") whenever none is
# provided. This is expected, documented Qwen2.5 behavior, and it's
# harmless since the same tokenizer/template is used in eval/accuracy.py
# too, keeping training and evaluation consistent.
#   https://huggingface.co/docs/trl/chat_templates
#
# v2 CHANGE: the user turn is now question + INSTRUCTION (Cell 4), the
# same text eval/accuracy.py sends at test time, so training and
# evaluation see the same prompt. v1 used the bare question here.
def format_example(example):
    return {
        "prompt": [
            {"role": "user", "content": f"{example['question']}\n\n{INSTRUCTION}"}
        ],
        "completion": [{"role": "assistant", "content": example["answer"]}],
    }


formatted_example = format_example(train_dataset[0])
print(formatted_example)

# ============================================================
# Cell 8 — Apply the formatting to the full train and validation sets
# ============================================================
# .map() applies format_example() to every row. remove_columns drops the
# original "question"/"answer" fields since SFTTrainer only needs the new
# "prompt"/"completion" fields we just created.
train_dataset_formatted = train_dataset.map(
    format_example, remove_columns=train_dataset.column_names
)
val_dataset_formatted = val_dataset.map(
    format_example, remove_columns=val_dataset.column_names
)

print(f"Formatted train examples: {len(train_dataset_formatted)}")
print(f"Formatted validation examples: {len(val_dataset_formatted)}")
print(train_dataset_formatted[0])

# ============================================================
# Cell 9 — LoRA config
# ============================================================
# Locked settings from sft_training_guide.md: the base model's weights
# stay frozen; only these small low-rank adapter matrices (attached to
# every linear layer) get trained.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

print(lora_config)

# ============================================================
# Cell 10 — SFTConfig (training hyperparameters)
# ============================================================
# fp16=True (not bf16) because Kaggle's T4 lacks native bf16 tensor core
# support (Turing architecture) — see docs/execution-plan.md's compute
# notes. completion_only_loss=True is actually the default for our
# prompt-completion dataset shape, set explicitly here for clarity.
training_args = SFTConfig(
    output_dir="/kaggle/working/qwen2.5-0.5b-gsm8k-sft-v2",
    max_length=MAX_SEQ_LENGTH,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size = 4 * 4 = 16
    fp16=True,
    completion_only_loss=True,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    # v2: a checkpoint every 200 steps (v1 saved only at epoch ends, which
    # left us unable to pick the real validation-loss minimum). save_steps
    # must be a multiple of eval_steps for load_best_model_at_end to work.
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    # After training, reload the checkpoint with the LOWEST validation loss
    # instead of keeping whatever the last step produced.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)

print(training_args)

# ============================================================
# Cell 11 — Instantiate SFTTrainer
# ============================================================
# Brings together everything built in Cells 5-10: the base model (LoRA
# adapters get attached automatically via peft_config), the formatted
# train/validation datasets, and the training hyperparameters. No
# training happens yet — this just constructs the trainer object.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_formatted,
    eval_dataset=val_dataset_formatted,
    peft_config=lora_config,
)

print(trainer.model.print_trainable_parameters())

# ============================================================
# Cell 12 — Train
# ============================================================
# The actual (expensive) training run. Watch the logged loss — it should
# generally trend downward over the 2 epochs. eval_loss (printed every
# 100 steps, per eval_steps in Cell 10) is computed on val_dataset_formatted
# and is what to watch for overfitting (train loss keeps dropping while
# eval loss stops improving or rises).
train_result = trainer.train()

print(train_result)

# ============================================================
# Cell 13 — Push the best checkpoint to Hugging Face Hub
# ============================================================
# Cell 10 set load_best_model_at_end=True with metric_for_best_model=
# "eval_loss", so after trainer.train() the model held by `trainer` is the
# checkpoint with the LOWEST validation loss (checkpoints are saved every
# 200 steps), not simply the last one. We print which step that was, save
# it, and push it to the NEW repo OUTPUT_REPO — the v1 repos
# (...-sft-epoch2 / ...-sft-epoch3) are left untouched.
#
# Write down the two printed values (best checkpoint + best eval_loss) in
# models/sft/sft_result.md.
from huggingface_hub import HfApi, create_repo

print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best eval_loss: ", trainer.state.best_metric)

BEST_DIR = "/kaggle/working/sft-v2-best"

# Saves the LoRA adapter (the best weights, now loaded in trainer.model).
trainer.save_model(BEST_DIR)
# Tokenizer files travel with it, like the v1 checkpoints did.
tokenizer.save_pretrained(BEST_DIR)

api = HfApi(token=HF_TOKEN)
# private=True: only you can see it. exist_ok=True: safe to re-run.
create_repo(repo_id=OUTPUT_REPO, repo_type="model", private=True, token=HF_TOKEN, exist_ok=True)
api.upload_folder(folder_path=BEST_DIR, repo_id=OUTPUT_REPO, repo_type="model")

print(f"Pushed best checkpoint -> https://huggingface.co/{OUTPUT_REPO}")
