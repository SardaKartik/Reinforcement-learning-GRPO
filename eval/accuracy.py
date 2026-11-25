"""
Evaluation harness: runs a given model/checkpoint on the held-out GSM8K
test set and reports exact-match accuracy on the final numeric answer.

Same script is reused for every checkpoint (base, SFT, GRPO, Distillation)
by changing MODEL_NAME below — this is what makes benchmark(0..3) a fair,
apples-to-apples comparison.

This file is written to be copy-pasted into a Kaggle notebook, one
"# Cell N" block per notebook cell (Kaggle already has torch/transformers
pre-installed, so only a couple of extra packages need installing).

Suggested Kaggle notebook name: "gsm8k-accuracy-eval"
(use the same notebook for benchmark(0..3) — just change MODEL_NAME in
Cell 3 and rerun, so results stay easy to find under one notebook name)
"""

# ============================================================
# Cell 1 — Kaggle setup: install packages not already pre-installed
# ============================================================
# peft is only needed when evaluating a LoRA checkpoint (ADAPTER_REPO set
# in Cell 3 below) — installing it unconditionally here is simplest and
# harmless for plain full-model runs (base/distilled) too.
#
# NOTE: Kaggle's pre-installed torchao (0.10.0) is older than what peft's
# LoRA injection code requires (>=0.16.0) — without upgrading it here,
# PeftModel.from_pretrained(...) raises:
#   ImportError: Found an incompatible version of torchao...
# Same root cause and fix as models/sft/sft.py's Cell 1 — each Kaggle
# notebook has its OWN separate pre-installed packages, so fixing it in
# one notebook does not carry over to this one.
# !pip install -q datasets huggingface_hub python-dotenv peft -U torchao

# ============================================================
# Cell 2 — Imports
# ============================================================
import json
import os
import re
import time

from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# Cell 3 — Config
# ============================================================
HF_DATASET_REPO = "sardakartik/gsm8k-split"

# MODEL_NAME is always the BASE/full model to load (unchanged behavior
# from benchmark(0) — never a LoRA adapter repo).
#   base / distilled -> "Qwen/Qwen2.5-0.5B-Instruct" (or the distilled
#                        checkpoint, if it was pushed as a full model)
#   SFT / GRPO        -> also "Qwen/Qwen2.5-0.5B-Instruct" — the actual
#                        fine-tuned checkpoint is a LoRA adapter, set via
#                        ADAPTER_REPO below instead, not here.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Leave as None to evaluate a plain full model exactly like benchmark(0)
# did (no code path changes for that case at all — see run_eval below).
# Set to a LoRA adapter repo id (e.g. one of our SFT checkpoints) to
# attach that adapter on top of MODEL_NAME before evaluating.
#   e.g. "sardakartik/qwen2.5-0.5b-gsm8k-sft-epoch2"
#     or "sardakartik/qwen2.5-0.5b-gsm8k-sft-epoch3"
ADAPTER_REPO = None

MAX_NEW_TOKENS = 512  # enough room for a full step-by-step solution

# Number of questions sent to the GPU together in one generate() call.
# Generating one question at a time leaves most of the GPU idle (it's
# built to do many calculations in parallel); batching several questions
# together lets the GPU work on all of them at roughly the same wall-clock
# cost as doing just one. Start at 8 — raise it if VRAM allows, lower it
# if you hit an out-of-memory error.
BATCH_SIZE = 8

# --- Diagnostic knobs -------------------------------------------------
# All four default to the ORIGINAL benchmark behavior (test set, with the
# instruction prompt, every question, nothing saved), so leaving them
# alone reproduces benchmark(0) exactly. They exist so we can run
# controlled experiments by changing one line between runs.

# Which split to evaluate on: "test" (final reported numbers) or
# "validation" (use this for diagnosing/making decisions, so we don't tune
# against the test set).
SPLIT = "test"

# "instruction": question + "Solve step by step... #### <number>" (the
#                original eval prompt).
# "bare":        the question only, exactly what SFT saw during training.
PROMPT_STYLE = "instruction"

# Evaluate only the first N questions of the split (None = all of them).
# Use the same N for every run you want to compare.
NUM_EXAMPLES = None

# If set (e.g. "/kaggle/working/base_instruction_val300.jsonl"), every
# generated answer is written to this file as it's produced (question,
# ground truth, predicted answer, correct?, full generated text) so we can
# analyze answer length / patterns afterward without re-running.
SAVE_OUTPUTS_PATH = None

# ============================================================
# Cell 4 — Helper functions: parsing answers + building the prompt
# ============================================================


# Matches an optional '-', digits with optional ',' thousand-separators,
# and an optional decimal part (e.g. "-1,234.5"). GSM8K's own ground-truth
# answers are always whole numbers, but the MODEL's generated text can
# still write a correct number in a slightly different shape (e.g. "18.0"
# instead of "18") — capturing the decimal part here means we don't
# silently truncate it, so the later numeric comparison can judge it fairly.
NUMBER_PATTERN = r"-?[\d,]+\.?\d*"


def extract_ground_truth(answer_text):
    """GSM8K answers always end with '#### <number>' — pull that number out."""
    match = re.search(rf"####\s*({NUMBER_PATTERN})", answer_text)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(generated_text):
    """
    Pull the model's final numeric answer out of its generated response.
    We prompt the model to use the same '#### <number>' convention as
    GSM8K itself, so the primary match looks for that; if the model
    doesn't follow the format, fall back to the last number it mentions.
    """
    match = re.search(rf"####\s*({NUMBER_PATTERN})", generated_text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(NUMBER_PATTERN, generated_text)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(predicted, ground_truth):
    """
    Compare the two extracted answers NUMERICALLY, not as raw strings.
    A strict string check would wrongly mark "18" vs "18.0" as different,
    even though they're the same number — comparing as floats avoids
    penalizing the model for harmless formatting differences.
    """
    if predicted is None or ground_truth is None:
        return False
    try:
        return float(predicted) == float(ground_truth)
    except ValueError:
        # If either string isn't a valid number at all, fall back to a
        # plain string comparison rather than crashing the eval loop.
        return predicted == ground_truth


def build_prompt(tokenizer, question, style="instruction"):
    """
    Wrap the raw GSM8K question in a chat-template prompt.

    style="instruction" (default, the original eval prompt): explicitly
    asks for the '#### <number>' format so extract_predicted_answer has
    something reliable to parse.

    style="bare": the question only, nothing appended — identical to the
    user turn SFT was trained on (see models/sft/sft.py's format_example).
    """
    if style == "instruction":
        # models/sft/sft.py (v2) appends this SAME instruction text to every
        # training question (its INSTRUCTION constant), so the SFT model
        # sees the same prompt in training and here. If you change this
        # wording, change it there too and retrain — a mismatch cost the v1
        # SFT model ~10 points (see models/sft/sft_result.md).
        content = (
            f"{question}\n\n"
            "Solve this step by step, then give the final answer "
            "on its own line in the format: #### <number>"
        )
    elif style == "bare":
        content = question
    else:
        raise ValueError(f"Unknown prompt style: {style!r} (use 'instruction' or 'bare')")

    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================
# Cell 5 — Main evaluation loop
# ============================================================


def run_eval(
    model_name,
    adapter_repo=None,
    split="test",
    prompt_style="instruction",
    num_examples=None,
    save_outputs_path=None,
):
    # Load HF_TOKEN from .env so we can pull the private test set.
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    # With the defaults this is the held-out test set — identical across
    # every checkpoint we benchmark. split="validation" is for diagnosis.
    print(f"Loading '{split}' split from {HF_DATASET_REPO} ...")
    test_set = load_dataset(HF_DATASET_REPO)[split]
    if num_examples is not None:
        # First N questions — deterministic, so every run sees the same ones.
        test_set = test_set.select(range(min(num_examples, len(test_set))))
    total = len(test_set)
    print(f"Loaded {total} '{split}' examples (prompt style: {prompt_style}).")

    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decoder-only models always predict the NEXT token after the LAST
    # token they see. When we batch questions of different lengths
    # together, shorter ones need padding to match the longest one in the
    # batch — that padding MUST go on the left (start), not the right.
    # Right-padding would put real tokens before padding tokens, so the
    # model would start generating from a padding token instead of from
    # the actual end of the question — producing garbage.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        # Some models don't define a pad token by default; reusing the
        # end-of-sequence token as padding is the standard workaround.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # ADDITIVE, OPT-IN ONLY: when adapter_repo is None (the default, and
    # exactly what benchmark(0)'s original run used), this block is
    # skipped entirely and `model` is the plain full model loaded above —
    # unchanged from the original base-model benchmark. Only pass
    # adapter_repo when evaluating a LoRA checkpoint (e.g. SFT/GRPO).
    if adapter_repo:
        print(f"Attaching LoRA adapter from {adapter_repo} ...")
        model = PeftModel.from_pretrained(model, adapter_repo)

    print(f"Model loaded on device: {model.device}")

    correct = 0
    start_time = time.time()

    # Opt-in: stream every generated answer to a .jsonl file as we go (one
    # line per question), so partial runs still leave usable data behind.
    out_file = open(save_outputs_path, "w", encoding="utf-8") if save_outputs_path else None

    # Walk through the test set in chunks of BATCH_SIZE instead of one
    # question at a time, so the GPU processes several questions in
    # parallel per generate() call instead of sitting mostly idle.
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)

        # Slicing a Hugging Face Dataset returns a dict of lists (e.g.
        # {"question": [...], "answer": [...]}), not a list of dicts.
        batch = test_set[batch_start:batch_end]
        questions = batch["question"]
        answers = batch["answer"]

        prompts = [build_prompt(tokenizer, q, prompt_style) for q in questions]

        # padding=True pads every prompt in this batch to the same length
        # (using left-padding, set above) so they can be processed together
        # as a single tensor.
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
            model.device
        )

        # Greedy decoding (do_sample=False) — deterministic output, which
        # is what we want for a reproducible accuracy benchmark (unlike
        # GRPO's rollouts, which need randomness to explore).
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Every prompt in the batch was padded to the same input length,
        # so this single slice index correctly strips the prompt off of
        # every sequence in the batch at once.
        input_len = inputs["input_ids"].shape[1]

        for j, (question, answer) in enumerate(zip(questions, answers)):
            generated_text = tokenizer.decode(
                outputs[j][input_len:], skip_special_tokens=True
            )

            predicted = extract_predicted_answer(generated_text)
            ground_truth = extract_ground_truth(answer)

            # Numeric comparison (not strict string equality) — see
            # answers_match() for why this matters.
            is_correct = answers_match(predicted, ground_truth)
            if is_correct:
                correct += 1

            if out_file:
                out_file.write(
                    json.dumps(
                        {
                            "question": question,
                            "ground_truth": ground_truth,
                            "predicted": predicted,
                            "correct": is_correct,
                            "generated_text": generated_text,
                        }
                    )
                    + "\n"
                )

        if out_file:
            out_file.flush()

        # Progress check-in after every batch, so the notebook output
        # shows it's actually running, not frozen.
        running_accuracy = correct / batch_end * 100
        elapsed_so_far = time.time() - start_time
        print(
            f"[{batch_end}/{total}] "
            f"running accuracy: {running_accuracy:.2f}% | "
            f"elapsed: {elapsed_so_far:.1f}s"
        )

    if out_file:
        out_file.close()
        print(f"Saved generated answers to {save_outputs_path}")

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    model_label = f"{model_name} + adapter {adapter_repo}" if adapter_repo else model_name
    print(f"Model: {model_label}")
    print(f"Split: {split} | prompt style: {prompt_style} | examples: {total}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Total eval time: {elapsed:.1f}s")


# ============================================================
# Cell 6 — Run it
# ============================================================
if __name__ == "__main__":
    run_eval(
        MODEL_NAME,
        ADAPTER_REPO,
        split=SPLIT,
        prompt_style=PROMPT_STYLE,
        num_examples=NUM_EXAMPLES,
        save_outputs_path=SAVE_OUTPUTS_PATH,
    )