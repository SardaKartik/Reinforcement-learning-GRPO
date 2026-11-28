# SFT vs Distillation vs GRPO on a Small Math Model

A controlled comparison of three ways to improve a small language model's
grade-school math reasoning:

- **SFT**: supervised fine-tuning on human-written step-by-step solutions
- **Distillation**: fine-tuning on solutions written by a larger teacher model
- **GRPO**: reinforcement learning, where the model grades its own attempts

Everything except the training method is held fixed, so the differences are
attributable to the method.

| | |
|---|---|
| Base model | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Task | GSM8K (grade-school math word problems) |
| Adapter | LoRA, r=16, alpha=32, dropout 0.05, all linear layers (same for every arm) |
| Metric | Exact match on the final numeric answer |
| Compared on | Accuracy, training cost, GPU memory, inference latency |

## Status

| Arm | Status |
|---|---|
| Evaluation harness | Done (`eval/accuracy.py`); supports the old `####` prompt and the GRPO `<think>` + `\boxed{}` prompt |
| Base model (control) | Measured with the old `####` prompt; the re-measurement under the GRPO prompt (test split) is pending |
| SFT | Done. Underperformed the base model, so GRPO starts from the base instead |
| GRPO | 500-step pilot trained on an A40: score on training questions rose from about 47% to about 62%, but the `<think>` format was not learned. Held-out evaluation is pending. |
| Distillation | Not started |
| Latency and memory benchmark | Not started |

### Results so far

Held-out accuracy, greedy decoding. **Only compare rows with the same prompt,
split and max-new-tokens.**

| Model | Prompt used at evaluation | Split | Accuracy |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (base) | old `####` instruction | test (1,319) | 42.15% (556/1319) |
| SFT v2 (LoRA, best checkpoint) | old `####` instruction | test (1,319) | 33.66% (444/1319) |
| Qwen2.5-0.5B-Instruct (base) | GRPO `<think>` + `\boxed{}` | test (1,319) | pending |
| GRPO pilot (500 steps) | GRPO `<think>` + `\boxed{}` | test (1,319) | pending |

The old 42.15% is not the baseline for GRPO. GRPO's baseline is the base model
under the GRPO prompt (the third row). The full table, the comparison rules and
the GRPO training curve are in [`docs/results.md`](docs/results.md) and
[`models/grpo/grpo_result.md`](models/grpo/grpo_result.md).

## How GRPO is set up here

For each training question the model writes **8 attempts**. Each attempt is
scored, and an attempt is reinforced if it scored better than the average of
its own group of 8 and discouraged if it scored worse. No reference answer text
and no learned reward model is needed.

- **Reward** = `1.0 × correct` + `0.2 × format`
  - *Correct*: the final number (last `\boxed{}` after `</think>`, otherwise the
    last number in the text) equals the GSM8K answer. It does not depend on
    tags, because the base model almost never writes them.
  - *Format*: exactly one `<think>...</think>` block followed by `\boxed{...}`.
- **Trainer**: TRL `GRPOTrainer` with LoRA and vLLM (colocated) for fast generation.
- **Loss**: `dr_grpo`, no reward scaling, KL penalty `beta = 0`, 2 updates per
  generation (`num_iterations = 2`), constant learning rate `1e-5`.
- **Batching**: 8 attempts per GPU pass × 4 accumulation = 32 attempts = 4 questions per step.

Every value and the reason for it is in
[`models/grpo/grpo-hyperparameters.md`](models/grpo/grpo-hyperparameters.md).

## Repository layout

```
data/
  prepare_gsm8k.py       build the train / validation / test split and upload it
  verify_upload.py       check the uploaded split loads back correctly
eval/
  accuracy.py            evaluation harness (exact-match accuracy, both prompts)
models/
  sft/                   SFT script, training guide, results
  grpo/
    prompt.py            the one prompt used for GRPO training and evaluation
    rewards.py           correctness and format reward functions
    train_grpo.py        GRPO training script
    grpo-hyperparameters.md
    grpo_result.md       reward functions and the pilot run, report by report
    requirements.txt     pinned versions that passed the smoke test
docs/
  project.md             problem statement and experiment design
  execution-plan.md      step-by-step plan and design decisions
  results.md             results table
  qwen_model.md          notes on the base model
```

## Data

The GSM8K split is **not stored in this repo**. It is loaded from a private
Hugging Face dataset (`sardakartik/gsm8k-split`) with `train`, `validation` and
`test` splits. To reproduce, run `data/prepare_gsm8k.py` with your own
Hugging Face token to build and upload your own copy, then change the
`DATASET_REPO` constant in the scripts to point to it.

## Running

You need a Hugging Face token in the `HF_TOKEN` environment variable (a
`.env` file works for the local scripts). **Never commit a token.** `.env` is
git-ignored.

**Reward and prompt tests (any machine, no GPU):**

```bash
python models/grpo/rewards.py     # prints OK/BAD for 13 test cases
python models/grpo/prompt.py      # prints the prompt and checks it
```

**GRPO training (Linux + NVIDIA GPU; developed for a RunPod Pod):**

```bash
python3 -m venv venv && source venv/bin/activate   # use a fresh venv: vLLM brings its own PyTorch
pip install -r models/grpo/requirements.txt
export HF_TOKEN=...

python models/grpo/train_grpo.py --smoke     # 5 steps, 16 questions, nothing pushed
python models/grpo/train_grpo.py             # pilot: 500 steps, pushes to the Hub
python models/grpo/train_grpo.py --full      # one full epoch
python models/grpo/train_grpo.py --full --resume /workspace/grpo_output/checkpoint-NNN
```

`HUB_MODEL_ID` in `train_grpo.py` is where checkpoints are pushed. Change it to
a repo you own.

**Evaluation:** `eval/accuracy.py` (written as Kaggle notebook cells). Set
`PROMPT_STYLE = "think_boxed"` for the GRPO prompt (the default) or
`"instruction"` for the old `####` prompt, `SPLIT = "validation"` or `"test"`,
and for a LoRA model set `ADAPTER_REPO` and `ADAPTER_SUBFOLDER`
(for the GRPO pilot: `sardakartik/qwen2.5-0.5b-gsm8k-grpo` and `final`). It
reports accuracy, and for the GRPO prompt also strict accuracy, format rate,
boxed rate, mean answer length and the share of truncated answers.

## Notes and caveats

- The 42.15% and 33.66% numbers were measured with the old `####` prompt and a
  512-token limit. Prompt wording alone moved accuracy by about 10 points in our
  SFT diagnostics ([`models/sft/sft_result.md`](models/sft/sft_result.md)), so
  they are not compared with anything measured under the GRPO prompt.
- The GRPO correctness numbers in the training logs are scores on the
  **training** questions, not held-out accuracy.
- Package versions for the GRPO run are pinned in `models/grpo/requirements.txt`
  (trl 1.19.0, vllm 0.34.0, peft 0.22.0, datasets 5.2.0, transformers 5.22.2).
- The prompt text is locked before training starts and must not be edited
  afterwards, so training and evaluation see identical wording.

## Further reading

- [`docs/project.md`](docs/project.md): experiment design, on-policy vs off-policy
- [`docs/execution-plan.md`](docs/execution-plan.md): decisions and plan
- [`models/sft/sft_training_guide.md`](models/sft/sft_training_guide.md): LoRA settings and reasoning
