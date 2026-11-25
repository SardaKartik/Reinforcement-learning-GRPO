# SFT Training Results

## Run summary

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `sardakartik/gsm8k-split` — train (6,725 examples), validation (748)
- Method: LoRA (r=16, alpha=32, dropout=0.05, all linear layers) via `trl.SFTTrainer`
- Hyperparameters: learning rate 1e-4, 3 epochs, effective batch size 16
  (per_device_train_batch_size=4 × gradient_accumulation_steps=4), fp16
- Trainable parameters: 8,798,208 / 502,830,976 total (**1.75%**) —
  confirms LoRA's efficiency (matches hand-calculated estimate in
  `sft_training_guide.md`)
- Platform: Kaggle, GPU T4 (restricted to 1 of the 2 available via
  `CUDA_VISIBLE_DEVICES=0` — see note below)
- Training time: 2,670.47s (**~44.5 minutes**)
- Throughput: 7.555 samples/sec, 0.473 steps/sec

## Training metrics (full run)

| Step | Training Loss | Validation Loss | Entropy | Mean Token Accuracy | Num Tokens |
|---|---|---|---|---|---|
| 100 | 0.4187 | 0.3996 | 0.3876 | 0.8855 | 338,461 |
| 200 | 0.4029 | 0.3920 | 0.3914 | 0.8870 | 676,804 |
| 300 | 0.4316 | 0.3866 | 0.3953 | 0.8889 | 1,017,117 |
| 400 | 0.4078 | 0.3842 | 0.3946 | 0.8892 | 1,359,674 |
| 500 | 0.3378 | 0.3858 | 0.3488 | 0.8887 | 1,696,806 |
| 600 | 0.3437 | 0.3876 | 0.3351 | 0.8889 | 2,032,797 |
| 700 | 0.3425 | 0.3849 | 0.3403 | 0.8897 | 2,376,791 |
| **800** | **0.3494** | **0.3829** ← lowest | **0.3402** | **0.8907** ← highest | 2,716,637 |
| 900 | 0.2865 | 0.4072 | 0.2890 | 0.8874 | 3,055,821 |
| 1000 | 0.2854 | 0.4061 | 0.2915 | 0.8873 | 3,397,488 |
| 1100 | 0.2810 | 0.4017 | 0.2963 | 0.8880 | 3,739,339 |
| 1200 | 0.2859 | 0.4031 | 0.2966 | 0.8878 | 4,077,807 |
| 1263 | 0.2854 | 0.4031 | 0.2972 | 0.8879 | 4,284,789 |

Final reported training loss (averaged over the whole run): **0.3489**.

## Key finding: overfitting in epoch 3

- Steps per epoch = ceil(6,725 / 16) = **421**. So epoch 1 ends at step
  421, epoch 2 at step 842, epoch 3 (final) at step 1263.
- **Validation loss decreased steadily through step 800** (best value:
  0.3829, near the end of epoch 2), then **increased** for the rest of
  training (up to 0.4031 by the final step) — while **training loss kept
  decreasing** the whole time (0.42 → 0.29).
- This is a textbook overfitting pattern: the model kept improving on
  training data it directly sees, but got *worse* at generalizing to the
  held-out validation set, starting partway through epoch 3.
- Supporting evidence: Mean Token Accuracy also peaked near step 800
  (0.8907) then slightly declined; Entropy dropped sharply after step
  800 (model became more confident specifically during the overfitting
  period — consistent with memorizing training-set specifics rather than
  generalizing).
- **Conclusion**: the final (epoch-3) checkpoint is likely NOT the best
  one. The epoch-2 checkpoint (closest saved checkpoint to the step-800
  low point — we only have checkpoints at epoch boundaries, since
  `save_strategy="epoch"`) is the better candidate by validation loss.
  This is being verified empirically on the actual test set (see
  Accuracy Results below), not assumed from validation loss alone.

## Checkpoints saved and pushed

Two checkpoints were pushed to Hugging Face Hub (private repos) for
comparison — epoch 1 was skipped (clearly undertrained):

| Checkpoint | Step | Repo |
|---|---|---|
| End of epoch 2 (best val loss) | 842 | [sardakartik/qwen2.5-0.5b-gsm8k-sft-epoch2](https://huggingface.co/sardakartik/qwen2.5-0.5b-gsm8k-sft-epoch2) |
| End of epoch 3 (final, overfit) | 1263 | [sardakartik/qwen2.5-0.5b-gsm8k-sft-epoch3](https://huggingface.co/sardakartik/qwen2.5-0.5b-gsm8k-sft-epoch3) |

Both are LoRA adapter-only repos (not full models) — evaluated by
loading the base model + attaching the adapter via
`peft.PeftModel.from_pretrained` (see `eval/accuracy.py`'s
`ADAPTER_REPO` option).

## Accuracy results (GSM8K test set, 1,319 examples)

| Checkpoint | Accuracy | Notes |
|---|---|---|
| Base model (benchmark(0)) | 42.15% (556/1319) | No training — the control |
| SFT epoch 2 | ~27% at 104/1319 examples (95% CI roughly 18–36%); run stopped early, was still climbing (9% at n=32 → 27% at n=104). PARTIAL, not a final number | Probably below base, not yet confirmed on the full test set |
| SFT epoch 3 | 12.50% at 32/1319 examples only (4/32; epoch 2 was 9.38% = 3/32 at the same point) | INCONCLUSIVE — n=32 cannot separate it from epoch 2 or from base; do NOT read this as "worse than epoch 2" |

Both SFT numbers above are partial. Full 1,319-question runs are needed
before any claim about epoch 2 vs epoch 3 vs base.

## SFT v2 — retrain with matched prompt and gentler settings

**Why:** v1 trained on the bare question but was evaluated with the
instruction prompt (a mismatch worth about 10 points on validation), and
its validation loss rose during epoch 3.

**What changed in v2:**
- The training question now includes the same instruction text that
  `eval/accuracy.py` uses.
- Learning rate 5e-5 (v1: 1e-4), 2 epochs (v1: 3).
- Checkpoint every 200 steps; the best one by validation loss is reloaded
  at the end.
- Unchanged: LoRA r=16, alpha=32, dropout 0.05, all linear layers
  (8,798,208 trainable parameters), effective batch size 16, fp16, Kaggle T4.

**Training (842 steps, 1,896s ≈ 31.6 min, 0.444 steps/s):**

| Step | Training Loss | Validation Loss | Entropy | Mean Token Accuracy |
|---|---|---|---|---|
| 100 | 0.4243 | 0.4061 | 0.3911 | 0.8830 |
| 200 | 0.4087 | 0.3977 | 0.3975 | 0.8858 |
| 300 | 0.4359 | 0.3921 | 0.3998 | 0.8871 |
| 400 | 0.4134 | 0.3893 | 0.3918 | 0.8882 |
| 500 | 0.3685 | 0.3887 | 0.3693 | 0.8884 |
| 600 | 0.3743 | 0.3876 | 0.3657 | 0.8883 |
| 700 | 0.3736 | 0.3860 | 0.3676 | 0.8892 |
| 800 | 0.3804 | **0.3849** (best) | 0.3673 | 0.8894 |
| 842 | 0.3619 | 0.3849 | 0.3645 | 0.8896 |

**Best checkpoint:** `checkpoint-800`, eval_loss 0.384918 (step 842 was
0.384922, a tie). Pushed to
[sardakartik/qwen2.5-0.5b-gsm8k-sft-v2](https://huggingface.co/sardakartik/qwen2.5-0.5b-gsm8k-sft-v2)
(LoRA adapter only; load it on top of `Qwen/Qwen2.5-0.5B-Instruct` via
`ADAPTER_REPO`).

**Observations:**
- Validation loss fell through the whole run (0.406 → 0.385) and was flat at
  the end. v1's rose from 0.383 to 0.403 in its last epoch. No overfitting
  signal in v2.
- Final training loss is 0.362 (v1 reached 0.285), so v2 memorized less.
- Validation loss only measures how well the model imitates the GSM8K
  solutions. v1 showed it can look fine while accuracy drops, so the
  test-set result below decides.

**Test accuracy (full 1,319 questions, instruction prompt, greedy):**

| Model | Accuracy |
|---|---|
| Base (no training) | 42.15% (556/1319) |
| SFT v2 (checkpoint-800) | **33.66% (444/1319)** — 8.5 points lower |

Question-by-question comparison on the same 1,319 test questions: base-only
right 258, SFT-v2-only right 146, both right 298, both wrong 617 (sign test
p ≈ 3e-8, so the drop is real). SFT v2 got 54% of the questions the base
got right, and 19% of the ones the base got wrong.

What changed in the answers: SFT v2 writes `####` in 1,318/1,319 answers
(base: 22) and never starts with the answer (v1 did, 18/300 on validation),
so the matched prompt fixed those problems. But its answers are about 3.7×
shorter (mean 266 characters vs 983 for the base). That is consistent with
the idea that SFT on the short GSM8K human solutions replaced the model's
own longer step-by-step solving, but it has not been tested directly.

**Conclusion for the SFT arm:** with matched prompts, a gentle learning
rate, 2 epochs and no overfitting, SFT on the human GSM8K solutions still
lowers test accuracy for Qwen2.5-0.5B-Instruct. This is the SFT arm's final
result. GRPO therefore starts from the base model.

### Why did SFT lower accuracy? (working explanation)

Qwen2.5-0.5B-Instruct is already a heavily post-trained model: Qwen's
technical report describes SFT on over 1 million samples plus offline DPO
and online GRPO (the 1M figure may be the total across those stages). Our
SFT used 6,725 short, human-written GSM8K solutions — far less data, in a
terser style than the model's own. Fine-tuning on that pulls the model
toward the short style and away from its own longer step-by-step solving.

Evidence consistent with this:
- Our experiment: answers became 3.7× shorter (266 vs 983 characters) and
  test accuracy fell 8.5 points (p ≈ 3e-8), with matched prompts, a gentle
  learning rate, 2 epochs and no overfitting (validation loss never rose).
  So the drop is not explained by overfitting or by a prompt mismatch.
- Research: fine-tuning on a small dataset causes forgetting, and LoRA does
  not prevent it (Scaling Laws for Forgetting When Fine-Tuning LLMs, arXiv
  2401.05605); human-written reasoning traces can sit far from a model's own
  output style (Shape of Thought, arXiv 2512.22255); training on the model's
  own outputs reduces the damage (self-distillation fine-tuning, arXiv
  2402.13669; rejection-sampling fine-tuning, arXiv 2308.01825).

Not verified: we have not run the control that would confirm this (for
example, the same SFT on the non-instruct Qwen2.5-0.5B base model, or SFT on
the model's own correct solutions). The sources are mostly preprints and
none test our exact setup. Also note: this SFT used LoRA in fp16, not
QLoRA (no 4-bit quantization), so quantization is not a factor.

## Diagnostic runs: prompt × model (validation split, first 300 questions)

All four runs: same 300 validation questions, greedy decoding,
`MODEL_NAME = Qwen/Qwen2.5-0.5B-Instruct`, checkpoint attached via
`ADAPTER_REPO` (None for the base model). Raw answers were saved with
`SAVE_OUTPUTS_PATH` (`base_instruction_val300`, `base_bare_val300`,
`ep2_instruction`, `ep2_bare` .jsonl).

| Model | Instruction prompt ("Solve step by step... #### <number>") | Bare prompt (= SFT training prompt) |
|---|---|---|
| Base | 60.67% (182/300) | 56.00% (168/300) |
| SFT epoch 2 | 38.33% (115/300) | 49.00% (147/300) |

Setup note: the first two SFT runs mistakenly had `MODEL_NAME` set to the
adapter repo (adapter attached twice, peft warning). Re-running with the
base model in `MODEL_NAME` reproduced identical numbers, so that slip was
harmless.

**Findings (paired comparisons, same questions):**
- Base beats SFT-epoch-2 under BOTH prompts. Instruction prompt: base-only
  right 96 vs SFT-only right 29 (sign test p < 0.0001). Bare prompt:
  59 vs 38 (p ≈ 0.04, borderline).
- SFT is prompt-sensitive; base is not. SFT: instruction → bare = +10.7
  points (62 vs 30 flips, p ≈ 0.001). Base: 60.7 vs 56.0 (36 vs 22
  flips, p ≈ 0.09, not significant), and in the opposite direction.
- SFT outputs almost always follow the `####` format (300/300 and
  299/300); the base model almost never does (2/300 and 1/300), so base
  answers are scored through the "last number in the text" fallback while
  SFT answers are scored on `####`. A spot check of 4 wrong base answers
  showed the extractor picked the number the model actually gave; only 3
  of 300 base outputs hit the 512-token cap. This is still a different
  scoring path for the two models — mention it in the write-up.
- Instruction prompt on SFT: 18/300 outputs start with `#### <number>`
  (answer before reasoning), only 1 of them correct. Upper bound on its
  effect ≈ 5.7 points of the 10.7-point prompt gap; scoring the last
  `####` instead of the first changes SFT-instruction by only ~1.3 points.
- The instruction prompt also changes the SFT model's writing style (210/300
  start with "First find..." vs 34/300 under the bare prompt); those
  outputs score about the same as the rest, so the mechanism behind the
  remaining gap is still open.
- The "×7 / 7 days" pattern appears at the same rate under both prompts
  (18 vs 21 of 300). The earlier "learned ×7 shortcut" story (from 5
  questions) is retracted.

**Open issue — base model validation vs test mismatch:** base +
instruction scores 60.67% on validation-300 but 42.15% on the full test
set (same prompt, same scoring). An 18-point gap is far beyond sampling
noise (about ±3). Hypothesis, NOT verified: validation questions come
from GSM8K's train split, which Qwen's own training data may have
included. Consequence: validation is not a trustworthy stand-in for
base-model ability, and the real comparison has to be made on the test set.

**Next:** run base + instruction on the first 300 TEST questions with the
current code (separates "split effect" from "code/version effect"), then
base + bare, SFT-epoch-2 + bare and SFT-epoch-2 + instruction on test.
Report every model × prompt combination; do not pick the best prompt per
model after seeing test results.

**Still unknown:** SFT epoch 2 vs base on the test set (only a partial
SFT run exists: ~27% at n=104); SFT epoch 3 on any full run; which prompt
is the official one for benchmark(0..3) (baseline used "instruction", SFT
trained on "bare").

(The sections below were written before these runs. Treat the
"overfitting" and "×7 shortcut" explanations in them as superseded.)

## Qualitative observation: SFT epoch-2 outputs contain invented steps (CAUSE NOT YET ESTABLISHED)

Ran a diagnostic printing full raw generated text (not just correct/
incorrect) for the first 5 test questions on the SFT epoch-2 checkpoint,
to look into why accuracy (~27%, partial) came in below the base model's
42%.

**Status of explanations (hypotheses, none confirmed):**
- *Prompt mismatch* (training saw the bare question; eval appends a
  "Solve step by step... #### <number>" instruction): NOT ruled out. Well-
  formed output only shows the model can follow the format, not that the
  mismatch costs no accuracy. Test: eval with the training-style prompt.
- *Overfitting*: does not explain epoch 2 well — epoch 2 sits at the
  validation-loss minimum, yet is already below base (see
  `sft_training_guide.md` / chat analysis). May contribute to epoch 3.
- *Style/distribution shift*: SFT on terse human GSM8K solutions may
  replace the instruct model's own, stronger chain-of-thought. Plausible,
  untested. Test: run the base model on the same 5 questions and compare.

**Pattern seen in 5 examples (n=5, anecdotal; base model's outputs on
these same questions were NOT inspected, so we cannot say these errors
are SFT-specific):** reflexively multiplying/dividing by numbers like 7
(days in a week) or 2 where the question doesn't call for it:

- *Janet's ducks* (16 eggs/day, no week conversion needed): model
  invented `16 × 7 = 112` before continuing — ground truth 18, predicted 190.
- *James running* ("3 times a week"): correctly identified `3 × 60 = 180`
  meters per session, then incorrectly multiplied by **7** again anyway
  (`180 × 7 = 1260`) — ground truth 540, predicted 1260.
- *House flipping*: skipped the actual required 150%-increase
  calculation and instead invented an unrelated "divide by 2" step —
  ground truth 70000, predicted 25000.
- *Wendi's chickens*: performed an irrelevant division instead of the
  required `(20 chickens × 3 cups) − (15+25 given) = 20` — ground truth
  20, predicted 2.
- *Robe (bolts)*: correct (`2 + 1 = 3`) — a simple single-step question
  with no "per day/week" framing for the shortcut to intrude on.

**Interpretation (tentative)**: the errors look like reasoning mistakes
(invented ×7 / ÷2 steps, one plain arithmetic slip: `112-3-4` written as
95), but five examples cannot tell us whether SFT caused them or the
base model makes the same ones. The "learned shortcut from training
data" story is a hypothesis; we have not checked how often ×7 patterns
appear in the GSM8K train set or what the base model does on these
questions. Next step is the three tests listed under "Status of
explanations", not further interpretation.

## Known issues hit during this run (for future reference)

- `DataCollatorForCompletionOnlyLM` was removed from current `trl` —
  switched to the prompt-completion dataset format with
  `completion_only_loss=True` instead (see `sft_training_guide.md`).
- `peft`'s LoRA injection failed with an `ImportError` about an
  incompatible `torchao` version pre-installed on Kaggle — fixed by
  adding `-U torchao` to the install cell.
- `SFTTrainer` raised a `ValueError` on Kaggle's `GPU T4 x2` (two GPUs
  visible to one process, unsupported `nn.DataParallel` path) — fixed by
  setting `CUDA_VISIBLE_DEVICES=0` before any other imports, restricting
  the process to a single GPU (sufficient for a 0.5B model).
