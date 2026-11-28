# Results

All held-out numbers use greedy decoding on GSM8K questions the model never
practised on. **Compare only rows that share the same prompt, split and
max-new-tokens setting.** A different prompt moves accuracy by itself (about
10 points in our SFT diagnostics, see `models/sft/sft_result.md`), so a number
from one group must not be read against a number from another group.

| Model | Prompt used at evaluation | Split | Max new tokens | Accuracy | Format rate | Status |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (base) | old `####` instruction | test (1,319) | 512 | 42.15% (556/1319) | n/a | measured |
| SFT v2 (LoRA, best checkpoint) | old `####` instruction (the prompt it was trained with) | test (1,319) | 512 | 33.66% (444/1319) | n/a | measured |
| Qwen2.5-0.5B-Instruct (base) | GRPO `<think>` + `\boxed{}` | test (1,319) | 1024 | 43.29% (571/1319) | 0.00% (boxed 95.98%) | measured |
| GRPO pilot (500 steps) | GRPO `<think>` + `\boxed{}` | test (1,319) | 1024 | **52.62% (694/1319)** | 0.00% (boxed 99.85%) | measured |
| GRPO, next training run (validation monitored during training) | GRPO `<think>` + `\boxed{}` | test (1,319) | 1024 | _todo_ | _todo_ | not started |
| Distilled | its own training prompt | test (1,319) | _tbd_ | _todo_ | n/a | not started |

## Which rows can be compared

- **Rows 1 and 2** (old `####` prompt, test split): base vs SFT. SFT scored
  lower than the base model, which is why GRPO starts from the base model.
- **Rows 3 and 4** (GRPO prompt, test split): base vs the GRPO pilot. This is
  the first held-out measure of what GRPO training did: 43.29% to 52.62%.
- **Row 5** is compared with row 3 (the same base-model baseline).
- The old 42.15% is **not** the baseline for GRPO. The baseline for GRPO is the
  base model measured under the GRPO prompt (row 3).

Each fine-tuned model is evaluated with the prompt it was trained with. Using a
different prompt than the one a model was trained on cost the first SFT
version about 10 points.

## Base model under the GRPO prompt (details)

Greedy decoding, test split, 1,024 new tokens, Kaggle T4, batch size 8:

| Metric | Value |
|---|---|
| Accuracy | 43.29% (571/1319) |
| Strict accuracy (right AND full format) | 0.00% |
| Format rate (`<think>...</think>` + `\boxed{}`) | 0.00% |
| `\boxed{}` rate | 95.98% |
| Mean answer length | 310 tokens |
| Truncated (hit the 1,024 limit) | 0.38% |
| Eval time | 3,493 s (about 58 min) |

Reading: the base model solves about 43% of the test questions, writes a
`\boxed{}` almost every time, and never writes the `<think>` tags. With 1,319
questions the statistical noise is about +/- 1.4 points (one standard error),
so a difference of less than about 3 points between two models should not be
treated as real. Per-question outputs were not saved for this run.

## GRPO pilot vs base (same prompt, test split, greedy, 1,024 tokens)

| Metric | Base | GRPO pilot (500 steps) | Change |
|---|---|---|---|
| Accuracy | 43.29% (571/1319) | **52.62% (694/1319)** | +9.33 points (+123 questions) |
| Strict accuracy | 0.00% | 0.00% | none |
| Format rate | 0.00% | 0.00% | none |
| `\boxed{}` rate | 95.98% | 99.85% | +3.87 points |
| Mean answer length | 310 tokens | 286 tokens | -24 |
| Truncated (hit 1,024) | 0.38% | 0.15% | -0.23 points |
| Eval time (Kaggle T4) | 3,493 s | 5,599 s | |

Adapter: `sardakartik/qwen2.5-0.5b-gsm8k-grpo`, subfolder `final`. Per-question
outputs of the pilot run were saved on Kaggle
(`grpo_pilot_think_boxed_test.jsonl`).

Reading:
- The gap is about 4-5 standard errors (noise is about +/- 1.4 points per
  model), so the improvement is real, not noise.
- It transfers to unseen questions: about 47% to 62% on training questions
  (sampled), 43% to 53% on test (greedy). The held-out gain is smaller, as
  expected.
- The gain comes from the correctness reward. The `<think>` format was never
  learned (0% before and after), so the +0.2 format reward had no effect. The
  correctness check falls back to the last number when there is no `\boxed{}`,
  so the +3.9-point boxed rate does not by itself explain the gain.
- Caveats: one run, one seed; the same test set will be used for later GRPO
  models; the pilot did not show that GRPO teaches `<think>` reasoning, only
  that accuracy improved.

## How the splits are used

- **Training:** the `train` split only. The 500-step pilot had no validation
  set.
- **Test:** the base-vs-pilot comparison and the final models.
- **Validation:** planned to be passed to the trainer in the next GRPO run, so
  progress on held-out questions is visible during training (the exact TRL
  settings still have to be checked). It is also used for choosing between
  options.
- **Caveat to report:** the same test set is used for more than one GRPO model
  (the pilot, then the next run), and the pilot's test result is seen before
  the next run is planned. With 1,319 questions and only a few models the effect
  is small, but it must be stated alongside the numbers.

The evaluation script is `eval/accuracy.py` (`PROMPT_STYLE = "think_boxed"` for
the GRPO prompt, `"instruction"` for the old `####` prompt, `SPLIT = "test"` or
`"validation"`).

## GRPO training curve (not a held-out result)

During the 500-step pilot the model's score on its **training** questions
(sampled at temperature 1.0) rose from about 47% to about 62%, but the
format reward stayed at about 0. This is a training signal, not a held-out
accuracy. All 50 reports and the reward functions are in
`models/grpo/grpo_result.md`.