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
| Qwen2.5-0.5B-Instruct (base) | GRPO `<think>` + `\boxed{}` | test (1,319) | 1024 | _todo_ | _todo_ | pending (Kaggle run) |
| GRPO pilot (500 steps) | GRPO `<think>` + `\boxed{}` | test (1,319) | 1024 | _todo_ | _todo_ | pending (Kaggle run) |
| GRPO, next training run (validation monitored during training) | GRPO `<think>` + `\boxed{}` | test (1,319) | 1024 | _todo_ | _todo_ | not started |
| Distilled | its own training prompt | test (1,319) | _tbd_ | _todo_ | n/a | not started |

## Which rows can be compared

- **Rows 1 and 2** (old `####` prompt, test split): base vs SFT. SFT scored
  lower than the base model, which is why GRPO starts from the base model.
- **Rows 3 and 4** (GRPO prompt, test split): base vs the GRPO pilot. This is
  the first held-out measure of what GRPO training did.
- **Row 5** is compared with row 3 (the same base-model baseline).
- The old 42.15% is **not** the baseline for GRPO. The baseline for GRPO is the
  base model measured under the GRPO prompt (row 3).

Each fine-tuned model is evaluated with the prompt it was trained with. Using a
different prompt than the one a model was trained on cost the first SFT
version about 10 points.

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