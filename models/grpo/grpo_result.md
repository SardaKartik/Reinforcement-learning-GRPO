# GRPO Results

Qwen2.5-0.5B-Instruct (base) trained with GRPO (TRL `GRPOTrainer`, LoRA, vLLM)
on GSM8K. This file records the pilot run: the reward functions, the settings,
and every logged report. All hyperparameters and the reason for each are in
`grpo-hyperparameters.md`.

**Important:** every "correct" number below is measured on the **training**
questions the model was practising on (the `train` split, 6,725 questions), with
random sampling at temperature 1.0. It is NOT a held-out score. The validation
split was not used during training at all (no eval dataset was given to the
trainer). The held-out accuracy is measured separately (see the last section).

## Reward functions

Two separate functions in `rewards.py`. TRL calls each one with the model's
answers (8 per question) and sums them with the weights `[1.0, 0.2]`:

```
total reward = 1.0 x correctness_reward + 0.2 x format_reward
```

| Function | Gives | Rule |
|---|---|---|
| `correctness_reward` | 1.0 if right, else 0.0 | Take the final answer: look only after the last `</think>` if there is one; use the last `\boxed{...}` there; otherwise fall back to the last number in the text. Compare with the GSM8K answer as numbers (`18` equals `18.0`, `1,234` equals `1234`). It does **not** need any tags, because the base model almost never writes them. |
| `format_reward` | 1.0 if the format is right, else 0.0 | Exactly one `<think>` and one `</think>`, followed later by a non-empty `\boxed{...}`. Returned as 0/1 so the logged mean equals the **format rate**; the 0.2 weight is applied by the trainer. |

Examples (answer is 18):

| Model output | correctness | format | total |
|---|---|---|---|
| `<think>9*2=18</think> The answer is \boxed{18}` | 1 | 1 | 1.2 |
| `Steps... so \boxed{18}` | 1 | 0 | 1.0 |
| `<think>...</think> \boxed{26}` | 0 | 1 | 0.2 |
| `Steps... the answer is 26` | 0 | 0 | 0.0 |

The reward functions were tested on 13 cases in `rewards.py` (all pass).

## Setup of the pilot

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Adapter | LoRA r=16, alpha=32, dropout 0.05, all 7 linear layers (about 8.8M trainable params) |
| Prompt | `models/grpo/prompt.py` (`<think>` + `\boxed{}`), with Qwen's default system message added by the chat template |
| Attempts per question | 8, temperature 1.0, max 1,024 new tokens |
| Batching | 8 attempts x 4 accumulation = 32 attempts per step = 4 questions, reused for 2 steps (`num_iterations=2`), so 2 NEW questions per step |
| Loss | `dr_grpo`, `scale_rewards="none"`, `beta=0`, `epsilon=0.2` |
| Learning rate | 1e-5, constant |
| Library versions | trl 1.19.0, vllm 0.34.0, torch 2.14.0, transformers 5.22.2, peft 0.22.0, datasets 5.2.0 |
| GPU | 1x A40 (46 GB usable), Secure Cloud, $0.60/hr |
| Steps | 500 (35 min 35 s, 4.27 s per step) |
| New questions seen | 500 x 2 = 1,000 (about 15% of the 6,725; each question seen once) |
| Peak GPU memory | 15.74 GB allocated, 16.23 GB reserved (PyTorch); at least 17,053 MiB of 46,068 MiB seen by `nvidia-smi` (sampled every 10 s) |
| Saved to | private Hub repo `sardakartik/qwen2.5-0.5b-gsm8k-grpo` (adapter at the top level and in `final/`, resumable `last-checkpoint/`, sample answers in `completions/`) |

Epoch arithmetic: `epoch = new questions / 6,725`, and 1 step = 2 new
questions. 10 steps = 20 questions = epoch 0.003. 500 steps = 1,000 questions =
epoch 0.149. A full epoch is about 3,362 steps (about 4.0 h, about $2.40).

## Column meanings

| Column | Meaning |
|---|---|
| step | optimizer steps done (each report covers the last 10 steps) |
| epoch | share of the 6,725 training questions used so far |
| correct | mean of `correctness_reward`: share of attempts with the right final answer |
| format | mean of `format_reward`: share of attempts in the `<think>...</think>` + `\boxed{}` format |
| len | mean answer length in tokens |
| clipped | share of answers cut off at the 1,024-token limit |
| zerostd | share of question groups where all 8 attempts got the same reward (no learning signal) |
| clip_ratio | share of tokens whose update hit the clipping limit (`clip_ratio/region_mean`) |

## Every report of the pilot (training questions)

| step | epoch | correct | format | len | clipped | zerostd | clip_ratio |
|---|---|---|---|---|---|---|---|
| 10 | 0.003 | 0.519 | 0.000 | 299 | 0.006 | 0.20 | 0.0010 |
| 20 | 0.006 | 0.438 | 0.000 | 308 | 0.000 | 0.25 | 0.0011 |
| 30 | 0.009 | 0.463 | 0.000 | 308 | 0.000 | 0.35 | 0.0011 |
| 40 | 0.012 | 0.362 | 0.000 | 326 | 0.006 | 0.15 | 0.0010 |
| 50 | 0.015 | 0.388 | 0.000 | 320 | 0.000 | 0.20 | 0.0012 |
| 60 | 0.018 | 0.494 | 0.000 | 338 | 0.000 | 0.25 | 0.0013 |
| 70 | 0.021 | 0.300 | 0.000 | 330 | 0.000 | 0.35 | 0.0010 |
| 80 | 0.024 | 0.556 | 0.000 | 321 | 0.006 | 0.45 | 0.0010 |
| 90 | 0.027 | 0.487 | 0.000 | 305 | 0.006 | 0.50 | 0.0009 |
| 100 | 0.030 | 0.700 | 0.000 | 276 | 0.000 | 0.35 | 0.0015 |
| 110 | 0.033 | 0.500 | 0.000 | 294 | 0.000 | 0.25 | 0.0015 |
| 120 | 0.036 | 0.613 | 0.000 | 277 | 0.006 | 0.25 | 0.0014 |
| 130 | 0.039 | 0.644 | 0.000 | 276 | 0.006 | 0.30 | 0.0015 |
| 140 | 0.042 | 0.562 | 0.000 | 269 | 0.000 | 0.30 | 0.0015 |
| 150 | 0.045 | 0.444 | 0.000 | 292 | 0.000 | 0.20 | 0.0013 |
| 160 | 0.048 | 0.588 | 0.000 | 292 | 0.000 | 0.55 | 0.0009 |
| 170 | 0.051 | 0.656 | 0.000 | 272 | 0.000 | 0.40 | 0.0012 |
| 180 | 0.054 | 0.613 | 0.000 | 244 | 0.000 | 0.15 | 0.0016 |
| 190 | 0.057 | 0.519 | 0.000 | 269 | 0.000 | 0.40 | 0.0013 |
| 200 | 0.059 | 0.481 | 0.000 | 269 | 0.000 | 0.20 | 0.0019 |
| 210 | 0.062 | 0.588 | 0.000 | 279 | 0.000 | 0.40 | 0.0015 |
| 220 | 0.065 | 0.519 | 0.000 | 319 | 0.000 | 0.20 | 0.0013 |
| 230 | 0.068 | 0.531 | 0.000 | 300 | 0.000 | 0.35 | 0.0014 |
| 240 | 0.071 | 0.600 | 0.000 | 302 | 0.000 | 0.20 | 0.0019 |
| 250 | 0.074 | 0.669 | 0.000 | 261 | 0.000 | 0.45 | 0.0014 |
| 260 | 0.077 | 0.662 | 0.000 | 258 | 0.000 | 0.40 | 0.0014 |
| 270 | 0.080 | 0.669 | 0.000 | 266 | 0.000 | 0.40 | 0.0015 |
| 280 | 0.083 | 0.506 | 0.006 | 289 | 0.000 | 0.50 | 0.0012 |
| 290 | 0.086 | 0.650 | 0.000 | 289 | 0.006 | 0.40 | 0.0015 |
| 300 | 0.089 | 0.506 | 0.000 | 282 | 0.000 | 0.45 | 0.0016 |
| 310 | 0.092 | 0.594 | 0.000 | 264 | 0.000 | 0.25 | 0.0021 |
| 320 | 0.095 | 0.606 | 0.000 | 278 | 0.000 | 0.40 | 0.0018 |
| 330 | 0.098 | 0.631 | 0.000 | 257 | 0.006 | 0.50 | 0.0013 |
| 340 | 0.101 | 0.588 | 0.000 | 290 | 0.000 | 0.15 | 0.0023 |
| 350 | 0.104 | 0.525 | 0.000 | 273 | 0.000 | 0.25 | 0.0022 |
| 360 | 0.107 | 0.700 | 0.000 | 252 | 0.000 | 0.40 | 0.0018 |
| 370 | 0.110 | 0.619 | 0.000 | 282 | 0.000 | 0.40 | 0.0020 |
| 380 | 0.113 | 0.700 | 0.000 | 271 | 0.000 | 0.45 | 0.0015 |
| 390 | 0.116 | 0.625 | 0.000 | 248 | 0.000 | 0.30 | 0.0021 |
| 400 | 0.119 | 0.562 | 0.000 | 274 | 0.000 | 0.45 | 0.0016 |
| 410 | 0.122 | 0.644 | 0.000 | 274 | 0.000 | 0.30 | 0.0016 |
| 420 | 0.125 | 0.569 | 0.000 | 258 | 0.000 | 0.35 | 0.0021 |
| 430 | 0.128 | 0.637 | 0.000 | 256 | 0.000 | 0.40 | 0.0022 |
| 440 | 0.131 | 0.531 | 0.000 | 302 | 0.000 | 0.35 | 0.0021 |
| 450 | 0.134 | 0.619 | 0.000 | 278 | 0.000 | 0.45 | 0.0018 |
| 460 | 0.137 | 0.681 | 0.000 | 263 | 0.000 | 0.45 | 0.0019 |
| 470 | 0.140 | 0.613 | 0.000 | 265 | 0.000 | 0.45 | 0.0023 |
| 480 | 0.143 | 0.606 | 0.000 | 271 | 0.000 | 0.70 | 0.0008 |
| 490 | 0.146 | 0.669 | 0.000 | 278 | 0.006 | 0.20 | 0.0028 |
| 500 | 0.149 | 0.669 | 0.000 | 282 | 0.000 | 0.15 | 0.0032 |

(The epoch column is copied from the log, which rounds it, so one step of 0.002
appears around step 200.)

## Summary by 100 steps

Each block is 10 reports, about 200 new questions, so expect roughly +/- 3-4
points of noise per block.

| Steps | correct (mean) | zerostd (mean) |
|---|---|---|
| 1-100 | 0.471 | 0.305 |
| 101-200 | 0.562 | - |
| 201-300 | 0.590 | - |
| 301-400 | 0.615 | - |
| 401-500 | 0.624 | 0.380 |

## What the pilot shows

- **Correctness improved** on the training questions, from about 47% to about
  62%, mostly in the first 200 steps; the curve flattens after that.
- **The format was never learned.** `format` is 0.000 in 49 of 50 reports (one
  report at 0.006, about 1 attempt in 160). With no attempt in the format, all 8
  attempts get the same format score, so the format reward gave no signal.
  Fallbacks if this matters: split the format reward into parts (`<think>`
  open, `</think>` close, `\boxed{}`), raise its weight, or prefill `<think>`.
- **Health:** answers stayed about 250-340 tokens (no rambling); `clipped` was
  almost always 0; `zerostd` rose from about 30% to about 38% as accuracy rose;
  `clip_ratio` stayed tiny (0.001 to 0.003) but crept up in the last reports.

## Smoke tests (5 steps, 16 questions)

| Run | step time | peak allocated | correct | format | clipped |
|---|---|---|---|---|---|
| First Pod | 4.5 s | 14.97 GB | 0.417 | 0 | 0 |
| Second Pod (pinned versions) | 4.56 s | 15.12 GB | 0.385 | 0 | 0 |

## Held-out evaluation (to be filled)

Not measured yet. Plan: compare the base model and the pilot adapter on the
test split (1,319 questions, greedy decoding, the GRPO prompt, 1,024 new
tokens). The validation split is planned to be passed to the trainer in the
next GRPO run to monitor progress during training. Note for the report: the same
test set is used for more than one GRPO model. The up-to-date table is in
`docs/results.md`.

| Model | Prompt | Split | Accuracy | Format rate | Notes |
|---|---|---|---|---|---|
| Base (re-measured) | GRPO prompt | test | _todo_ | _todo_ | |
| GRPO pilot adapter | GRPO prompt | test | _todo_ | _todo_ | |
| Base (old measurement) | `####` prompt | test | 42.15% (556/1319) | n/a | different prompt, not comparable |