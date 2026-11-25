# SFT Training Guide

## Dataset

### Source

- Dataset: GSM8K (Cobbe et al., 2021), MIT-licensed.
- We use our own private split, pushed earlier to Hugging Face Hub:
  `sardakartik/gsm8k-split` (see `data/prepare_gsm8k.py`).
- Splits: `train` (6,725 examples), `validation` (748 examples),
  `test` (1,319 examples).

### Which split SFT uses

- **Training data**: `train` split only (6,725 examples).
- **Validation**: `validation` split (748 examples) — used only for
  monitoring during training (loss curves, checkpoint selection), never
  used to report final results.
- **Test set is NOT used during SFT at all** — it stays completely
  untouched, reserved only for the final `eval/accuracy.py` benchmark run
  after training is done (benchmark(1)).

### Input (X) and Target (Y) for each training example

Each GSM8K example looks like this (real example from our train split):

```json
{
  "question": "...",
  "answer": "In two years, Lennon will be 2+8 = <<2+8=10>>10 years old.\nOphelia will be 4*10 = <<4*10=40>>40 years old in two years.\nCurrently, Ophelia is 40-2 = <<40-2=38>>38 years old.\n#### 38"
}
```

- **X (input)** = the `question` field plus the evaluation instruction
  ("Solve this step by step, then give the final answer on its own line in
  the format: #### <number>"), exactly as `eval/accuracy.py` sends it.
  *(v1 trained on the bare question only; that mismatch with the
  evaluation prompt cost the SFT model ~10 points — see `sft_result.md`.
  v2 trains on question + instruction.)*
- **Y (target)** = the **full `answer` field**, as-is — all reasoning
  steps plus the final `#### <number>` line. NOT just the final number.

### Why Y is the full reasoning text, not just the final number

If the model were trained only on the bare final number (e.g. `38`), it
would learn to associate specific question wording with a number —
essentially memorization, not reasoning. It would not learn *how* to
break a word problem into steps, and would not generalize to new,
differently-worded problems at test time.

Training on the full worked solution teaches the model the actual
step-by-step reasoning *process* (what to compute first, what depends on
what, in what order) — the actual skill we want it to apply to unseen
test questions.

This also matters structurally: our evaluation (`eval/accuracy.py`) and
GRPO's reward function both expect the model to produce reasoning ending
in `#### <number>`. If SFT only ever taught the model to output a bare
number, it wouldn't know how to produce this format or any reasoning
structure at all — which would break both evaluation and the later GRPO
phase (which starts from the SFT checkpoint).

## Loss Function

**What it is**: Cross-Entropy Loss (= Negative Log-Likelihood) on
next-token prediction. Same loss used in original LLM pretraining — SFT
just continues it on GSM8K-specific data.

### Per-position loss

```
Loss_t = -log( P(y_t given y_<t) )
```

- `y_t` = actual correct token at position t
- `y_<t` = all tokens before position t (context seen so far)
- `P(y_t given y_<t)` = probability the model assigned to the correct token
- High confidence in the right answer → loss near 0. Low/wrong confidence
  → large loss.

### Masked loss — only count answer tokens, not question tokens

```
Loss = -(1 / sum of all m_t) * sum over t=1 to T of [ m_t * log( P(y_t given y_<t) ) ]
```

- `m_t` = mask: 1 = count this position (answer token), 0 = ignore
  (question token)
- Implemented via PyTorch's `CrossEntropyLoss` using `ignore_index = -100`
  (its default value) — question-token labels get set to `-100`, the loss
  function skips them automatically, and averages only over the
  non-ignored (answer) positions.
- This is why Y is the full reasoning text but the loss only trains on
  the answer portion — see "Why Y is the full reasoning text" above for
  the data side of this; this is the mechanism that enforces it during
  training.

### Batched version (what actually runs in training)

```
Loss_batch = (1 / B) * sum over i=1 to B of [ Loss for example i ]
```

One single number per batch → used to compute gradients → updates only
the LoRA adapter weights (base model frozen, see "Fine-Tuning Method"
below).

### Key terms

- **Teacher forcing**: during training, the model is always shown the
  *real* previous tokens from the dataset (never its own guesses) — this
  is what makes one-shot parallel computation of the loss possible
  (vs. token-by-token generation at inference time, which is sequential).
- **Completion-only loss**: the common name for this masking technique
  (loss only on the "completion"/answer, not the "prompt"/question).

### Sources

- [CrossEntropyLoss — PyTorch official documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html)
- [Will Trainer loss functions automatically ignore -100? — Hugging Face Forums](https://discuss.huggingface.co/t/will-trainer-loss-functions-automatically-ignore-100/36134)
- [Memory saving by upcasting logits for only non-ignored positions · Issue #38452 — huggingface/transformers](https://github.com/huggingface/transformers/issues/38452)
- [Why is inference sequential while training is much more parallel? — Sebastian Raschka](https://sebastianraschka.com/faq/docs/inference-sequential-vs-training-parallel.html)

## Fine-Tuning Method: LoRA

We do **not** update all ~494M base model weights. We use **LoRA**
(Low-Rank Adaptation): the base model is frozen, and small trainable
low-rank adapter matrices are added to all linear layers
(q/k/v/o_proj, gate/up/down_proj). Only these adapters are trained.

- rank (r) = 16, alpha = 32, dropout = 0.05
- Why LoRA: far less VRAM/compute than full fine-tuning, tiny checkpoint
  size (easy to push/pull between Kaggle/RunPod/HF Hub), preserves the
  base model's general knowledge (less catastrophic forgetting).

## Framing: SFT is off-policy, imitation learning

Training data here is fixed and external (human-written, never generated
by the model being trained) — this is supervised imitation learning, not
reinforcement learning. See `docs/project.md` Section 5 for the full
on-policy vs. off-policy comparison.
