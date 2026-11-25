# Qwen2.5 Model — Reference Notes

## Naming

| Part of name | Meaning |
|---|---|
| `Qwen` | Model family (made by Alibaba's Qwen team) |
| `2.5` | Generation/version number — NOT parameter count |
| `0.5B` / `1.5B` / etc. | Actual size — the real parameter count |
| `Instruct` | Instruction/chat fine-tuned version (not the raw base model) |

Qwen2.5 is a whole family of models sharing the same generation name but
different sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B — all "Qwen2.5", just
different parameter counts.

## This project uses: Qwen2.5-0.5B-Instruct (primary); Qwen2.5-1.5B-Instruct is an optional scale-up only

### Qwen2.5-0.5B-Instruct — parameter breakdown

| Component | Value |
|---|---|
| Total parameters | ~0.49 billion (494 million) |
| Non-embedding parameters | ~0.36 billion (360 million) |
| Embedding table parameters (approx.) | ~0.13 billion (134 million) |

### Architecture config (0.5B)

| Config | Value |
|---|---|
| Hidden size | 896 |
| Number of layers | 24 |
| Attention heads (Q) | 14 |
| Key-value heads | 2 (Grouped Query Attention) |
| Intermediate (FFN) size | 4,864 |
| Vocabulary size | ~151,936 tokens |
| Max context length | 32,768 tokens |
| Max generation length | 8,192 tokens |
| Activation function | SiLU |
| Position embeddings | RoPE |
| Other features | SwiGLU, RMSNorm, QKV bias, tied word embeddings |

## What "weights / parameters" means (plain terms)

| Concept | Explanation |
|---|---|
| Weight / parameter | One individual adjustable number inside the model — one knob out of millions |
| Embedding table | Knobs that convert each of the ~152K known tokens into a numeric vector ("dictionary") |
| Attention layers (Q/K/V/O) | Knobs deciding which other words in a sentence to "pay attention to" |
| Feed-forward layers | Knobs that further process/refine what attention gathered |
| Why embedding table is a big chunk here | Vocabulary size (~152K tokens) doesn't shrink just because the rest of the model is small — so for a 0.5B model, the embedding table is a disproportionately large share of total parameters |
| Training (SFT, via LoRA) | We don't adjust all 494M knobs — LoRA adds and trains a small set of extra low-rank matrices on top of the frozen original weights (see `models/sft/sft_training_guide.md`) |

## Sources

- [Qwen/Qwen2.5-0.5B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [config.json · Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blame/main/config.json)
- [Qwen2.5-0.5B Architecture Teardown — KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-2-5-0-5b-architecture-teardown)