# GRPO Hyperparameters

One table per group: **hyperparameter, value, reason** (one line each).
Full discussion, fallbacks and evidence are in `docs/execution-plan.md`
(section "GRPO Final Config"). Argument names come from the TRL docs
(v1.18.2) and source (main); re-check them against the TRL version installed
on the Pod. "default" means TRL's default, kept on purpose.

## Model and adapter

| Hyperparameter | Value | Reason |
|---|---|---|
| model | `Qwen/Qwen2.5-0.5B-Instruct` (base) | Small enough for the budget; the SFT checkpoint scored below base, so GRPO starts from base |
| dtype (`model_init_kwargs`) | `bfloat16` (proposed, confirm on the Pod) | TRL loads float32 if no dtype is given; bf16 is lighter and matches TRL's `bf16` default |
| LoRA rank `r` | 16 | Same LoRA recipe as SFT and distillation, for a fair comparison |
| LoRA `alpha` | 32 | Scaling factor, 2 x rank |
| LoRA dropout | 0.05 | Same as SFT |
| LoRA target modules | all linear layers (q, k, v, o, gate, up, down proj) | Same as SFT; LoRA works best on all layers, especially MLP |

## Prompt and rewards

| Hyperparameter | Value | Reason |
|---|---|---|
| prompt | Option B (`<think>...</think>` then `\boxed{number}`), text in `models/grpo/prompt.py` | Qwen already writes `\boxed{}`; the same prompt is used for training and eval |
| `reward_funcs` | `[correctness_reward, format_reward]` | Two separate checks, so correctness never depends on the tags |
| `reward_weights` | `[1.0, 0.2]` | Correctness drives accuracy; format is a small nudge toward the `<think>` habit |

## Generation (rollouts)

| Hyperparameter | Value | Reason |
|---|---|---|
| `use_vllm` | `True` | About 11x faster generation, which dominates GRPO cost |
| `vllm_mode` | `"colocate"` (default) | vLLM shares the single GPU with training |
| `vllm_gpu_memory_utilization` | `0.3` (default) | Share of GPU memory vLLM may claim; plenty for a 0.5B model |
| `num_generations` | 8 | Attempts per question (the group GRPO compares) |
| `temperature` | 1.0 (default) | Gives varied attempts; fall back to 0.7 if most groups score all-zero |
| `top_p` / `top_k` | 1.0 / 0 (default) | No filtering, so sampling stays true to the model's own probabilities |
| `repetition_penalty` | 1.0 (default) | Must stay 1.0 while the vLLM importance-sampling correction is on |
| `max_completion_length` | 1024 | Room for `<think>` reasoning plus answer; the old 320 would cut most answers |
| `mask_truncated_completions` | `False` (default) | Cut-off answers score 0, which discourages rambling |
| `vllm_importance_sampling_correction` | `True` (default) | Guards against small numeric differences between vLLM and the trainer |

## Batching

| Hyperparameter | Value | Reason |
|---|---|---|
| `per_device_train_batch_size` | 8 | Attempts per GPU pass (the memory knob); always divides evenly by 8 |
| `gradient_accumulation_steps` | 4 | 8 x 4 = 32 attempts = 4 questions per step, instead of TRL's noisy 1 |
| `num_iterations` | 2 | Reuse each costly generation for 2 weight updates (1 round = 2 steps) |

## Loss and optimisation

| Hyperparameter | Value | Reason |
|---|---|---|
| `loss_type` | `"dr_grpo"` | Removes the length bias of the original GRPO loss |
| `scale_rewards` | `"none"` | Avoids the question-level difficulty bias (Dr. GRPO paper); TRL's default is `"group"`, so set explicitly |
| `beta` (KL) | 0.0 | Saves memory and time (no reference model); rule-based reward is hard to hack |
| `epsilon` (clip) | 0.2 (default) | Limits how far one update can move the model; matters because `num_iterations` is 2 |
| `learning_rate` | 1e-5 | TRL's recommended GRPO rate for LoRA (10x the full fine-tuning rate) |
| `lr_scheduler_type` | `"constant"` | Never shrinks, so a successful pilot can be continued without a learning-rate jump |

## Run length, saving and logging

| Hyperparameter | Value | Reason |
|---|---|---|
| `max_steps` | 5 (smoke test), 500 (pilot) | Smoke test checks the code runs; pilot checks the model is learning (judge at ~300 steps) |
| `num_train_epochs` | 1, with `max_steps = -1` (full run) | One pass over the 6,725 training questions after the pilot passes |
| `save_strategy` | `"steps"` | Save at regular intervals |
| `save_steps` | 24 | About 12 rounds (~48 different questions); re-tune after the smoke test |
| `save_total_limit` | 3 | Keeps local disk use small |
| `push_to_hub` | `True` | Backs up each save in case the Pod is stopped |
| `hub_strategy` | `"checkpoint"` | Also pushes a `last-checkpoint` folder so a fresh Pod can resume |
| `hub_private_repo` | `True` | TRL's default would make the repo public |
| `hub_model_id` | `sardakartik/qwen2.5-0.5b-gsm8k-grpo` (proposed, not confirmed) | Follows the naming of the SFT adapters |
| `logging_steps` | 10 (default) | Frequent enough to watch reward, format rate and length |
| `log_completions` | `True` | Lets us read sample outputs, since there is no `kl` metric at beta 0 |

## Derived numbers (for reference)

| Quantity | Value | Reason |
|---|---|---|
| Attempts per step | 32 | 8 per GPU pass x 4 accumulation steps |
| Questions per step | 4 | 32 attempts / 8 attempts per question |
| Steps per round of generation | 2 | `num_iterations = 2` |
| Different questions in 500 steps | about 1,000 | 250 rounds x 4 questions (about 15% of the training set) |
| Steps in one epoch | about 3,400 | Our estimate from TRL's sampler code; unverified |
