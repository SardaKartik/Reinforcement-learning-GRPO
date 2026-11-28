# Execution Plan — Step by Step

This is the concrete "what we're actually going to do" checklist, derived
from `project.md`. Update this file as decisions get locked in or revised.

---

## 0. Finalized Choices So Far

| Decision | Choice | Status |
|---|---|---|
| Base model (primary) | Qwen2.5-0.5B-Instruct | Locked |
| Base model (optional scale-up, only if time allows) | Qwen2.5-1.5B-Instruct | Optional |
| Dataset | GSM8K (train + test split) | Locked |
| Teacher model (for distillation) | Google Gemini 2.5 Flash (free API tier, 1,500 requests/day) | Locked |
| Implementation order | SFT → GRPO (RL) → Distillation (last) | Locked |
| Fine-tuning method (SFT + Distillation) | Plain LoRA, fp16 (no QLoRA — not needed at this model size) | Locked |
| Fine-tuning method (GRPO) | `trl.GRPOTrainer`, `use_vllm=True` for rollout generation | Locked |
| SFT/Distillation training compute | Kaggle (free T4 GPU) | Locked |
| GRPO training + all latency/memory benchmarking compute | RunPod (RTX PRO 4000) | Locked |
| Eval dataset | GSM8K test set (exact-match on final numeric answer) | Locked |
| GRPO reward design | Outcome reward: `1.0 × answer correct + 0.2 × format correct` (rule-based; no process reward model / learned judge) | Locked |
| GRPO KL beta | 0.0 (no KL penalty; fallback 0.001 if drift is seen) | Locked |
| GRPO sampling temperature | 1.0 (TRL default; fallback 0.7 if too few correct answers per group) | Locked |
| GRPO num_iterations | 2 (two update passes per generated batch; saves generation cost, clipping at epsilon 0.2 now matters) | Locked |
| GRPO loss_type | `dr_grpo` (removes length bias; constant = max_completion_length = 1024) | Locked |
| GRPO scale_rewards | `"none"` (no division by group std, following the Dr. GRPO paper). NOTE: TRL's default is `"group"` (std scaling ON), so this must be set explicitly. | Locked |
| GRPO max completion length | 1024 tokens (see Locked Hyperparameters) | Locked |
| GRPO batch size / accumulation | `per_device_train_batch_size = 8`, `gradient_accumulation_steps = 4` → 32 attempts = 4 questions per step (see "GRPO Final Config") | Locked |
| GRPO saving / pushing | `save_steps = 24`, `save_total_limit = 3`, push to a PRIVATE Hub repo with `hub_strategy = "checkpoint"` | Locked (repo name still to confirm) |
| GRPO output format | `<think>...</think>` reasoning, then the final answer as `\boxed{number}` ("Option B"; exact prompt text in `models/grpo/prompt.py`). Correctness is read from `\boxed{}` / last number independent of the tags; the format reward is a separate check. | Locked |
| Baseline re-measure | Re-run the BASE model under the Option B prompt on the TEST split and compare the GRPO pilot (and later the final GRPO model) against that new baseline, not the old 42.15% "####"-prompt number. The validation split is planned to be passed to the trainer in the next GRPO run to monitor progress during training. `eval/accuracy.py` now supports this (`PROMPT_STYLE = "think_boxed"`, 1,024 new tokens); the Kaggle runs are pending. All results go in `docs/results.md`. | Required (harness updated; runs pending) |
| GRPO starting point | The **base** Qwen2.5-0.5B-Instruct, NOT the SFT checkpoint (SFT scored below base on the validation diagnostics, see `models/sft/sft_result.md`) | Locked |

---

## Step 0 — Data Prep (done)

- [x] `data/prepare_gsm8k.py` — downloads GSM8K, splits official train into
      train (6,725) / validation (748), keeps official test (1,319)
      untouched, pushes all three to the private HF dataset repo
      `sardakartik/gsm8k-split`.
- [x] `data/verify_upload.py` — confirms the split loads back correctly
      from the Hub.
- **Note for all future scripts**: `datasets` auto-normalizes the
  `val.jsonl` file to the dict key `"validation"` (not `"val"`) when
  loading this repo — use `splits["validation"]`, not `splits["val"]`.

---

## Step 1 — Build the Evaluation Harness First

Before training anything, build the shared eval harness so Model (0)'s
baseline numbers exist for comparison.

- [x] Script to load GSM8K test set (from `sardakartik/gsm8k-split`,
      `splits["test"]`) — `eval/accuracy.py`.
- [x] Script to run inference on a given model/checkpoint and extract the
      final numeric answer (regex extraction) from generated text —
      numeric comparison (not strict string match), batched generation
      (batch size 8, left-padding) for reasonable Kaggle runtime.
- [x] Accuracy = exact-match (numeric) against ground truth, reported as %.
- [ ] Hooks to record: peak VRAM, tokens/sec, time-to-first-token — these
      are deferred to the dedicated `eval/latency.py` (Step 5, vLLM on
      RunPod), not part of this accuracy script.
- [x] Run this harness on the **base model, untouched** → benchmark(0):

  **benchmark(0) — Qwen2.5-0.5B-Instruct (base, zero-shot, greedy), OLD
  `####` instruction prompt, 512 new tokens:**
  - Accuracy: **42.15%** (556/1,319) on the GSM8K test set
  - Eval wall-clock time: ~52 min (3,119.9s) on Kaggle T4, batch size 8
  - This number is only comparable with other runs under the same old prompt
    (for example the SFT model). It is NOT the baseline for GRPO, which uses a
    different prompt; see `docs/results.md`.
- [x] Extend the harness for the GRPO prompt: `PROMPT_STYLE = "think_boxed"`
      (the text of `models/grpo/prompt.py`), answer extraction identical to the
      GRPO correctness reward, 1,024 new tokens, LoRA adapter loading from a
      Hub subfolder, and extra metrics (strict accuracy, format rate, boxed rate,
      mean length, truncated rate). The old `instruction` style is kept so SFT
      can still be evaluated with the prompt it was trained on.
- [ ] Run the **base model** under the GRPO prompt on the test split (the
      baseline for GRPO), then the GRPO pilot adapter on the same split. Note
      in the report that the same test set is used for more than one GRPO
      model.

---

## Step 2 — SFT (Model 1)

**Platform: Kaggle (T4)**

- [ ] Load `Qwen2.5-0.5B-Instruct`.
- [ ] Load GSM8K train split — use the human-written CoT solutions as
      targets (question → full reasoning + final answer).
- [ ] Fine-tune with LoRA (fp16, not QLoRA) using `trl.SFTTrainer`.
- [ ] Save LoRA adapter + push to Hugging Face Hub (private repo).
- [ ] Record training cost: wall-clock time, GPU-hours, peak VRAM.
- [ ] Run Step 1's eval harness on this checkpoint → benchmark(1).

---

## Step 3 — GRPO (Model 2)

**Platform: RunPod (RTX PRO 4000)**

- [ ] Start from the **base** `Qwen2.5-0.5B-Instruct` (decision changed:
      the SFT checkpoint scored below base in the diagnostics, so GRPO no
      longer chains off SFT). Attach a fresh LoRA adapter for GRPO.
- [ ] Prompt the model to reason inside `<think>...</think>` and then give
      the final answer; the +0.2 format reward checks for these tags.
- [ ] Set up `trl.GRPOTrainer` with `use_vllm=True` for fast rollout
      generation (colocated on the same GPU).
- [x] Reward functions written and tested locally: `models/grpo/rewards.py`
      (`correctness_reward` + `format_reward`, weights [1.0, 0.2]; 13 test
      cases pass). Still to check against REAL model output on the Pod.
- [ ] Use only GSM8K **questions + ground-truth final answers** as
      training input — no written solutions needed (on-policy: the model
      generates its own completions at each step).
- [ ] Smoke test first (max_steps = 5, a few questions): confirms the code
      runs. Then a PILOT (max_steps = 500, about 1,000 different questions,
      judged at ~300 steps) to confirm the model is actually learning
      before scaling to the full training set. Constant learning rate, so a
      successful pilot can be continued instead of restarted.
- [ ] Checkpoint frequently (RunPod bills per second while a Pod runs and
      sessions can be interrupted): save_steps = 24, pushed to a private Hub
      repo. Keep outputs under `/workspace` (a volume disk survives a stop
      but not a terminate; the container disk is lost on stop).
- [ ] Push final GRPO model to Hugging Face Hub.
- [ ] Record training cost: wall-clock time, GPU-hours, peak VRAM.
- [ ] Run Step 1's eval harness on this checkpoint → benchmark(2).

---

## Step 4 — Distillation (Model 3)

**Platform: Kaggle (T4)**

- [ ] Use Gemini 2.5 Flash (free API tier) to generate CoT rationales for
      the **same GSM8K training questions** used in Step 2 (not the test
      set). This generation step does not need a GPU — it's just API calls.
- [ ] Fine-tune a **fresh copy** of the base model (same base as Step 2,
      not the SFT or GRPO output) on this teacher-generated data — same
      LoRA/fp16 recipe as Step 2, only the data source differs.
- [ ] Save LoRA adapter + push to Hugging Face Hub.
- [ ] Record training cost: wall-clock time, GPU-hours, peak VRAM.
- [ ] Run Step 1's eval harness on this checkpoint → benchmark(3).

---

## Step 5 — Unified Latency & Memory Benchmarking

**Platform: RunPod (RTX PRO 4000) — all 4 checkpoints, same GPU, for a fair comparison**

- [ ] Serve each of the 4 checkpoints (base, SFT, distilled, GRPO) via
      vLLM on the RTX PRO 4000.
- [ ] Measure tokens/sec and time-to-first-token at a fixed batch
      size/sequence length for each.
- [ ] Measure peak inference VRAM at a given batch size for each.
- [ ] **Do not** mix GPUs for this step (e.g. don't benchmark SFT on
      Kaggle's T4 and GRPO on RunPod's RTX PRO 4000) — hardware
      differences would corrupt the comparison.

---

## Step 6 — Final Comparison & Write-Up

- [ ] Compile a single table: accuracy / training cost / GPU memory /
      inference latency across all 4 checkpoints (base, SFT, distilled,
      GRPO).
- [ ] Write up trade-off analysis — expected narrative: SFT/Distillation
      cheap but capped by imitation quality; GRPO more expensive to
      train (on-policy generation overhead) but potentially higher
      ceiling on accuracy.
- [ ] Note on-policy vs off-policy framing as the explanation for the
      training cost gap between GRPO and the other two arms.

---

## Locked Hyperparameters

Derived from reproducing reference implementations that ran the same
SFT/Distillation/GRPO comparison on Qwen2.5-0.5B/1.5B + GSM8K (see
research notes — repos: hennyi-yin/llm-reasoning-grpo,
DrEternity/gsm8k-post-training, tahaBerkBeton/improving-a-small-slm-on-gsm8k,
why015/qwen-grpo-gsm8k).

**LoRA config** (reused identically across SFT, Distillation and GRPO for a
fair comparison; GRPO attaches a fresh adapter to the base model with the
same settings. GRPO batch size / gradient accumulation are decided: 8 and 4,
see "GRPO Final Config" below):
- rank (r) = 16
- alpha = 32
- dropout = 0.05
- target_modules = all linear layers (q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj)

**SFT hyperparameters** (v1 values below; the v2 retrain uses lr 5e-5,
2 epochs, the evaluation instruction in the training prompt, a checkpoint
every 200 steps and best-validation-loss selection — see
`models/sft/sft_result.md`):
- learning rate = 1e-4
- epochs = 2-3
- max sequence length = 512
- effective batch size ≈ 16 (via gradient accumulation)

**GRPO hyperparameters:**
- group size (num_generations) = 8
- num_iterations = 2 (TRL default is 1). Each generated batch gets two
  update passes, so the costly generation is reused. The second pass learns
  from answers written by the slightly older model, so the clipping limit
  (epsilon = 0.2, TRL default) now matters. Watch the clip metrics; fall
  back to 1 if training is unstable.
- loss_type = "dr_grpo" (TRL default is "dapo"). Both remove the length
  bias of the original "grpo" loss; dr_grpo divides by a fixed constant,
  which TRL says corresponds to max_completion_length (1024). Fallbacks:
  "dapo", then "sapo" only if instability appears.
- scale_rewards = "none" (set explicitly; TRL's default is "group", which
  divides advantages by the within-group standard deviation, verified in
  the TRL docs v1.18.2 / main). The Dr. GRPO paper argues std scaling
  creates a question-level difficulty bias. Not tested in isolation and not
  tested on 0.5B models, so keep "group" as the fallback if learning looks
  unstable.
- Sampling and correction settings left at TRL defaults: top_p = 1.0,
  top_k = 0, repetition_penalty = 1.0 (must stay 1.0 while the vLLM
  importance-sampling correction is on), vllm_importance_sampling_correction
  = True (mode "sequence_mask"), mask_truncated_completions = False.
- sampling temperature = 1.0 (TRL default). Evidence from a Kaggle sampling
  exercise on one question (Janet's ducks, 8 samples per setting, Qwen2.5-
  0.5B-Instruct base, pure sampling): correct answers 3/8 at T=0.3, 2/8 at
  0.7, 1/8 at 1.0, 0/8 at 1.5 (output was multilingual noise, all rewards 0
  = no learning signal). Groups at 0.3-1.0 had mixed rewards, so T=1.0 is
  usable but on the high side. n=8 on one question is very noisy. Watch the
  in-group reward std; if most groups are all-zero, lower to 0.7.
- learning rate = 1e-5 (changed from 5e-6). Source: TRL's official PEFT docs
  (huggingface.co/docs/trl/peft_integration) say LoRA needs about 10x the
  full fine-tuning rate; their GRPO row is 1e-6 full fine-tuning vs 1e-5
  with LoRA. Thinking Machines' "LoRA Without Regret" also finds the best
  LoRA rate is about 10x full fine-tuning, for RL as well. Fallback: 5e-6
  (Unsloth's RL recommendation) if training is unstable. Avoid 1e-4 or
  higher (collapsed in one blog). Unsloth advises waiting at least ~300
  steps before judging whether reward is rising, so judge the pilot at
  ~300 steps, not 100.
- lr_scheduler_type = "constant" (never shrink the learning rate). Reason:
  a decaying schedule (e.g. linear) is tied to the PLANNED total number of
  steps. A 500-step pilot would decay to ~0 by step 500; if we then extend
  the same run toward ~3,400 steps, the schedule would suddenly think the
  end is far away and the rate could jump back up (for a linear schedule,
  to ~85% of the starting rate at step 500). A constant rate avoids this, so
  the pilot can flow into the full run. The Thinking Machines "LoRA Without
  Regret" experiments also used a constant learning rate. NOT verified: TRL's
  default schedule (assumed decaying, like TrainingArguments' default) and
  exactly how the Trainer rebuilds the scheduler on resume. Setting
  "constant" explicitly makes both questions irrelevant.
- Continue-or-restart rule: if the pilot passes its gate (reward rising,
  format rate moving off 1-2%, groups not all-zero), CONTINUE instead of
  restarting from scratch, to save budget. Way 1 (preferred): resume the same
  run with `trainer.train(resume_from_checkpoint=...)` from the last
  checkpoint (hub_strategy = "checkpoint" pushes a `last-checkpoint`
  folder) and a larger target (`num_train_epochs = 1`, `max_steps = -1`).
  Way 2: start a NEW run from the pilot's saved adapter (load base model +
  adapter as a PeftModel and pass it to the trainer WITHOUT `peft_config`;
  TRL raises an error if both are given). Use Way 2 if any hyperparameter
  has to change. Do not change hyperparameters between pilot and
  continuation otherwise. For the cost comparison in the final write-up,
  GRPO training cost = pilot + continuation.
- max_steps vs num_train_epochs: smoke test `max_steps = 5`; pilot
  `max_steps = 500` (about 15% of one pass over the 6,725 training
  questions; one epoch is about 3,400 steps by our estimate from TRL's
  sampler code, unverified); full run `num_train_epochs = 1` with
  `max_steps = -1`. A positive `max_steps` overrides `num_train_epochs`, and
  TRL's default is 3 epochs (about 10,000 steps), so always set one
  explicitly.
- KL beta = 0.0 (changed from 0.02). Reasons: budget / limited GPU (beta=0
  means TRL does not load the reference model, saving memory and time), it
  matches TRL's default and DAPO's choice to drop the KL term, and the
  reward is rule-based (hard to hack). Known trade-off: no leash on drift,
  and TRL's `kl` metric is only logged when beta is non-zero, so we lose
  that warning signal. Mitigations: modest learning rate (1e-5, the LoRA rate in TRL's docs), LoRA, short
  initial run (500 steps), frequent checkpoints, and watching reward, format
  rate, completion length and sample outputs (`log_completions`) instead.
  Fallback if drift is seen: beta = 0.001 (DeepSeek-R1's value).
- max completion length = 1024 tokens (raised from 320: the model now writes
  `<think>` reasoning plus the answer, and a truncated completion has no
  final answer and scores 0. Model limits are 32,768 context / 8,192
  generation, so 1024 is well inside them. Monitor TRL's
  `completions/clipped_ratio`; fall back to 768 if rollouts are too slow,
  raise it only if clipping stays high.)
- training steps = 500 initially (validate pipeline first, scale up if
  time/compute allows). "1 step" = 1 weight update. (The TRL docs never
  define "step"; we assume it is one optimizer update, which is how TRL's
  `step_time` metric is logged. Confirm with the step counter in the first
  logs.)
- per_device_train_batch_size = 8, gradient_accumulation_steps = 4.
  TRL counts ATTEMPTS (completions), not questions: 8 x 4 = 32 attempts per
  step = 32 / num_generations (8) = 4 questions per step. TRL's defaults (8 x
  1) would give only 1 question per step, which is too noisy. Keep
  per_device_train_batch_size at 8 (it is the GPU-memory knob); raise
  gradient_accumulation_steps to 8 (8 questions/step) only if a step is
  cheap. TRL requires generation_batch_size (= per_device_train_batch_size x
  gradient_accumulation_steps x number of GPUs) to divide evenly by
  num_generations: 8 x 4 = 32 does.
- How num_iterations = 2 interacts with steps: each round generates 8
  attempts for each of 4 questions ONCE (the costly part), then does 2
  weight updates (2 steps) on those same attempts, then moves to 4 new
  questions. So 1 round = 2 steps, and N steps cover about N x 2 DIFFERENT
  questions: 500 steps = 250 rounds = about 1,000 different questions (about
  15% of the 6,725 training questions).
- save_steps = 24 (every 24 updates = 12 rounds = about 48 different
  questions), save_total_limit = 3, push_to_hub = True, hub_strategy =
  "checkpoint" (pushes each save plus a `last-checkpoint` folder so a fresh
  Pod can resume with `trainer.train(resume_from_checkpoint="last-
  checkpoint")`), hub_private_repo = True (TRL's default would make the repo
  PUBLIC). Use an even save_steps so saves land at the end of a full round
  (each round = 2 steps; this is our inference, not a documented rule). The
  LoRA adapter is small (the SFT run had 8.8M trainable parameters, about
  35 MB in fp32), so frequent saves are cheap. Re-tune save_steps after the
  smoke test: aim to lose at most ~10 minutes of work if a Pod is
  interrupted, i.e. save_steps = about 600 s / seconds-per-step.
  logging_steps = 10 (TRL default).
- `use_vllm=True` for rollout generation — reference repos measured an
  ~11x throughput gain (10,196 tok/s via vLLM vs. 912 tok/s via plain HF
  `.generate()`), confirming this is not optional for GRPO to run in
  reasonable time.

**Reward function:**
```
reward = 1.0 * (final answer correct) + 0.2 * (format correct)
```
- +1.0 for a correct final answer
- +0.2 for correctly formatted/parseable output (e.g. proper `\boxed{}` or
  equivalent final-answer formatting, or `<think>...</think>` + answer tags
  if a thinking format is used)
- 0 otherwise

**Implementation rule (from the base-model prompt test on validation, 40
questions x 4 samples, temperature 1.0):** the two terms must be SEPARATE
reward checks (TRL sums a list of reward functions). The correctness term
must NOT depend on the tags: read the final number from `\boxed{}` (or the
last number in the text) whether or not `<think>` tags are present. Reason:
the base model almost never writes `<think>` tags (format rate 1.9% with the
`<answer>` prompt, 0.0% with the `\boxed{}` prompt), while its answers are
correct about 40-47% of the time when read loosely. If correctness were only
credited inside the tags, nearly every answer would score 0 at the start, all
groups would have zero advantage, and nothing would be learned. Main
priority is accuracy: the 1.0 correctness term drives it; the 0.2 format
term is a nudge toward the `<think>` habit. If the format rate stays near
1-2% after ~300 steps, options: raise the format weight, split the format
reward into parts (open tag / close tag / answer), or prefill `<think>`.

**Why outcome + format only (no reward on reasoning quality):** the format
term only checks that the thinking is present, not that it is good.
Reasoning quality improves indirectly because good reasoning leads to
correct answers. A process reward model (scoring each reasoning step) was
considered and rejected: it costs more to train and run, needs step labels
or a judge model, and is easier to reward-hack. The DeepSeek-R1 paper
(arXiv 2501.12948, Sec. 2.2) made the same choice for the same reasons.

**Known risk, flagged by reference implementations — worth tracking in
our own results:** findings conflict across repos on whether SFT-before-GRPO
helps or hurts (one repo found SFT actively hurt GRPO's final accuracy,
another found it clearly helped). This is a real open question in the
literature, not a mistake to avoid — our own benchmark(1) vs. benchmark(2)
comparison will produce a real answer for our specific setup. One repo
also flagged a **configuration-selection bias risk**: GRPO improvements
measured on a dev/validation set did not generalize to a true held-out
test set — reinforces why our test set must stay completely untouched
until final benchmarking (see Step 1 and the data-split discussion).

---

## GRPO Final Config — Single Reference (with comments)

Everything decided for the first GRPO run in one place. Argument names are
from the TRL docs (v1.18.2) and source (main) read during planning; re-check
them against the TRL version actually installed on the Pod (the smoke test
will show any renamed argument). Anything marked "default" is left at TRL's
default on purpose.

```
# ---- Model and adapter ---------------------------------------------------
model                  = "Qwen/Qwen2.5-0.5B-Instruct"  # base model, NOT the SFT checkpoint
LoRA r                 = 16      # same LoRA recipe as SFT/Distillation (fair comparison)
LoRA alpha             = 32      # scaling; alpha = 2 x r
LoRA dropout           = 0.05
LoRA target_modules    = all linear layers (q,k,v,o,gate,up,down proj)

# ---- Prompt and rewards --------------------------------------------------
prompt                 = Option B text in models/grpo/prompt.py  # <think>..</think> then \boxed{}
dataset                = GSM8K train split from sardakartik/gsm8k-split (NOT committed to git)
                         # columns: "prompt" = [{"role":"user","content": prompt text}]
                         #          "ground_truth" = the number after "####"
reward_funcs           = [correctness_reward, format_reward]   # models/grpo/rewards.py
reward_weights         = [1.0, 0.2]   # correctness drives accuracy; format is a small nudge
                         # correctness does NOT depend on tags (reads \boxed{} or last number)

# ---- Generation (rollouts) -----------------------------------------------
use_vllm               = True        # ~11x faster generation (reference repos); colocate mode
num_generations        = 8           # attempts per question (the "group")
temperature            = 1.0         # default; fallback 0.7 if most groups score all-zero
top_p, top_k           = 1.0, 0      # default (no filtering)
repetition_penalty     = 1.0         # default; MUST stay 1.0 while vLLM correction is on
max_completion_length  = 1024        # room for <think> + answer; watch completions/clipped_ratio
mask_truncated_completions = False   # default: cut-off answers score 0 (discourages rambling)
vllm_importance_sampling_correction = True  # default; guards vLLM-vs-trainer numeric mismatch

# ---- Batching ------------------------------------------------------------
per_device_train_batch_size = 8      # attempts per GPU pass (memory knob)
gradient_accumulation_steps = 4      # 8 x 4 = 32 attempts = 4 questions per step
num_iterations         = 2           # 2 weight updates per generated batch (1 round = 2 steps)
                         # 1 step = 1 weight update; 500 steps = ~1,000 different questions

# ---- Loss and optimisation -----------------------------------------------
loss_type              = "dr_grpo"   # removes length bias; fallback "dapo", then "sapo"
scale_rewards          = "none"      # TRL default is "group", so set explicitly
beta (KL)              = 0.0         # no KL penalty; fallback 0.001 if drift is seen
epsilon (clip)         = 0.2         # default; matters now because num_iterations = 2
learning_rate          = 1e-5        # LoRA rate per TRL docs (10x full FT); fallback 5e-6
lr_scheduler_type      = "constant"  # never shrink the rate: pilot can flow into the full run (no jump on resume)

# ---- Run length, saving, logging ------------------------------------------
max_steps              = 500         # pilot; judge at ~300 steps, not earlier
save_strategy          = "steps"
save_steps             = 24          # ~48 different questions; re-tune after smoke test
save_total_limit       = 3
push_to_hub            = True
hub_strategy           = "checkpoint"   # also pushes last-checkpoint for resuming
hub_private_repo       = True           # TRL default would be PUBLIC
hub_model_id           = "sardakartik/qwen2.5-0.5b-gsm8k-grpo"   # PROPOSED, confirm
logging_steps          = 10
log_completions        = True           # read sample outputs; we have no kl metric at beta=0
```

**Watch during the run** (TRL log names): `rewards/correctness_reward/mean`
(training accuracy), `rewards/format_reward/mean` (format rate),
`frac_reward_zero_std` (share of questions where all 8 attempts scored the
same = no learning signal), `reward_std`, `completions/clipped_ratio`,
`entropy`, completion length, and sample outputs. (How TRL names a plain
function in the log keys was not verified; check on the Pod.)

**Gates:**
1. Smoke test (about 5 steps on a few questions): model loads, vLLM
   generates, rewards look sane on REAL outputs (print a few completions),
   a checkpoint is written and pushed. Record seconds per step and peak VRAM.
2. Pilot judged at ~300 steps: reward rising? format rate moving off 1-2%?
   groups not all-zero? Otherwise use the fallbacks above before the full run.
3. Only then continue to the full run. If the pilot passed, CONTINUE from
   its last checkpoint (resume, or start from its adapter) instead of
   restarting from scratch; see "Continue-or-restart rule" above. Run
   stages: smoke test (max_steps = 5, checks the code runs) -> pilot
   (max_steps = 500, checks the model is learning) -> full run
   (num_train_epochs = 1, max_steps = -1).

---

## Open Decisions to Resolve Before Starting

- [x] MATH subset: decided as an OPTIONAL, evaluation-only extra. If time
      allows, evaluate the checkpoints on MATH-500 (500 problems, 7
      subjects, levels 1-5) to see how accuracy changes. No training on
      MATH. Note: MATH answers are not always integers (e.g. fractions,
      text), so the GSM8K numeric checker would need a different answer
      comparison. The 0.5B study (arXiv 2506.13404) saw only about +1 point
      on MATH500 from RL, so expect a small change.
- [x] GRPO batch size / gradient accumulation: 8 and 4 (4 questions per
      step). Decided.
- [x] GRPO save / push settings: save_steps = 24, private Hub repo,
      hub_strategy = "checkpoint". Decided.
- [ ] Hugging Face repo name for the GRPO adapter (proposed:
      `sardakartik/qwen2.5-0.5b-gsm8k-grpo`).
- [ ] Time per step and peak VRAM: unknown until the smoke test on the Pod;
      save_steps and gradient_accumulation_steps may be re-tuned then.
- [ ] RTX PRO 4000 availability and price on RunPod (not found on RunPod's
      GPU list or pricing pages while researching; check in the Pod console).
- [ ] Compatible trl / vllm / torch versions on the Pod (not verified).