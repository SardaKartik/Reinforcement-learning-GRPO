"""
GRPO training script: TRL GRPOTrainer + LoRA + vLLM, for Qwen2.5-0.5B-Instruct
on GSM8K. It runs on the RunPod Pod (Linux + GPU), from the project folder:

    python models/grpo/train_grpo.py --smoke    # ~5 steps: does the code run?
    python models/grpo/train_grpo.py            # the pilot: 500 steps

All settings, and the reason for each, are in models/grpo/grpo-hyperparameters.md
"""
import argparse
import os

from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Our own two files in this same folder. This import works because Python puts
# the script's folder on its search path when you run the script.
from prompt import build_messages
from rewards import correctness_reward,format_reward

# ---- Names ------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"      # base model (NOT the SFT checkpoint)
DATASET_REPO = "sardakartik/gsm8k-split"       # private GSM8K split on Hugging Face
HUB_MODEL_ID = "sardakartik/qwen2.5-0.5b-gsm8k-grpo"   # PROPOSED name, not confirmed
OUTPUT_DIR = "/workspace/grpo_output"          # /workspace survives a Pod stop

# ---- Run lengths ------------------------------------------------------------
PILOT_STEPS = 500       # the pilot (judge it at ~300 steps)
SMOKE_STEPS = 5         # smoke test: just check that the code runs
SMOKE_QUESTIONS = 16    # smoke test uses only this many training questions

def get_ground_truth(answer_text):
    """
    GSM8K answers end with '#### <number>'. Pull that number out as a string.

    EXAMPLE
      input : "Janet sells 9 eggs ... <<9*2=18>>18\n#### 18"   -> output: "18"
      input : "... \n#### 1,234"                               -> output: "1234" (comma removed)
      input : "... \n#### -5"                                  -> output: "-5"
    """
    return answer_text.split("####")[-1].strip().replace(",", "")

def make_example(row):
    """
    Turn one GSM8K row into the two columns TRL needs.

    EXAMPLE
      input : {"question": "Janet's ducks ...?", "answer": "... #### 18"}
      output: {"prompt": [{"role": "user", "content": "<question + our prompt>"}],
               "ground_truth": "18"}

    "prompt"       -> chat-style messages; TRL applies Qwen's chat template itself.
    "ground_truth" -> a column TRL passes to our reward functions as a keyword
                      argument with the same name (correctness_reward uses it).
    """
    return {
        "prompt": build_messages(row["question"]),
        "ground_truth": get_ground_truth(row["answer"]),
    }

def load_train_data(smoke):
    """
    Load the GSM8K train split from the Hub and convert every row.
    In smoke mode only the first SMOKE_QUESTIONS rows are used.
    """
    ds = load_dataset(DATASET_REPO)["train"]
    if smoke:
        ds = ds.select(range(SMOKE_QUESTIONS))
    # remove columns drops the old "question" and "answer" columns, so only
    # "prompt" and "ground_truth" remain.
    return ds.map(make_example, remove_columns=ds.column_names)

def make_lora_config():
    """
    LoRA = the base model's weights stay frozen; we only train small extra
    "adapter" matrices attached to the model's layers. This is the SAME config
    as the SFT run (models/sft/sft.py), so the comparison stays fair.

    r              : size of the adapter (bigger = more capacity, more memory)
    lora_alpha     : scaling of the adapter's effect (alpha / r = 2 here)
    lora_dropout   : randomly ignore 5% of adapter signals while training
    target_modules : which layers get an adapter. These are the 4 attention
                     layers (q, k, v, o) and the 3 feed-forward layers
                     (gate, up, down) of every transformer block, i.e. every
                     linear layer.
    task_type      : tells peft this is a text-generation (causal LM) model
    """

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

def make_grpo_config(max_steps,push):
    """
    All GRPO settings in one place. Reasons for each value are in
    models/grpo/grpo-hyperparameters.md.

    max_steps : how many weight updates to run (5 smoke, 500 pilot, -1 = full epoch)
    push      : True = back up checkpoints to the Hugging Face Hub
    """
    return GRPOConfig(
        output_dir=OUTPUT_DIR,

        # ---- load the model in bfloat16 (lighter than TRL's float32 default) ---
        # PROPOSED: confirm the key name ("dtype") on the Pod's TRL version.
        model_init_kwargs={"dtype": "bfloat16"},
        bf16=True,

        # ---- generation: vLLM makes the 8 attempts per question much faster ---
        use_vllm=True,
        vllm_mode="colocate",              # vLLM shares the one GPU with training
        vllm_gpu_memory_utilization=0.3,   # share of GPU memory vLLM may claim
        num_generations=8,                 # attempts per question (the "group")
        temperature=1.0,                   # varied attempts
        max_completion_length=1024,        # room for <think> + answer
        # (no max_prompt_length: that argument no longer exists in TRL)

        # ---- batching: 8 x 4 = 32 attempts = 4 questions per step -------------
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_iterations=2,                  # generate once, update twice

        # ------- Loss and optimization ------------------------------------------------
        loss_type = "dr_grpo", # no length bias
        scale_rewards = "none", # TRL's default is "group", so set it
        beta = 0.0, # No kl penalty, No reference model
        epsilon = 0.2, # clipping range
        learning_rate=1e-5,
        lr_scheduler_type="constant",      # lets a good pilot be continued

        # ---- rewards: correctness x 1.0 + format x 0.2 (same order as in the
        #      reward_funcs list passed to the trainer in Step 6) --------------
        reward_weights=[1.0, 0.2],

        # -------- run length,saving,logging --------------------------------------
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=24,
        save_total_limit=3,
        logging_steps=10,
        log_completions=True,              # print sample answers while training

        # --------Hugging face Hub backup ----------------------------------------------
        push_to_hub=push,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="checkpoint",         # also pushes "last-checkpoint" for resuming
        hub_private_repo=True,             # TRL's default would be public
    )

def run_training(max_steps,push,smoke,resume = None):
    """
    Build the GRPOTrainer, train, then save (and optionally push) the result.

    max_steps : passed to make_grpo_config (5 smoke, 500 pilot, -1 = full epoch)
    push      : True = push checkpoints and the final adapter to the Hub
    smoke     : True = use only SMOKE_QUESTIONS training questions
    resume    : None = start fresh; a checkpoint folder path = continue from it
    """
    train_ds = load_train_data(smoke)
    print("training questions:", len(train_ds))

    trainer = GRPOTrainer(
        model = MODEL_NAME,   # TRL downloads and loads the model itself
        reward_funcs = [correctness_reward,format_reward],  # ORDER matters:
                                        # it matches reward_weights=[1.0, 0.2]
        args = make_grpo_config(max_steps, push),
        train_dataset = train_ds, # columns : "prompt" and "ground_truth"
        peft_config=make_lora_config(),    # train a LoRA adapter, not the full model
    )

    # DTYPE CHECK (see the dtype question): which number formats are in the model?
    #   bfloat16 present -> the dtype setting worked.
    #   LoRA adapter weights may show as float32; peft often keeps them that way,
    #   which is normal. If you see ONLY float32, the "dtype" key was ignored.
    print("model dtypes:", {p.dtype for p in trainer.model.parameters()})

    # The actual training. resume=None starts fresh.
    trainer.train(resume_from_checkpoint=resume)

    # Save the final LoRA adapter to the Pod's disk (/workspace survives a stop).
    final_dir = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    print("adapter saved to:", final_dir)

    # Upload the final adapter to the Hugging Face Hub (skipped in smoke mode).
    if push:
        trainer.push_to_hub()
        print("pushed to:", HUB_MODEL_ID)

def main():
    """
    Read the command-line choice, log in to Hugging Face, then train.

      python models/grpo/train_grpo.py --smoke     5 steps, 16 questions, no push
      python models/grpo/train_grpo.py             the pilot: 500 steps, pushes
      python models/grpo/train_grpo.py --full      one full epoch (max_steps = -1)
      python models/grpo/train_grpo.py --resume PATH   continue from a checkpoint
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny test run: 5 steps, 16 questions, nothing pushed")
    parser.add_argument("--full", action="store_true",
                        help="one full epoch over the training set (max_steps = -1)")
    parser.add_argument("--resume", default=None,
                        help="path of a checkpoint folder to continue from")
    args = parser.parse_args()

    if args.smoke and args.full:
        parser.error("choose --smoke or --full, not both")

    # Login: the token comes from an environment variable, never from the code.
    # On the Pod you set HF_TOKEN as a secret. On a missing token this raises a
    # clear KeyError right away, instead of failing later at the first push.
    login(token=os.environ["HF_TOKEN"])

    if args.smoke:
        max_steps, push = SMOKE_STEPS, False
    elif args.full:
        max_steps, push = -1, True
    else:
        max_steps, push = PILOT_STEPS, True

    run_training(max_steps=max_steps, push=push, smoke=args.smoke,
                 resume=args.resume)

if __name__ == "__main__":
    main()
