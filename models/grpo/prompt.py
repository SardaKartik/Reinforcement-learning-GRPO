# =============================================================================
# GRPO prompt (Option B: <think> + \boxed{})
# =============================================================================
#
# This file records the prompt we use for GRPO, and WHY it matters. The
# comments come first (agreed and written down before any training script
# existed); the actual code (PROMPT_TEMPLATE, build_prompt, build_messages)
# is at the bottom of the file.
#
# STATUS: Option B is LOCKED (see "Why B" below). The exact wording below is
# the one used for GRPO training AND for every evaluation. Do not edit the
# wording after training starts.
#
# -----------------------------------------------------------------------------
# THE PROMPT (sent as the user message; {question} is the GSM8K question)
# -----------------------------------------------------------------------------
#
#   {question}
#
#   Think through the problem step by step inside <think> </think> tags.
#   After </think>, give the final answer as a number inside \boxed{}.
#   Format: <think> your reasoning </think> The answer is \boxed{number}
#
# (In Python source, the two \boxed{} occurrences are written as \\boxed{{}}
#  and \\boxed{{number}} when the string goes through str.format, exactly as
#  in the Kaggle prompt test.)
#
# Expected model output shape:
#
#   <think>
#   ...step-by-step reasoning...
#   </think>
#   The answer is \boxed{42}
#
# -----------------------------------------------------------------------------
# WHY B (\boxed{}) and not A (<answer> tags)
# -----------------------------------------------------------------------------
#
# We tested both on the base Qwen2.5-0.5B-Instruct (Kaggle, validation split,
# 40 questions x 4 samples, temperature 1.0, max 1024 new tokens):
#
#   A (<think> + <answer>):  format rate 1.9%, strict acc 0.0%, loose acc 40.0%
#   B (<think> + \boxed{}):  format rate 0.0%, strict acc 0.0%, loose acc 47.5%
#
# The difference between A and B is within noise, so the test did not pick a
# winner by itself. We chose B because:
#   - Qwen already writes \boxed{} on its own (seen in the sample outputs, and
#     Qwen's own math prompt asks for \boxed{}), so it is one less new habit
#     for a 0.5B model to learn.
#   - The optional MATH-500 evaluation uses \boxed{} answers, so the same
#     extractor can be reused.
#
# Main finding of that test (applies to BOTH prompts): the base model almost
# never writes <think> tags. It does reason step by step in plain text and is
# right about 40-47% of the time. So:
#   - The correctness reward must NOT depend on the tags. Read the final
#     number from \boxed{} (or the last number in the text) whether or not
#     <think> tags are present. Otherwise nearly every answer scores 0 at the
#     start and nothing is learned.
#   - The <think> format is a SEPARATE small reward (+0.2) that nudges the
#     model toward the habit. It may be learned slowly. If the format rate is
#     still ~1-2% after ~300 steps: raise the format weight, split the format
#     reward into parts (open tag / close tag / boxed answer), or prefill
#     "<think>" at the start of the model's reply.
#
# -----------------------------------------------------------------------------
# WHY we keep this prompt fixed and written down
# -----------------------------------------------------------------------------
#
# 1. The model only writes <think> reasoning because the prompt asks for it.
#    The base model (Qwen2.5-0.5B-Instruct) is not a reasoning model and has
#    no built-in <think> mode. No prompt -> no tags -> no format reward.
#
# 2. The reward function depends on this exact format.
#      reward = 1.0 * (final answer correct) + 0.2 * (format correct)
#    "Format correct" = a <think>...</think> block followed by a \boxed{...}
#    answer. Check it loosely (do not require the WHOLE output to match a
#    strict ^...$ pattern: a 0.5B model often adds stray text). Read the
#    answer from the LAST \boxed{} after </think>, so the model cannot score
#    by putting guesses inside its thinking. If the prompt changes, the
#    reward checks must change with it.
#
# 3. The prompt must be IDENTICAL for GRPO training and for evaluation.
#    Prompt wording changes accuracy by itself. Our own SFT diagnostics
#    (models/sft/sft_result.md) showed SFT-epoch-2 scoring 38.33% under the
#    instruction prompt and 49.00% under the bare prompt, on the same 300
#    validation questions. If training and eval use different prompts, we
#    cannot tell how much of a score change came from training and how much
#    from the wording.
#
# 4. Fair comparison across all four arms (base, SFT, distilled, GRPO).
#    The old baseline (42.15%) used the "Solve step by step... #### <number>"
#    prompt from eval/accuracy.py.
#
# -----------------------------------------------------------------------------
# THE BASE MODEL UNDER THIS PROMPT (the baseline for GRPO)
# -----------------------------------------------------------------------------
#
# The 42.15% baseline (benchmark(0)) was measured with the OLD "####" prompt,
# so it is NOT comparable to a GRPO model evaluated with THIS prompt. GRPO is
# therefore compared against the BASE model measured with exactly this prompt,
# on the same split, with the same decoding (greedy) and the same token limit
# (1,024 new tokens, the limit used in training).
#
# STATUS: eval/accuracy.py now supports this prompt (PROMPT_STYLE =
# "think_boxed") and reads answers with the same rule as rewards.py. The runs
# themselves (base, then the GRPO pilot adapter, on the test split) are
# pending; results are recorded in docs/results.md.
#
# Reported for each model:
#   - accuracy under THIS prompt,
#   - strict accuracy (right AND full format), format rate and boxed rate, so
#     we can see how much comes from following the format vs solving the
#     problem,
#   - for reference only, the old "####" numbers (42.15% base, 33.66% SFT on
#     test), which stay in a separate group because they used a different
#     prompt.
# Split plan: the base-vs-pilot comparison and the final models use the TEST
# split (the same test set serves more than one GRPO model, which must be
# stated in the report). The VALIDATION split is planned to be passed to the
# trainer in the next GRPO run, to monitor progress during training.
#
# -----------------------------------------------------------------------------
# RULES (decided in discussion — keep following them)
# -----------------------------------------------------------------------------
#
# - The exact prompt text is locked before GRPO training starts. It cannot be
#   changed after training based on the results: the model would then be
#   evaluated on wording it never trained on.
# - Do NOT pick the "best" prompt per model after looking at test results.
#   Report every model x prompt combination.
# - If we ever want to tune the prompt wording, compare options on the
#   VALIDATION split, never the test set, and decide before touching test.
# - eval/accuracy.py has a matching "think_boxed" prompt style and \boxed{}
#   extraction. Its copy of the prompt text and answer rules must stay
#   identical to this file and to rewards.py; change them together.
#
# -----------------------------------------------------------------------------
# STATUS / TODO
# -----------------------------------------------------------------------------
#
# [x] Finalize and lock the prompt wording (Option B).
# [x] Add a matching "think_boxed" prompt style and \boxed{} extraction to
#     eval/accuracy.py (done; also reports strict accuracy, format rate,
#     boxed rate, mean length and truncated rate).
# [ ] Run the BASE model under this prompt on the test split = the baseline
#     for GRPO, then the GRPO pilot adapter on the same split. Pending
#     (Kaggle).
# [x] Write the GRPO reward function: separate correctness check (reads
#     \boxed{} or last number, independent of tags) and format check
#     (<think>...</think> then \boxed{}), weights 1.0 and 0.2.
#     DONE: models/grpo/rewards.py (13 test cases pass).
# [x] Turn the prompt into code (PROMPT_TEMPLATE / build_prompt /
#     build_messages, below) so training and eval share one source of truth.
# [~] Count tag rates separately: eval/accuracy.py now reports the full
#     format rate and the \boxed{} rate separately. The <think> open and
#     </think> close rates on their own are not counted yet.


# =============================================================================
# CODE (the comments above explain WHY; this is the actual prompt)
# =============================================================================

# The locked Option B prompt. {question} is replaced by the GSM8K question.
# Written with normal strings, so a literal backslash is written as \\ :
#   "\\boxed{}"  in the source  ->  \boxed{}  in the real text
# (A single "\b" would be read by Python as a BACKSPACE character, which is a
#  classic bug. The test at the bottom checks that this did not happen.)
# Note: this is the exact text used in the Kaggle prompt test, where the first
# two sentences sit on one line.
PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Think through the problem step by step inside <think> </think> tags. "
    "After </think>, give the final answer as a number inside \\boxed{}.\n"
    "Format: <think> your reasoning </think> The answer is \\boxed{number}"
)


def build_prompt(question):
    """
    Put one question into the prompt and return the full prompt text.

    EXAMPLE
      input : "Janet's ducks lay 16 eggs per day. How much does she make?"
      output: "Janet's ducks lay 16 eggs per day. How much does she make?\n\n
               Think through the problem step by step inside <think> </think> tags. ...
               Format: <think> your reasoning </think> The answer is \\boxed{number}"

    We use .replace() and NOT .format(): the template contains curly braces
    (\\boxed{}), and .format() would try to treat them as placeholders and crash.
    """
    return PROMPT_TEMPLATE.replace("{question}", question)


def build_messages(question):
    """
    Wrap the prompt as a chat message. This is what goes in the dataset's
    "prompt" column, so TRL applies Qwen's chat template itself.

    EXAMPLE
      input : "What is 2+2?"
      output: [{"role": "user", "content": "<the full prompt text>"}]
    """
    return [{"role": "user", "content": build_prompt(question)}]


if __name__ == "__main__":
    q = "Janet's ducks lay 16 eggs per day. How much does she make?"
    print(build_prompt(q))
    # should be 2: one backslash for each \boxed. If it is 0, a backslash was eaten.
    print("backslashes:", build_prompt(q).count("\\"))
    # a question that contains braces must not break anything
    tricky = "Solve {x} + 2"
    print("braces OK:", build_prompt(tricky).startswith(tricky))
