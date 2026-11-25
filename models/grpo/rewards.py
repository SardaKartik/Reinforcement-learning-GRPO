"""
Reward functions for GRPO.

TRL calls each function with the model's answers (8 per question) and expects
back ONE number per answer, as a list of floats.

Two separate rewards, added together by TRL with weights [1.0, 0.2]:
  - correctness_reward : is the final number right?  (does NOT need any tags)
  - format_reward      : <think>...</think> followed by \\boxed{...}
"""
import re

# A number like 18, -3, 1,234 or 18.5 (commas allowed as thousand separators).
NUMBER = r"-?\d[\d,]*\.?\d*"

# Matches \boxed{...} and allows ONE level of nested braces, so
# \boxed{18 \text{ dollars}} is matched too. The pieces:
#   \\boxed\{      the literal text \boxed{
#   (...)          the content we capture
#   [^{}] | \{[^{}]*\}   either a normal character, or a small {...} group
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

# Format check: a <think>...</think> block, then later a \boxed{...} with
# something inside it. re.DOTALL makes "." match line breaks too. Without it,
# multi-line reasoning would fail (the TRL docs' example has this problem).
FORMAT = re.compile(r"<think>.*?</think>.*?\\boxed\{.+?\}", re.DOTALL)

# step 2 : small helpers 

# EXAMPLE (get_text)
#   input  (chat style): [{"role": "assistant", "content": "The answer is 18"}]
#   output             : "The answer is 18"
#   input  (plain text): "The answer is 18"
#   output             : "The answer is 18"      (returned unchanged)
def get_text(completion):
    """
    Get the plain text of one answer.
    With a chat style dataset TRL gives [{"role" : "assistant", "content": "..."}].
    With a plain-text dataset it gives just a string. Handle both.
    """
    if isinstance(completion,list):
        return completion[0]["content"]
    return completion 

# EXAMPLE (last_number): returns the LAST number in the text, as a string
#   input : "The answer is 18"                  -> output: "18"
#   input : "she makes $1,234.50 today"         -> output: "1234.50"   ($ and commas removed)
#   input : "16 - 3 = 13, then 13 - 4 ="        -> output: "4"         (last number wins)
#   input : "no numbers here"                   -> output: None
def last_number(s):
    """ Return the Last Number in a string (as text) or None if there is none."""
    nums = re.findall(NUMBER, s.replace("$", "")) # drop $ signs first
    return nums[-1].replace(",", "") if nums else None  # "1,234" -> "1234" 

# extracting the model final answer 
# EXAMPLE (extract_final_answer): the model's final answer, as a string (or None)
#   input : "<think>16-3-4=9, 9*2=18</think> The answer is \boxed{18}"
#   output: "18"     (took the \boxed{} that comes AFTER </think>)
#
#   input : "<think>maybe \boxed{5}</think> The answer is \boxed{18}"
#   output: "18"     (the guess inside <think> is ignored)
#
#   input : "so \boxed{18 \text{ dollars}}"
#   output: "18"     (number taken from inside the box)
#
#   input : "Steps... the answer is 26"
#   output: "26"     (no \boxed, so fallback: last number in the text)
#
#   input : "<think>16 - 3 = 13, then 13 - 4 ="      (cut off, no answer)
#   output: "4"      (fallback grabs the last number; it is wrong, so it scores 0 later)
def extract_final_answer(text):
    """
    Where we look for the answer:
    1. Only after the last </think> , if there is one. This stops the model 
       from scoring by writing guesses inside the thinking 
       If there is no </think> we look at the whole text.
    2. Inside the Last \\boxed{...} if there is one.
    3. Otherwise the last number in that text (the fallback).
    The fallback matters: the base model rarely writes tags, and without it
    nearly every answer would score 0 and GRPO would learn nothing.
    """
    tail = text.split("</think>")[-1] if "</think>" in text else text
    boxed = BOXED.findall(tail)
    if boxed:
        return last_number(boxed[-1])  
    return last_number(tail)

# correctness reward 

# EXAMPLE (same): are two numbers equal? (compared as numbers, not as text)
#   same("18",   "18")   -> True
#   same("18.0", "18")   -> True      (18.0 == 18)
#   same("1234", "1,234")-> True      (commas removed from the truth)
#   same(None,   "18")   -> False     (no answer found)
#   same("26",   "18")   -> False
def same(pred,truth):
    """ compare as numbers, so 18 and 18.0 count as equal and 1,234 == 1234"""
    if pred is None or truth is None:
        return False 
    try:
        return float(pred) == float(str(truth).replace(",",""))
    except ValueError:
        return False

# EXAMPLE (correctness_reward): one score per answer, 1.0 right / 0.0 wrong
#   completions  (what TRL passes in; 4 answers to ONE question):
#     [
#       [{"role": "assistant", "content": "<think>9*2=18</think> The answer is \boxed{18}"}],
#       [{"role": "assistant", "content": "Steps... so \boxed{18}"}],
#       [{"role": "assistant", "content": "Steps... the answer is 26"}],
#       [{"role": "assistant", "content": "<think>maybe \boxed{5}</think> \boxed{18}"}],
#     ]
#   ground_truth (one correct answer per completion): ["18", "18", "18", "18"]
#
#   step 1, answers extracted : ["18", "18", "26", "18"]
#   step 2, compared with truth: [ ok ,  ok , wrong,  ok ]
#   OUTPUT: [1.0, 1.0, 0.0, 1.0]
#   (log_extra also records the column "extracted_answer" = ["18", "18", "26", "18"])
def correctness_reward(completions,ground_truth,log_extra = None, **kwargs):
    """
    1.0 if the final number matches ground_truth, else 0.0.
    `ground_truth` is a dataset column; TRL passes it in automatically
    (one value per answer). `log_extra` is optional: it adds a column to
    TRL's completions table so you can SEE what was extracted.
    """
    answers = [extract_final_answer(get_text(c)) for c in completions]
    if log_extra is not None:
        log_extra("extracted_answer", [a if a is not None else "[none]" for a in answers])
    # strict=True raises a clear error if the two lists differ in length,
    # instead of silently dropping answers (needs Python 3.10+).
    return [1.0 if same(a, t) else 0.0 for a, t in zip(answers, ground_truth, strict=True)]

# the format reward 
# EXAMPLE (has_format): True only for  <think>...</think>  then  \boxed{...}
#   "<think>9*2=18</think> The answer is \boxed{18}"  -> True
#   "Steps... so \boxed{18}"                          -> False   (no think block)
#   "<think>a<think>b</think> \boxed{18}"             -> False   (two <think> tags)
#   "<think>x</think> The answer is 18"               -> False   (no \boxed{})
def has_format(text):
    """
    True only if the text has exactly ONE <think> and ONE </think>, and a
    \\boxed{...} comes after them. The count check matters: the regex alone
    would accept "<think>a<think>b</think> \\boxed{18}".
    """
    one_pair = text.count("<think>") == 1 and text.count("</think>") == 1 
    return one_pair and FORMAT.search(text) is not None

# EXAMPLE (format_reward): same 4 completions as in correctness_reward above
#   OUTPUT: [1.0, 0.0, 0.0, 1.0]
#     answer 1 -> think + boxed          -> 1.0
#     answer 2 -> boxed but no <think>   -> 0.0
#     answer 3 -> plain text, no tags    -> 0.0
#     answer 4 -> think + boxed          -> 1.0
#   With the trainer weights [1.0, 0.2], the TOTAL reward per answer is
#   correctness*1.0 + format*0.2 :  [1.2, 1.0, 0.0, 1.2]
def format_reward(completions, **kwargs):
    """
    1.0 if the format is right, else 0.0. We return 0/1 on purpose: the
    0.2 weight is set in the trainer config (reward_weights=[1.0, 0.2]),
    and the logged average of this function then equals the FORMAT RATE.
    """

    return [1.0 if has_format(get_text(c)) else 0.0 for c in completions]


if __name__ == "__main__":
    # (name, model output, ground truth, want correctness, want format)
    cases = [
        ("1 good",            "<think>\n16-3-4=9\n9*2=18\n</think>\nThe answer is \\boxed{18}", "18", 1, 1),
        ("2 box, no think",   "Steps... so \\boxed{18}",                                        "18", 1, 0),
        ("3 think, wrong",    "<think>stuff</think> \\boxed{26}",                               "18", 0, 1),
        ("4 plain wrong",     "Steps... the answer is 26",                                      "18", 0, 0),
        ("5 truncated",       "<think>16 - 3 = 13, then 13 - 4 =",                              "18", 0, 0),
        ("6 boxed w/ text",   "<think>x</think> \\boxed{18 \\text{ dollars}}",                  "18", 1, 1),
        ("7 plain $18.",      "So she makes $18.",                                              "18", 1, 0),
        ("8 guess in think",  "<think>maybe \\boxed{5}</think> The answer is \\boxed{18}",      "18", 1, 1),
        ("9 commas",          "<think>x</think> \\boxed{1,234}",                                "1234", 1, 1),
        ("10 two <think>",    "<think>a<think>b</think> \\boxed{18}",                           "18", 1, 0),
        ("11 empty box",      "<think>x</think> \\boxed{}",                                     "18", 0, 0),
        ("12 18.0",           "<think>x</think> \\boxed{18.0}",                                 "18", 1, 1),
        ("13 only in think",  "<think>so \\boxed{18}</think>",                                  "18", 0, 0),
    ]
    # `comps` imitates what TRL passes to the reward functions. Each answer is
    # wrapped as a one-message list, e.g. for case "2 box, no think":
    #   comps[1]                  = [{"role": "assistant", "content": "Steps... so \boxed{18}"}]
    #   comps[1][0]               = {"role": "assistant", "content": "Steps... so \boxed{18}"}
    #   comps[1][0]["content"]    = "Steps... so \boxed{18}"      <- the model's text
    # c[1] below is the 2nd item of each case tuple, i.e. the model output.
    comps = [[{"role": "assistant", "content": c[1]}] for c in cases]
    truth = [c[2] for c in cases]
    got_c = correctness_reward(comps, truth)
    got_f = format_reward(comps)
    ok = True
    for (name, _, _, ec, ef), gc, gf in zip(cases, got_c, got_f):
        flag = "OK " if (gc == ec and gf == ef) else "BAD"
        ok = ok and flag == "OK "
        print(f"{flag} {name:20s} correctness={gc} (want {ec})  format={gf} (want {ef})")
    print("ALL OK" if ok else "SOME FAILED")



    