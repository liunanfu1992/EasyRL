import json
import os
import random
import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


# format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"
acc_format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"


format_reward_factor = float(os.getenv("FORMAT_REWARD_FACTOR", 0.1))


def format_reward(content, reward_func):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count1 = content.count("<think>")
    think_count2 = content.count("</think>")
    answer_count1 = content.count("<answer>")
    answer_count2 = content.count("</answer>")
    if think_count1 == 1 and think_count2 == 1 and answer_count1 == 1 and answer_count2 == 1:
        if reward_func == "acc":
            return bool(re.match(acc_format_pattern, content, re.DOTALL))
    return 0

    

def process_expression(s):
    # 使用正则表达式移除所有运算符（=+、-、*、/）周围的空格
    return re.sub(r'\s*([=+\-*/])\s*', r'\1', s)


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None

def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def default_accuracy_reward(content, sol):
    reward = 0.0
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(content)
        if float(verify(answer, parse(sol))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails

    if reward == 0.0:
        gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                threshold = float(verify(answer_parsed, gold_parsed))
                if threshold > 0:
                    reward = 1.0
            except Exception:
                pass
                
    # If symbolic verification failed, perform strict exact checks only (no fuzzy matching)
    if reward == 0.0:
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            # Extract answer from content if it has think/answer tags
            content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_matches[-1].strip() if content_matches else content.strip()

            if ground_truth and ground_truth[0] == "$" and ground_truth[-1] == "$":
                ground_truth = ground_truth[1:-1]
            if student_answer and student_answer[0] == "$" and student_answer[-1] == "$":
                student_answer = student_answer.replace("$", "")

            student_answer = process_expression(student_answer)
            ground_truth = process_expression(ground_truth)

            # Strict equality
            if student_answer == ground_truth:
                reward = 1.0
            else:
                # For numeric answers, use exact numeric equality
                numeric_eq = numeric_reward(student_answer, ground_truth)
                reward = 1.0 if numeric_eq == 1.0 else 0.0

        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward if reward else 0.0


def acc_reward(content, sol):
    return default_accuracy_reward(content, sol)


def my_reward_fn(solution_str, ground_truth):
    acc_score = acc_reward(solution_str, ground_truth)
    return acc_score 