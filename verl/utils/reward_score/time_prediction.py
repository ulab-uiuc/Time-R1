import re
import sys
import numpy as np

# Temporarily remove the current working directory (usually sys.path[0])
orig_path = sys.path.pop(0)
import math as builtin_math  # What is loaded here is the built-in math module
sys.path.insert(0, orig_path)  # Restore the original sys.path

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_answer_format(solution_str):
    """Extract the content in the middle of the <answer>...</answer> tag from the answer text.
    If not found, return None."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if match:
        return match.group(1).strip()
    return None

def is_valid_date_format(date_str):
    """Verify that date_str conforms to the "YYYY-MM" format, where the months must be 01 to 12."""
    pattern = r'^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$'
    return re.match(pattern, date_str) is not None

def date_prediction_reward(prediction, target, alpha=0.1):
    """Rewards are calculated based on the number of months gap between the predicted date and the real date.
    
    parameter:
        prediction (str): predicted date in the format "YYYY-MM"
        target (str): Real date in the format "YYYY-MM"
        alpha (float): Decay rate, default 0.1, more stringent than time inference task
        
    return:
        float: The reward value, 1 when both are exactly consistent, and the reward index decays as the month gap increases."""
    try:
        pred_year, pred_month = map(int, prediction.split("-"))
        target_year, target_month = map(int, target.split("-"))
    except Exception:
        return 0.0
    
    diff_in_months = abs((pred_year - target_year) * 12 + (pred_month - target_month))
    reward = builtin_math.exp(-alpha * diff_in_months)
    return reward

def format_reward_single(solution_str):
    """Check whether solution_str strictly includes:
    <think> and </think> and <answer> and </answer> tags.
    Returns 1.0 if it matches exactly, otherwise returns 0.0."""
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer><|im_end|>$"
    return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

def tag_count_reward_single(solution_str):
    """Calculate the score based on the number of occurrences of the tag in solution_str"""
    count = 0.0
    if solution_str.count("</think>") == 1:
        count += 0.33
    if solution_str.count("<answer>") == 1:
        count += 0.33
    if solution_str.count("</answer>") == 1:
        count += 0.33
    return count

def compute_length_repetition_penalty(solution_str):
    """Calculate the penalty value for excessive answers and duplicate content
    Return a penalty value, 0 means no penalty, the larger the value, the more serious the penalty"""
    # No initial punishment
    length_penalty = 0.0
    repetition_penalty = 0.0
    
    # Partialization
    tokens = solution_str.split()
    token_count = len(tokens)

    # length penalty
    if token_count > 900:
        excess_ratio = min(1.0, (token_count - 900) / 124)
        length_penalty = excess_ratio * 0.3

    # if token_count > 400:
    #     tokens = tokens[:400]  
    
    # 1. Detect continuous repetition at word level
    if token_count > 50:
        # Maximum number of consecutive same words
        max_repeat_count = 1
        current_word = None
        current_count = 0
        
        for word in tokens:
            if word == current_word:
                current_count += 1
                max_repeat_count = max(max_repeat_count, current_count)
            else:
                current_word = word
                current_count = 1
        
        # If there are more than 5 consecutive words, apply punishment
        if max_repeat_count >= 5:
            repetition_penalty = 0.1 * min(5, max_repeat_count - 4)  # Up to 0.5 points penalty
    
    # 2. Detect continuous repetitions at phrase level
    if token_count > 100:
        # Incremental repeat detection within the window size range
        for window_size in [3,5,7,9]:  # Detect duplication of 3-9 words
            for i in range(len(tokens) - window_size * 3):
                phrase = ' '.join(tokens[i:i+window_size])
                
                # Check for continuous duplication in the following text
                next_text = ' '.join(tokens[i+window_size:i+window_size*4])
                repeat_count = next_text.count(phrase)
                
                if repeat_count >= 2:  # The same phrase appears twice immediately afterwards
                    repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)  # Up to 0.45 penalty
    
    # 3. Global n-gram diversity detection
    if token_count > 200:
        chunks = [' '.join(tokens[i:i+5]) for i in range(0, min(len(tokens)-5, 500))]  # Sliding window
        if chunks:
            unique_chunks = set(chunks)
            unique_ratio = len(unique_chunks) / len(chunks)
            
            # Punishment starts when the duplication exceeds 50%
            if unique_ratio < 0.5:
                repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)  # Increase punishment
    
    # Combined length punishment and repeated punishment
    total_penalty = max(length_penalty, repetition_penalty)  # Take a greater punishment
    
    return total_penalty

def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the time prediction task
    
    parameter:
        solution_str (str): Solution text
        ground_truth (dict): Dictionary containing real dates in the format {"event_pub_date": "YYYY-MM"}
        bonus (float): Additional reward when formatted correctly, default is 0.05
        alpha (float): The decay rate used for date_prediction_reward, default is 0.1 (stricter than inference task)
        tag_bonus_each (float): The reward for correct format for each tag, default is 0.025
        
    return:
        tuple: (Total score, accuracy score, format reward, tag format score, tag count score, consistency penalty, task type)"""
    answer = extract_answer_format(solution_str)

    # "No event" punishment
    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.2  # Punish "No event" more strictly in prediction tasks
    else:
        no_event_penalty = 0.3

    # If the answer is extracted and meets the "YYYY-MM" format, you will get the format reward first
    format_bonus, pred_reward = 0.0, 0.0
    if answer and is_valid_date_format(answer):
        format_bonus = bonus
        true_pub_date = ground_truth.get("event_pub_date")
        # Make sure that the real date in ground_truth also conforms to the "YYYY-MM" format, otherwise the predicted reward will not be calculated
        if true_pub_date and is_valid_date_format(true_pub_date):
            pred_reward = date_prediction_reward(answer, true_pub_date, alpha=alpha)
    
    accuracy_score = pred_reward

    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_prediction")

# Test cases
if __name__ == "__main__":
    # Time prediction test
    solution = "<think> Based on the technological development process and related industry trends described in the article, I expect this event to happen in early 2025. </think>\n<answer>2025-03</answer><|im_end|>"
    
    ground_truth = {
        "event_pub_date": "2025-03",
    }
    
    print("Time prediction score:", compute_score(solution, ground_truth))