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

def date_prediction_reward(prediction, target, alpha=0.05):
    """Rewards are calculated based on the number of months gap between the predicted date and the real date.
    
    parameter:
        prediction (str): predicted date in the format "YYYY-MM"
        target (str): Real date in the format "YYYY-MM"
        alpha (float): Attenuation rate, default 0.05
        
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

def month_diff_reward(pred_diff, true_diff, alpha=0.06):
    """Calculate the reward for the month's difference prediction, exponential decay based on the difference"""
    diff = abs(pred_diff - true_diff)
    reward = builtin_math.exp(-alpha * diff)
    return reward

def compute_inconsistency_penalty0(date1, date2, claimed_diff):
    """Calculate the inconsistency penalty between claimed month difference and actual month difference"""
    try:
        year1, month1 = map(int, date1.split("-"))
        year2, month2 = map(int, date2.split("-"))
        
        actual_diff = abs((year2 - year1) * 12 + (month2 - month1))
        error = abs(actual_diff - claimed_diff)
        
        # The penalty coefficient increases as the error increases
        if error == 0:
            return 1.0  # No error, no punishment
        elif error <= 2:
            return 0.7  # Small error, minor punishment
        elif error <= 5:
            return 0.4  # Medium error, medium punishment
        else:
            return 0.2  # Large error, serious punishment
    except Exception:
        return 0.1  # An exception occurs, severe punishment
    
def compute_inconsistency_penalty(date1, date2, claimed_diff):
    """Calculate the inconsistency penalty between the claimed month difference and the actual month difference, using the exponential decay function"""
    try:
        year1, month1 = map(int, date1.split("-"))
        year2, month2 = map(int, date2.split("-"))
        
        actual_diff = abs((year2 - year1) * 12 + (month2 - month1))
        error = abs(actual_diff - claimed_diff)
        
        # Basic consistency evaluation parameters
        base_alpha = 0.1  # Base attenuation rate
        
        # Apply a more relaxed penalty factor to the difference in the big month
        if claimed_diff >= 25:
            # The larger the difference in the month, the smaller the penalty factor
            scaling_factor = min(1.0, 25.0 / claimed_diff)  # As the month difference increases, it decreases
            alpha = base_alpha * scaling_factor
        else:
            alpha = base_alpha
            
        # Use exponential function to calculate the penalty value (1.0 is the best, no penalty)
        consistency_score = builtin_math.exp(-alpha * error)
        
        return consistency_score
    except Exception:
        return 0.1  # An exception occurs, severe punishment

def is_valid_order_format(order_str):
    """Verify that order_str is a string containing 1, 2, 3 without duplicate numbers separated by '-'"""
    if not re.match(r'\d-\d-\d', order_str):
        return False
    
    # Extract the numbers and verify that only 1, 2, 3 are included without duplication
    digits = [int(d) for d in order_str.split('-')]
    return sorted(digits) == [1, 2, 3]

def compute_order_accuracy(predicted_order, true_order):
    """Calculate the accuracy of order prediction, compare the order relationship between three groups of pairs of events
    One group gets 0.333 points, two groups gets 0.666 points, all three groups get 0.999 points"""
    if predicted_order == true_order:
        return 0.999  # All correct, get full marks (slightly reduce to avoid floating point accuracy issues)
    
    pred_parts = predicted_order.split('-')
    true_parts = true_order.split('-')
    
    # Check whether the relative order between each pair of events is correct
    pairs_correct = 0
    
    # Check the relative order of event 1 and event 2
    pred_1_before_2 = pred_parts.index('1') < pred_parts.index('2')
    true_1_before_2 = true_parts.index('1') < true_parts.index('2')
    if pred_1_before_2 == true_1_before_2:
        pairs_correct += 1
    
    # Check the relative order of event 1 and event 3
    pred_1_before_3 = pred_parts.index('1') < pred_parts.index('3')
    true_1_before_3 = true_parts.index('1') < true_parts.index('3')
    if pred_1_before_3 == true_1_before_3:
        pairs_correct += 1
    
    # Check the relative order of event 2 and event 3
    pred_2_before_3 = pred_parts.index('2') < pred_parts.index('3')
    true_2_before_3 = true_parts.index('2') < true_parts.index('3')
    if pred_2_before_3 == true_2_before_3:
        pairs_correct += 1
    
    # Calculate the score based on the correct logarithm
    return pairs_correct * 0.333

def compute_order_consistency_penalty(event1_date, event2_date, event3_date, claimed_order):
    """Calculate inconsistency penalty between claimed event sort and actual date sort"""
    try:
        # parse three dates
        year1, month1 = map(int, event1_date.split("-"))
        year2, month2 = map(int, event2_date.split("-"))
        year3, month3 = map(int, event3_date.split("-"))
        
        # Calculate the total number of months for each event to compare order
        event1_months = year1 * 12 + month1
        event2_months = year2 * 12 + month2
        event3_months = year3 * 12 + month3
        
        # Calculate the actual order based on dates
        events_by_time = [(1, event1_months), (2, event2_months), (3, event3_months)]
        events_by_time.sort(key=lambda x: x[1])
        actual_order = '-'.join(str(event[0]) for event in events_by_time)
        
        # Comparison of the order of claim and actual order
        if claimed_order == actual_order:
            return 1.0  # Completely consistent, no punishment
        
        # Calculate the degree of inconsistency between the claimed order and the actual order
        # Here we can use compute_order_accuracy to calculate the similarity of two orders
        similarity = compute_order_accuracy(claimed_order, actual_order)
        
        # Set the penalty coefficient according to similarity
        if similarity >= 0.666:  # At least 2 groups are correct
            return 0.7
        elif similarity >= 0.333:  # At least 1 group is correct
            return 0.4
        else:  # All Errors
            return 0.2
    except Exception:
        return 0.1  # An exception occurs, severe punishment

# Define standard month names and all their variant maps
months_and_variants = {
            "January": ["january", "jan", "jan."],
            "February": ["february", "feb", "feb."],
            "March": ["march", "mar", "mar."],
            "April": ["april", "apr", "apr."],
            "May": ["may"],
            "June": ["june", "jun", "jun."],
            "July": ["july", "jul", "jul."],
            "August": ["august", "aug", "aug."],
            "September": ["september", "sept", "sept.", "sep", "sep."],
            "October": ["october", "oct", "oct."],
            "November": ["november", "nov", "nov."],
            "December": ["december", "dec", "dec."]
        }

def entity_match_score(predicted_entity, true_entity, entity_type, alpha=0.3):
    """Calculate the match score for the missing entity prediction using the exponential decay function"""
    if predicted_entity is None or true_entity is None:
        return 0.0
    
    if entity_type == "year":
        # For years, use exponential decay of numerical gaps
        try:
            pred_year = int(predicted_entity)
            true_year = int(true_entity)
            diff = abs(pred_year - true_year)
            return builtin_math.exp(-alpha * diff)
        except ValueError:
            return 0.0
    
    elif entity_type == "month":
        # For months, use looser matching criteria, including all month variations
        
        # Map month numbers to month names
        month_numbers = {1: "January", 2: "February", 3: "March", 4: "April", 
                        5: "May", 6: "June", 7: "July", 8: "August", 
                        9: "September", 10: "October", 11: "November", 12: "December"}
        
        # Standardized forecasts and real months are lowercase
        pred_lower = predicted_entity.lower()
        true_lower = true_entity.lower()
        
        # Try to parse the number month
        try:
            if pred_lower.isdigit():
                pred_month_num = int(pred_lower)
                if 1 <= pred_month_num <= 12:
                    pred_lower = month_numbers[pred_month_num].lower()
            
            if true_lower.isdigit():
                true_month_num = int(true_lower)
                if 1 <= true_month_num <= 12:
                    true_lower = month_numbers[true_month_num].lower()
        except (ValueError, KeyError):
            pass
        
        # For each standard month, check whether the prediction and the real value match any of its variants
        pred_standard_month = None
        true_standard_month = None
        
        for standard_month, variants in months_and_variants.items():
            if pred_lower in variants or pred_lower == standard_month.lower():
                pred_standard_month = standard_month
            
            if true_lower in variants or true_lower == standard_month.lower():
                true_standard_month = standard_month
        
        # If the forecast and the real month belong to the same standard month, match
        if pred_standard_month and true_standard_month and pred_standard_month == true_standard_month:
            return 1.0
        
        # If there is no exact match, try to calculate the distance between months
        month_order = list(months_and_variants.keys())
        
        if pred_standard_month and true_standard_month:
            try:
                pred_idx = month_order.index(pred_standard_month)
                true_idx = month_order.index(true_standard_month)
                
                # Calculate the ring distance - Take the minimum value of the direct distance and the distance around a circle
                direct_diff = abs(pred_idx - true_idx)
                circular_diff = 12 - direct_diff  # 12-month cycle
                month_diff = min(direct_diff, circular_diff)
                
                # Use the index decay of month gaps
                return builtin_math.exp(-alpha * month_diff)
            except (ValueError, IndexError):
                pass
    
    return 0.0

def is_valid_year(year_str):
    """Verify that the input is valid for 4-digit years"""
    try:
        year = int(year_str)
        return 1900 <= year <= 2100  # Set a reasonable range of years
    except ValueError:
        return False

def is_valid_month(month_str):
    """Verify that the input is a valid month name or variant"""
    month_str_lower = month_str.lower()
    
    # Check if any month or its variants are matched
    for standard_month, variants in months_and_variants.items():
        if month_str_lower in variants or month_str_lower == standard_month.lower():
            return True
    
    # If it is a number, check whether it is within the range of 1-12
    try:
        month_num = int(month_str)
        return 1 <= month_num <= 12
    except ValueError:
        pass
    
    return False

def compute_length_repetition_penalty(solution_str):
    """Calculate the penalty value for excessive answers and duplicate content
    Return a penalty value, 0 means no penalty, the larger the value, the more serious the penalty"""
    # No initial punishment
    length_penalty = 0.0
    repetition_penalty = 0.0
    
    # # Extract some of the text to think
    # think_content = ""
    # if "<think>" in solution_str and "</think>" in solution_str:
    #     try:
    #         think_content = solution_str.split("<think>")[1].split("</think>")[0]
    #     except:
    #         pass
    
    # Partialization
    tokens = solution_str.split()
    token_count = len(tokens)

    # Length penalty remains, but calculated separately
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
        for window_size in [3,5,7,9]:  # Detect duplication of 3-7 words
            for i in range(len(tokens) - window_size * 3):
                phrase = ' '.join(tokens[i:i+window_size])
                
                # Check for continuous duplication in the following text
                next_text = ' '.join(tokens[i+window_size:i+window_size*4])
                repeat_count = next_text.count(phrase)
                
                if repeat_count >= 2:  # The same phrase appears twice immediately afterwards
                    repetition_penalty = max(repetition_penalty, 0.15 * repeat_count)  # Up to 0.45 penalty
    
    # 3. Still retain global n-gram diversity detection, but use sliding windows instead of jump sampling
    if token_count > 200:
        chunks = [' '.join(tokens[i:i+5]) for i in range(0, min(len(tokens)-5, 500))]  # No more jumps every 5 words
        if chunks:
            unique_chunks = set(chunks)
            unique_ratio = len(unique_chunks) / len(chunks)
            
            # Punishment starts when the duplication exceeds 60%
            if unique_ratio < 0.5:
                repetition_penalty = max(repetition_penalty, (0.5 - unique_ratio) * 1.0)  # Increase punishment
    
    # Combined length punishment and repeated punishment
    total_penalty = repetition_penalty * 0.8 # Maximum 0.4
        
    total_penalty = max(length_penalty, repetition_penalty)  # Take a greater punishment
    
    return total_penalty                


#------------------------------------------------------------------------------------------------------------------------------

def extract_time_diff_answer(solution_str):
    """Extract time difference answers from the answer text, including two date and month difference
    Format: 'Event 1: YYYY-MM, Event 2: YYYY-MM. Month difference: XX.'
    Return (event1_date, event2_date, month_diff) or None"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if not match:
        return None, None, None
    
    answer_text = match.group(1).strip()
    
    # Extract the difference between two dates and months
    date_diff_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2})\. Month difference: (\d{1,3})\.'
    match = re.search(date_diff_pattern, answer_text)
    
    if match:
        event1_date = match.group(1)
        event2_date = match.group(2)
        month_diff = int(match.group(3))
        return event1_date, event2_date, month_diff
    
    return None, None, None

def compute_time_diff_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the time difference task
    
    parameter:
        solution_str (str): Solution text
        ground_truth (dict): A dictionary containing the difference between real date and month, in the format:
                           {"event1_pub_date": "YYYY-MM",
                            "event2_pub_date": "YYYY-MM",
                            "month_difference": XX}
        bonus (float): Additional reward when formatted correctly, default is 0.05
        alpha (float): The attenuation rate used for date_prediction_reward, default is 0.05
        tag_bonus_each (float): The reward for correct format for each tag, default is 0.025
        
    return:
        float: total rating"""
    # Analyze the answer
    event1_date, event2_date, month_diff = extract_time_diff_answer(solution_str)
    
    # "No event" punishment
    no_event_penalty = 0
    if event1_date and event2_date:
        if "no event" in event1_date.lower() or "no event" in event2_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # Format Rewards Section
    format_bonus = 0.0
    if (event1_date and event2_date and month_diff is not None and 
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date)):
        format_bonus = bonus
    
    # Accuracy calculation part
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    diff_accuracy = 0.0
    consistency_penalty = 1.0
    
    if format_bonus > 0:  # Calculate accuracy only if the format is correct
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_month_diff = ground_truth.get("month_difference")
        
        if true_event1_date and true_event2_date and true_month_diff is not None:
            # Calculate the accuracy of two date predictions
            if is_valid_date_format(true_event1_date):
                event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.25
            
            if is_valid_date_format(true_event2_date):
                event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.25
            
            # # Calculate the accuracy of the forecast for the difference in the month
            # diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha) * 0.5
            # Calculate the accuracy of the month difference prediction, and use different alphas according to the predicted value size
            if month_diff >= 25:
                # Use smaller alpha=0.05 for large months to provide looser ratings
                diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha=0.05) * 0.5
            else:
                # Use larger alpha=0.1 for small months, requiring stricter accuracy
                diff_accuracy = month_diff_reward(month_diff, true_month_diff, alpha=0.1) * 0.5
            
            # Check consistency and apply penalties
            consistency_penalty = compute_inconsistency_penalty(event1_date, event2_date, month_diff)
            
            # Apply penalties
            accuracy_score = (event1_accuracy + event2_accuracy + diff_accuracy) * consistency_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            consistency_penalty, "time_difference")

def compute_time_diff_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the time difference task, supporting the use of different alpha values ​​for each event"""
    # Analyze the answer
    event1_date, event2_date, month_diff = extract_time_diff_answer(solution_str)
    
    # "No event" punishment
    no_event_penalty = 0
    if event1_date and event2_date:
        if "no event" in event1_date.lower() or "no event" in event2_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # Format Rewards Section
    format_bonus = 0.0
    if (event1_date and event2_date and month_diff is not None and 
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date)):
        format_bonus = bonus
    
    # Accuracy calculation part
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    diff_accuracy = 0.0
    consistency_penalty = 1.0

    # Get training progress information
    current_alpha = ground_truth.get("current_alpha", None)
    global_step = ground_truth.get("global_step", 0)
    
    if format_bonus > 0:  # Calculate accuracy only if the format is correct
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_month_diff = ground_truth.get("month_difference")
        
        # Get difficulty information
        difficulty_info = ground_truth.get("difficulty", {})
        events_difficulty = difficulty_info.get("events_difficulty", [])
        
        # Set the default alpha value
        alpha1 = alpha
        alpha2 = alpha
        diff_alpha = alpha
        
        # If there is difficulty information, adjust the alpha according to the difficulty of each event
        if len(events_difficulty) == 2:
            # alpha1 = 0.1 if events_difficulty[0] else 0.07
            # alpha2 = 0.1 if events_difficulty[1] else 0.07
            if events_difficulty[0]:  # Simple sample
                alpha1 = 0.1
            else:  # Difficulty Sample
                # If current_alpha information is available, use progressive value, otherwise use original value 0.07
                alpha1 = current_alpha if current_alpha is not None else 0.07
                
            if events_difficulty[1]:  # Simple sample
                alpha2 = 0.1
            else:  # Difficulty Sample
                # If current_alpha information is available, use progressive value, otherwise use original value 0.07
                alpha2 = current_alpha if current_alpha is not None else 0.07
            # Use average alpha or default alpha for month difference
            if month_diff >= 25:
                diff_alpha = 0.05  # Use looser alpha in big months
            else:
                diff_alpha = (alpha1 + alpha2) / 2  # Use the average of two events alphas
        
        if true_event1_date and true_event2_date and true_month_diff is not None:
            # Calculate the accuracy of two date predictions, using their respective alpha
            if is_valid_date_format(true_event1_date):
                event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha1) * 0.25
            
            if is_valid_date_format(true_event2_date):
                event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha2) * 0.25
            
            # Calculate the accuracy of the month difference prediction, using diff_alpha
            diff_accuracy = month_diff_reward(month_diff, true_month_diff, diff_alpha) * 0.5
            
            # Check consistency and apply penalties
            consistency_penalty = compute_inconsistency_penalty(event1_date, event2_date, month_diff)
            
            # Apply penalties
            accuracy_score = (event1_accuracy + event2_accuracy + diff_accuracy) * consistency_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each

    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)
    
    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            consistency_penalty, "time_difference")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_time_order_answer(solution_str):
    """Extract chronological answers from the answer text, including three dates and time orders
    Format: 'Event 1: YYYY-MM, Event 2: YYYY-MM, Event 3: YYYY-MM. Event order: X-X-X.'
    Return (event1_date, event2_date, event3_date, event_order) or None"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if not match:
        return None, None, None, None
    
    answer_text = match.group(1).strip()
    
    # Extract three dates and order
    date_order_pattern = r'Event 1: (\d{4}-\d{2}), Event 2: (\d{4}-\d{2}), Event 3: (\d{4}-\d{2})\. Event order: (\d-\d-\d)\.'
    match = re.search(date_order_pattern, answer_text)
    
    if match:
        event1_date = match.group(1)
        event2_date = match.group(2)
        event3_date = match.group(3)
        event_order = match.group(4)
        return event1_date, event2_date, event3_date, event_order
    
    return None, None, None, None

def compute_date_diversity_penalty(event1_date, event2_date, event3_date, event_order):
    """Check if the dates are diverse and if the event order is not a simple default order 1-2-3"""
    try:
        # parse date
        year1, month1 = map(int, event1_date.split("-"))
        year2, month2 = map(int, event2_date.split("-"))
        year3, month3 = map(int, event3_date.split("-"))
        
        # Convert dates to total months
        total_months1 = year1 * 12 + month1
        total_months2 = year2 * 12 + month2
        total_months3 = year3 * 12 + month3
        
        # Check if all dates are the same
        if total_months1 == total_months2 == total_months3:
            return 0.2  # Serious Punishment - All dates are the same
        
        # Check if it is a simple continuous month pattern (e.g. 2018-11, 2018-12, 2019-01)
        is_sequential_0 = False  # Incremental mode initialization
        is_sequential_1 = False  # Decreasing mode initialization
        
        # Check if it is monthly incremental mode
        if (total_months2 == total_months1 + 1 and total_months3 == total_months2 + 1):
            is_sequential_0 = True
        # or monthly decreasing mode
        elif (total_months2 == total_months1 - 1 and total_months3 == total_months2 - 1):
            is_sequential_1 = True
        
        # Check whether the event order is just the default 1-2-3
        is_default_order_0 = (event_order == "1-2-3")
        is_default_order_1 = (event_order == "3-2-1")
        
        # Combination penalty
        if is_sequential_0 and is_default_order_0:
            return 0.2  # Moderate penalty - Continuous months and default order
        elif is_sequential_1 and is_default_order_1:
            return 0.2  # Mild punishment - just for consecutive months
        # elif is_default_order and abs(total_months1 - total_months2) < 3 and abs(total_months2 - total_months3) < 3:
        # return 0.7 # Very light punishment - default order and dates are close
        
        # Date is diverse and the order is not the default
        return 1.0  # No punishment
        
    except Exception:
        # If parsing errors occur, they should also be punished.
        return 0.5

def compute_time_order_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of chronological tasks
    
    parameter:
        solution_str (str): Solution text
        ground_truth (dict): A dictionary containing real date and time order, in the format:
                           {"event1_pub_date": "YYYY-MM",
                            "event2_pub_date": "YYYY-MM",
                            "event3_pub_date": "YYYY-MM",
                            "event_order": "X-X-X"}
        bonus (float): Additional reward when formatted correctly, default is 0.05
        alpha (float): The attenuation rate used for date_prediction_reward, default is 0.05
        tag_bonus_each (float): The reward for correct format for each tag, default is 0.025
        
    return:
        float: total rating"""
    # Analyze the answer
    event1_date, event2_date, event3_date, event_order = extract_time_order_answer(solution_str)
    
    # "No event" punishment
    no_event_penalty = 0
    if event1_date and event2_date and event3_date:
        if ("no event" in event1_date.lower() or "no event" in event2_date.lower() or 
            "no event" in event3_date.lower()):
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # Format Rewards section - Use the new is_valid_order_format function
    format_bonus = 0.0
    if (event1_date and event2_date and event3_date and event_order and
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date) and 
        is_valid_date_format(event3_date) and is_valid_order_format(event_order)):
        format_bonus = bonus
    
    # Accuracy calculation part
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    event3_accuracy = 0.0
    order_accuracy = 0.0
    consistency_penalty = 1.0
    combined_penalty = 1.0
    
    if format_bonus > 0:  # Calculate accuracy only if the format is correct
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_event3_date = ground_truth.get("event3_pub_date")
        true_event_order = ground_truth.get("event_order")
        
        if (true_event1_date and true_event2_date and true_event3_date and true_event_order and
            is_valid_date_format(true_event1_date) and is_valid_date_format(true_event2_date) and 
            is_valid_date_format(true_event3_date)):
            
            # Calculate the accuracy of the three date predictions (27% each)
            event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha) * 0.2
            event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha) * 0.2
            event3_accuracy = date_prediction_reward(event3_date, true_event3_date, alpha) * 0.2
            
            # Calculate the accuracy of sequential predictions (19%) - Use the new compute_order_accuracy function
            order_accuracy = compute_order_accuracy(event_order, true_event_order) * 0.4
            
            # Add consistency penalty - use the new compute_order_consistency_penalty function
            consistency_penalty = compute_order_consistency_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # Add date diversity penalty
            diversity_penalty = compute_date_diversity_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # Combining two punishments
            combined_penalty = consistency_penalty * diversity_penalty
            
            # Apply penalties to calculate total accuracy scores
            accuracy_score = (event1_accuracy + event2_accuracy + event3_accuracy + order_accuracy) * combined_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Return total score and scores of each part, including consistency penalty
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            combined_penalty, "time_ordering")

def compute_time_order_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the chronological task, supporting the use of different alpha values ​​for each event"""
    # Analyze the answer
    event1_date, event2_date, event3_date, event_order = extract_time_order_answer(solution_str)
    
    # "No event" punishment
    no_event_penalty = 0
    if event1_date and event2_date and event3_date:
        if ("no event" in event1_date.lower() or "no event" in event2_date.lower() or 
            "no event" in event3_date.lower()):
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # Format Rewards Section
    format_bonus = 0.0
    if (event1_date and event2_date and event3_date and event_order and
        is_valid_date_format(event1_date) and is_valid_date_format(event2_date) and 
        is_valid_date_format(event3_date) and is_valid_order_format(event_order)):
        format_bonus = bonus
    
    # Accuracy calculation part
    event1_accuracy = 0.0
    event2_accuracy = 0.0
    event3_accuracy = 0.0
    order_accuracy = 0.0
    consistency_penalty = 1.0
    combined_penalty = 1.0
    
    if format_bonus > 0:  # Calculate accuracy only if the format is correct
        true_event1_date = ground_truth.get("event1_pub_date")
        true_event2_date = ground_truth.get("event2_pub_date")
        true_event3_date = ground_truth.get("event3_pub_date")
        true_event_order = ground_truth.get("event_order")
        
        # Get difficulty information
        difficulty_info = ground_truth.get("difficulty", {})
        events_difficulty = difficulty_info.get("events_difficulty", [])

        # Get training progress information
        current_alpha = ground_truth.get("current_alpha", None)
        global_step = ground_truth.get("global_step", 0)
        
        # Set the default alpha value
        alpha1 = alpha
        alpha2 = alpha
        alpha3 = alpha
        
        # If there is difficulty information, adjust the alpha according to the difficulty of each event
        if len(events_difficulty) == 3:
            # alpha1 = 0.1 if events_difficulty[0] else 0.07
            # alpha2 = 0.1 if events_difficulty[1] else 0.07
            # alpha3 = 0.1 if events_difficulty[2] else 0.07

            # Added dynamic alpha code
            if events_difficulty[0]:  # Simple sample
                alpha1 = 0.1
            else:  # Difficulty Sample
                # If current_alpha information is available, use progressive value, otherwise use original value 0.07
                alpha1 = current_alpha if current_alpha is not None else 0.07
                
            if events_difficulty[1]:  # Simple sample
                alpha2 = 0.1
            else:  # Difficulty Sample
                # If current_alpha information is available, use progressive value, otherwise use original value 0.07
                alpha2 = current_alpha if current_alpha is not None else 0.07
                
            if events_difficulty[2]:  # Simple sample
                alpha3 = 0.1
            else:  # Difficulty Sample
                # If current_alpha information is available, use progressive value, otherwise use original value 0.07
                alpha3 = current_alpha if current_alpha is not None else 0.07
        
        if (true_event1_date and true_event2_date and true_event3_date and true_event_order and
            is_valid_date_format(true_event1_date) and is_valid_date_format(true_event2_date) and 
            is_valid_date_format(true_event3_date)):
            
            # Calculate the accuracy of the three date predictions, each using the corresponding alpha
            event1_accuracy = date_prediction_reward(event1_date, true_event1_date, alpha1) * 0.2
            event2_accuracy = date_prediction_reward(event2_date, true_event2_date, alpha2) * 0.2
            event3_accuracy = date_prediction_reward(event3_date, true_event3_date, alpha3) * 0.2
            
            # Calculate the accuracy of sequential predictions
            order_accuracy = compute_order_accuracy(event_order, true_event_order) * 0.4
            
            # Add consistency penalty
            consistency_penalty = compute_order_consistency_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # Add date diversity penalty
            diversity_penalty = compute_date_diversity_penalty(
                event1_date, event2_date, event3_date, event_order)
            
            # Combining two punishments
            combined_penalty = consistency_penalty * diversity_penalty
            
            # Apply penalties to calculate total accuracy scores
            accuracy_score = (event1_accuracy + event2_accuracy + event3_accuracy + order_accuracy) * combined_penalty
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each

    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)
    
    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Return total score and scores of each part, including consistency penalty
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            combined_penalty, "time_ordering")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def extract_time_completion_answer(solution_str):
    """Extract time-completion answers from the answer text, including event date and missing entity
    Format: 'Event: YYYY-MM. Missing entity: XXXXX.'
    Return (event_date, missing_entity) or None"""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str)
    if not match:
        return None, None
    
    answer_text = match.group(1).strip()
    
    # Extract event date and missing entity
    completion_pattern = r'Event: (\d{4}-\d{2})\. Missing entity: (.+?)\.'
    match = re.search(completion_pattern, answer_text)
    
    if match:
        event_date = match.group(1)
        missing_entity = match.group(2).strip()
        return event_date, missing_entity
    
    return None, None

def compute_time_completion_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the time completion task
    
    parameter:
        solution_str (str): Solution text
        ground_truth (dict): A dictionary containing the real date and missing entity in the format:
                           {"event_pub_date": "YYYY-MM",
                            "mask_type": "year" or "month",
                            "masked_entity": "XXXX"}
        bonus (float): Additional reward when formatted correctly, default is 0.05
        alpha (float): The attenuation rate used for date_prediction_reward, default is 0.05
        tag_bonus_each (float): The reward for correct format for each tag, default is 0.025
        
    return:
        float: total rating"""
    # Analyze the answer
    event_date, missing_entity = extract_time_completion_answer(solution_str)
    
    # "No event" punishment
    no_event_penalty = 0
    if event_date:
        if "no event" in event_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # Get mask type
    mask_type = ground_truth.get("mask_type", "")
    
    # Format Rewards Section - Add verification of missing_entity type
    format_bonus = 0.0
    if event_date and missing_entity and is_valid_date_format(event_date):
        # Verify the format of missing_entity based on mask_type
        if mask_type == "year" and is_valid_year(missing_entity):
            format_bonus = bonus
        elif mask_type == "month" and is_valid_month(missing_entity):
            format_bonus = bonus
        elif not mask_type:  # If there is no mask_type, loose processing
            format_bonus = bonus
    
    # Accuracy calculation part
    date_accuracy = 0.0
    entity_accuracy = 0.0
    
    if format_bonus > 0:  # Calculate accuracy only if the format is correct
        true_event_date = ground_truth.get("event_pub_date")
        masked_entity = ground_truth.get("masked_entity", "")
        
        if true_event_date and mask_type and masked_entity and is_valid_date_format(true_event_date):
            # Calculate the accuracy of date predictions (50%)
            date_accuracy = date_prediction_reward(event_date, true_event_date, alpha) * 0.5
            
            # Calculate the accuracy of missing entity predictions (50%)
            entity_accuracy = entity_match_score(missing_entity, masked_entity, mask_type, alpha*3) * 0.5
            
            # Total accuracy score
            accuracy_score = date_accuracy + entity_accuracy
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_completion")  # Consistency penalty is 1.0, because this task does not require consistency checks

def compute_time_completion_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the time completion task, supporting dynamic alpha"""
    # Analyze the answer
    event_date, missing_entity = extract_time_completion_answer(solution_str)
    
    # Get training progress information
    current_alpha = ground_truth.get("current_alpha", None)
    global_step = ground_truth.get("global_step", 0)

    # Get difficulty information and determine alpha value
    difficulty_info = ground_truth.get("difficulty", {})
    if difficulty_info:
        # Override the default parameters using the alpha value of the event itself
        event_alpha = difficulty_info.get("alpha", alpha)
        # If the original alpha is 0.07 (difficult sample), apply dynamic alpha
        if abs(event_alpha - 0.07) < 0.001 and current_alpha is not None:
            event_alpha = current_alpha  # Use dynamic alpha
        alpha = event_alpha  # Use dynamic alpha
    
    # "No event" punishment
    no_event_penalty = 0
    if event_date:
        if "no event" in event_date.lower():
            no_event_penalty = 0.1
    else:
        no_event_penalty = 0.2
    
    # Get mask type
    mask_type = ground_truth.get("mask_type", "")
    
    # Format Rewards Section
    format_bonus = 0.0
    if event_date and missing_entity and is_valid_date_format(event_date):
        # Verify the format of missing_entity based on mask_type
        if mask_type == "year" and is_valid_year(missing_entity):
            format_bonus = bonus
        elif mask_type == "month" and is_valid_month(missing_entity):
            format_bonus = bonus
        elif not mask_type:  # If there is no mask_type, loose processing
            format_bonus = bonus
    
    # Accuracy calculation part
    date_accuracy = 0.0
    entity_accuracy = 0.0
    
    if format_bonus > 0:  # Calculate accuracy only if the format is correct
        true_event_date = ground_truth.get("event_pub_date")
        masked_entity = ground_truth.get("masked_entity", "")
        
        if true_event_date and mask_type and masked_entity and is_valid_date_format(true_event_date):
            # Calculate the accuracy of date predictions (50%), using dynamic alpha
            date_accuracy = date_prediction_reward(event_date, true_event_date, alpha) * 0.5
            
            # Calculate the accuracy of missing entity predictions (50%), using the same dynamic alpha
            entity_accuracy = entity_match_score(missing_entity, masked_entity, mask_type, alpha*3) * 0.5
            
            # Total accuracy score
            accuracy_score = date_accuracy + entity_accuracy
        else:
            accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    
    # Tag Rewards Section
    tag_format_score = tag_bonus_each if format_reward_single(solution_str) == 1.0 else 0.0
    tag_count_score = tag_count_reward_single(solution_str) * tag_bonus_each
    
    # Apply length and repetition penalties
    length_repetition_penalty = compute_length_repetition_penalty(solution_str)

    # Total score calculation
    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_completion")

#---------------- Task 4: Time Inference -------------------#

def compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Compute the total score of the time inference task (exactly the same as the existing compute_score)
    
    parameter:
        solution_str (str): Solution text
        ground_truth (dict): Dictionary containing real dates in the format {"event_pub_date": "YYYY-MM"}
        bonus (float): Additional reward when formatted correctly, default is 0.05
        alpha (float): The attenuation rate used for date_prediction_reward, default is 0.05
        tag_bonus_each (float): The reward for correct format for each tag, default is 0.025
        
    return:
        float: total rating"""
    answer = extract_answer_format(solution_str)

    # "No event" punishment
    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.1  
    else:
        no_event_penalty = 0.2

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

    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_inferring")

def compute_time_inferring_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """Calculate the total score of the time inference task, supporting dynamic alpha"""
    answer = extract_answer_format(solution_str)

    # Get training progress information
    current_alpha = ground_truth.get("current_alpha", None)
    global_step = ground_truth.get("global_step", 0)

    # Get event difficulty information and determine alpha value
    difficulty_info = ground_truth.get("difficulty", {})
    if difficulty_info:
        # Override the default parameters using the alpha value of the event itself
        event_alpha = difficulty_info.get("alpha", alpha)
        # If the original alpha is 0.07 (difficult sample), apply dynamic alpha
        if abs(event_alpha - 0.07) < 0.001 and current_alpha is not None:
            event_alpha = current_alpha  # Use dynamic alpha
        alpha = event_alpha  # Use dynamic alpha
    
    # "No event" punishment
    no_event_penalty = 0
    if answer:
        if "no event" in answer.lower() or "none" in answer.lower():
            no_event_penalty = 0.1  
    else:
        no_event_penalty = 0.2

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

    total_score = format_bonus + accuracy_score + tag_format_score + tag_count_score - no_event_penalty - length_repetition_penalty
    
    # Returns the total score and the scores of each part, for easy debugging
    return (total_score, accuracy_score, format_bonus, tag_format_score, tag_count_score, 
            1.0, "time_inferring")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_score(solution_str, ground_truth, bonus=0.05, alpha=0.1, tag_bonus_each=0.025):
    """A unified scoring entry function, automatically selecting the appropriate scoring function according to the task type
    
    parameter:
        solution_str (str): Solution text
        ground_truth (dict): a dictionary containing the true situation
        bonus (float): Additional reward when formatted correctly, default is 0.05
        alpha (float): The attenuation rate used for date_prediction_reward, default is 0.05
        tag_bonus_each (float): The reward for correct format for each tag, default is 0.025
        
    return:
        tuple: (Total score, accuracy score, format reward, tag format score, tag count score, consistency penalty, task type)"""
    # Find task types from extra_info of ground_truth
    task_type = ""
    
    # First try to get the task type from the extra_info of non_tensor_batch
    if isinstance(ground_truth, dict) and "task" in ground_truth:
        task_type = ground_truth.get("task", "")
    
    # If not found, try to judge the task type from the characteristics of the ground_truth itself
    if not task_type:
        if "event1_pub_date" in ground_truth and "event2_pub_date" in ground_truth:
            if "event3_pub_date" in ground_truth:
                task_type = "time_ordering"
            else:
                task_type = "time_difference"
        elif "mask_type" in ground_truth and "masked_entity" in ground_truth:
            task_type = "time_completion"
        elif "event_pub_date" in ground_truth:
            task_type = "time_inferring"
    
    
    # Select the appropriate scoring function according to the task type
    if task_type == "time_difference":
        return compute_time_diff_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    elif task_type == "time_ordering":
        return compute_time_order_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    elif task_type == "time_completion":
        return compute_time_completion_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    elif task_type == "time_inferring":
        return compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)
    else:
        # Default time inference score
        return compute_time_inferring_score_fixed_alpha(solution_str, ground_truth, bonus, alpha, tag_bonus_each)




# Test cases
if __name__ == "__main__":
    # Task 1: Time difference calculation test
    solution1 = """<think>Analyze the time of the two events... It was judged that the first event occurred in March 2019 and the second event occurred in January 2020, a 10-month difference. </think>
<answer>Event 1: 2019-03, Event 2: 2020-01. Month difference: 10.</answer><|im_end|>"""
    
    ground_truth1 = {
        "event1_pub_date": "2019-03",
        "event2_pub_date": "2020-01", 
        "month_difference": 10,
        "extra_info": {"task": "time_difference"}
    }
    
    # Task 2: Time sorting test
    solution2 = """<think>Analyze the time of three events...</think>
<answer>Event 1: 2021-05, Event 2: 2019-11, Event 3: 2022-08. Event order: 2-1-3.</answer><|im_end|>"""
    
    ground_truth2 = {
        "event1_pub_date": "2021-05",
        "event2_pub_date": "2019-11",
        "event3_pub_date": "2022-08",
        "event_order": "2-1-3",
        "extra_info": {"task": "time_ordering"}
    }
    
    # Task 3: Time Completion Test
    solution3 = """<think>Analyze event time and missing information...</think>
<answer>Event: 2021-03. Missing entity: 2019.</answer><|im_end|>"""
    
    ground_truth3 = {
        "event_pub_date": "2021-03",
        "mask_type": "year",
        "masked_entity": "2019",
        "extra_info": {"task": "time_completion"}
    }
    
    # Task 4: Time Inference Test
    solution4 = "<think>Some reasoning...</think>\n<answer>2025-03</answer><|im_end|>"
    
    ground_truth4 = {
        "event_pub_date": "2025-03",
        "extra_info": {"task": "time_inferring"}
    }
    
    print("Task 1 Score:", compute_score(solution1, ground_truth1))
    print("Task 2 score:", compute_score(solution2, ground_truth2))
    print("Task 3 Score:", compute_score(solution3, ground_truth3))
    print("Task 4 Score:", compute_score(solution4, ground_truth4))









