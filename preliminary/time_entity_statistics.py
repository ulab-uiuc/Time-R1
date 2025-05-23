import os
import re
import json
import pickle
import random
from datetime import datetime
from collections import Counter

# Define a list of months and its variants
MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

# Includes abbreviation and variations of the month
MONTH_VARIANTS = {
    "January": ["January", "Jan", "Jan."],
    "February": ["February", "Feb", "Feb."],
    "March": ["March", "Mar."],
    "April": ["April", "Apr", "Apr."],
    "May": ["May"],
    "June": ["June", "Jun", "Jun."],
    "July": ["July", "Jul", "Jul."],
    "August": ["August", "Aug", "Aug."],
    "September": ["September", "Sept", "Sept.", "Sep", "Sep."],
    "October": ["October", "Oct", "Oct."],
    "November": ["November", "Nov", "Nov."],
    "December": ["December", "Dec", "Dec."]
}

# Create a map of month alias to standard month
MONTH_MAPPING = {}
for standard, variants in MONTH_VARIANTS.items():
    for variant in variants:
        MONTH_MAPPING[variant.lower()] = standard

def parse_year_month_from_true(date_str: str):
    """
    Parse a date string in the format 'YYYY-MM'. Returns (year, month) or (None, None) if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def load_events(input_dir, years, split_ratio=0.9, random_seed=1024):
    """
    Load events from files and split them into train and test sets.
    Returns a tuple of (train_events, test_events)
    """
    all_events = []
    
    # Iterate over each year's file
    for year in years:
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue
                
            for line in lines:
                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue

                true_pub_date = data.get("true_pub_date", "")
                year_val, month_val = parse_year_month_from_true(true_pub_date)
                if year_val is None or month_val is None:
                    continue

                event = {
                    "headline": data.get("headline", ""),
                    "abstract": data.get("abstract", ""),
                    "true_pub_date": true_pub_date,
                    "year": year_val,
                    "month": month_val
                }
                all_events.append(event)
    
    # Split events into train and test sets - exactly the same as event_order_nyt.py
    random.seed(random_seed)
    random.shuffle(all_events)
    split_index = int(split_ratio * len(all_events))
    
    return all_events[:split_index], all_events[split_index:]

def extract_years(text):
    """Extract the year (four digits) in the text and filter the four digits of non-years using context rules"""
    years = []
    
    # Find four digits in the shape of yyyy
    year_matches = re.finditer(r'\b(19\d\d|20\d\d)\b', text)
    
    for match in year_matches:
        year = match.group(1)
        start, end = match.span()
        
        # Get the context (30 characters before and after)
        context_start = max(0, start - 15)
        context_end = min(len(text), end + 15)
        context = text[context_start:context_end].lower()
        
        # Check if the context supports this year
        # For example, there is a time vocabulary or preposition in the context that suggests time
        is_year = (
            # Time Vocabulary
            any(word in context for word in ["year""in", "on", "of", "since", "during", "before", "after", "through", "until", "for", "by", "at", "from"]) or
            # Month name
            any(month.lower() in context for month in MONTHS) or
            # Season Name
            any(season in context for season in ["spring", "summer", "fall", "autumn", "winter"])
        )
        
        if is_year:
            years.append(year)
    
    return years

def extract_months_0(text):
    """Extract the month name in the text and filter words with enhanced context rules that are not meaningful to month"""
    found_months = []
    text_lower = text.lower()
    
    # Ambiguity of the month words
    ambiguous_months = ["may", "march", "august"]
    
    # Common patterns of modal verb "may"
    modal_may_patterns = [
        r'may\s+(be|have|not|also|include|seem|become|need|want|make|take|get|help|cause|vary|\w+\s+to)',  # may followed by verb or infinitive
        r'(it|this|that|they|we|you|he|she|which|who)\s+may\s+\w+',  # Pronoun + may + verb
        r'(and|but|or|then|however|therefore)\s+may\s+\w+',  # conjunction + may + verb
        r'may\s+(not|never|only|also|still|yet)'  # may followed by adverb
    ]
    
    # Check every month
    for standard_month, variants in MONTH_VARIANTS.items():
        for variant in variants:
            variant_lower = variant.lower()
            
            # Simply check if the variant appears in the text
            if not re.search(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                continue
                
            # For months that may be ambiguity, additional context checks are required
            if variant_lower in ambiguous_months:
                # Extract the context containing the month
                contexts = []
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    # Use a larger context window (30 characters) to capture more context information
                    context_start = max(0, start - 15)
                    context_end = min(len(text_lower), end + 15)
                    context = text_lower[context_start:context_end]
                    contexts.append(context)
                
                # Check if any context meets the time-related conditions
                is_month = False
                for context in contexts:
                    # For "may" special check whether it is a modal verb
                    if variant_lower == "may":
                        # Check whether the pattern matches the modal verb may
                        is_modal_verb = any(re.search(pattern, context) for pattern in modal_may_patterns)
                        if is_modal_verb:
                            continue  # If it is a modal verb, skip this match
                    
                    # Check the time indication in the context
                    is_time_context = (
                        # Clear time format
                        re.search(r'(in|on|of|since|during|before|after|through|until|for|by)\s+' + re.escape(variant_lower), context) or
                        re.search(re.escape(variant_lower) + r'\s+\d{1,2}(st|nd|rd|th)?', context) or
                        # Year is nearby
                        re.search(r'\b(19\d\d|20\d\d)\b.*' + re.escape(variant_lower), context) or
                        re.search(re.escape(variant_lower) + r'.*\b(19\d\d|20\d\d)\b', context) or
                        # Other seasons or months are nearby
                        any(month.lower() != variant_lower and month.lower() in context for month in MONTHS) or
                        any(season in context for season in ["spring", "summer", "fall", "autumn", "winter"])
                    )
                    
                    if is_time_context:
                        is_month = True
                        break
                
                if not is_month:
                    continue  # If there is no time context, skip this month that may be ambiguous
            
            # Add to list of found months
            found_months.append(standard_month)
            break  # Find a variant and stop checking for other variants
    
    return found_months

def get_word_context(text, position, n_words=3):
    """Get the context of n words before and after the specified position"""
    # Split text into a word list and its position
    word_positions = []
    for match in re.finditer(r'\b\w+\b', text):
        word_positions.append((match.start(), match.end(), match.group()))
    
    if not word_positions:
        return [], "", []
    
    # Find the word index containing the target position
    target_index = -1
    for i, (start, end, _) in enumerate(word_positions):
        if start <= position < end:
            target_index = i
            break
    
    # If no exact match is found, find the closest word
    if target_index == -1:
        for i, (start, end, _) in enumerate(word_positions):
            if start > position:
                target_index = i
                break
        if target_index == -1:
            target_index = len(word_positions) - 1
    
    # Get n words before and after
    start_index = max(0, target_index - n_words)
    end_index = min(len(word_positions), target_index + n_words + 1)
    
    before_words = [word for _, _, word in word_positions[start_index:target_index]]
    current_word = word_positions[target_index][2] if target_index < len(word_positions) else ""
    after_words = [word for _, _, word in word_positions[target_index+1:end_index]]
    
    return before_words, current_word, after_words

def extract_months(text):
    """Extract the month names in the text and use word-based context rules for strict filtering"""
    found_months = []
    text_lower = text.lower()
    
    # Strictly filtered months
    strict_months = ["may", "march", "august"]
    
    for standard_month, variants in MONTH_VARIANTS.items():
        for variant in variants:
            variant_lower = variant.lower()
            
            # Simply check if the variant appears in the text
            if not re.search(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                continue
            
            # Apply special rules for months that require strict filtering
            if variant_lower in strict_months:
                is_month = False
                
                # Find all matching positions
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # Get word-level context (3 words in front and after)
                    before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                    
                    # Exclude special circumstances - Mar-a-Lago
                    if variant_lower == "mar" and "a-lago" in " ".join(after_words):
                        continue
                    
                    # Condition 1: Preposition + Month Format
                    if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                        is_month = True
                        break
                    
                    # Condition 2: Month + Number Format
                    if after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                        is_month = True
                        break
                    
                    # Condition 3: Month + Year format
                    if after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                        is_month = True
                        break
                    
                    # # Condition 4: Year + Month Format (New)
                    # if before_words and re.match(r'(19|20)\d{2}', before_words[-1]):
                    #     is_month = True
                    #     break
                
                # If no evidence is found to meet the criteria, skip this month
                if not is_month:
                    continue
            else:
                # For other months, use looser rules
                is_month = True  # Non-ambiguous months are accepted by default
            
            # Add to list of found months
            found_months.append(standard_month)
            break  # Find a variant and stop checking for other variants
    
    return found_months

def count_time_entity_distribution(events):
    """Statistics the distribution of time entities in each news to ensure that each month appears strictly verified"""
    
    # Used to store the number of time entities occur in each news
    year_counts_per_article = []
    month_counts_per_article = []  # Total number of occurrences in the month (strict verification)
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # Merge titles and abstracts for analysis
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # Extract year
        years = extract_years(full_text)
        year_counts_per_article.append(len(years))
        
        # count the number of occurrences in the month (using strict rules)
        month_count = 0
        
        # traverse all months and their variations
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # Find all matching positions for this variant
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # Apply verification rules
                    is_valid_month = False
                    
                    # Strict month requires special treatment
                    strict_months = ["may", "march", "august"]
                    if variant_lower in strict_months:
                        # Get word-level context
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # Exclude Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            continue
                        
                        # Verification Rules
                        # 1. Preposition + Month
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        
                        # 2. Month + Number
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        
                        # 3. Month + Year
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                        
                        # 4. Year + Month (optional activation)
                        # elif before_words and re.match(r'(19|20)\d{2}', before_words[-1]):
                        #     is_valid_month = True
                    else:
                        # Non-strict months can be counted directly
                        is_valid_month = True
                    
                    # If verification is passed, add 1
                    if is_valid_month:
                        month_count += 1
        
        # Record the number of valid months in this news
        month_counts_per_article.append(month_count)
    
    # Statistical distribution
    year_count_distribution = Counter(year_counts_per_article)
    month_count_distribution = Counter(month_counts_per_article)
    
    return year_count_distribution, month_count_distribution

def count_combined_time_entity_distribution(events):
    """Statistics the distribution of the total number of occurrences of time entities (year + month) in each news"""
    
    # Used to store the total number of time entities occurring in each news
    combined_counts_per_article = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # Merge titles and abstracts for analysis
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # Extract year
        years = extract_years(full_text)
        year_count = len(years)
        
        # count the number of occurrences in the month (using strict rules)
        month_count = 0
        
        # traverse all months and their variations
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # Find all matching positions for this variant
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # Apply verification rules
                    is_valid_month = False
                    
                    # Strict month requires special treatment
                    strict_months = ["may", "march", "august"]
                    if variant_lower in strict_months:
                        # Get word-level context
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # Exclude Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            continue
                        
                        # Verification Rules
                        # 1. Preposition + Month
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        
                        # 2. Month + Number
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        
                        # 3. Month + Year
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                    else:
                        # Non-strict months can be counted directly
                        is_valid_month = True
                    
                    # If verification is passed, add 1
                    if is_valid_month:
                        month_count += 1
        
        # Record the total number of years and months in this news
        combined_counts_per_article.append(year_count + month_count)
    
    # Statistical distribution
    combined_count_distribution = Counter(combined_counts_per_article)
    
    return combined_count_distribution

def analyze_time_entities(events):
    """Analyze the time entities in events"""
    year_counter = Counter()
    month_counter = Counter()
    
    # Record the number of events containing time entities
    events_with_years = 0
    events_with_months = 0
    
    # Save an example containing the time entity
    examples_with_years = []
    examples_with_months = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # Merge titles and abstracts for analysis
        full_text = headline + " " + abstract
        
        # Extract year and month
        years = extract_years(full_text)
        months = extract_months(full_text)
        
        # Update counter
        year_counter.update(years)
        month_counter.update(months)
        
        # Record the number of events containing time entities
        if years:
            events_with_years += 1
            # Store all examples without limiting the number
            examples_with_years.append({
                "text": full_text,
                "years": years
            })
        if months:
            events_with_months += 1
            # Store all examples without limiting the number
            examples_with_months.append({
                "text": full_text,
                "months": months
            })

        # # Record the number of events containing time entities
        # if years:
        #     events_with_years += 1
        # if len(examples_with_years) < 5: # Save the first 5 examples
        #         examples_with_years.append({
        #             "text": full_text,
        #             "years": years
        #         })
        # if months:
        #     events_with_months += 1
        # if len(examples_with_months) < 5: # Save the first 5 examples
        #         examples_with_months.append({
        #             "text": full_text,
        #             "months": months
        #         })
    
    return year_counter, month_counter, events_with_years, events_with_months, examples_with_years, examples_with_months

def print_results(train_years, train_months, test_years, test_months):
    """Print statistics results"""
    print("=== Year Statistics ===")
    print("Training set:")
    for year, count in sorted(train_years.items()):
        print(f"  {year}: {count}")
    
    print("\nTest Set:")
    for year, count in sorted(test_years.items()):
        print(f"  {year}: {count}")
    
    print("\n=== Month Statistics ===")
    print("Training set:")
    for month in MONTHS:
        print(f"  {month}: {train_months.get(month, 0)}")
    
    print("\nTest Set:")
    for month in MONTHS:
        print(f"  {month}: {test_months.get(month, 0)}")

# Add in main function, after print_results
def print_random_examples(examples_with_years, examples_with_months, count=100):
    """Randomly prints a specified number of year and month identification examples"""
    print(f"\n=== Random year identification examples ({count}) ===")
    if len(examples_with_years) > 0:
        # Randomly select the most count example
        sample_size = min(count, len(examples_with_years))
        sampled_year_examples = random.sample(examples_with_years, sample_size)
        
        for i, example in enumerate(sampled_year_examples, 1):
            # Truncate too long text for readability
            text_sample = example['text'][:249] + "..." if len(example['text']) > 249 else example['text']
            print(f"Example {i}/{sample_size}:")
            print(f"Text: {text_sample}")
            print(f"Recognized year: {example['years']}")
            
            # Print context to help verify the correctness of identification
            for year in example['years']:
                # Find the context around the year
                year_pos = example['text'].find(year)
                if year_pos != -1:
                    context_start = max(0, year_pos - 30)
                    context_end = min(len(example['text']), year_pos + len(year) + 30)
                    print(f"'{year}' context: ...{example['text'][context_start:context_end]}...")
            print()
    else:
        print("No examples containing the year were found")
        
    print(f"\n=== Random month identification examples ({count}) ===")
    if len(examples_with_months) > 0:
        # Randomly select the most count example
        sample_size = min(count, len(examples_with_months))
        sampled_month_examples = random.sample(examples_with_months, sample_size)
        
        for i, example in enumerate(sampled_month_examples, 1):
            # Truncate too long text for readability
            text_sample = example['text'][:249] + "..." if len(example['text']) > 249 else example['text']
            print(f"Example {i}/{sample_size}:")
            print(f"Text: {text_sample}")
            print(f"Recognized month: {example['months']}")
            
            # Print context to help verify the correctness of identification
            for month in example['months']:
                # Try to find the context around the month and its variants
                variants = MONTH_VARIANTS[month]
                for variant in variants:
                    if variant.lower() in example['text'].lower():
                        pos = example['text'].lower().find(variant.lower())
                        if pos != -1:
                            context_start = max(0, pos - 30)
                            context_end = min(len(example['text']), pos + len(variant) + 30)
                            print(f"'{month}' context: ...{example['text'][context_start:context_end]}...")
                            break
            print()
    else:
        print("No examples containing months were found")

def print_specific_month_examples(examples_with_months, month_name, count=30):
    """Random examples for printing a specific month"""
    # Filter examples containing specified months
    matching_examples = [example for example in examples_with_months 
                        if month_name in example['months']]
    
    print(f"\n=== Random {month_name} identification examples ({count}) ===")
    if len(matching_examples) > 0:
        # Randomly select the most count example
        sample_size = min(count, len(matching_examples))
        sampled_examples = random.sample(matching_examples, sample_size)
        
        for i, example in enumerate(sampled_examples, 1):
            # Truncate too long text for readability
            text_sample = example['text'][:249] + "..." if len(example['text']) > 249 else example['text']
            print(f"Example {i}/{sample_size}:")
            print(f"Text: {text_sample}")
            
            # Print context to help verify the correctness of identification
            # Try to find the context around the month and its variants
            variants = MONTH_VARIANTS[month_name]
            for variant in variants:
                if variant.lower() in example['text'].lower():
                    pos = example['text'].lower().find(variant.lower())
                    if pos != -1:
                        context_start = max(0, pos - 30)
                        context_end = min(len(example['text']), pos + len(variant) + 30)
                        print(f"'{month_name}' context: ...{example['text'][context_start:context_end]}...")
                        break
            print()
    else:
        print(f"No examples containing month {month_name} were found")

def create_masked_dataset0(events, output_file):
    """Perform time entity mask processing on event sets and save them in jsonl format"""
    
    masked_events = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # Merge titles and abstracts for analysis
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # Extract year and month and its location in text
        year_entities = []
        month_entities = []
        
        # Find the year
        for match in re.finditer(r'\b(19\d\d|20\d\d)\b', full_text):
            year_value = match.group(1)
            start, end = match.span()
            
            # Verify that it is the actual year
            is_year = True
            
            # Get the context
            context_start = max(0, start - 15)
            context_end = min(len(full_text), end + 15)
            context = full_text[context_start:context_end].lower()
            
            # Simple verification of context
            if any(word in context for word in ["year", "in", "on", "of", "since", "during", "before", "after"]) or \
               any(month.lower() in context for month in MONTHS) or \
               any(season in context for season in ["spring", "summer", "fall", "autumn", "winter"]):
                year_entities.append({
                    "value": year_value,
                    "start_in_full": start,
                    "end_in_full": end,
                    "type": "year"
                })
        
        # Find months
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                for match in re.finditer(r'\b' + re.escape(variant) + r'\b', full_text, re.IGNORECASE):
                    start, end = match.span()
                    
                    # Verify that it is a real month, especially for ambiguous months
                    is_valid_month = True
                    variant_lower = variant.lower()
                    
                    if variant_lower in ["may", "march", "august"]:
                        # Get the context
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # Exclude special situations such as Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            is_valid_month = False
                            continue
                        
                        # Apply verification rules
                        is_valid_month = False
                        
                        # Verification rules: preposition + month, month + number, month + year
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                    
                    if is_valid_month:
                        # Determine which part of the actual month is (headline or abstract)
                        if start < len(headline):
                            # in the title
                            start_in_component = start
                            component = "headline"
                        else:
                            # in the summary
                            start_in_component = start - len(headline) - 1  # minus 1 because there is a space in the middle
                            component = "abstract"
                        
                        end_in_component = start_in_component + (end - start)
                        
                        month_entities.append({
                            "value": standard_month,
                            "start_in_full": start,
                            "end_in_full": end,
                            "start_in_component": start_in_component,
                            "end_in_component": end_in_component,
                            "component": component,
                            "type": "month"
                        })
                    
                    # Skip other variants of that month after finding a valid month variant
                    if is_valid_month:
                        break
        
        # Merge all time entities
        all_entities = year_entities + month_entities
        
        # If there is no time entity, skip the event
        if not all_entities:
            # masked_events.append(event)
            continue
        
        # Randomly select a time entity for masking
        entity_to_mask = random.choice(all_entities)
        
        # Create a masked news object
        masked_event = event.copy()
        
        # Determine the mask mark
        mask_token = "<YEAR>" if entity_to_mask["type"] == "year" else "<MONTH>"
        
        # Apply mask
        if "start_in_component" in entity_to_mask:
            # Month situation - determined in which component
            component = entity_to_mask["component"]
            start = entity_to_mask["start_in_component"]
            end = entity_to_mask["end_in_component"]
            
            text = masked_event[component]
            masked_text = text[:start] + mask_token + text[end:]
            masked_event[component] = masked_text
        else:
            # Year situation - Need to determine which component to be in
            start = entity_to_mask["start_in_full"]
            end = entity_to_mask["end_in_full"]
            
            if start < len(headline):
                # in the title
                masked_text = headline[:start] + mask_token + headline[end:]
                masked_event["headline"] = masked_text
            else:
                # in the summary
                start_in_abstract = start - len(headline) - 1
                end_in_abstract = end - len(headline) - 1
                masked_text = abstract[:start_in_abstract] + mask_token + abstract[end_in_abstract:]
                masked_event["abstract"] = masked_text
        
        # Add mask information
        masked_event["masked_info"] = entity_to_mask["value"]
        masked_event["mask_type"] = entity_to_mask["type"]  # Optional: Record mask type
        
        masked_events.append(masked_event)
    
    # Save as jsonl format
    with open(output_file, 'w', encoding='utf-8') as f:
        for event in masked_events:
            f.write(json.dumps(event) + '\n')
    
    print(f"The saved data set processed by masks is {output_file}, with a total of {len(masked_events)} records.")
    return masked_events

def create_masked_dataset(events, output_file):
    """Perform time entity mask processing on event sets and save them in jsonl format"""
    
    masked_events = []
    
    for event in events:
        headline = event.get("headline", "")
        abstract = event.get("abstract", "")
        
        # Merge titles and abstracts for analysis
        full_text = headline + " " + abstract
        text_lower = full_text.lower()
        
        # Use a unified function to extract years and months
        years = extract_years(full_text)
        
        # Extract year and month and its location in text
        year_entities = []
        month_entities = []
        
        # Find the year and its location
        for year in years:
            # Find its position in the text for each year
            for match in re.finditer(r'\b' + re.escape(year) + r'\b', full_text):
                start, end = match.span()
                
                # Determine which component the year is in
                if start < len(headline):
                    # in the title
                    start_in_component = start
                    end_in_component = end
                    component = "headline"
                else:
                    # in the summary
                    start_in_component = start - len(headline) - 1
                    end_in_component = end - len(headline) - 1
                    component = "abstract"
                
                year_entities.append({
                    "value": year,
                    "start_in_full": start,
                    "end_in_full": end,
                    "start_in_component": start_in_component,
                    "end_in_component": end_in_component,
                    "component": component,
                    "type": "year"
                })
        
        # Find the month and its location
        for standard_month, variants in MONTH_VARIANTS.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # Find all matching positions for this variant
                for match in re.finditer(r'\b' + re.escape(variant_lower) + r'\b', text_lower):
                    start, end = match.span()
                    
                    # Apply verification rules
                    is_valid_month = False
                    
                    # Strict month requires special treatment
                    strict_months = ["may", "march", "august"]
                    if variant_lower in strict_months:
                        # Get word-level context
                        before_words, current_word, after_words = get_word_context(text_lower, start, n_words=3)
                        
                        # Exclude Mar-a-Lago
                        if variant_lower == "mar" and any("a-lago" in word for word in after_words):
                            continue
                        
                        # Verification Rules
                        # 1. Preposition + Month
                        if before_words and before_words[-1] in ["in", "on", "of", "since", "during", "before", "after", "through", "until", "by", "at", "from"]:
                            is_valid_month = True
                        
                        # 2. Month + Number
                        elif after_words and re.match(r'\d{1,2}(st|nd|rd|th)?', after_words[0]):
                            is_valid_month = True
                        
                        # 3. Month + Year
                        elif after_words and re.match(r'(19|20)\d{2}', after_words[0]):
                            is_valid_month = True
                    else:
                        # Non-strict months can be counted directly
                        is_valid_month = True
                    
                    # If verification is passed, add to the month entity list
                    if is_valid_month:
                        # Determine which part of the actual month is (headline or abstract)
                        if start < len(headline):
                            # in the title
                            start_in_component = start
                            end_in_component = end
                            component = "headline"
                        else:
                            # in the summary
                            start_in_component = start - len(headline) - 1  # minus 1 because there is a space in the middle
                            end_in_component = end - len(headline) - 1
                            component = "abstract"
                        
                        month_entities.append({
                            "value": standard_month,
                            "start_in_full": start,
                            "end_in_full": end,
                            "start_in_component": start_in_component,
                            "end_in_component": end_in_component,
                            "component": component,
                            "type": "month"
                        })
                
                # # After all matches in a month are processed, the inner loop will break out.
                # break
        
        # Merge all time entities
        all_entities = year_entities + month_entities
        
        # If there is no time entity, skip the event
        if not all_entities:
            continue
        
        # Randomly select a time entity for masking
        entity_to_mask = random.choice(all_entities)
        
        # Create a masked news object
        masked_event = event.copy()
        
        # Determine the mask mark
        mask_token = "<YEAR>" if entity_to_mask["type"] == "year" else "<MONTH>"
        
        # Apply mask
        component = entity_to_mask["component"]
        start = entity_to_mask["start_in_component"]
        end = entity_to_mask["end_in_component"]
        
        text = masked_event[component]
        masked_text = text[:start] + mask_token + text[end:]
        masked_event[component] = masked_text
        
        # Add mask information
        masked_event["masked_info"] = entity_to_mask["value"]
        masked_event["mask_type"] = entity_to_mask["type"]  # Optional: Record mask type
        
        masked_events.append(masked_event)
    
    # Save as jsonl format
    with open(output_file, 'w', encoding='utf-8') as f:
        for event in masked_events:
            f.write(json.dumps(event) + '\n')
    
    print(f"The saved data set processed by masks is {output_file}, with a total of {len(masked_events)} records.")
    return masked_events

def main():
    # Use the same data loading logic as event_order_nyt.py
    input_dir = "Time-R1/preliminary/original_ability_result"
    years = range(2016, 2024)  # Using data from 2016 to 2023
    
    # Check whether there are saved training/test data divisions
    split_cache_file = "Time-R1/datasets/nyt_train_test_split.pkl"
    
    if os.path.exists(split_cache_file):
        # If a cache file exists, directly load the divided data
        print(f"Loading train/test split from cache: {split_cache_file}")
        with open(split_cache_file, 'rb') as f:
            train_events, test_events = pickle.load(f)
    else:
        # If the cache file does not exist, create and save the segmentation
        print(f"Creating new train/test split and saving to: {split_cache_file}")
        train_events, test_events = load_events(input_dir, years)
        
        # Save split results for future use
        with open(split_cache_file, 'wb') as f:
            pickle.dump((train_events, test_events), f)
    
    # Add after statistics output
    print("\n=== Create time entity mask dataset ===")
    
    # Create an output directory
    masked_output_dir = "Time-R1/datasets/masked_time_entity"
    os.makedirs(masked_output_dir, exist_ok=True)
    
    # Process training sets and test sets
    train_output_file = os.path.join(masked_output_dir, "train_masked.jsonl")
    test_output_file = os.path.join(masked_output_dir, "test_masked.jsonl")
    
    create_masked_dataset(train_events, train_output_file)
    create_masked_dataset(test_events, test_output_file)
    
    print(f"\nMask dataset has been saved to {masked_output_dir}")



    # print(f"Analyze {len(train_events)} training events and {len(test_events)} test events...")
    
    # train_results = analyze_time_entities(train_events)
    # test_results = analyze_time_entities(test_events)
    
    # train_years, train_months, train_events_with_years, train_events_with_months, train_year_examples, train_month_examples = train_results
    # test_years, test_months, test_events_with_years, test_events_with_months, test_year_examples, test_month_examples = test_results

    # # Print results
    # print_results(train_years, train_months, test_years, test_months)
    
    # # # Save the results
    # # output_dir = "Time-R1/datasets/time_entity_stats"
    # # os.makedirs(output_dir, exist_ok=True)
    
    # # output_file = os.path.join(output_dir, "time_entity_statistics.pkl")
    
    # # results = {
    # #     'train_years': dict(train_years),
    # #     'train_months': dict(train_months),
    # #     'test_years': dict(test_years),
    # #     'test_months': dict(test_months)
    # # }
    
    # # with open(output_file, 'wb') as f:
    # #     pickle.dump(results, f)
    
    # # print(f"\nResult saved to {output_file}")
    
    # # Output overall statistics
    # print("\n=== Overall Statistics ===")
    # print(f"Total number of years mentioned in the training set: {sum(train_years.values())}")
    # print(f"Total number of months mentioned in the training set: {sum(train_months.values())}")
    # print(f"Total number of years mentioned in the test set: {sum(test_years.values())}")
    # print(f"Total number of months mentioned in the test set: {sum(test_months.values())}")
    
    # print(f"\nNumber of events in the training set contains years: {train_events_with_years} ({train_events_with_years/len(train_events)*100:.2f}%)")
    # print(f"Number of events in the training set contains months: {train_events_with_months} ​​({train_events_with_months/len(train_events)*100:.2f}%)")
    # print(f"Number of events in the test set contains years: {test_events_with_years} ({test_events_with_years/len(test_events)*100:.2f}%)")
    # print(f"Number of events in the test set contains months: {test_events_with_months} ​​({test_events_with_months/len(test_events)*100:.2f}%)")
    
    # print("\nMost common year:")
    # for year, count in train_years.most_common(5):
    #     print(f"  {year}: {count}")
    
    # print("\nMost Common Month:")
    # for month, count in train_months.most_common(5):
    #     print(f"  {month}: {count}")

    # # # Print recognition example
    # # print("\n=== Year Identification Example ===")
    # # for i, example in enumerate(train_year_examples[:3], 1):
    # # print(f"Example {i}:")
    # # print(f"text: {example['text']}...")
    # # print(f"Recognized year: {example['years']}")
    # #     print()
    
    # # print("\n=== Month recognition example ===")
    # # for i, example in enumerate(train_month_examples[:3], 1):
    # # print(f"Example {i}:")
    # # print(f"text: {example['text']}...")
    # # print(f"Recognized month: {example['months']}")
    # #     print()
    # # print("\nStop selecting and printing time entity recognition example...")
    # # print_random_examples(train_year_examples, train_month_examples, count=100) 


    # print("\nSelecting and printing a specific month's identification example...")
    # print_specific_month_examples(train_month_examples, "June", count=30)  
    # print_specific_month_examples(train_month_examples, "July", count=30)


    # print("\n=== Statistics on the total number of occurrences of time entities (year + month) in each news ===")
    # train_combined_dist = count_combined_time_entity_distribution(train_events)
    # test_combined_dist = count_combined_time_entity_distribution(test_events)

    # print("\nDistribution of total occurrences of time entities (year + month) in each news:")
    # print("training set:")
    # for count in sorted(train_combined_dist.keys()):
    # print(f" Number of news that the {count} time entity appears: {train_combined_dist[count]} ({train_combined_dist[count]/len(train_events)*100:.2f}%)")

    # print("\nTest Set:")
    # for count in sorted(test_combined_dist.keys()):
    # print(f" Number of news that the {count} time entity appears: {test_combined_dist[count]} ({test_combined_dist[count]/len(test_events)*100:.2f}%)")

    # # print("\n=== Statistics on the number of time entities appearing in each news ===")
    # # train_year_dist, train_month_dist = count_time_entity_distribution(train_events)
    # # test_year_dist, test_month_dist = count_time_entity_distribution(test_events)
    
    # # print("\nDistribution of years in each news:")
    # # print("training set:")
    # # for count in sorted(train_year_dist.keys()):
    # # print(f" Number of news in the following year of {count}: {train_year_dist[count]} ({train_year_dist[count]/len(train_events)*100:.2f}%)")
    
    # # print("\nTest Set:")
    # # for count in sorted(test_year_dist.keys()):
    # # print(f" Number of news in the following year of {count}: {test_year_dist[count]} ({test_year_dist[count]/len(test_events)*100:.2f}%)")
    
    # # print("\nDistribution of the number of occurrences in each news (after strict verification):")
    # # print("training set:")
    # # for count in sorted(train_month_dist.keys()):
    # # print(f" Number of news in the month after {count} appears: {train_month_dist[count]} ({train_month_dist[count]/len(train_events)*100:.2f}%)")
    
    # # print("\nTest Set:")
    # # for count in sorted(test_month_dist.keys()):
    # # print(f" Number of news in the month after {count} appears: {test_month_dist[count]} ({test_month_dist[count]/len(test_events)*100:.2f}%)")


def print_random_masked_examples(file_path, count=30):
    """Examples of randomly printing a specified number from the masked dataset"""
    # Read JSONL file
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"There are {len(examples)} records in the file")
    
    print(examples[0], examples[1])
    # # Example of randomly selecting a specified number
    # sample_count = min(count, len(examples))
    # sampled_examples = random.sample(examples, sample_count)
    
    # # Print selected example
    # for i, example in enumerate(sampled_examples, 1):
    #     headline = example.get("headline", "")
    #     abstract = example.get("abstract", "")
    #     masked_info = example.get("masked_info", "")
    #     mask_type = example.get("mask_type", "")
        
    # print(f"\nExample {i}/{sample_count}:")
    # print(f"Title: {headline}")
    # print(f"Abstract: {abstract}")
    # print(f"Mask Information: {masked_info}")
    # print(f"Mask type: {mask_type}")
    #     print("-" * 80)



if __name__ == "__main__":
    # main()

    train_file = "Time-R1/datasets/masked_time_entity/train_masked.jsonl"
    test_file = "Time-R1/datasets/masked_time_entity/test_masked.jsonl"

    print("=== Random example of training set ===")
    print_random_masked_examples(train_file, count=30)

    print("\n=== Random example of test set ===")
    print_random_masked_examples(test_file, count=30)