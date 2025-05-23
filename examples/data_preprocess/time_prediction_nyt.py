import os
import re
import json
import random
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict

def construct_prefix(event):
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Please carefully read the following news article information:\n"
        f"Headline: {event['headline']}\n"
        f"Abstract: {event['abstract']}\n"
        "For the purpose of this prediction, assume that the event described in the article definitely will occur within the next few months or years. "
        "Based on the information provided and your general knowledge, determine the most likely specific future occurrence date of the event.\n"
        "- You can recall relevant and similar events in the past and their occurrence dates and identify the development patterns to help you predict.\n"
        "- Output the event's predicted occurrence date in the format 'YYYY-MM'.\n"
        "- Do not output 'No event' under any circumstances. Always provide your best prediction, even if the information is ambiguous.\n"
        "- Show your reasoning process in <think> </think> tags, and return the final answer on a new line in <answer> </answer> tags, for example <answer>2025-03</answer>.\n"
        "Your answer must strictly follow the above format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me carefully review all the relevant details and systematically work through the reasoning process.\n"
        "<think>"
    )

def parse_year_month_from_true(date_str: str):
    """
    Parse a date string in the format 'YYYY-MM'. Returns (year, month) or (None, None) if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m")
        return dt.year, dt.month
    except Exception:
        return None, None

def month_difference(year1, month1, year2, month2):
    """Calculate the difference in the month between two dates"""
    return (year2 - year1) * 12 + (month2 - month1)

def calculate_time_distance(event_date, reference_date="2024-01"):
    """Calculate the distance between the event date and the reference date in months
    Positive values ​​indicate the event after the reference date, negative values ​​indicate the event before the reference date"""
    ref_year, ref_month = parse_year_month_from_true(reference_date)
    event_year, event_month = event_date["year"], event_date["month"]
    
    return month_difference(ref_year, ref_month, event_year, event_month)

def load_events_by_year_month(input_dir, years_range=(2024, 2026)):
    """Load events for the specified year range and organize by year and month
    Returns a dictionary in the format: {(year, month): [events]}"""
    events_by_year_month = defaultdict(list)
    
    for year in range(years_range[0], years_range[1]):
        file_path = os.path.join(input_dir, f"{year}-0.jsonl")
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}, skip.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
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
                events_by_year_month[(year_val, month_val)].append(event)
    
    return events_by_year_month


def create_small_test_set(test_file_path, output_file_path, sample_count=1024, random_seed=1024):
    """Take a specified number of samples from the test set to ensure that the number of samples is equal each month
    
    parameter:
        test_file_path: path to the original test set
        output_file_path: The output small test set path
        sample_count: The total number of samples to be drawn, default 1024
        random_seed: Random seeds to ensure that the results are reproducible"""
    # Set random seeds
    random.seed(random_seed)
    
    # Read the original test set
    df_test = pd.read_parquet(test_file_path)
    test_data = df_test.to_dict('records')
    
    # Group by month
    samples_by_month = defaultdict(list)
    for sample in test_data:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        samples_by_month[(year, month)].append(sample)
    
    # Confirm what months are there
    available_months = sorted(samples_by_month.keys())
    print(f"Months included in the test set: {available_months}")
    
    # Calculate the number of samples required for each month
    months_count = len(available_months)  # It should be 7 months
    samples_per_month = sample_count // months_count  # 128 items per month
    
    print(f"{samples_per_month} pieces of data need to be extracted every month")
    
    #Take samples
    selected_samples = []
    insufficient_months = []
    
    for month_key in available_months:
        month_samples = samples_by_month[month_key]
        
        if len(month_samples) >= samples_per_month:
            # If the sample is sufficient, randomly select the specified number
            month_selected = random.sample(month_samples, samples_per_month)
        else:
            # If the sample is insufficient, use all and record insufficient months
            month_selected = month_samples
            insufficient_months.append((month_key, samples_per_month - len(month_samples)))
            print(f"Warning: {month_key} monthly samples are insufficient, only {len(month_samples)} entries, and {samples_per_month - len(month_samples)} entries are required")
        
        selected_samples.extend(month_selected)
    
    # Handle in case of insufficient samples
    if insufficient_months:
        shortage = sum(shortage for _, shortage in insufficient_months)
        print(f"The total is missing {shortage} samples and will be supplemented from other months")
        
        # Find out the months with surplus samples
        surplus_months = []
        for month_key in available_months:
            remaining = len(samples_by_month[month_key]) - samples_per_month
            if remaining > 0:
                surplus_months.append((month_key, remaining))
        
        # Calculate the number of additional extracts from each month with surplus samples
        extra_samples = []
        if surplus_months:
            # Allocation of additional quantities required by the surplus sample ratio
            total_surplus = sum(surplus for _, surplus in surplus_months)
            
            for month_key, surplus in surplus_months:
                # Calculate how many additional samples you need to draw this month
                extra_count = min(shortage * surplus // total_surplus, surplus)
                if extra_count > 0:
                    # Selected sample
                    already_selected = [s for s in selected_samples if 
                                       s["extra_info"]["year"] == month_key[0] and 
                                       s["extra_info"]["month"] == month_key[1]]
                    
                    # Optional samples (exclude selected samples)
                    available = [s for s in samples_by_month[month_key] if s not in already_selected]
                    
                    # Extra sampling
                    extra = random.sample(available, extra_count)
                    extra_samples.extend(extra)
                    shortage -= extra_count
                    
                    print(f"Extra {extra_count} samples from {month_key} months")
        
        # Add additional sample drawn
        selected_samples.extend(extra_samples)
    
    # Final confirmed total number of samples
    final_count = len(selected_samples)
    print(f"The total number of final samples taken: {final_count}")
    
    # Random disruption order
    random.shuffle(selected_samples)
    
    # count the final number of samples for each month
    final_month_dist = defaultdict(int)
    for sample in selected_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        final_month_dist[(year, month)] += 1
    
    print("\nFinally distributed month:")
    for month_key in sorted(final_month_dist.keys()):
        count = final_month_dist[month_key]
        print(f"{month_key[0]}-{month_key[1]:02d}: {count} ({count/final_count*100:.2f}%)")
    
    # Save as parquet file
    df_small_test = pd.DataFrame(selected_samples)
    df_small_test.to_parquet(output_file_path, index=False)
    print(f"Saved small test set to: {output_file_path}")
    
    return selected_samples


def parse_articles_from_v3_generation(jsonl_file):
    """Parses articles from v3 generation results and records filtered entries, especially those that succeed in simple matches but fail in detailed matches."""
    articles_by_month = defaultdict(list)
    filtered_out_articles = [] 
    extracted_articles_count = 0 
    
    with open(jsonl_file, 'r', encoding='utf-8') as f: # Added encoding
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id', '')
                response_content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

                if not custom_id or '_' not in custom_id:
                    filtered_out_articles.append({
                        "line_num": line_num,
                        "custom_id": custom_id,
                        "reason": "Invalid or missing custom_id",
                        "original_content_block": response_content
                    })
                    continue
                    
                date_str, desk, _ = custom_id.split('_', 2)
                
                if '-' in date_str:
                    year, month_num = map(int, date_str.split('-'))
                else:
                    filtered_out_articles.append({
                        "line_num": line_num,
                        "custom_id": custom_id,
                        "reason": f"Incorrect date format in custom_id: {date_str}",
                        "original_content_block": response_content
                    })
                    continue
                
                article_pattern_detailed = r'ARTICLE (\d+):\s*(?:\*\*)?Headline:(?:\*\*)?\s*(.*?)\s*(?:\*\*)?Abstract:(?:\*\*)?\s*(.*?)(?=\n\nARTICLE \d+:|$)'
                
                potential_article_blocks = re.split(r'(ARTICLE \d+:)', response_content)
                
                if potential_article_blocks and not potential_article_blocks[0].strip().startswith("ARTICLE"):
                    potential_article_blocks.pop(0)

                processed_article_indices_in_block = set()

                for match_detailed in re.finditer(article_pattern_detailed, response_content, re.DOTALL):
                    article_num_detailed = match_detailed.group(1)
                    headline_detailed = match_detailed.group(2).strip()
                    abstract_detailed = match_detailed.group(3).strip()

                    cleaned_headline = headline_detailed.replace('\n', ' ').strip()
                    cleaned_headline = re.sub(r'\*\*|\*', '', cleaned_headline).strip() # Remove all * and **
                    cleaned_abstract = abstract_detailed.replace('\n', ' ').strip()
                    cleaned_abstract = re.sub(r'\*\*|\*', '', cleaned_abstract).strip() # Remove all * and **

                    if not cleaned_headline or not cleaned_abstract:
                        filtered_out_articles.append({
                            "line_num": line_num,
                            "custom_id": custom_id,
                            "article_num_in_source": article_num_detailed,
                            "reason": "Detailed match but missing headline/abstract after cleaning",
                            "original_headline": headline_detailed,
                            "original_abstract": abstract_detailed,
                            "full_article_text_match": match_detailed.group(0)
                        })
                        processed_article_indices_in_block.add(article_num_detailed)
                        continue

                    extracted_articles_count += 1
                    true_pub_date = f"{year}-{month_num:02d}"
                    article = {
                        'headline': cleaned_headline,
                        'abstract': cleaned_abstract,
                        'desk': desk,
                        'true_pub_date': true_pub_date,
                        'year': year,
                        'month': month_num
                    }
                    articles_by_month[true_pub_date].append(article)
                    processed_article_indices_in_block.add(article_num_detailed)

                simple_matches = []
                for i in range(0, len(potential_article_blocks) -1, 2):
                    identifier = potential_article_blocks[i]
                    text_block = potential_article_blocks[i+1]
                    article_num_match_simple = re.match(r'ARTICLE (\d+):', identifier)
                    if article_num_match_simple:
                        simple_matches.append({
                            "id_text": identifier.strip(),
                            "num": article_num_match_simple.group(1),
                            "text_content": text_block.strip() 
                        })
                
                for pot_article in simple_matches:
                    article_num_simple = pot_article["num"]
                    if article_num_simple not in processed_article_indices_in_block:
                        text_block_after_id = pot_article["text_content"]
                        
                        headline_extracted_raw = ""
                        abstract_extracted_raw = ""

                        # Attempt 1: Match format like "**Headline:** content"
                        h_match_bold_keyword = re.search(r'\*\*Headline:\*\*\s*(.*?)(?=\s*\*\*Abstract:\*\*|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                        if h_match_bold_keyword:
                            headline_extracted_raw = h_match_bold_keyword.group(1).strip()

                        a_match_bold_keyword = re.search(r'\*\*Abstract:\*\*\s*(.*?)(?=\n\nARTICLE \d+:|\n\n---|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                        if a_match_bold_keyword:
                            abstract_extracted_raw = a_match_bold_keyword.group(1).strip()
                        
                        # Attempt 2: Fallback for formats like "Headline: content" (keywords not bold)
                        if not headline_extracted_raw: # Only if first attempt failed for headline
                            h_match_plain_keyword = re.search(r'(?<!\*)\bHeadline:\s*(.*?)(?=(?<!\*)\bAbstract:|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                            if h_match_plain_keyword:
                                headline_extracted_raw = h_match_plain_keyword.group(1).strip()
                        
                        if not abstract_extracted_raw: # Only if first attempt failed for abstract
                            a_match_plain_keyword = re.search(r'(?<!\*)\bAbstract:\s*(.*?)(?=\n\nARTICLE \d+:|\n\n---|$)', text_block_after_id, re.DOTALL | re.IGNORECASE)
                            if a_match_plain_keyword:
                                abstract_extracted_raw = a_match_plain_keyword.group(1).strip()
                        
                        final_headline = ""
                        if headline_extracted_raw:
                            final_headline = headline_extracted_raw.replace('\n', ' ').strip()
                            final_headline = re.sub(r'\*\*|\*', '', final_headline).strip()

                        final_abstract = ""
                        if abstract_extracted_raw:
                            final_abstract = abstract_extracted_raw.replace('\n', ' ').strip()
                            final_abstract = re.sub(r'\*\*|\*', '', final_abstract).strip()
                        
                        if final_headline and final_abstract:
                            extracted_articles_count += 1
                            true_pub_date = f"{year}-{month_num:02d}"
                            article = {
                                'headline': final_headline,
                                'abstract': final_abstract,
                                'desk': desk,
                                'true_pub_date': true_pub_date,
                                'year': year,
                                'month': month_num
                            }
                            articles_by_month[true_pub_date].append(article)
                            # Mark as processed to avoid re-logging if it somehow appeared again
                            processed_article_indices_in_block.add(article_num_simple) 
                        else:
                            # Log if still not parsable
                            filtered_out_articles.append({
                                "line_num": line_num,
                                "custom_id": custom_id,
                                "article_num_in_source": article_num_simple,
                                "reason": "Simple match (ARTICLE N:) found, but content extraction failed with new logic.",
                                "potential_article_identifier": pot_article["id_text"],
                                "following_text_block": text_block_after_id,
                                "attempted_headline_raw": headline_extracted_raw,
                                "attempted_abstract_raw": abstract_extracted_raw 
                            })
            
            except Exception as e:
                filtered_out_articles.append({
                    "line_num": line_num,
                    "custom_id": custom_id if 'custom_id' in locals() else "Unknown",
                    "reason": f"Exception during processing: {str(e)}",
                    "original_content_block": line 
                })
    
    return articles_by_month, filtered_out_articles

def build_new_training_dataset(original_parquet, v3_jsonl, output_parquet, filtered_output_jsonl="filtered_v3_samples.jsonl"):
    """Build a new training dataset"""
    # 1. Read the original training data
    print("Read the original training data...")
    df_original = pd.read_parquet(original_parquet)
    original_samples = df_original.to_dict('records')
    
    # 2. Group by month and randomly select 1,000 items per month
    print("Group by month and extract data...")
    samples_by_month = defaultdict(list)
    for sample in original_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        
        # Only keep data from January to July 2024
        if year == 2024 and 1 <= month <= 7:
            samples_by_month[(year, month)].append(sample)
    
    # Randomly draw 1,000 items per month
    selected_original_samples = []
    for (year, month), samples in samples_by_month.items():
        if len(samples) > 1000:
            selected = random.sample(samples, 1000)
        else:
            selected = samples
        selected_original_samples.extend(selected)
        print(f"{len(selected)} pieces of data were extracted from {year}-{month:02d}")

    # 3. Analyze the data generated by v3
    print("Parsing the data generated by v3...")
    articles_by_month, filtered_samples = parse_articles_from_v3_generation(v3_jsonl) # Get filtered samples
    
    # Save the filtered sample to a file
    if filtered_samples:
        print(f"{len(filtered_samples)} filed v3 generated entry is recorded to: {filtered_output_jsonl}")
        with open(filtered_output_jsonl, 'w', encoding='utf-8') as f_out:
            for item in filtered_samples:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 4. Convert v3 generated articles into training samples
    print("Convert v3 generated articles to training samples...")
    v3_samples = []
    index_counter = len(selected_original_samples)
    
    for month, articles in articles_by_month.items():
            
        for article in articles:
            year, month_num = article['year'], article['month'] 
            # if year is None or month_num is None: # This check has actually been done in parse_articles_from_v3_generation
            #     continue
            # Calculate the time distance from the reference date
            # time_distance = calculate_time_distance(article) # Make sure calculate_time_distance accepts dictionary
            time_distance = month_difference(2024, 1, year, month_num) # Use month_difference directly
            # time_distance = calculate_time_distance(article)
            
            # Construct prefix
            event = {
                'headline': article['headline'],
                'abstract': article['abstract']
            }
            prefix = construct_prefix(event)

            # Create training samples
            sample = {
                "data_source": "new_york_times",
                "prompt": [{
                    "role": "user",
                    "content": prefix
                }],
                "ability": "news_time_prediction",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "event_pub_date": article['true_pub_date'],
                        "time_distance": time_distance
                    }
                },
                "extra_info": {
                    "split": "train",
                    "index": index_counter,
                    "layer": "generated",  # Tagged as generated data
                    "time_distance": time_distance,
                    "year": year,
                    "month": month_num,
                    "desk": article['desk']  # Add news category information
                }
            }
            v3_samples.append(sample)
            index_counter += 1
    
    print(f"{len(v3_samples)} samples were extracted from v3 generation results")
    
    # 5. Merge data
    all_samples = selected_original_samples + v3_samples
    random.shuffle(all_samples)  # Random disruption
    
    # 6. Save as new train_time_prediction.parquet
    df_new = pd.DataFrame(all_samples)
    df_new.to_parquet(output_parquet, index=False)
    
    # 7. Statistical dataset status
    print("\nNew training set statistics:")
    print(f"Total number of samples: {len(all_samples)}")
    print(f"Number of raw data samples: {len(selected_original_samples)}")
    print(f"v3 generates samples: {len(v3_samples)}")

    # Statistics by month
    month_stats = defaultdict(int)
    for sample in all_samples:
        year = sample["extra_info"]["year"]
        month = sample["extra_info"]["month"]
        month_stats[(year, month)] += 1
    
    print("\nMonth distribution:")
    for (year, month), count in sorted(month_stats.items()):
        print(f"{year}-{month:02d}: {count}")
    
    # If there are v3 generated samples, statistics on the distribution of news categories
    if v3_samples:
        desk_stats = defaultdict(int)
        for sample in v3_samples:
            desk = sample["extra_info"].get("desk", "Unknown")
            desk_stats[desk] += 1
        
        print("\nv3 Generate sample news category distribution:")
        for desk, count in sorted(desk_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{desk}: {count}")
    
    return all_samples
            

if __name__ == "__main__":
    # Set random seeds to ensure that the results are reproducible
    random.seed(1024)
    
    # File path
    original_parquet = "Time-R1/datasets/train_time_prediction.parquet"
    v3_jsonl = "Time-R1/future_news_generation/v3_generation_4prediction_results.jsonl"
    output_parquet = "Time-R1/datasets/train_time_prediction_with_generated_1.parquet"
    filtered_output_jsonl = "Time-R1/examples/data_preprocess/filtered_v3_samples.jsonl" # Specify the output path
    
    # Build a new training dataset
    build_new_training_dataset(original_parquet, v3_jsonl, output_parquet, filtered_output_jsonl) # Pass new parameters







