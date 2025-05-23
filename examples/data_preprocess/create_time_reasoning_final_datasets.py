import pandas as pd
import random
import os
from typing import List, Dict, Any

def random_sample_from_parquet(file_path: str, sample_size: int) -> pd.DataFrame:
    """Randomly extract the specified number of samples from the parquet file
    
    parameter:
        file_path: parquet file path
        sample_size: The number of samples to be drawn
        
    return:
        DataFrame containing random sampling results"""
    print(f"Reading {file_path}...")
    df = pd.read_parquet(file_path)
    
    total_samples = len(df)
    if total_samples < sample_size:
        print(f"Warning: Requested {sample_size} samples but only {total_samples} available in {file_path}.")
        sample_size = total_samples
    
    print(f"Randomly sampling {sample_size} from {total_samples} samples in {file_path}")
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    return sampled_df

def create_combined_dataset(
    task_files: Dict[str, str], 
    sample_sizes: Dict[str, int],
    output_file: str,
    shuffle: bool = True
) -> None:
    """Sample from multiple task datasets and merge into one comprehensive dataset
    
    parameter:
        task_files: Mapping of task name to data file path
        sample_sizes: Map of task name to sample number
        output_file: output file path
        shuffle: Whether to disrupt the final dataset"""
    combined_df = pd.DataFrame()
    
    for task_name, file_path in task_files.items():
        if task_name not in sample_sizes:
            print(f"Warning: No sample size specified for {task_name}. Skipping.")
            continue
            
        sample_size = sample_sizes[task_name]
        sampled_df = random_sample_from_parquet(file_path, sample_size)
        
        # Record the size of each dataset so that it can be counted
        print(f"  - {task_name}: {len(sampled_df)} samples")
        
        # Merge into total dataset
        combined_df = pd.concat([combined_df, sampled_df], ignore_index=True)
    
    # Whether to disrupt data
    if shuffle:
        combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save the merged dataset
    print(f"Saving combined dataset with {len(combined_df)} samples to {output_file}")
    combined_df.to_parquet(output_file, index=False)
    
    # Print the distribution of each task in the final dataset
    task_counts = combined_df['extra_info'].apply(lambda x: x.get('task', 'unknown')).value_counts()
    print("\nTask distribution in the final dataset:")
    for task, count in task_counts.items():
        print(f"  - {task}: {count} samples ({count/len(combined_df)*100:.2f}%)")

def main():
    # Dataset root directory
    dataset_dir = "Time-R1/datasets"
    
    # Training dataset for each subtask
    train_files = {
        "time_difference": os.path.join(dataset_dir, "train_time_difference.parquet"),
        "time_ordering": os.path.join(dataset_dir, "train_time_ordering.parquet"),
        "time_completion": os.path.join(dataset_dir, "train_time_completion.parquet"),
        "time_inferring": os.path.join(dataset_dir, "train_time_inferring.parquet")
    }
    
    # Test dataset for each subtask
    test_files = {
        "time_difference": os.path.join(dataset_dir, "test_time_difference.parquet"),
        "time_ordering": os.path.join(dataset_dir, "test_time_ordering.parquet"),
        "time_completion": os.path.join(dataset_dir, "test_time_completion.parquet"),
        "time_inferring": os.path.join(dataset_dir, "test_time_inferring.parquet")
    }
    
    # Number of samples for training set
    train_samples = {
        "time_difference": 13000,
        "time_ordering": 13000,
        "time_completion": 13000,
        "time_inferring": 10000
    }
    
    # Number of samples for test set
    test_samples = {
        "time_difference": 256,
        "time_ordering": 256,
        "time_completion": 256,
        "time_inferring": 256
    }
    
    # Output file path
    train_output = os.path.join(dataset_dir, "train_time_reasoning_combined.parquet")
    test_output = os.path.join(dataset_dir, "test_time_reasoning_combined.parquet")
    
    # Create a training set
    print("\n===== Creating Combined Training Dataset =====")
    create_combined_dataset(train_files, train_samples, train_output)
    
    # Create a test set
    print("\n===== Creating Combined Test Dataset =====")
    create_combined_dataset(test_files, test_samples, test_output)
    
    print("\nProcess completed successfully!")

# import pandas as pd
# import os

def create_time_inferring_easy_dataset():
    """Extract samples that match train_easy_nyt.parquet from train_time_inferring.parquet,
    Create train_time_inferring_easy.parquet for first-stage training."""
    # File path
    easy_samples_path = "Time-R1/datasets/train_easy_nyt.parquet"
    all_samples_path = "Time-R1/datasets/train_time_inferring.parquet"
    output_path = "Time-R1/datasets/train_time_inferring_easy.parquet"
    
    # Load simple sample dataset
    easy_df = pd.read_parquet(easy_samples_path)
    
    # Create a matching set of titles and summary
    easy_samples_set = set()
    for _, row in easy_df.iterrows():
        content = row['prompt'][0]['content']
        
        # Extract headline and abstract from content
        try:
            headline_start = content.find("Headline: ") + len("Headline: ")
            headline_end = content.find("\n", headline_start)
            headline = content[headline_start:headline_end].strip()
            
            abstract_start = content.find("Abstract: ") + len("Abstract: ")
            abstract_end = content.find("\n", abstract_start)
            abstract = content[abstract_start:abstract_end].strip()
            
            easy_samples_set.add((headline, abstract))
        except Exception:
            continue
    
    print(f"Loaded {len(easy_samples_set)} unique samples from easy dataset.")
    
    # Load all time inference samples
    all_df = pd.read_parquet(all_samples_path)
    print(f"Loaded {len(all_df)} samples from all time_inferring dataset.")
    
    # Filter matching samples
    matched_indices = []
    
    for idx, row in all_df.iterrows():
        content = row['prompt'][0]['content']
        
        try:
            headline_start = content.find("Headline: ") + len("Headline: ")
            headline_end = content.find("\n", headline_start)
            headline = content[headline_start:headline_end].strip()
            
            abstract_start = content.find("Abstract: ") + len("Abstract: ")
            abstract_end = content.find("\n", abstract_start)
            abstract = content[abstract_start:abstract_end].strip()
            
            # Check if it is in a simple sample collection
            if (headline, abstract) in easy_samples_set:
                matched_indices.append(idx)
        except Exception:
            continue
    
    # Create a new DataFrame and save it
    matched_df = all_df.iloc[matched_indices].copy()
    
    # Update split field in extra_info
    for i in range(len(matched_df)):
        matched_df.iloc[i]['extra_info']['split'] = 'train_easy'
    
    # Save as a Parquet file
    matched_df.to_parquet(output_path, index=False)
    print(f"Finished! {len(matched_df)} matched samples saved to {output_path}.")

# if __name__ == "__main__":
#     create_time_inferring_easy_dataset()


# if __name__ == "__main__":
# # Set random seeds to ensure repeatability
#     random.seed(1024)
#     main()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_month_difference_distribution():
    """Analyze the month difference distribution of time_difference task in train_time_reasoning_combined.parquet"""
    # File path
    dataset_path = "Time-R1/datasets/train_time_reasoning_combined.parquet"
    
    # Read the dataset
    print(f"Reading {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # Filter time_difference task
    time_diff_df = df[df['extra_info'].apply(lambda x: x.get('task') == 'time_difference')]
    print(f"Found {len(time_diff_df)} time_difference samples in the dataset.")
    
    # Extract the month difference
    month_diffs = []
    for _, row in time_diff_df.iterrows():
        ground_truth = row['reward_model']['ground_truth']
        if isinstance(ground_truth, dict) and 'month_difference' in ground_truth:
            month_diff = ground_truth['month_difference']
            if month_diff is not None:
                month_diffs.append(float(month_diff))
    
    month_diffs = np.array(month_diffs)
    print(f"Extracted {len(month_diffs)} valid month difference values.")
    
    # Statistical basic information
    print("\n--- Month Difference Statistics ---")
    print(f"Mean: {np.mean(month_diffs):.2f}")
    print(f"Median: {np.median(month_diffs):.2f}")
    print(f"Min: {np.min(month_diffs):.2f}")
    print(f"Max: {np.max(month_diffs):.2f}")
    print(f"Std Dev: {np.std(month_diffs):.2f}")
    
    # interval distribution
    ranges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, np.inf]
    range_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', 
                    '41-50', '51-75', '76-100', '100+']
    
    counts = []
    for i in range(len(ranges)-1):
        count = np.sum((month_diffs >= ranges[i]) & (month_diffs < ranges[i+1]))
        counts.append(count)
        percentage = count / len(month_diffs) * 100
        print(f"{range_labels[i]}: {count} samples ({percentage:.2f}%)")
    
    # Return statistics
    return {
        'month_diffs': month_diffs,
        'ranges': ranges,
        'range_labels': range_labels,
        'counts': counts
    }

# if __name__ == "__main__":
#     stats = analyze_month_difference_distribution()
    
    # If you want to visualize, uncomment the following
    # plt.figure(figsize=(12, 6))
    # plt.bar(stats['range_labels'], stats['counts'])
    # plt.title('Month Difference Distribution in time_difference Task')
    # plt.xlabel('Month Difference Range')
    # plt.ylabel('Number of Samples')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig('month_diff_distribution.png')


def extract_headline_abstract(content):
    """Extract title and summary from prompt content"""
    try:
        headline_start = content.find("Headline: ") + len("Headline: ")
        headline_end = content.find("\n", headline_start)
        headline = content[headline_start:headline_end].strip()
        
        abstract_start = content.find("Abstract: ") + len("Abstract: ")
        abstract_end = content.find("\n", abstract_start)
        abstract = content[abstract_start:abstract_end].strip()
        
        return headline, abstract
    except Exception:
        return None, None

def extract_two_events(content):
    """Extract the title and summary of two events from the time_difference task prompt"""
    try:
        # Extract the first event
        event1_headline_start = content.find("News article 1:\nHeadline: ") + len("News article 1:\nHeadline: ")
        event1_headline_end = content.find("\n", event1_headline_start)
        event1_headline = content[event1_headline_start:event1_headline_end].strip()
        
        event1_abstract_start = content.find("Abstract: ", event1_headline_end) + len("Abstract: ")
        event1_abstract_end = content.find("\n", event1_abstract_start)
        event1_abstract = content[event1_abstract_start:event1_abstract_end].strip()
        
        # Extract the second event
        event2_headline_start = content.find("News article 2:\nHeadline: ") + len("News article 2:\nHeadline: ")
        event2_headline_end = content.find("\n", event2_headline_start)
        event2_headline = content[event2_headline_start:event2_headline_end].strip()
        
        event2_abstract_start = content.find("Abstract: ", event2_headline_end) + len("Abstract: ")
        event2_abstract_end = content.find("\n", event2_abstract_start)
        event2_abstract = content[event2_abstract_start:event2_abstract_end].strip()
        
        return (event1_headline, event1_abstract), (event2_headline, event2_abstract)
    except Exception:
        return None, None
    
def add_difficulty_tags_to_dataset(combined_path, easy_samples_path, output_path):
    """Add difficulty labels for each sample in the mixed dataset"""
    # Load the dataset
    combined_df = pd.read_parquet(combined_path)
    easy_df = pd.read_parquet(easy_samples_path)
    
    # Create a collection of identifications for simple samples (title + abstract)
    easy_identifiers = set()
    for _, row in easy_df.iterrows():
        content = row['prompt'][0]['content']
        try:
            headline, abstract = extract_headline_abstract(content)
            if headline and abstract:
                easy_identifiers.add((headline, abstract))
        except Exception:
            continue
            
    print(f"Loaded {len(easy_identifiers)} unique easy samples for matching")
    
    # Record the difficulty statistics of various types of tasks
    difficulty_stats = {
        "time_inferring": {"easy": 0, "hard": 0},
        "time_difference": {"easy": 0, "hard": 0},
        "time_ordering": {"easy": 0, "hard": 0},
        "time_completion": {"easy": 0, "hard": 0}
    }
    
    # traverse each sample of the mixed dataset and add difficulty labels
    for idx, row in combined_df.iterrows():
        task_type = row['extra_info']['task']
        content = row['prompt'][0]['content']
        
        # Initialization defaults to "difficult"
        difficulty = {
            "is_easy": False,
            "alpha": 0.07  # Use loose alpha for difficult samples
        }
        
        # Process according to task type
        if task_type == "time_inferring":
            try:
                headline, abstract = extract_headline_abstract(content)
                if headline and abstract and (headline, abstract) in easy_identifiers:
                    difficulty["is_easy"] = True
                    difficulty["alpha"] = 0.1  # Simple sample use strict alpha
                    difficulty_stats[task_type]["easy"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
                
        elif task_type == "time_difference":
            # Extract the title and summary of two events
            try:
                event1, event2 = extract_two_events(content)
                if event1 and event2:
                    event1_easy = (event1[0], event1[1]) in easy_identifiers
                    event2_easy = (event2[0], event2[1]) in easy_identifiers
                    
                    # Record the difficulty of each event
                    difficulty["events_difficulty"] = [int(event1_easy), int(event2_easy)]
                    # If both events are simple, the overall task is also simple.
                    if event1_easy and event2_easy:
                        difficulty["is_easy"] = True
                        difficulty["alpha"] = 0.1
                        difficulty_stats[task_type]["easy"] += 1
                    else:
                        difficulty_stats[task_type]["hard"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
                
        elif task_type == "time_ordering":
            # Handle three events
            try:
                # Extract three event information
                events_easy = []
                
                # Extract the first event
                event1_headline, event1_abstract = extract_event_info(content, 1)
                event1_easy = (event1_headline, event1_abstract) in easy_identifiers if event1_headline and event1_abstract else False
                events_easy.append(int(event1_easy))
                
                # Extract the second event
                event2_headline, event2_abstract = extract_event_info(content, 2)
                event2_easy = (event2_headline, event2_abstract) in easy_identifiers if event2_headline and event2_abstract else False
                events_easy.append(int(event2_easy))
                
                # Extract the third event
                event3_headline, event3_abstract = extract_event_info(content, 3)
                event3_easy = (event3_headline, event3_abstract) in easy_identifiers if event3_headline and event3_abstract else False
                events_easy.append(int(event3_easy))
                
                # Record the difficulty of each event
                difficulty["events_difficulty"] = events_easy
                
                # If all events are simple, the overall task is simple.
                if all(events_easy):
                    difficulty["is_easy"] = True
                    difficulty["alpha"] = 0.1
                    difficulty_stats[task_type]["easy"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
            
        elif task_type == "time_completion":
            # Special processing time_completion task
            try:
                # Extract title and summary
                headline, abstract = extract_headline_abstract(content)
                
                # Because it is difficult to restore the original content, the title or summary is marked as simple in a simple sample
                # traverse a simple sample collection
                found_match = False
                for easy_headline, easy_abstract in easy_identifiers:
                    # Check if the title or summary has a high match
                    if (headline and easy_headline and (headline in easy_headline or easy_headline in headline)) or \
                       (abstract and easy_abstract and (abstract in easy_abstract or easy_abstract in abstract)):
                        found_match = True
                        break
                
                if found_match:
                    difficulty["is_easy"] = True
                    difficulty["alpha"] = 0.1
                    difficulty_stats[task_type]["easy"] += 1
                else:
                    difficulty_stats[task_type]["hard"] += 1
            except Exception:
                difficulty_stats[task_type]["hard"] += 1
            
        # Add difficulty information to ground_truth
        combined_df.at[idx, 'reward_model']['ground_truth']['difficulty'] = difficulty
    
    # Print difficulty statistics
    print("\n=== Difficulty Statistics ===")
    for task_type, stats in difficulty_stats.items():
        total = stats["easy"] + stats["hard"]
        if total > 0:
            easy_percent = stats["easy"] / total * 100
            print(f"{task_type}: {stats['easy']} easy ({easy_percent:.1f}%), {stats['hard']} hard ({100-easy_percent:.1f}%)")
    
    # Save the dataset with difficulty tags added
    combined_df.to_parquet(output_path, index=False)
    print(f"Saved dataset with difficulty tags to {output_path}")

def extract_event_info(content, event_num):
    """Extract the title and summary of the specified event from the time_ordering task prompt"""
    try:
        event_headline_start = content.find(f"News article {event_num}:\nHeadline: ") + len(f"News article {event_num}:\nHeadline: ")
        event_headline_end = content.find("\n", event_headline_start)
        event_headline = content[event_headline_start:event_headline_end].strip()
        
        event_abstract_start = content.find("Abstract: ", event_headline_end) + len("Abstract: ")
        event_abstract_end = content.find("\n", event_abstract_start)
        event_abstract = content[event_abstract_start:event_abstract_end].strip()
        
        return event_headline, event_abstract
    except Exception:
        return None, None
    
def main_difficulty_tagging():
    """Add difficulty labels and dynamic alpha values ​​to the time inference training dataset.
    Used to implement the second phase hybrid training in a multi-stage training strategy"""
    # Set file path
    dataset_dir = "Time-R1/datasets"
    
    # Enter a file
    combined_path = os.path.join(dataset_dir, "train_time_reasoning_combined.parquet")
    easy_samples_path = os.path.join(dataset_dir, "train_easy_nyt.parquet")
    
    # Output file
    output_path = os.path.join(dataset_dir, "train_time_reasoning_dynamic_alpha.parquet")
    
    print("\n===== Adding Difficulty Tags and Dynamic Alpha Values =====")
    print(f"Input combined dataset: {combined_path}")
    print(f"Easy samples reference: {easy_samples_path}")
    print(f"Output dataset: {output_path}")
    
    # Call the difficulty tag to add functions
    add_difficulty_tags_to_dataset(combined_path, easy_samples_path, output_path)
    
    print("\nDifficulty tagging completed successfully!")
    
    # # The same processing is done to the test set
    # test_combined_path = os.path.join(dataset_dir, "test_time_reasoning_combined.parquet")
    # test_output_path = os.path.join(dataset_dir, "test_time_reasoning_dynamic_alpha.parquet")
    
    # print("\n===== Adding Difficulty Tags to Test Dataset =====")
    # add_difficulty_tags_to_dataset(test_combined_path, easy_samples_path, test_output_path)
    
    # print("\nTest dataset difficulty tagging completed!")


if __name__ == "__main__":
    # Uncomment the following functions as needed to run different dataset generation tasks
    
    # Generate a simple time inference dataset
    # create_time_inferring_easy_dataset()
    
    # Generate mixed datasets
    # random.seed(1024)
    # main()
    
    # Analyze the month difference distribution
    # stats = analyze_month_difference_distribution()
    
    # Add difficulty labels and dynamic alpha values
    main_difficulty_tagging()