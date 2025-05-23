import os
import json
import random
import pandas as pd
from collections import defaultdict
from datetime import datetime

def create_future_news_prompt(target_date, seed_examples, news_desk):
    """
    Create a prompt for generating future news based on past examples.
    
    Args:
        target_date (str): Target date in YYYY-MM format
        seed_examples (list): List of 3 past news examples with headline, abstract and pub_date
        news_desk (str): The news desk category
        
    Returns:
        str: The formatted prompt
    """
    # Format the examples section
    examples_text = ""
    for i, example in enumerate(seed_examples):
        pub_date = example.get("pub_date", "").split("T")[0]  # Extract just the date part
        examples_text += f"EXAMPLE {i+1} [{pub_date}]:\n"
        examples_text += f"Headline: {example.get('headline', '')}\n"
        examples_text += f"Abstract: {example.get('abstract', '')}\n\n"
    
    # Create the prompt
    prompt = (
        f"You are an expert New York Times journalist for the {news_desk} desk.\n\n"
        f"I'm going to show you three recent news articles from the New York Times {news_desk} section "
        f"from May-July 2024. Study these examples to understand the NYT writing style and content patterns.\n\n"
        f"{examples_text}"
        f"Based on your general knowledge or prediction of current events and trends, generate SIX distinct and "
        f"plausible news headlines and abstracts for the {news_desk} section that might be published in {target_date}.\n\n"
        f"Guidelines:\n"
        f"- Create realistic and plausible content for {target_date}\n"
        f"- Write from the perspective of {target_date}, not as predictions\n"
        f"- Make each of the six articles substantially different from each other\n"
        f"- Maintain the New York Times style and tone for {news_desk} content\n"
        f"- Avoid extreme or unlikely scenarios\n\n"
        f"Provide your output in this exact format:\n\n"
        f"ARTICLE 1:\n"
        f"Headline: [News headline 1]\n"
        f"Abstract: [1-2 sentence news abstract 1]\n\n"
        f"ARTICLE 2:\n"
        f"Headline: [News headline 2]\n"
        f"Abstract: [1-2 sentence news abstract 2]\n\n"
        f"And so on through ARTICLE 6."
    )
    return prompt

def load_nyt_data(input_folder):
    """
    Load NYT data from May, June, July 2024 jsonl files
    """
    data = []
    
    # Look for files matching our criteria
    for filename in os.listdir(input_folder):
        if filename.startswith("2024") and filename.endswith(".jsonl"):
            file_path = os.path.join(input_folder, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        # Extract publication date
                        pub_date = article.get("pub_date", "")
                        if pub_date:
                            # Check if this article is from May-July 2024
                            date_obj = datetime.strptime(pub_date.split('T')[0], "%Y-%m-%d")
                            if date_obj.year == 2024 and 5 <= date_obj.month <= 7:
                                data.append(article)
                    except json.JSONDecodeError:
                        continue
    
    return data

def calculate_desk_distribution(articles):
    """
    Calculate distribution of news desk categories
    """
    # Count articles per desk
    desk_counts = {}
    for article in articles:
        desk = article.get("news_desk", "")
        if desk in desk_counts:
            desk_counts[desk] += 1
        else:
            desk_counts[desk] = 1
    
    # Calculate percentages
    total = sum(desk_counts.values())
    desk_distribution = {desk: count / total for desk, count in desk_counts.items() if count > 0}
    
    return desk_distribution

def generate_requests(articles, target_months=None, total_per_month=600):
    """
    Generate batch requests for generating future news
    
    Args:
        articles (list): List of past news articles
        target_months (list): List of target months in YYYY-MM format
        total_per_month (int): Total articles to generate per month
    """
    if target_months is None:
        target_months = ["2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
    
    # Group articles by news desk
    desk_articles = defaultdict(list)
    for article in articles:
        desk = article.get("news_desk", "")
        if desk:
            desk_articles[desk].append(article)
    
    # Use predefined distribution from chart
    desk_distribution = {
        "Foreign": 0.208,
        "Business": 0.165, 
        "OpEd": 0.142,
        "National": 0.109,
        "Washington": 0.096,
        "Metro": 0.086,
        "Politics": 0.055,
        "Science": 0.046
    }
    
    # Calculate how many articles to generate for each desk per month
    desk_counts = {}
    for desk, percentage in desk_distribution.items():
        desk_counts[desk] = max(1, round(total_per_month * percentage / 6))  # Divide by 6 as each prompt generates 6 articles
    
    # Generate requests
    requests = []
    used_examples = set()  # Track used examples to avoid duplication
    
    for target_month in target_months:
        month_requests = []
        
        for desk, count in desk_counts.items():
            if desk in desk_articles and len(desk_articles[desk]) >= 3:
                available_articles = [a for a in desk_articles[desk] 
                                     if json.dumps((a.get('headline', ''), a.get('abstract', ''))) not in used_examples]
                
                if len(available_articles) < 3:
                    # If we run out of unused articles, reset the tracking
                    used_examples = set()
                    available_articles = desk_articles[desk]
                
                for i in range(count):
                    # Select 3 random articles for examples
                    if len(available_articles) >= 3:
                        examples = random.sample(available_articles, 3)
                        
                        # Mark these examples as used
                        for example in examples:
                            used_examples.add(json.dumps((example.get('headline', ''), example.get('abstract', ''))))
                        
                        # Create the prompt
                        prompt = create_future_news_prompt(target_month, examples, desk)
                        
                        # Create the request
                        request = {
                            "custom_id": f"{target_month}_{desk}_{i}",
                            "body": {
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 2048,
                                "temperature": 1.2,
                                "top_p": 1
                            }
                        }
                        month_requests.append(request)
        
        print(f"Generated {len(month_requests)} requests for {target_month}")
        requests.extend(month_requests)
    
    return requests

def main():
    # Parameters
    input_folder = "Time-R1/datasets/nyt_years"
    output_file = "Time-R1/future_news_generation/v3_generation_4prediction_api_batch.jsonl"
    
    # Load past articles
    print("Loading articles from 2024 May-July...")
    articles = load_nyt_data(input_folder)
    print(f"Loaded {len(articles)} articles")
    
    # Generate requests
    requests = generate_requests(articles)
    print(f"Generated {len(requests)} total requests")
    
    # Write output file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    print(f"Output written to {output_file}")
    
    # Calculate expected articles
    total_articles = sum(len(req["body"]["messages"][0]["content"].split("ARTICLE ")) - 1 for req in requests)
    print(f"Expected to generate approximately {total_articles} articles across all months")

if __name__ == "__main__":
    main()