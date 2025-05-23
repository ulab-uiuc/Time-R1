import json

def count_news_desk_categories(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = data["response"]["docs"]
    desk_counts = {}

    for doc in docs:
        # Get the news_desk field, if it does not exist or is empty, use "Undefined" as the identifier
        desk = doc.get("news_desk") or "Undefined"
        desk_counts[desk] = desk_counts.get(desk, 0) + 1

    print(f"There are {len(desk_counts)} different 'news_desk' categories in this JSON file:")
    for desk, count in desk_counts.items():
        print(f"- {desk}: {count}")

if __name__ == "__main__":
    # Replace archive.json with your JSON file path
    count_news_desk_categories("nyt_archives/2016_02.json")