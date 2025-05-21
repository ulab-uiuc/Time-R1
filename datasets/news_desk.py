import json

def count_news_desk_categories(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = data["response"]["docs"]
    desk_counts = {}

    for doc in docs:
        # 获取 news_desk 字段，如果不存在或为空，则使用 "Undefined" 作为标识
        desk = doc.get("news_desk") or "Undefined"
        desk_counts[desk] = desk_counts.get(desk, 0) + 1

    print(f"该 JSON 文件中共有 {len(desk_counts)} 个不同的 'news_desk' 类别：")
    for desk, count in desk_counts.items():
        print(f" - {desk}: {count} 条")

if __name__ == "__main__":
    # 将 archive.json 替换为你的 JSON 文件路径
    count_news_desk_categories("nyt_archives/2016_02.json")