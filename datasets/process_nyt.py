import os
import json

def filter_and_combine_nyt_data(input_folder, output_folder):
    """
    从指定文件夹中读取所有 .json 文件，过滤并保留特定 news_desk 的文章，
    并将结果按年份写入 .jsonl 文件（一年一个文件）。
    """
    # 需要保留的 news_desk 列表（可根据实际需求增减或改写）
    allowed_desks = {
        "Politics", "National", "Washington", "U.S.",
        "Business", "SundayBusiness", "RealEstate",
        "Foreign", "World", "Metro", "Science", "Health", "Climate",
        "Opinion", "OpEd"
    }

    # 存储 {year: [filtered_docs, ...]} 的字典
    year_docs = {}

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历 input_folder 下的所有文件
    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue

        # 假设文件名格式为 "YYYY_MM.json"（如 2010_01.json）
        base_name = filename[:-5]  # 去掉 ".json"
        try:
            year_str, month_str = base_name.split("_")
            year_int = int(year_str)
        except ValueError:
            # 如果文件名格式不符合 "YYYY_MM.json"，则跳过
            print(f"文件名 {filename} 不符合 'YYYY_MM.json' 格式，已跳过。")
            continue

        # 仅处理 2025 年的数据
        if year_int != 2025:
            continue

        file_path = os.path.join(input_folder, filename)

        # 读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 假设新闻列表位于 data["response"]["docs"]
        docs = data.get("response", {}).get("docs", [])

        # 逐条过滤
        for doc in docs:
            news_desk = doc.get("news_desk", "")
            if news_desk in allowed_desks:
                # 仅保留以下字段
                filtered_doc = {
                    "abstract": doc.get("abstract", ""),
                    "snippet": doc.get("snippet", ""),
                    "lead_paragraph": doc.get("lead_paragraph", ""),
                    # headline -> headline["main"]
                    "headline": doc.get("headline", {}).get("main", ""),
                    "news_desk": news_desk,
                    "pub_date": doc.get("pub_date", "")
                }

                # 将结果存入对应年份的列表
                if year_int not in year_docs:
                    year_docs[year_int] = []
                year_docs[year_int].append(filtered_doc)

    # 按年份将结果写入 JSON Lines 文件
    for year in sorted(year_docs.keys()):
        out_path = os.path.join(output_folder, f"{year}_until_apr.jsonl")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for doc in year_docs[year]:
                out_f.write(json.dumps(doc, ensure_ascii=False))
                out_f.write("\n")
        print(f"已生成 {out_path}")

def main():
    input_folder = "nyt_archives"     # 存放原始数据文件的文件夹
    output_folder = "nyt_years" # 输出结果的文件夹

    filter_and_combine_nyt_data(input_folder, output_folder)

if __name__ == "__main__":
    main()