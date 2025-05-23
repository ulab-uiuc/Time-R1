import os
import json

def filter_and_combine_nyt_data(input_folder, output_folder):
    """Read all .json files from the specified folder, filter and preserve articles for specific news_desk,
    And write the results to a .jsonl file (one file per year)."""
    # The news_desk list that needs to be retained (can be added or reduced or rewritten according to actual needs)
    allowed_desks = {
        "Politics", "National", "Washington", "U.S.",
        "Business", "SundayBusiness", "RealEstate",
        "Foreign", "World", "Metro", "Science", "Health", "Climate",
        "Opinion", "OpEd"
    }

    # store dictionary of {year: [filtered_docs, ...]}
    year_docs = {}

    # If the output directory does not exist, create
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # traverse all files under input_folder
    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue

        # Assume that the file name format is "YYYY_MM.json" (such as 2010_01.json)
        base_name = filename[:-5]  # Remove ".json"
        try:
            year_str, month_str = base_name.split("_")
            year_int = int(year_str)
        except ValueError:
            # If the file name format does not conform to "YYYY_MM.json", skip
            print(f"Filename {filename} does not conform to the 'YYYY_MM.json' format and has been skipped.")
            continue

        # Process only data for 2025
        if year_int != 2025:
            continue

        file_path = os.path.join(input_folder, filename)

        # Read JSON files
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Assume that the news list is at data["response"]["docs"]
        docs = data.get("response", {}).get("docs", [])

        # Filter one by one
        for doc in docs:
            news_desk = doc.get("news_desk", "")
            if news_desk in allowed_desks:
                # Only the following fields are preserved
                filtered_doc = {
                    "abstract": doc.get("abstract", ""),
                    "snippet": doc.get("snippet", ""),
                    "lead_paragraph": doc.get("lead_paragraph", ""),
                    # headline -> headline["main"]
                    "headline": doc.get("headline", {}).get("main", ""),
                    "news_desk": news_desk,
                    "pub_date": doc.get("pub_date", "")
                }

                # Save the results into the list of the corresponding year
                if year_int not in year_docs:
                    year_docs[year_int] = []
                year_docs[year_int].append(filtered_doc)

    # Write results to JSON Lines file by year
    for year in sorted(year_docs.keys()):
        out_path = os.path.join(output_folder, f"{year}_until_apr.jsonl")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for doc in year_docs[year]:
                out_f.write(json.dumps(doc, ensure_ascii=False))
                out_f.write("\n")
        print(f"{out_path} has been generated")

def main():
    input_folder = "nyt_archives"     # folder where original data files are stored
    output_folder = "nyt_years" # The output folder

    filter_and_combine_nyt_data(input_folder, output_folder)

if __name__ == "__main__":
    main()