import os
import requests
import time

def download_nyt_archive(year, month, api_key, dest_folder):
    """Download the New York Times archive data for the specified year and month and save it as a JSON file"""
    # Construct the request URL
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # If the return status code is not 200, an exception will be thrown
    except requests.RequestException as e:
        print(f"Download {year}-{month:02d} Data failed: {e}")
        return

    # Splice the saved file path, for example: nyt_archives/2019_01.json
    filename = os.path.join(dest_folder, f"{year}_{month:02d}.json")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Successfully downloaded the data of {year}-{month:02d} and saved in {filename}")
    except Exception as e:
        print(f"Save {year}-{month:02d} Data failed: {e}")

def main():
    # API Key and Storage Directory
    api_key = ""  # Your NYT API Key
    dest_folder = "nyt_archives"
    
    # Create if the directory does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # # Set the time range of data to be downloaded
    # start_year = 2016
    # end_year = 2024
    
    # for year in range(start_year, end_year + 1):
    #     for month in range(1, 13):
    #         download_nyt_archive(year, month, api_key, dest_folder)
    # # To avoid requests more than 5 times per minute, sleep for 12 seconds after each call
    #         time.sleep(12)

    download_nyt_archive(2025, 3, api_key, dest_folder)
    download_nyt_archive(2025, 4, api_key, dest_folder)

if __name__ == "__main__":
    main()