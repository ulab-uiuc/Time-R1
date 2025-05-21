import os
import requests
import time

def download_nyt_archive(year, month, api_key, dest_folder):
    """
    下载指定年份和月份的纽约时报archive数据，并保存为JSON文件
    """
    # 构造请求URL
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果返回状态码不是200，将抛出异常
    except requests.RequestException as e:
        print(f"下载 {year}-{month:02d} 数据失败: {e}")
        return

    # 拼接保存的文件路径，例如：nyt_archives/2019_01.json
    filename = os.path.join(dest_folder, f"{year}_{month:02d}.json")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"成功下载 {year}-{month:02d} 的数据，保存在 {filename}")
    except Exception as e:
        print(f"保存 {year}-{month:02d} 数据失败: {e}")

def main():
    # API Key 和存储目录
    api_key = ""  # Your NYT API Key
    dest_folder = "nyt_archives"
    
    # 如果目录不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # # 设置需要下载的数据时间范围
    # start_year = 2016
    # end_year = 2024
    
    # for year in range(start_year, end_year + 1):
    #     for month in range(1, 13):
    #         download_nyt_archive(year, month, api_key, dest_folder)
    #         # 为避免每分钟请求超过5次，每次调用后休眠12秒
    #         time.sleep(12)

    download_nyt_archive(2025, 3, api_key, dest_folder)
    download_nyt_archive(2025, 4, api_key, dest_folder)

if __name__ == "__main__":
    main()