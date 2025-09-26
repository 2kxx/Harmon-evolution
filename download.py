import os
import requests
from bs4 import BeautifulSoup
import shutil

baseurl = "https://hf-mirror.com/wusize/Harmon-1_5B/resolve/main/"
tree_url = "https://hf-mirror.com/wusize/Harmon-1_5B/tree/main"
dataset_path = "/hd2/tangzhenchen/model/harmon/harmon"

# 创建本地目录
os.makedirs(dataset_path, exist_ok=True)

# 获取网页内容
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
response = requests.get(tree_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# 解析符合 "pytorch_model-x-of-33.bin" 格式的文件
file_list = []
for link in soup.find_all('a'):
    href = link.get('href')
    if href and '/resolve/main/' in href:
        filename = href.split('/resolve/main/')[-1].split('?')[0]
        if filename.startswith("model-"):
            file_list.append(filename)

print(f"[i] 共找到 {len(file_list)} 个符合条件的文件")


# 下载符合条件的文件
def download_file(url, local_path):
    """支持断点续传的大文件下载"""
    temp_path = local_path + ".part"

    headers = {'User-Agent': 'Mozilla/5.0'}
    if os.path.exists(temp_path):
        # 获取已下载大小，支持断点续传
        downloaded_size = os.path.getsize(temp_path)
        headers['Range'] = f'bytes={downloaded_size}-'
    else:
        downloaded_size = 0

    with requests.get(url, headers=headers, stream=True) as r:
        if r.status_code == 416:
            # 416 表示请求范围超出（说明已经下载完整）
            os.rename(temp_path, local_path)
            print(f"[✓] 续传完成：{local_path}")
            return

        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0)) + downloaded_size

        with open(temp_path, 'ab') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB 块
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    print(f"\r[↓] 下载进度：{downloaded_size}/{total_size} ({downloaded_size / total_size:.2%})", end='',
                          flush=True)

    os.rename(temp_path, local_path)
    print(f"\n[✓] 下载完成：{local_path}")


for filename in file_list:
    local_path = os.path.join(dataset_path, filename)

    # 如果文件已下载，跳过
    if os.path.exists(local_path):
        print(f"[✓] 已存在，跳过：{filename}")
        continue

    print(f"[↓] 正在下载：{filename}")
    try:
        download_url = baseurl + filename
        download_file(download_url, local_path)
    except Exception as e:
        print(f"[!] 下载失败：{filename}, 错误：{e}")
