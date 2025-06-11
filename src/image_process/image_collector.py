import os
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse

from tqdm.auto import tqdm

import pandas as pd


root_url = "https://www.learningdisabilityservice-leeds.nhs.uk/easy-on-the-i/image-bank/page/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def get_page_images(page_no=1):
    response = requests.get(root_url+str(page_no), headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    ul = soup.find("ul", class_="grid-responsive grid-responsive--image-bank")

    results = []
    for li in ul.find_all("li", class_="image-bank-item"):
        img = li.find("img")
        a_tags = li.find_all("a", class_="btn--download")

        if img and len(a_tags) >= 1:
            entry = {
                "alt": img.get("alt", ""),
                "a1_url": a_tags[0].get("href"),
                "a1_text": a_tags[0].get_text(strip=True)
            }
            if len(a_tags) >= 2:
                entry["a2_url"] = a_tags[1].get("href")
                entry["a2_text"] = a_tags[1].get_text(strip=True)
                
            results.append(entry)
    
    return results


def download_image(url, folder):
    os.makedirs(folder, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    if not filename:
        raise ValueError("Cannot determine filename from URL.")

    filepath = os.path.join(folder, filename)

    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filepath



data = []

start = 1
end = 118

for i in tqdm(range(start, end+1)):
    data.extend(get_page_images(page_no=i))


for item in tqdm(data):
    try:
        filepath = download_image(item["a1_url"], "./images")
        item["filepath"] = filepath
    except:
        item["filepath"] = None


# !zip -r easyread_images.zip ./images

df = pd.DataFrame(data)
print(df.shape)

df.to_csv("easy_read_images_nhs_uk.csv", index=False)

print("Finished!")
