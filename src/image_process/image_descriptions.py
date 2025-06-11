import os
import ast

import pandas as pd
from google import genai
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

# WE use Gemini to get image descriptions
GEMINI_API_KEY = os.getenv(GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)


def generate_description(image_path, alt):
    my_file = client.files.upload(file=image_path)
    prompt = f"The following is an easy read compatible image. The Alt is '{alt}'. Please provide a description of it for easy read usage and a category. Only return a dict with keys description and category"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, prompt],
    )

    result = response.text
    result = result.lstrip("```json").rstrip("```").strip()
    return result


if __name__ == "__main__":
    df = pd.read_csv("./easy_read_images_nhs_uk.csv") # col: alt, a1_url

    data = []
    for idx, item in tqdm(df.iterrows(), total=len(df)):
        item = item.to_dict()
        if pd.isna(item["description"]):
            try:
                result = generate_description(item["filepath"], item["alt"])
                result = ast.literal_eval(result)

                item["description"] = result["description"]
                item["category"] = result["category"]
            except:
                item["description"] = None
                item["category"] = None

        data.append(item)
    
    df = pd.DataFrame(data)
    df.to_csv("./easy_read_images_nhs_uk.csv", index=False)
