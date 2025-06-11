import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-large')

def create_embedding(df: pd.DataFrame):
    data = []

    for _, item in tqdm(df.iterrows(), total=len(df), desc="Creating Embeddings..."):
        item = item.to_dict()
        if not pd.isna(item["description"]):
            embedding = model.encode(item["description"], normalize_embeddings=True)
            item["embedding"] = embedding.tolist()  # Convert to list now
        else:
            item["embedding"] = None
        data.append(item)

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = pd.read_csv("./easy_read_images_nhs_uk_emb_with_s3.csv")
    df = create_embedding(df)
    df.to_parquet("./easy_read_images_nhs_uk_emb_with_s3.parquet", index=False)