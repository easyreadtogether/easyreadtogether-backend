import boto3
import pandas as pd
import os
import uuid
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_IMAGE_BUCKET")

PREFIX = "images/"
INPUT_CSV = "./easy_read_images_nhs_uk_emb.csv"
OUTPUT_CSV = "./easy_read_images_nhs_uk_emb_with_s3.csv"

# S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def upload_file_to_s3(local_path):
    filename = os.path.basename(local_path)
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{uuid.uuid4().hex}{ext}"
    s3_key = os.path.join(PREFIX, unique_filename)
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"

def process_row(row):
    filepath = str(row["filepath"])
    if pd.notnull(filepath) and os.path.exists(filepath):
        row["s3_image_url"] = upload_file_to_s3(filepath)
    else:
        row["s3_image_url"] = None
    return row

def main():
    df = pd.read_csv(INPUT_CSV)
    tqdm.pandas(total=len(df), desc="Uploading to S3")
    df = df.progress_apply(process_row, axis=1)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved to {OUTPUT_CSV}")


def delete_all_images(bucket_name, prefix="images/"):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" in response:
        to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
        s3.delete_objects(Bucket=bucket_name, Delete={"Objects": to_delete})
        print(f"Deleted {len(to_delete)} objects from {bucket_name}/{prefix}")
    else:
        print(f"No objects found in {bucket_name}/{prefix}")

if __name__ == "__main__":
    main()
    ## delete_all_images(BUCKET_NAME, PREFIX)
