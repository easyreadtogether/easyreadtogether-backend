import os
import io
import re
from uuid import uuid4
from datetime import datetime, timedelta
import wave
from dotenv import load_dotenv

load_dotenv()

import torch
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from markitdown import MarkItDown

from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse

from sentence_transformers import SentenceTransformer

from src.tts import generate_tts
from model_aws import generate_easy_read, count_tokens
from model_swa import translate

import sqlite3


DATABASE_PATH = "analytics.db"
def init_db():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                ip TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contact (
                name TEXT,
                email TEXT,
                organization TEXT
            )
        """)
init_db()


device = "cuda" if torch.cuda.is_available() else "cpu"


md = MarkItDown(enable_plugins=False)
emb_model = SentenceTransformer('intfloat/multilingual-e5-large').to(device)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = "/tmp/easyread"
os.makedirs(CACHE_DIR, exist_ok=True)


MODEL_OPTIONS = {
    "English": "meta.llama3-3-70b-instruct-v1:0",
    "Swahili": "swahili"
}
MAX_ALLOWED_TOKENS = 4096

default_prompt_path = "./prompts/prompt_2.txt"

with open(default_prompt_path, "r") as f:
    default_system_prompt = f.read()  # 392 llama tokens

# Embeddings and Metadata
image_emb_df = pd.read_parquet("./src/image_process/easy_read_images_nhs_uk_emb_with_s3.parquet")
image_emb_df = image_emb_df[image_emb_df["embedding"].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)].reset_index(drop=True)

desc_embs = np.stack(image_emb_df["embedding"])


def simplify_text(text: str, lang: str):
    model_id = MODEL_OPTIONS[lang]

    group_prompt = "Per each paragprah or block of content group, enclose the block content with <block> tag at the start and </block> tag at the end to mark a distinct separation. Make sure to keep this format."

    if model_id == "swahili":
        # system_prompt = default_prompt_path
        # custom_prompt = f"Please follow the above instruction. Regardless of whether the content is in English or Swahili, your response should only be in Swahili only. Only return the easyread in Swahili. {group_prompt} Content:"
        # output = get_easyread(
        #     text=text,
        #     prompt=f"{system_prompt}\n{custom_prompt}"
        # )

        text = translate(text, source_language="swa", target_language="eng")
        output = generate_easy_read(
                content=text, model_id=model_id, context_info=None, system_prompt=f"{default_system_prompt}\n{group_prompt}"
            )
        output = translate(output, source_language="eng", target_language="swa")
    else:
        output = generate_easy_read(
            content=text, model_id=model_id, context_info=None, system_prompt=f"{default_system_prompt}\n{group_prompt}"
        )
    

    return output


def find_closest_image(text, threshold=0.85):
    text_emb = emb_model.encode(text, normalize_embeddings=True)

    sims = cosine_similarity([text_emb], desc_embs)[0]

    max_idx = np.argmax(sims)

    if sims[max_idx] >= threshold:
        return image_emb_df.iloc[max_idx]["s3_image_url"], sims[max_idx]
    
    return None, None


@app.post("/api/simplify")
async def simplify(text: str = Form(None), lang: str = Form(...), file: UploadFile = File(None), image_threshold: float = Form(0.75)):
    if file and file.filename:
        file_id = f"{uuid4().hex}_{file.filename}"
        cache_path = os.path.join(CACHE_DIR, file_id)
        with open(cache_path, "wb") as f_out:
            f_out.write(await file.read())
        
        try:
            result = md.convert(cache_path)
        except Exception as e:
            return JSONResponse({"error": f"Failed to convert file: {str(e)}"}, status_code=400)
        text = result.markdown
        # remove file
        os.remove(cache_path)
    elif not text:
        return JSONResponse({"error": "No input text or file provided."}, status_code=400)
    
    output = simplify_text(text=text, lang=lang)
    contents = re.findall(r"<block>(.*?)</block>", output, re.DOTALL)

    if not image_threshold:
        image_threshold = 0.75

    data = []
    for content in contents:
        image_url, sim_score = find_closest_image(content, threshold=image_threshold)
        data.append({
            "content": content,
            "image_url": image_url,
            "similarity": float(sim_score) if image_url else None
        })

    return data


@app.post("/api/listen")
async def listen(text: str = Form(...), speed: float = Form(default=0.8)):
    audio_data = generate_tts(text=text.strip().replace("\n", ". "), speed=speed, speaker_name="bf_lily")
    if audio_data:
        return StreamingResponse(audio_data, media_type="audio/wav")
    return {"error": "TTS generation failed"}


@app.post("/api/authenticate") # mock authentication
async def listen(password: str = Form(...)):
    if password == os.getenv("MOCK_PASSWORD"):
        return {"success": True}
    return {"success": False}


@app.post("/api/analytics")
async def listen(ip: str = Form(...)):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("INSERT INTO analytics (ip) VALUES (?)", (ip,))
    return {"message": f"Logged IP {ip}"}


@app.get("/api/analytics")
def get_analytics():
    timeframes = {
        "1hr": datetime.utcnow() - timedelta(hours=1),
        "24hr": datetime.utcnow() - timedelta(hours=24),
        "7day": datetime.utcnow() - timedelta(days=7),
        "1month": datetime.utcnow() - timedelta(days=30),
        "all": None
    }

    result = {}
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        for label, since in timeframes.items():
            if since:
                cursor.execute("SELECT ip FROM analytics WHERE timestamp >= ?", (since,))
            else:
                cursor.execute("SELECT ip FROM analytics")

            ips = cursor.fetchall()
            ip_list = [ip[0] for ip in ips]
            unique_ips = set(ip_list)

            result[label] = {
                "unique_ips": len(unique_ips),
                "total_requests": len(ip_list)
            }

    return JSONResponse(result)


@app.post("/api/contact-us")
async def submit_contact(
    name: str = Form(...), 
    email: str = Form(...), 
    organization: str = Form(...)
):
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            "INSERT INTO contact (name, email, organization) VALUES (?, ?, ?)",
            (name.strip(), email.strip(), organization.strip())
        )
    return {"success": True}

@app.get("/api/contact-us")
async def get_contacts():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT name, email, organization FROM contact")
        contacts = [dict(row) for row in cursor.fetchall()]
    return {"contacts": contacts}

# uvicorn server:app --host 0.0.0.0 --port 8050 --reload

