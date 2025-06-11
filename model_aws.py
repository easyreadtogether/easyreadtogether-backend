import os

from transformers import AutoTokenizer

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


SUPPORTED_MODELS = [
    "meta.llama3-2-1b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-3-70b-instruct-v1:0",
]

aws_region = "us-east-1"

brt = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=aws_region,
)
model_id = "meta.llama3-3-70b-instruct-v1:0"

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


default_prompt_path = "./prompts/prompt_2.txt"

with open(default_prompt_path, "r") as f:
    default_system_prompt = f.read()  # 392 tokens

system_prompt = {
    "text": f"{default_system_prompt} \nOnly return the easyread version.",
}


def generate_easy_read(
    content: str, model_id=model_id, system_prompt=None, context_info=None
) -> str:

    print(f"INFO: Generating easy read for '{model_id}' model.")

    if model_id not in SUPPORTED_MODELS:
        print(f"ERROR: Model '{model_id}' is not supported.")
        return None

    model_id = f"us.{model_id}"

    if not system_prompt:
        system_prompt = {
            "text": f"{default_system_prompt} \nFollow the instruction and only return the easyread version.",
        }
    else:
        system_prompt = {
            "text": f"{system_prompt} \nOnly return the easyread version.",
        }

    if context_info is not None:
        content = f"Here is a context for the main content:\n{context_info}\nI want you to convert the following to easyread version:\n{content}"

    conversation = [
        {"role": "user", "content": [{"text": content}]},
    ]

    try:
        response = brt.converse(
            modelId=model_id,
            system=[system_prompt],
            messages=conversation,
            inferenceConfig={"maxTokens": 4096, "temperature": 0.5, "topP": 0.9},
        )

        response_text = response["output"]["message"]["content"][0]["text"]

        return response_text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        # exit(1)
        return ""


def count_tokens(text: str) -> int:
    return len(llama_tokenizer(text)["input_ids"])


if __name__ == "__main__":
    user_input = "Describe the purpose of a 'hello world' program in one line."
    output = generate_easy_read(user_input)
    print(output)
