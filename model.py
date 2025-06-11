import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


default_prompt_path = "./prompts/prompt_2.txt"

with open(default_prompt_path, "r") as f:
    default_system_prompt = f.read()  # 392 tokens


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
    )
    terminators = [
        tokenizer.eos_token_id,
    ]

    return model_pipeline, tokenizer, terminators


def generate_easy_read(
    content: str, model_pipeline, tokenizer, terminators, system_prompt=None
) -> str:

    if not system_prompt:
        system_prompt = default_system_prompt

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    content = tokenizer.apply_chat_template(conversation)

    outputs = model_pipeline(
        conversation,
        max_new_tokens=2048,
        eos_token_id=terminators,
    )

    response = outputs[0]["generated_text"]

    return response[-1]["content"]
