import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="eng_Latn", tgt_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("EzekielMW/LuoKslGloss").to(device)


LANGUAGE_CODES = ["eng", "swa", "ksl", "luo"]
code_mapping = {
    'eng': 'eng_Latn',
    'swa': 'swh_Latn',
    'ksl': 'ace_Latn',
    'luo': 'luo_Latn',
}

# offset = tokenizer.sp_model_size + tokenizer.fairseq_offset
for code in LANGUAGE_CODES:
    i = tokenizer.convert_tokens_to_ids(code_mapping[code])
    tokenizer._added_tokens_encoder[code] = i


def translate(text, source_language="eng", target_language="swa"):
    inputs = tokenizer(text.lower(), return_tensors="pt").to(device)
    inputs['input_ids'][0][0] = tokenizer.convert_tokens_to_ids(source_language)  # source: English
    translated_tokens = model.to(device).generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language),  # target: Swahili
        max_length=100,
        num_beams=5,
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result