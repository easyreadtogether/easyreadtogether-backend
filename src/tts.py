from kokoro import KModel, KPipeline
import torch, numpy as np, io
from scipy.io.wavfile import write

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline_cache = {}
model_cache = {}

def get_pipeline(speaker):
    lang_code = speaker[0]
    if lang_code not in pipeline_cache:
        pipeline_cache[lang_code] = KPipeline(lang_code=lang_code, model=False)
    return pipeline_cache[lang_code]

def get_model():
    if device not in model_cache:
        model_cache[device] = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        print("TTS model loaded!")
    return model_cache[device]

def generate_tts(text: str, speed: float = 1.0, speaker_name: str = 'af_heart'):
    try:
        pipeline = get_pipeline(speaker_name)
        tts_model = get_model()
        pack = pipeline.load_voice(speaker_name)

        for _, ps, _ in pipeline(text, speaker_name, speed):
            ref_s = pack[len(ps) - 1]
            audio = tts_model(ps, ref_s, speed).numpy()
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

            buffer = io.BytesIO()
            write(buffer, 24000, audio_int16)
            buffer.seek(0)
            return buffer
    except Exception as e:
        return None


if __name__ == "__main__":
    # text = "Hello, this is a test using Kokoro TTS."
    text = "\n\nThe ministers agree to:\n1. Create more jobs for refugees, returnees, and host communities.\n2. Use a plan to help these people.\n3. Make sure governments include the needs of these people in their plans.\n4. Help these people get good education and find jobs.\n".strip().replace("\n", ". ").strip()
    speaker = "bf_lily"
    speed = 0.8

    output = generate_tts(text, speed, speaker)
    if output is None:
        print("TTS generation failed.")
    else:
        out_path = "tts_output.wav"
        with open(out_path, "wb") as f:
            f.write(output.read())
        print(f"Saved audio to: {out_path}")