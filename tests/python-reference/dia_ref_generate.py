#!/usr/bin/env python3
"""Reference Dia generation via HF transformers — does the REFERENCE loop on this text?
Builds the byte-level DiaTokenizer directly (the HF repo ships no tokenizer files) + the
feature extractor for DAC decode. Writes a WAV to STT-compare against our C# engine."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, scipy.io.wavfile as wav
from transformers import (DiaForConditionalGeneration, DiaTokenizer,
                          DiaFeatureExtractor, DiaProcessor, DacModel)

REPO = "nari-labs/Dia-1.6B"
TEXT = ("[S1] Hello there! This is a test of the Dia text to speech model. "
        "[S2] It really does sound quite natural, doesn't it? "
        "[S1] Yes, the dialogue flows nicely between the two speakers.")
OUT = "/tmp/hartsyinference_tts_to_stt/dia_REF_transformers.wav"

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} building processor + loading {REPO} ...", flush=True)
tok = DiaTokenizer()
try:
    fe = DiaFeatureExtractor.from_pretrained(REPO)
except Exception:
    fe = DiaFeatureExtractor()
audio_tok = DacModel.from_pretrained("descript/dac_44khz").to(dev).eval()
proc = DiaProcessor(feature_extractor=fe, tokenizer=tok, audio_tokenizer=audio_tok)
model = DiaForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float32).to(dev).eval()
print("loaded. generating (sampled, temp 1.2 / top_p 0.95 / top_k 45, seed 42) ...", flush=True)

inputs = proc(text=[TEXT], padding=True, return_tensors="pt").to(dev)
torch.manual_seed(42)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=True,
                         temperature=1.2, top_p=0.95, top_k=45)
outputs = proc.batch_decode(out)
a = outputs[0]
if hasattr(a, "detach"): a = a.detach().cpu().numpy()
a = np.asarray(a, dtype=np.float32).flatten()
sr = int(getattr(fe, "sampling_rate", 44100))
peak = float(np.abs(a).max()) + 1e-8
wav.write(OUT, sr, (a / peak * 0.95 * 32767).astype(np.int16))
print(f"WROTE {OUT}  ({len(a)/sr:.2f}s @ {sr}Hz, {len(a)} samples, peak {peak:.3f})", flush=True)
