#!/usr/bin/env python3
"""Kokoro-82M reference on GPU: gen a fixed sentence, save wav, print time+dur+RMS, STT-verify."""
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, soundfile as sf, torch
from kokoro import KPipeline

TEXT = "The speech synthesizer is now working correctly."
OUT = "/tmp/hartsyinference_tts_to_stt/kokoro_REF_python.wav"
import os; os.makedirs("/tmp/hartsyinference_tts_to_stt", exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
pipe = KPipeline(lang_code="a", device=dev)
# warm
for _ in pipe("warm up", voice="af_heart"): pass
torch.cuda.synchronize() if dev == "cuda" else None

t0 = time.perf_counter()
audio = np.concatenate([a for _, _, a in pipe(TEXT, voice="af_heart")])
torch.cuda.synchronize() if dev == "cuda" else None
dt = time.perf_counter() - t0

audio = np.asarray(audio, dtype=np.float32).flatten()
sf.write(OUT, audio, 24000)
dur = len(audio) / 24000
rms = float(np.sqrt((audio ** 2).mean()))
print(f"KOKORO REF [{dev}]: gen {dt:.3f}s | audio {dur:.2f}s | RTF {dt/dur:.3f} | RMS {rms:.4f} -> {OUT}", flush=True)

import whisper
m = whisper.load_model("base")
print("Heard:", m.transcribe(OUT, language="en", fp16=False)["text"].strip(), flush=True)
