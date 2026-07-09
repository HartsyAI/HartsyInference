#!/usr/bin/env python3
"""Python-side Kokoro timing (onnx runtime). Pure generation time, excluding model load,
averaged over 3 runs on a fixed sentence."""
import time, os, urllib.request, numpy as np
from kokoro_onnx import Kokoro

D = "/home/kalebbroo/.cache/hartsyinference/models/_bench"
os.makedirs(D, exist_ok=True)
FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
for f, url in FILES.items():
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print(f"downloading {f} ...", flush=True); urllib.request.urlretrieve(url, p)
TEXT = "Hello world. This is a test of the speech synthesizer."
k = Kokoro(os.path.join(D, "kokoro-v1.0.onnx"), os.path.join(D, "voices-v1.0.bin"))
# warmup
s, sr = k.create(TEXT, voice="af_heart", speed=1.0, lang="en-us")
dur = len(s) / sr
ts = []
for _ in range(3):
    t0 = time.perf_counter(); k.create(TEXT, voice="af_heart", speed=1.0, lang="en-us"); ts.append(time.perf_counter() - t0)
print(f"KOKORO(onnx-cpu): audio {dur:.2f}s | gen {np.mean(ts):.3f}s (min {min(ts):.3f}) | RTF {np.mean(ts)/dur:.3f}", flush=True)
