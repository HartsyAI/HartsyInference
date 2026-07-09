#!/usr/bin/env python3
"""Python Piper timing on the same voice our engine uses. Pure gen time, warmup + 3 runs."""
import time, io, wave, numpy as np
from piper import PiperVoice

VOICE = "/home/kalebbroo/.cache/hartsyinference/models/rhasspy--piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
TEXT = "Hello world. This is a test of the speech synthesizer."

voice = PiperVoice.load(VOICE)

def synth():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(TEXT, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        n = wf.getnframes(); sr = wf.getframerate()
    return n, sr

n, sr = synth()  # warmup
ts = []
for _ in range(3):
    t0 = time.perf_counter(); synth(); ts.append(time.perf_counter() - t0)
dur = n / sr
print(f"PIPER(onnx-cpu): audio {dur:.2f}s | gen {np.mean(ts):.3f}s (min {min(ts):.3f}) | RTF {np.mean(ts)/dur:.3f}", flush=True)
