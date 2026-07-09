#!/usr/bin/env python3
"""Authoritative nari-labs Dia reference on the SAME text + SAME params as our C# engine
(cfg 3.0 / temp 1.2 / top_p 0.95 / top_k 45 — identical to DiaConfig). If this is clean and
ours loops, the bug is definitively in our engine's AR loop."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, scipy.io.wavfile as wav
from dia.model import Dia

TEXT = ("[S1] Hello there! This is a test of the Dia text to speech model. "
        "[S2] It really does sound quite natural, doesn't it? "
        "[S1] Yes, the dialogue flows nicely between the two speakers.")
OUT = "/tmp/hartsyinference_tts_to_stt/dia_REF_nari.wav"

print("loading nari-labs/Dia-1.6B (float32) ...", flush=True)
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32")
print("generating (max_tokens=512, cfg3.0/temp1.2/top_p.95/top_k45, seed42) ...", flush=True)
torch.manual_seed(42)
audio = model.generate(TEXT, max_tokens=512, cfg_scale=3.0, temperature=1.2,
                       top_p=0.95, cfg_filter_top_k=45, use_torch_compile=False, verbose=True)
a = np.asarray(audio, dtype=np.float32).flatten()
sr = 44100
peak = float(np.abs(a).max()) + 1e-8
wav.write(OUT, sr, (a / peak * 0.95 * 32767).astype(np.int16))
print(f"WROTE {OUT}  ({len(a)/sr:.2f}s @ {sr}Hz, peak {peak:.3f})", flush=True)
