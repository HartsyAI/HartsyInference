#!/usr/bin/env python3
"""Reference F5-TTS on a good input. If THIS is clean and ours is garbled, ours has a bug.
Also dumps the generated mel (pre-vocoder) so we can localize DiT-vs-vocoder later."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
from f5_tts.api import F5TTS

REF = "/home/kalebbroo/.cache/hartsyinference/models/test-clips/jfk.wav"
REF_TEXT = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
GEN_TEXT = "The speech synthesizer is now working correctly."
OUT = "/tmp/hartsyinference_tts_to_stt/f5_REF_python.wav"

f5 = F5TTS(model="F5TTS_v1_Base", device="cuda" if torch.cuda.is_available() else "cpu")
print("model loaded; generating (nfe=32, cfg=2, sway=-1) ...", flush=True)
torch.manual_seed(7)
wav, sr, spec = f5.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=GEN_TEXT,
                         nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1, seed=7)
wav = np.asarray(wav, dtype=np.float32).flatten()
sf.write(OUT, wav, sr)
np.save("/tmp/hartsyinference_tts_to_stt/f5_ref_mel.npy", np.asarray(spec))
print(f"WROTE {OUT} ({len(wav)/sr:.2f}s @ {sr}Hz) | mel shape {np.asarray(spec).shape}", flush=True)
