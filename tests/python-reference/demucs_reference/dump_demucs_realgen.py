"""Real-generation reference: separate a real audio clip (jfk speech, resampled to 44.1k stereo) with BOTH
demucs apply_model (the production pipeline: segment chunking + overlap-add) and the raw model forward
(use_train_segment=False, = what the C# DemucsPipeline.Separate currently does). Saves the input + both stem
sets so the C# can be compared to each, plus the vocals stem as a WAV artifact."""
import os, torch, numpy as np
import soundfile as sf, scipy.signal as sps
from demucs.pretrained import get_model
from demucs.apply import apply_model
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
m = get_model("htdemucs")
sub = m.models[0].eval()
sr = 44100

w, orig_sr = sf.read(os.path.expanduser("~/.cache/sharpinference/models/test-clips/jfk.wav"))
if w.ndim > 1: w = w.mean(1)
w = w[: orig_sr * 6]                                  # 6 s
w = sps.resample(w, int(len(w) * sr / orig_sr)).astype(np.float32)
wav = np.stack([w, w * 0.9])                          # fake-stereo [2, L]
L = wav.shape[1]
x = torch.from_numpy(wav).float()
print("clip", tuple(x.shape), "dur", L / sr)

with torch.no_grad():
    prod = apply_model(m, x[None], split=True, overlap=0.25, progress=False)[0]   # [4,2,L] production
    sub.use_train_segment = False
    raw = sub(x[None])[0]                                                          # [4,2,L] raw model forward
print("prod", tuple(prod.shape), "raw", tuple(raw.shape), "sources", m.sources)

save_file({
    "wav": x[None].contiguous(),          # [1,2,L]
    "stems_apply": prod.contiguous(),     # production pipeline
    "stems_model": raw.contiguous(),      # raw forward (matches C# Separate)
}, os.path.join(OUT, "demucs_realgen.safetensors"))
vi = list(m.sources).index("vocals")
sf.write(os.path.join(OUT, "vocals_demucs.wav"), prod[vi].T.numpy(), sr)
print("wrote demucs_realgen.safetensors + vocals_demucs.wav; L =", L)
