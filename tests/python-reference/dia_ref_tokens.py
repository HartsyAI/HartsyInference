#!/usr/bin/env python3
"""Reference Dia TOKEN generation (no DAC) — does the reference degenerate / emit EOS?
Compares directly against our C# channel-0 dump. If the reference progresses cleanly and
emits EOS while ours degenerates, our engine's AR loop has a bug."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from collections import Counter
from transformers import DiaForConditionalGeneration, DiaTokenizer

REPO = "nari-labs/Dia-1.6B"
TEXT = ("[S1] Hello there! This is a test of the Dia text to speech model. "
        "[S2] It really does sound quite natural, doesn't it? "
        "[S1] Yes, the dialogue flows nicely between the two speakers.")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} loading model ...", flush=True)
tok = DiaTokenizer()
model = DiaForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float32).to(dev).eval()
enc = tok([TEXT], padding=True, return_tensors="pt").to(dev)
print(f"input_ids {tuple(enc['input_ids'].shape)}; generating 512 (temp1.2/top_p.95/top_k45, seed42) ...", flush=True)
torch.manual_seed(42)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=512, do_sample=True,
                         temperature=1.2, top_p=0.95, top_k=45)
out = out.detach().cpu()
print("raw generate output shape:", tuple(out.shape), "dtype", out.dtype, flush=True)

# Dia output is the delayed audio-code grid. Reduce to a per-step channel-0 view.
arr = out.numpy()
if arr.ndim == 3:          # [B, T, C]
    ch0 = arr[0, :, 0]
    print("channels:", arr.shape[2])
elif arr.ndim == 2:        # [B, T] flattened or single-channel
    ch0 = arr[0]
else:
    ch0 = arr.flatten()
ch0 = [int(x) for x in ch0]
print("n steps:", len(ch0), "| unique:", len(set(ch0)))
print("EOS(1024) count:", ch0.count(1024), "| PAD(1025):", ch0.count(1025), "| BOS(1026):", ch0.count(1026))
print("most common:", Counter(ch0).most_common(6))
print("first 40:", ch0[:40])
print("last 40:", ch0[-40:])
np.save("/tmp/hartsyinference_tts_to_stt/dia_ref_ch0.npy", np.array(ch0))
print("saved ch0 tokens", flush=True)
