"""EnCodec 24 kHz per-decoder-stage dump for C# parity debugging.
Captures the dequantized latent (decoder input) and the output of every
EncodecDecoder.layers[i] so the C# SeaNetDecoder DebugStageHook can be diffed
stage-by-stage. Reuses the codes from encodec24_ref_io.safetensors so the input
is identical to the failing test."""
import os, torch
from transformers import EncodecModel
from safetensors.torch import save_file, load_file

OUT = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("SCRATCH", OUT)
m = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()

ref = load_file(os.path.join(SCRATCH, "encodec24_ref_io.safetensors"))
codes = ref["codes"]  # [nQ, B, T] int32 (engine order)
nq, B, T = codes.shape
# HF quantizer.decode wants [nQ, B, T] as well (codes per quantizer layer).
ac = codes.to(torch.long)

dumps = {}
with torch.no_grad():
    emb = m.quantizer.decode(ac)  # [B, latent_dim, T]
    dumps["latent_in"] = emb.float().contiguous()
    print("latent_in", tuple(emb.shape), "mean", float(emb.mean()), "std", float(emb.std()))

    x = emb
    for i, layer in enumerate(m.decoder.layers):
        x = layer(x)
        dumps[f"stage_{i:02d}"] = x.float().contiguous()
        print(f"stage_{i:02d} {type(layer).__name__:24s} {tuple(x.shape)}  mean={float(x.mean()):+.4f} std={float(x.std()):.4f}")

save_file(dumps, os.path.join(OUT, "encodec24_stages.safetensors"))
print("wrote encodec24_stages.safetensors")
