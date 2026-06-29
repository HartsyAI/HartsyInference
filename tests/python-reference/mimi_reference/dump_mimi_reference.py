"""Mimi (kyutai/mimi) decode parity oracle + structure dump."""
import os, re, torch
from transformers import MimiModel
from safetensors.torch import save_file

OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
model = MimiModel.from_pretrained("kyutai/mimi").eval()
cfg = model.config
print("num_quantizers", cfg.num_quantizers, "semantic", cfg.num_semantic_quantizers,
      "codebook_size", cfg.codebook_size, "frame_rate", cfg.frame_rate, "hidden", cfg.hidden_size)

sd = {k: v.detach().contiguous().float().cpu() for k, v in model.state_dict().items()}
save_file(sd, os.path.join(OUT, "mimi.safetensors"))
print("weights:", len(sd))

def tree(prefix):
    ks = [k for k in sd if k.startswith(prefix)]
    return sorted(ks)

print("\n-- top groups --", sorted({k.split('.')[0] for k in sd}))
print("\n-- upsample --", tree("upsample"))
print("-- downsample --", tree("downsample"))
print("\n-- quantizer (first 24) --")
for k in tree("quantizer")[:24]:
    print("  ", k, tuple(sd[k].shape))
print("\n-- decoder_transformer.layers.0 --")
for k in tree("decoder_transformer.layers.0"):
    print("  ", k, tuple(sd[k].shape))
print("\n-- decoder (first 16) --")
for k in tree("decoder")[:16]:
    if not k.startswith("decoder_transformer"):
        print("  ", k, tuple(sd[k].shape))

# fixed codes: [B, num_quantizers, T]
nq = cfg.num_quantizers
T = 16
torch.manual_seed(1)
codes = torch.randint(0, cfg.codebook_size, (1, nq, T), dtype=torch.long)
with torch.no_grad():
    audio = model.decode(codes)[0]
print("\naudio", tuple(audio.shape), "rms", float(audio.pow(2).mean().sqrt()))

ref = {"codes": codes.to(torch.int32).cpu().contiguous(), "audio_out": audio.float().cpu().contiguous(),
       "nq": torch.tensor([nq], dtype=torch.int32)}
save_file(ref, os.path.join(OUT, "mimi_ref_io.safetensors"))
print("wrote ref; nq", nq, "T", T)
