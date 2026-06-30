"""
TEMPORARY DIAGNOSTIC (ERNIE-Image haze bug localization). Safe to delete.

Rewritten to match the INSTALLED diffusers API (self.layers / self.final_norm /
forward(hidden_states, timestep, text_bth, text_lens)). Loads the fp8 checkpoint
and casts to float32 (torch decodes F8_E4M3 -> f32, the reference decode), runs one
forward on CPU with the SAME synthetic inputs the C# diff test loads, and dumps
per-component f32 tensors aligned to the C# ErnieImageDebugDump names.

Output: tests/python-reference/ernie_image_reference_tensors/full_forward/
"""
import os, sys, json, torch
from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModel

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/ErnieImage/v1"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ernie_image_reference_tensors/full_forward")
os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)

def save(t, path):
    t.float().detach().cpu().contiguous().numpy().tofile(path)

def stats(name, t):
    f = t.float().flatten()
    return {"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
            "mean": float(f.mean()), "std": float(f.std()),
            "min": float(f.min()), "max": float(f.max()),
            "abs_mean": float(f.abs().mean()), "first_8": [float(x) for x in f[:8]]}

print(f"Building ErnieImageTransformer2DModel (V1 config) and loading shards from "
      f"{MODEL_DIR}/transformer ...", flush=True)
# No config.json ships in the transformer dir; construct explicitly to match
# C# ErnieImageConfig.V1. Build on the META device to avoid a 32GB random-init
# allocation, then load f32 weights with assign=True (no double alloc -> avoids OOM).
with torch.device("meta"):
    xfm = ErnieImageTransformer2DModel(
        hidden_size=4096, num_attention_heads=32, num_layers=36, ffn_hidden_size=12288,
        in_channels=128, out_channels=128, patch_size=1, text_in_dim=3072,
        rope_theta=256, rope_axes_dim=(32, 48, 48), eps=1e-6, qk_layernorm=True,
    )

import glob
from safetensors.torch import load_file
sd = {}
for shard in sorted(glob.glob(f"{MODEL_DIR}/transformer/*.safetensors")):
    part = load_file(shard)
    for k, v in part.items():
        sd[k] = v.to(torch.float32)  # torch decodes F8_E4M3 -> f32 (reference decode)
        del v
    del part
missing, unexpected = xfm.load_state_dict(sd, strict=False, assign=True)
print(f"  Loaded {len(sd)} tensors. missing={len(missing)} unexpected={len(unexpected)}", flush=True)
if missing: print("   MISSING:", missing[:12], flush=True)
if unexpected: print("   UNEXPECTED:", unexpected[:12], flush=True)
xfm = xfm.eval()
print(f"  hidden={xfm.config.hidden_size} layers={xfm.config.num_layers} "
      f"text_in={xfm.config.text_in_dim} heads={xfm.config.num_attention_heads}", flush=True)

# ── Synthetic inputs (MUST match ErnieImageDiffTests.cs) ──
torch.manual_seed(42)
B = 1; H = W = 32; T = 64; TEXT_LEN = 50
HIDDEN_TEXT = xfm.config.text_in_dim
latent = torch.randn(B, 128, H, W, dtype=torch.float32)
text_embeds = torch.randn(B, T, HIDDEN_TEXT, dtype=torch.float32)
text_lens = torch.tensor([TEXT_LEN], dtype=torch.long)
timestep = torch.tensor([500.0], dtype=torch.float32)
save(latent, f"{OUT_DIR}/inputs/latent.bin")
save(text_embeds, f"{OUT_DIR}/inputs/text_embeds.bin")
save(text_lens.to(torch.int32), f"{OUT_DIR}/inputs/text_lens.bin")
save(timestep, f"{OUT_DIR}/inputs/timestep.bin")
print(f"Saved inputs latent{tuple(latent.shape)} text{tuple(text_embeds.shape)} t={float(timestep)}", flush=True)

cap = {}
def hook(name, transpose_sb=False):
    def fn(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        out = out.detach().clone()
        cap[name] = out
    return fn

# Component hooks aligned to C# dump names.
xfm.x_embedder.register_forward_hook(hook("patch_embed"))            # [B, N_img, hidden]
if xfm.text_proj is not None:
    xfm.text_proj.register_forward_hook(hook("text_proj"))           # [B, Tmax, hidden]
xfm.time_embedding.register_forward_hook(hook("time_embedding"))     # [B, hidden]
xfm.adaLN_modulation.register_forward_hook(hook("adaLN_modulation")) # [B, 6*hidden]
xfm.pos_embed.register_forward_hook(hook("pos_embed"))               # [B, S, 1, head_dim] ANGLES
for i in range(xfm.config.num_layers):
    xfm.layers[i].register_forward_hook(hook(f"layers.{i}"))         # [S, B, hidden]
xfm.final_norm.register_forward_hook(hook("final_norm"))             # [S, B, hidden]
xfm.final_linear.register_forward_hook(hook("final_linear_full"))    # [S, B, out_dim]

print("Running forward ...", flush=True)
with torch.no_grad():
    out = xfm(hidden_states=latent, timestep=timestep,
              text_bth=text_embeds, text_lens=text_lens).sample
save(out, f"{OUT_DIR}/output_velocity.bin")
print(f"Output shape={tuple(out.shape)} mean={float(out.mean()):.4e} std={float(out.std()):.4e}", flush=True)

N_img = (H // xfm.config.patch_size) * (W // xfm.config.patch_size)
hidden = xfm.config.hidden_size
index = []

def dump(name, t):
    fname = f"{name.replace('.', '_')}.bin"
    save(t, f"{OUT_DIR}/layers/{fname}")
    index.append({"name": name, "file": f"layers/{fname}", **stats(name, t)})

# patch_embed [B,N,hidden] matches C#
dump("patch_embed", cap["patch_embed"])
if "text_proj" in cap: dump("text_proj", cap["text_proj"])
dump("time_embedding", cap["time_embedding"])
# modulation: chunk(6) -> shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp
mod = cap["adaLN_modulation"].chunk(6, dim=-1)
for nm, m in zip(["modulation_shift_msa","modulation_scale_msa","modulation_gate_msa",
                  "modulation_shift_mlp","modulation_scale_mlp","modulation_gate_mlp"], mod):
    dump(nm, m.contiguous())
# rope: pos_embed outputs ANGLES [B,S,1,headdim]; produce cos/sin block layout like C# rope_freqs [B,S,2*headdim]
ang = cap["pos_embed"]            # [B, S, 1, head_dim]
ang = ang.squeeze(2)             # [B, S, head_dim]
cos_blk = torch.cos(ang); sin_blk = torch.sin(ang)
rope_freqs = torch.cat([cos_blk, sin_blk], dim=-1)  # [B, S, 2*head_dim]
dump("rope_freqs", rope_freqs.contiguous())
dump("rope_angles", ang.contiguous())
# layers: block out [S,B,H] -> [B,S,H]
for i in range(xfm.config.num_layers):
    bo = cap[f"layers.{i}"]
    if bo.dim() == 3:  # [S,B,H]
        bo = bo.permute(1, 0, 2).contiguous()
    dump(f"layers.{i}", bo)
# final_linear: full [S,B,out_dim] -> image front [:N_img] -> [B,N_img,out_dim]
fl = cap["final_linear_full"]     # [S, B, out_dim]
fl_img = fl[:N_img].permute(1, 0, 2).contiguous()
dump("final_linear", fl_img)
fn = cap["final_norm"]
if fn.dim() == 3: fn = fn.permute(1, 0, 2).contiguous()
dump("final_norm", fn)
index.append({"name": "output_velocity", "file": "output_velocity.bin", **stats("output_velocity", out)})
with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump(index, f, indent=2)
print(f"Done. {len(index)} entries -> {OUT_DIR}/", flush=True)
