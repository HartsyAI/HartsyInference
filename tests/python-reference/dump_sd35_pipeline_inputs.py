"""
Dumps the pipeline inputs for the SD3.5 transformer's first denoise step from
diffusers. Specifically: CLIP-L hidden + pooled, CLIP-G hidden + pooled, T5
hidden, AND the final concatenated context + pooled tensors that the
transformer actually consumes.

Used to diff the C# Sd3Pipeline.EncodePrompt against the diffusers reference,
since the transformer math itself is already verified (see Sd35DiffTests).

Output:
  tests/python-reference/sd35_reference_tensors/pipeline_inputs/
    clip_l_hidden_penultimate.bin     [1, 77, 768]
    clip_l_pooled.bin                  [1, 768]
    clip_g_hidden_penultimate.bin      [1, 77, 1280]
    clip_g_pooled.bin                  [1, 1280]
    t5_hidden.bin                      [1, 77, 4096]
    final_context_pre.bin              [1, 154, 4096]   (the input to the transformer's context_embedder)
    final_pooled.bin                   [1, 2048]        (the pooled CLIP_L+G concat)
    initial_latent.bin                 [1, 16, 64, 64]  (random noise * init_sigma)
    timesteps.bin                      [28]             F32
    sigmas.bin                         [29]             F32
    info.json
"""
import os
import json
import torch
from safetensors.torch import load_file
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextConfig, T5EncoderModel, T5TokenizerFast, T5Config
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

CKPT = "/home/kalebbroo/Desktop/Projects/SharpInference/Models/Stable-Diffusion/SD3/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"
OUT_DIR = "tests/python-reference/sd35_reference_tensors/pipeline_inputs"
os.makedirs(OUT_DIR, exist_ok=True)


def save(t, path):
    t.float().detach().cpu().contiguous().numpy().tofile(path)


def stats(t):
    f = t.float().flatten()
    return dict(shape=list(t.shape), mean=float(f.mean()), std=float(f.std()), min=float(f.min()), max=float(f.max()))


# ── 1. Load raw safetensors and pick out CLIP-L, CLIP-G keys ──
print("Loading raw safetensors ...")
raw = load_file(CKPT, device="cpu")
print(f"  {len(raw)} keys")

# Strip "text_encoders.clip_l.transformer." prefix
clip_l_state = {}
for k, v in raw.items():
    if k.startswith("text_encoders.clip_l.transformer."):
        nk = k[len("text_encoders.clip_l.transformer."):]
        if nk.endswith("position_ids"): continue
        clip_l_state[nk] = v
print(f"  CLIP-L keys: {len(clip_l_state)}")

# Strip "text_encoders.clip_g.transformer." prefix and convert OpenCLIP→HF format if needed
clip_g_raw = {}
for k, v in raw.items():
    if k.startswith("text_encoders.clip_g.transformer."):
        nk = k[len("text_encoders.clip_g.transformer."):]
        if nk.endswith("position_ids"): continue
        clip_g_raw[nk] = v
print(f"  CLIP-G keys: {len(clip_g_raw)}")

# CLIP-L config (HF openai/clip-vit-large-patch14)
clip_l_config = CLIPTextConfig(
    vocab_size=49408, hidden_size=768, intermediate_size=3072,
    num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77,
    hidden_act="quick_gelu", layer_norm_eps=1e-5,
    projection_dim=768,
)
print(f"\nLoading CLIP-L (HF CLIPTextModelWithProjection) ...")
clip_l = CLIPTextModelWithProjection(clip_l_config).eval()
missing_l, unex_l = clip_l.load_state_dict(clip_l_state, strict=False)
print(f"  CLIP-L missing: {len(missing_l)}, unexpected: {len(unex_l)}")
if missing_l: print(f"    missing sample: {missing_l[:3]}")
if unex_l: print(f"    unexpected sample: {unex_l[:3]}")

# CLIP-G config
clip_g_config = CLIPTextConfig(
    vocab_size=49408, hidden_size=1280, intermediate_size=5120,
    num_hidden_layers=32, num_attention_heads=20, max_position_embeddings=77,
    hidden_act="gelu", layer_norm_eps=1e-5,
    projection_dim=1280,
)
print(f"\nLoading CLIP-G (HF CLIPTextModelWithProjection) ...")
clip_g = CLIPTextModelWithProjection(clip_g_config).eval()
missing_g, unex_g = clip_g.load_state_dict(clip_g_raw, strict=False)
print(f"  CLIP-G missing: {len(missing_g)}, unexpected: {len(unex_g)}")
if missing_g: print(f"    missing sample: {missing_g[:3]}")
if unex_g: print(f"    unexpected sample: {unex_g[:3]}")

# ── 2. Build deterministic token IDs (the same way C# would tokenize "A photograph of an astronaut riding a horse") ──
prompt = "A photograph of an astronaut riding a horse"
tok_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
inp_l = tok_l(prompt, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
print(f"\nCLIP-L tokenized: {list(inp_l['input_ids'][0].tolist()[:12])} ... len {inp_l['input_ids'].shape[1]}")
eos_positions = (inp_l['input_ids'][0] == tok_l.eos_token_id).nonzero(as_tuple=True)[0]
clip_l_eos = int(eos_positions[0].item())
print(f"  EOS pos (first occurrence): {clip_l_eos}")

# Save token ids as int32 for reproducibility
clip_l_tok_ids = inp_l["input_ids"].to(torch.int32)
clip_l_tok_ids.cpu().numpy().tofile(os.path.join(OUT_DIR, "clip_l_token_ids.bin"))

# CLIP-G tokenizer is OpenCLIP — but the tokenizer is the same as CLIP-L (same vocab). Just use the same.
clip_g_tok_ids = clip_l_tok_ids  # same tokens for both

# ── 3. Encode through CLIP-L and CLIP-G ──
print("\nEncoding with CLIP-L ...")
with torch.no_grad():
    out_l = clip_l(input_ids=inp_l["input_ids"], output_hidden_states=True)
    clip_l_hidden_penult = out_l.hidden_states[-2]   # diffusers convention (penultimate)
    clip_l_pooled = out_l.text_embeds                # text_projection(eos token of last_hidden_state)

print(f"  hidden_penult: {tuple(clip_l_hidden_penult.shape)} mean={clip_l_hidden_penult.mean():.4f} std={clip_l_hidden_penult.std():.4f}")
print(f"  pooled:        {tuple(clip_l_pooled.shape)} mean={clip_l_pooled.mean():.4f} std={clip_l_pooled.std():.4f}")
print(f"  pooled first 8: {clip_l_pooled[0, :8].tolist()}")

print("\nEncoding with CLIP-G ...")
with torch.no_grad():
    out_g = clip_g(input_ids=inp_l["input_ids"], output_hidden_states=True)
    clip_g_hidden_penult = out_g.hidden_states[-2]
    clip_g_pooled = out_g.text_embeds

print(f"  hidden_penult: {tuple(clip_g_hidden_penult.shape)} mean={clip_g_hidden_penult.mean():.4f} std={clip_g_hidden_penult.std():.4f}")
print(f"  pooled:        {tuple(clip_g_pooled.shape)} mean={clip_g_pooled.mean():.4f} std={clip_g_pooled.std():.4f}")
print(f"  pooled first 8: {clip_g_pooled[0, :8].tolist()}")

# ── 4. Save ──
save(clip_l_hidden_penult, os.path.join(OUT_DIR, "clip_l_hidden_penultimate.bin"))
save(clip_l_pooled, os.path.join(OUT_DIR, "clip_l_pooled.bin"))
save(clip_g_hidden_penult, os.path.join(OUT_DIR, "clip_g_hidden_penultimate.bin"))
save(clip_g_pooled, os.path.join(OUT_DIR, "clip_g_pooled.bin"))

# ── 5. Pooled concat = [clip_l_pooled, clip_g_pooled] → [1, 2048] ──
final_pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)
save(final_pooled, os.path.join(OUT_DIR, "final_pooled.bin"))
print(f"\nfinal_pooled: {tuple(final_pooled.shape)} mean={final_pooled.mean():.4f}")

# ── 6. Skip T5 for now (FP8, complex). Pass zeros to mimic the C# noT5 path. ──
t5_hidden = torch.zeros(1, 77, 4096, dtype=torch.float32)
save(t5_hidden, os.path.join(OUT_DIR, "t5_hidden.bin"))

# ── 7. Build context: concat([clip_l, clip_g], dim=-1) → pad to 4096 → concat with t5 along seq ──
clip_combined = torch.cat([clip_l_hidden_penult, clip_g_hidden_penult], dim=-1)  # [1, 77, 2048]
clip_padded = torch.nn.functional.pad(clip_combined, (0, 4096 - 2048))            # [1, 77, 4096]
final_context_pre = torch.cat([clip_padded, t5_hidden], dim=-2)                    # [1, 154, 4096]
save(final_context_pre, os.path.join(OUT_DIR, "final_context_pre.bin"))
print(f"final_context_pre: {tuple(final_context_pre.shape)} mean={final_context_pre.mean():.4f} std={final_context_pre.std():.4f}")

# ── 8. Initial noise + scheduler ──
torch.manual_seed(42)
latent_shape = (1, 16, 64, 64)
init_noise = torch.randn(latent_shape, dtype=torch.float32)
save(init_noise, os.path.join(OUT_DIR, "init_noise.bin"))
print(f"\ninit_noise: {tuple(init_noise.shape)} mean={init_noise.mean():.4f} std={init_noise.std():.4f}")

scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
scheduler.set_timesteps(28)
init_sigma = scheduler.init_noise_sigma
sigmas = scheduler.sigmas.float()
timesteps = scheduler.timesteps.float()
print(f"init_noise_sigma: {init_sigma}")
print(f"timesteps[:8]: {timesteps[:8].tolist()}")
print(f"sigmas[:8]: {sigmas[:8].tolist()}")

timesteps.cpu().numpy().tofile(os.path.join(OUT_DIR, "timesteps.bin"))
sigmas.cpu().numpy().tofile(os.path.join(OUT_DIR, "sigmas.bin"))

# ── 9. info.json ──
info = {
    "prompt": prompt,
    "seed": 42,
    "init_noise_sigma": float(init_sigma),
    "timesteps_count": len(timesteps),
    "clip_l_pooled_first_8": clip_l_pooled[0, :8].tolist(),
    "clip_g_pooled_first_8": clip_g_pooled[0, :8].tolist(),
    "final_pooled_stats": stats(final_pooled),
    "final_context_pre_stats": stats(final_context_pre),
    "init_noise_stats": stats(init_noise),
    "clip_l_eos_pos": clip_l_eos,
}
with open(os.path.join(OUT_DIR, "info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"\nDone. Saved to {OUT_DIR}/")
