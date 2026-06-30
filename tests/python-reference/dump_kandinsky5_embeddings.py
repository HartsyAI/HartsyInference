#!/usr/bin/env python
"""Generate pre-computed Kandinsky 5.0 T2I-Lite text embeddings for the C# end-to-end test.

Reproduces the embedding extraction in diffusers' Kandinsky5T2IPipeline
(pipeline_kandinsky_t2i.py: _encode_prompt_qwen / _encode_prompt_clip):

  Qwen2.5-VL-7B (hidden=3584):
    full_text   = prompt_template.format(prompt)
    inputs      = processor(text=full_text, images=None, videos=None, padding=True, ...)
    hidden      = text_encoder(input_ids, output_hidden_states=True)["hidden_states"][-1]
    embeds      = hidden[:, prompt_template_encode_start_idx:]      # drop 41-token system preamble
    -> saved as raw F32 [seq_len, 3584]  (batch dim stripped)

  CLIP-L (openai/clip-vit-large-patch14, pooled=768):
    inputs      = tokenizer(prompt, max_length=77, truncation, padding="max_length", add_special_tokens)
    pooled      = text_encoder_2(**inputs)["pooler_output"]
    -> saved as raw F32 [768]

The Qwen2.5-VL weights on disk are a ComfyUI fp8_scaled checkpoint
(Models/text_encoders/qwen2_5_vl_7b.safetensors): big matmul weights are
float8_e4m3 with a per-tensor `*.scale_weight`; dequant = w.float() * scale_weight.
Flat keys (model.layers.* / model.embed_tokens / model.norm) are remapped to the
transformers-5.x Qwen2_5_VL layout (model.language_model.*). Visual tower is skipped
(text-only forward never touches it).

No header is written; the C# loader (Kandinsky5GenerationTests.LoadF32Tensor /
LoadPooled) reads raw little-endian F32 and infers seq_len from file size.
"""
import os, struct, json, sys
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(os.cpu_count() or 8)

REPO = "/home/hartsy/Desktop/HartsyInference"
QWEN_FP8 = os.path.join(REPO, "Models/text_encoders/qwen2_5_vl_7b.safetensors")
OUT_DIR = os.path.join(REPO, "Models/Stable-Diffusion/Kandinsky5/TestEmbeddings")
QWEN_HF = "Qwen/Qwen2.5-VL-7B-Instruct"     # ungated; config + tokenizer/processor only
CLIP_HF = "openai/clip-vit-large-patch14"   # ungated

# Matches pipeline_kandinsky_t2i.py exactly.
PROMPT_TEMPLATE = ("<|im_start|>system\nYou are a promt engineer. Describe the image by detailing "
                   "the color, shape, size, texture, quantity, text, spatial relationships of the "
                   "objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>")
START_IDX = 41
MAX_SEQ = 512

PROMPT = ("A majestic snow leopard perched on a rocky cliff at golden hour, ultra-detailed fur, "
          "dramatic snow-capped mountains in the background, photorealistic, sharp focus, "
          "cinematic lighting")
NEG_PROMPT = ""   # diffusers Kandinsky5 default negative_prompt


def save_raw_f32(path, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    arr.tofile(path)
    n = os.path.getsize(path) // 4
    print(f"  wrote {path}  ({os.path.getsize(path)} bytes = {n} floats, shape {arr.shape})")


def remap_key(k):
    if k.startswith("model.layers."):
        return "model.language_model.layers." + k[len("model.layers."):]
    if k == "model.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if k == "model.norm.weight":
        return "model.language_model.norm.weight"
    if k == "lm_head.weight":
        return "lm_head.weight"
    if k.startswith("visual."):
        return None  # skip vision tower (text-only forward)
    return None


def load_qwen_text_state_dict(path):
    """Dequantize fp8_scaled comfy checkpoint -> bf16 state_dict for transformers Qwen2_5_VL text model."""
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        scale_keys = {k for k in keys if k.endswith(".scale_weight") or k.endswith(".scale_input")}
        for k in keys:
            if k in scale_keys:
                continue
            tk = remap_key(k)
            if tk is None:
                continue
            w = f.get_tensor(k)
            if w.dtype == torch.float8_e4m3fn:
                sw = k.rsplit(".weight", 1)[0] + ".scale_weight"
                scale = f.get_tensor(sw).float() if sw in scale_keys else 1.0
                w = w.float() * scale
            sd[tk] = w.to(torch.bfloat16)
    print(f"  built text state_dict: {len(sd)} tensors")
    return sd


def build_qwen():
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(QWEN_HF)
    torch.set_default_dtype(torch.bfloat16)              # random-init in bf16 (~15GB, not 28GB)
    model = Qwen2_5_VLForConditionalGeneration(cfg)
    torch.set_default_dtype(torch.float32)
    sd = load_qwen_text_state_dict(QWEN_FP8)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Expect ONLY visual.* in missing; nothing unexpected.
    miss_nonvisual = [m for m in missing if "visual" not in m]
    print(f"  load: missing(non-visual)={len(miss_nonvisual)} unexpected={len(unexpected)}")
    if miss_nonvisual[:5]:
        print("   missing sample:", miss_nonvisual[:5])
    if unexpected[:5]:
        print("   unexpected sample:", unexpected[:5])
    model.eval()
    return model


@torch.no_grad()
def encode_qwen(model, tok, prompt):
    # Text-only: AutoTokenizer yields identical input_ids to Qwen2VLProcessor(text=...).
    full = PROMPT_TEMPLATE.format(prompt)
    inputs = tok([full], max_length=START_IDX + MAX_SEQ, truncation=True,
                 return_tensors="pt", padding=True)
    out = model(input_ids=inputs["input_ids"], return_dict=True, output_hidden_states=True)
    embeds = out["hidden_states"][-1][:, START_IDX:]   # [1, seq, 3584]
    return embeds[0].float().cpu().numpy()             # [seq, 3584]


@torch.no_grad()
def encode_clip(model, tok, prompt):
    inputs = tok(prompt, max_length=77, truncation=True, add_special_tokens=True,
                 padding="max_length", return_tensors="pt")
    pooled = model(**inputs)["pooler_output"]          # [1, 768]
    return pooled[0].float().cpu().numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Prompt: {PROMPT!r}")
    print(f"Neg   : {NEG_PROMPT!r}")

    # ---- Qwen2.5-VL ----
    from transformers import AutoTokenizer
    print("[Qwen] loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(QWEN_HF)
    print("[Qwen] building model + loading dequantized fp8 weights ...")
    qwen = build_qwen()
    print("[Qwen] encoding prompt ...")
    q_pos = encode_qwen(qwen, tok, PROMPT)
    print(f"  qwen prompt embeds shape {q_pos.shape}")
    q_neg = encode_qwen(qwen, tok, NEG_PROMPT)
    print(f"  qwen neg embeds shape {q_neg.shape}")
    del qwen
    import gc; gc.collect()

    # ---- CLIP-L ----
    from transformers import CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
    print("[CLIP] loading openai/clip-vit-large-patch14 ...")
    clip_tok = CLIPTokenizer.from_pretrained(CLIP_HF)
    clip = CLIPTextModel.from_pretrained(CLIP_HF).eval()   # pooler_output = [1,768]
    c_pos = encode_clip(clip, clip_tok, PROMPT)
    c_neg = encode_clip(clip, clip_tok, NEG_PROMPT)
    print(f"  clip pooled shape {c_pos.shape}")

    # ---- write ----
    print("Writing .bin files:")
    save_raw_f32(os.path.join(OUT_DIR, "prompt_qwen.bin"), q_pos)
    save_raw_f32(os.path.join(OUT_DIR, "prompt_clip.bin"), c_pos)
    save_raw_f32(os.path.join(OUT_DIR, "neg_prompt_qwen.bin"), q_neg)
    save_raw_f32(os.path.join(OUT_DIR, "neg_prompt_clip.bin"), c_neg)

    # ---- verify by re-reading ----
    print("Verify re-read:")
    for name, dim in [("prompt_qwen.bin", 3584), ("neg_prompt_qwen.bin", 3584),
                      ("prompt_clip.bin", 768), ("neg_prompt_clip.bin", 768)]:
        p = os.path.join(OUT_DIR, name)
        data = np.fromfile(p, dtype="<f4")
        assert data.size % dim == 0, f"{name}: {data.size} not multiple of {dim}"
        seq = data.size // dim
        print(f"  {name}: {data.size} floats -> [{seq}, {dim}]  (finite={np.isfinite(data).all()})")


if __name__ == "__main__":
    main()
