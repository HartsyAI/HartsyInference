"""
Encodes a text prompt for Lumina-Image-2.0 generation: produces Gemma 2 2B caption embeddings in the
exact format the diffusers Lumina2 pipeline feeds the transformer via `prompt_embeds`, and that the
HartsyInference C# e2e test loads as raw F32 [seq_len, cap_feat_dim=2304].

Parity points mirrored from diffusers/pipelines/lumina2/pipeline_lumina2.py:
  - System-prompt template:
      "You are an assistant designed to generate superior images with the superior degree of
       image-text alignment based on textual prompts or user prompts. <Prompt Start> " + prompt
  - output_hidden_states=True, then take hidden_states[-2]  (SECOND-to-last layer, not last_hidden_state)
  - hidden dim = 2304 (Gemma 2 2B hidden_size = Lumina2Config.CapFeatDim)

Padding note: diffusers pads to max_length=256 and passes an attention mask. The C# pipeline has NO mask,
so we emit ONLY the real (unpadded) tokens. Gemma 2 is causal with right padding, so the real-token rows
of hidden_states[-2] are bit-identical whether or not trailing pad tokens are appended — i.e. the unpadded
output equals the valid-token slice of the padded output. This is the correct, mask-free input for C#.

Text encoder + tokenizer are the exact Gemma 2 2B weights bundled in the (ungated, Apache-2.0)
Alpha-VLLM/Lumina-Image-2.0 repo under text_encoder/ and tokenizer/.

Usage:
  /home/hartsy/hfvenv/bin/python tests/python-reference/encode_lumina2_prompt.py [prompt] [neg_prompt]

Output (under Models/Stable-Diffusion/Lumina2/TestEmbeddings/):
  prompt.bin       — positive Gemma2 hidden_states[-2] (F32, raw [seq_len, 2304])
  neg_prompt.bin   — negative (unconditional) Gemma2 hidden_states[-2] (F32, raw [seq_len, 2304])
  prompt.txt       — prompts + token bookkeeping (reproducibility)
"""
import os
import sys
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "Models/Stable-Diffusion/Lumina2/TestEmbeddings")
HF_REPO = "Alpha-VLLM/Lumina-Image-2.0"
SYSTEM_PROMPT = ("You are an assistant designed to generate superior images with the superior "
                 "degree of image-text alignment based on textual prompts or user prompts.")
CAP_FEAT_DIM = 2304


def encode(model, tok, text, device):
    full = SYSTEM_PROMPT + " <Prompt Start> " + text
    enc = tok(full, padding="longest", return_tensors="pt")  # no max_length pad: real tokens only
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)
    with torch.no_grad():
        out = model(ids, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[-2]  # [1, S, 2304] — diffusers takes second-to-last layer
    return h.squeeze(0).contiguous().to(torch.float32).cpu().numpy(), ids.shape[1]


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A serene mountain lake at sunset, photorealistic, golden light"
    neg_prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"Prompt:     {prompt!r}")
    print(f"Neg prompt: {neg_prompt!r}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nResolving Gemma 2 2B text encoder from {HF_REPO} ...")
    snap = snapshot_download(repo_id=HF_REPO, allow_patterns=["text_encoder/*", "tokenizer/*"])
    te_dir = os.path.join(snap, "text_encoder")
    tk_dir = os.path.join(snap, "tokenizer")
    print(f"  text_encoder: {te_dir}")

    tok = AutoTokenizer.from_pretrained(tk_dir)
    print("Loading Gemma 2 2B (F32, CPU)...")
    model = AutoModel.from_pretrained(te_dir, torch_dtype=torch.float32).eval()
    device = torch.device("cpu")
    model = model.to(device)
    print(f"  hidden_size = {model.config.hidden_size} (expect {CAP_FEAT_DIM})")
    assert model.config.hidden_size == CAP_FEAT_DIM, "hidden size mismatch vs Lumina2Config.CapFeatDim"

    pos_arr, pos_n = encode(model, tok, prompt, device)
    neg_arr, neg_n = encode(model, tok, neg_prompt, device)
    print(f"\n  pos hidden[-2]: shape={pos_arr.shape}  mean={pos_arr.mean():+.4f} std={pos_arr.std():.4f}  (tokens={pos_n})")
    print(f"  neg hidden[-2]: shape={neg_arr.shape}  mean={neg_arr.mean():+.4f} std={neg_arr.std():.4f}  (tokens={neg_n})")

    pos_arr.tofile(os.path.join(OUT_DIR, "prompt.bin"))
    neg_arr.tofile(os.path.join(OUT_DIR, "neg_prompt.bin"))
    with open(os.path.join(OUT_DIR, "prompt.txt"), "w") as f:
        f.write(f"system_prompt = {SYSTEM_PROMPT}\n")
        f.write(f"prompt = {prompt}\nneg_prompt = {neg_prompt}\n")
        f.write(f"template = <system> <Prompt Start> <prompt>\n")
        f.write(f"layer = hidden_states[-2]\n")
        f.write(f"pos_seq_len = {pos_arr.shape[0]}\nneg_seq_len = {neg_arr.shape[0]}\n")
        f.write(f"cap_feat_dim = {pos_arr.shape[1]}\n")

    print(f"\nSaved to {OUT_DIR}")
    print(f"  prompt.bin      ({pos_arr.nbytes} bytes, seq_len={pos_arr.shape[0]}, dim={pos_arr.shape[1]})")
    print(f"  neg_prompt.bin  ({neg_arr.nbytes} bytes, seq_len={neg_arr.shape[0]}, dim={neg_arr.shape[1]})")


if __name__ == "__main__":
    main()
