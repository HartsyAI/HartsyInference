"""
Encodes a text prompt for Anima generation: produces BOTH Qwen-3 hidden states (cross-attn K/V source)
AND T5-tokenized input IDs (the LlmAdapter main stream lookup keys).

Anima's text encoder stack is dual:
  - Qwen3-0.6B-Base hidden states → AnimaLlmAdapter context (cross-attn K/V)
  - T5-XXL SentencePiece tokenization → AnimaLlmAdapter.embed[ids] (main stream)
Both must be derived from the same prompt; the LlmAdapter fuses them.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/encode_anima_prompt.py [prompt] [neg_prompt]

Output (under Models/Stable-Diffusion/Anima/TestEmbeddings/):
  prompt.bin              — positive Qwen3 hidden states (F32, raw [seq_len, 1024])
  neg_prompt.bin          — negative Qwen3 hidden states (F32, raw [seq_len, 1024])
  prompt_t5_ids.bin       — positive T5 token IDs (int64, raw [seq_len])
  neg_prompt_t5_ids.bin   — negative T5 token IDs (int64, raw [seq_len])
  prompt.txt              — the prompts + token decoding (for reproducibility)
"""
import os
import sys
import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3 import Qwen3Config, Qwen3Model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Local checkpoint name; falls back to the historical SwarmUI name if the symlink differs.
_W1 = os.path.join(REPO_ROOT, "Models/text_encoders/qwen_3_06b_base.safetensors")
_W2 = os.path.join(REPO_ROOT, "Models/text_encoders/qwen_3_600m.safetensors")
WEIGHTS = _W1 if os.path.exists(_W1) else _W2
TOKENIZER_DIR = os.path.join(REPO_ROOT, "Models/Tokenizers/Qwen3")
T5_SPM = os.path.join(REPO_ROOT, "Models/Tokenizers/T5/t5_xxl_spiece.model")
OUT_DIR = os.path.join(REPO_ROOT, "Models/Stable-Diffusion/Anima/TestEmbeddings")

# Probed Qwen3-0.6B Base config from the local safetensors:
#   embed_tokens: [151936, 1024] → vocab=151936, hidden=1024
#   q_proj:       [2048, 1024]   → num_heads = 2048/128 = 16
#   k_proj:       [1024, 1024]   → num_kv_heads = 1024/128 = 8
#   mlp gate:     [3072, 1024]   → intermediate=3072
#   28 layers, RMSNorm eps=1e-6, RoPE theta=1000000 (Qwen3 default)
QWEN3_06B_CONFIG = Qwen3Config(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=False,
    rope_theta=1000000.0,
    sliding_window=None,
    attention_dropout=0.0,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Anime girl with blue hair in a garden"
    neg_prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"Prompt:     {prompt!r}")
    print(f"Neg prompt: {neg_prompt!r}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load tokenizer ──
    print(f"\nLoading tokenizer from {TOKENIZER_DIR}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def encode_ids(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids
        # Qwen3 has no BOS; an empty string yields a zero-length sequence which crashes the
        # attention-mask builder and gives the DiT cross-attn no K/V. The CFG unconditional
        # branch needs at least one token, so fall back to a single <|endoftext|> (151643).
        if ids.shape[1] == 0:
            eos = tok.eos_token_id if tok.eos_token_id is not None else 151643
            ids = torch.tensor([[eos]], dtype=torch.long)
        return ids

    pos_ids = encode_ids(prompt)
    neg_ids = encode_ids(neg_prompt)
    print(f"  pos ids: {pos_ids.shape} = {pos_ids.tolist()}")
    print(f"  neg ids: {neg_ids.shape} = {neg_ids.tolist()}")

    # ── 2. Build the Qwen3 model + load weights ──
    print(f"\nBuilding Qwen3-0.6B Base model (F32 inference)...")
    torch.manual_seed(0)
    model = Qwen3Model(QWEN3_06B_CONFIG)
    model = model.to(torch.float32).eval()
    print(f"  Constructed: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    print(f"Loading weights: {WEIGHTS}")
    state = {}
    with safe_open(WEIGHTS, framework="pt", device="cpu") as f:
        for k in f.keys():
            # Strip "model." prefix — Qwen3Model expects keys without the wrapper.
            new_k = k[len("model."):] if k.startswith("model.") else k
            state[new_k] = f.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing[:5]:
        print(f"    missing[:5] = {missing[:5]}")
    if unexpected[:5]:
        print(f"    unexpected[:5] = {unexpected[:5]}")

    # ── 3. Encode prompts ──
    print(f"\nRunning Qwen3 forward...")
    with torch.no_grad():
        pos_out = model(pos_ids, output_hidden_states=False).last_hidden_state  # [1, S_pos, 1024]
        neg_out = model(neg_ids, output_hidden_states=False).last_hidden_state  # [1, S_neg, 1024]
    print(f"  pos hidden: shape={tuple(pos_out.shape)}  mean={pos_out.mean():+.4f} std={pos_out.std():.4f}")
    print(f"  neg hidden: shape={tuple(neg_out.shape)}  mean={neg_out.mean():+.4f} std={neg_out.std():.4f}")

    # ── 4. T5-tokenize both prompts (same SentencePiece tokenizer used during Anima training) ──
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(T5_SPM)
    pos_t5_ids = sp.encode(prompt, add_eos=True)
    neg_t5_ids = sp.encode(neg_prompt, add_eos=True)
    print(f"\nT5 ids (pos): {pos_t5_ids}  pieces={[sp.id_to_piece(i) for i in pos_t5_ids]}")
    print(f"T5 ids (neg): {neg_t5_ids}  pieces={[sp.id_to_piece(i) for i in neg_t5_ids]}")

    # ── 5. Save raw F32 [S, 1024] hidden states + raw int64 [S] T5 ids ──
    pos_arr = pos_out.squeeze(0).contiguous().to(torch.float32).numpy()
    neg_arr = neg_out.squeeze(0).contiguous().to(torch.float32).numpy()
    pos_arr.tofile(os.path.join(OUT_DIR, "prompt.bin"))
    neg_arr.tofile(os.path.join(OUT_DIR, "neg_prompt.bin"))
    import numpy as np
    np.asarray(pos_t5_ids, dtype=np.int64).tofile(os.path.join(OUT_DIR, "prompt_t5_ids.bin"))
    np.asarray(neg_t5_ids, dtype=np.int64).tofile(os.path.join(OUT_DIR, "neg_prompt_t5_ids.bin"))
    with open(os.path.join(OUT_DIR, "prompt.txt"), "w") as f:
        f.write(f"prompt = {prompt}\nneg_prompt = {neg_prompt}\n")
        f.write(f"pos_seq_len = {pos_arr.shape[0]}\n")
        f.write(f"neg_seq_len = {neg_arr.shape[0]}\n")
        f.write(f"hidden_size = {pos_arr.shape[1]}\n")
        f.write(f"pos_t5_ids = {pos_t5_ids}\n")
        f.write(f"neg_t5_ids = {neg_t5_ids}\n")

    print(f"\nSaved to {OUT_DIR}")
    print(f"  prompt.bin             ({pos_arr.nbytes} bytes, qwen3_seq_len={pos_arr.shape[0]})")
    print(f"  neg_prompt.bin         ({neg_arr.nbytes} bytes, qwen3_seq_len={neg_arr.shape[0]})")
    print(f"  prompt_t5_ids.bin      (int64 x {len(pos_t5_ids)})")
    print(f"  neg_prompt_t5_ids.bin  (int64 x {len(neg_t5_ids)})")
    print(f"  prompt.txt")


if __name__ == "__main__":
    main()
