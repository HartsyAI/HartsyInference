"""
Dumps the canonical Anima `net.llm_adapter` forward pass for layer-by-layer C# comparison.

The canonical source is `tdrussell/diffusion-pipe/models/llm_adapter.py`. The LlmAdapter is a
6-block transformer with TWO input streams:
  - `x`       = self.embed(t5_token_ids) — main stream, [B, S_target, 1024]
  - `context` = qwen3_hidden_states       — cross-attn K/V source, [B, S_source, 1024]

Each block does self-attn (Q,K,V from x with RoPE) → cross-attn (Q from x, K,V from context, RoPE
applied with separate target/source position embeddings) → MLP (GELU). Final: RMSNorm(out_proj(x)).

This dump records:
  - Inputs: t5_input_ids [1, S_target] (int64), qwen3_hidden [1, S_source, 1024] (F32)
  - After embed[ids]:           x_post_embed
  - After each block (0..5):    block_N
  - After out_proj:             post_out_proj
  - After final norm:           final_output

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_anima_llm_adapter.py [prompt]

Output: tests/python-reference/anima_llm_adapter_reference/
  inputs/t5_input_ids.bin                [1, S_target] int64
  inputs/qwen3_hidden.bin                [1, S_source, 1024] F32
  inputs/prompt.txt                       the prompt + token decoding
  layers/x_post_embed.bin                F32
  layers/block_0.bin … block_5.bin       F32
  layers/post_out_proj.bin               F32
  final_output.bin                       F32
  index.json                              stats per dump
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANIMA_PATH = "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/Stable-Diffusion/Anima/anima-preview3-base.safetensors"
QWEN3_PATH = "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/text_encoders/qwen_3_600m.safetensors"
QWEN3_TOK = "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/Tokenizers/Qwen3"
T5_SPM = "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/Tokenizers/T5/t5_xxl_spiece.model"
OUT_DIR = os.path.join(REPO_ROOT, "tests/python-reference/anima_llm_adapter_reference")

os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)


# ── Canonical LlmAdapter (copied verbatim from diffusion-pipe/models/llm_adapter.py) ──
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)
    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)
        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb(key_states, cos, sin)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16):
        super().__init__()
        self.norm_self_attn = RMSNorm(model_dim)
        self.self_attn_layer = Attention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = RMSNorm(model_dim)
        self.cross_attn = Attention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = RMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )
    def forward(self, x, context, position_embeddings, position_embeddings_context):
        normed = self.norm_self_attn(x)
        attn_out = self.self_attn_layer(normed, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings)
        x = x + attn_out
        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(normed, context=context, position_embeddings=position_embeddings, position_embeddings_context=position_embeddings_context)
        x = x + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16, vocab_size=32128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, target_dim)
        self.in_proj = nn.Identity()  # model_dim == target_dim
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([TransformerBlock(source_dim, model_dim, num_heads) for _ in range(num_layers)])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = RMSNorm(target_dim)

    def forward(self, source_hidden_states, target_input_ids, dump_callback=None):
        x = self.in_proj(self.embed(target_input_ids))
        if dump_callback: dump_callback("x_post_embed", x)
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for i, block in enumerate(self.blocks):
            x = block(x, context, position_embeddings, position_embeddings_context)
            if dump_callback: dump_callback(f"block_{i}", x)
        x = self.out_proj(x)
        if dump_callback: dump_callback("post_out_proj", x)
        return self.norm(x)


# ── Helpers ──
def save(t, path):
    t.float().detach().cpu().contiguous().numpy().tofile(path)

def stats(name, t, file=None):
    f = t.float().flatten()
    s = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": float(f.min()),
        "max": float(f.max()),
        "abs_mean": float(f.abs().mean()),
    }
    if file: s["file"] = file
    return s


# ── Map our checkpoint key naming → diffusion-pipe naming ──
# Our checkpoint (under `net.llm_adapter.` already stripped by AnimaCheckpointConverter):
#   embed.weight                              [32128, 1024]
#   norm.weight                               [1024]
#   out_proj.weight                           [1024, 1024]
#   out_proj.bias                             [1024]
#   blocks.N.norm_self_attn.weight            [1024]
#   blocks.N.self_attn.{q,k,v,o}_proj.weight  [1024, 1024]
#   blocks.N.self_attn.{q,k}_norm.weight      [64]
#   blocks.N.norm_cross_attn.weight           [1024]
#   blocks.N.cross_attn.{q,k,v,o}_proj.weight [1024, 1024]
#   blocks.N.cross_attn.{q,k}_norm.weight     [64]
#   blocks.N.norm_mlp.weight                  [1024]
#   blocks.N.mlp.0.weight, mlp.0.bias         [4096, 1024], [4096]
#   blocks.N.mlp.2.weight, mlp.2.bias         [1024, 4096], [1024]
# diffusion-pipe uses attribute `self_attn` (we used `self_attn_layer` above to dodge the Python name
# clash with the `self_attn` __init__ flag in their code); rename loading-side so state_dict aligns.
def load_llm_adapter_weights(raw):
    state = {}
    for k, v in raw.items():
        if not k.startswith("net.llm_adapter."):
            continue
        tail = k[len("net.llm_adapter."):]
        # blocks.N.self_attn.* → blocks.N.self_attn_layer.*
        if ".self_attn." in tail:
            tail = tail.replace(".self_attn.", ".self_attn_layer.")
        state[tail] = v
    return state


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Anime girl with blue hair in a garden"
    print(f"Prompt: {prompt!r}")

    # ── 1. Load Anima checkpoint, extract llm_adapter weights ──
    print(f"\nLoading Anima checkpoint: {ANIMA_PATH}")
    raw_state = {}
    with safe_open(ANIMA_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("net.llm_adapter."):
                raw_state[k] = f.get_tensor(k).float()
    print(f"  Found {len(raw_state)} llm_adapter tensors")
    adapter_state = load_llm_adapter_weights(raw_state)
    del raw_state

    # ── 2. Build the canonical LLMAdapter, load weights ──
    print("\nConstructing canonical LLMAdapter (F32)...")
    adapter = LLMAdapter().to(torch.float32).eval()
    missing, unexpected = adapter.load_state_dict(adapter_state, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing[:8]:
        print(f"    missing[:8] = {missing[:8]}")
    if unexpected[:8]:
        print(f"    unexpected[:8] = {unexpected[:8]}")

    # ── 3. Tokenize prompt with T5 SentencePiece (32000 base + 128 sentinels = 32128 embed rows) ──
    # We use the same tokenizer the canonical diffusion-pipe training uses.
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(T5_SPM)
    print(f"\nT5 tokenizer loaded: vocab={sp.vocab_size()}")
    # T5 typically adds </s> (eos_id=1) at the end, no BOS.
    t5_ids = sp.encode(prompt, add_eos=True)
    t5_ids_t = torch.tensor([t5_ids], dtype=torch.long)
    print(f"  T5 token ids: {t5_ids}  (len={len(t5_ids)})")
    decoded_tokens = [sp.id_to_piece(i) for i in t5_ids]
    print(f"  T5 pieces:    {decoded_tokens}")

    # ── 4. Build Qwen3 hidden states (re-use the same logic as encode_anima_prompt.py) ──
    print(f"\nEncoding Qwen3 hidden states for the same prompt...")
    from transformers.models.qwen3 import Qwen3Config, Qwen3Model
    from transformers import AutoTokenizer
    qwen_cfg = Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        hidden_act="silu", max_position_embeddings=40960, initializer_range=0.02,
        rms_norm_eps=1e-6, use_cache=False, rope_theta=1000000.0,
        sliding_window=None, attention_dropout=0.0, tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )
    qwen = Qwen3Model(qwen_cfg).to(torch.float32).eval()
    qwen_state = {}
    with safe_open(QWEN3_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            new_k = k[len("model."):] if k.startswith("model.") else k
            qwen_state[new_k] = f.get_tensor(k).float()
    qwen.load_state_dict(qwen_state, strict=False)
    qwen_tok = AutoTokenizer.from_pretrained(QWEN3_TOK)
    qwen_ids = qwen_tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    with torch.no_grad():
        qwen_hidden = qwen(qwen_ids).last_hidden_state  # [1, S_source, 1024]
    print(f"  qwen_hidden: shape={tuple(qwen_hidden.shape)} mean={qwen_hidden.mean():+.4f} std={qwen_hidden.std():.4f}")

    # ── 5. Save inputs ──
    save(qwen_hidden, f"{OUT_DIR}/inputs/qwen3_hidden.bin")
    t5_ids_t.cpu().numpy().astype("int64").tofile(f"{OUT_DIR}/inputs/t5_input_ids.bin")
    with open(f"{OUT_DIR}/inputs/prompt.txt", "w") as f:
        f.write(f"prompt = {prompt}\n")
        f.write(f"qwen3_seq_len = {qwen_hidden.shape[1]}\n")
        f.write(f"qwen3_hidden_size = {qwen_hidden.shape[2]}\n")
        f.write(f"t5_seq_len = {t5_ids_t.shape[1]}\n")
        f.write(f"t5_ids = {t5_ids}\n")
        f.write(f"t5_pieces = {decoded_tokens}\n")

    # ── 6. Hook the forward to dump every captured tensor ──
    saved = []
    def dump_callback(name, t):
        safe = name.replace(".", "_").replace("/", "_")
        path_rel = f"layers/{safe}.bin"
        save(t, f"{OUT_DIR}/{path_rel}")
        saved.append(stats(name, t, file=path_rel))

    # ── 7. Forward ──
    print(f"\nRunning LLMAdapter forward (F32, layer dumps enabled)...")
    with torch.no_grad():
        out = adapter(qwen_hidden, t5_ids_t, dump_callback=dump_callback)
    print(f"  output: shape={tuple(out.shape)} mean={out.mean():+.4f} std={out.std():.4f}")
    save(out, f"{OUT_DIR}/final_output.bin")

    # ── 8. Save index.json ──
    index = {
        "model": "Anima net.llm_adapter (canonical from tdrussell/diffusion-pipe)",
        "prompt": prompt,
        "t5_input_ids": t5_ids,
        "qwen3_seq_len": int(qwen_hidden.shape[1]),
        "t5_seq_len": int(t5_ids_t.shape[1]),
        "hidden_size": 1024,
        "num_blocks": 6,
        "final_output_stats": stats("final_output", out),
        "layers": saved,
    }
    with open(f"{OUT_DIR}/index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDONE. LlmAdapter reference saved to: {OUT_DIR}")
    print(f"  {len(saved)} per-layer captures")


if __name__ == "__main__":
    main()
