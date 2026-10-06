"""Writes a tiny random Fish Audio S2 checkpoint (in the released key layout) plus the logits the official
fish-speech DualARTransformer produces for it, for FishAudioS2DualArParityTests.

Usage: python s2_dual_ar_reference.py <fish-speech checkout> <output dir>
Needs torch, safetensors, einops, loguru, loralib. Rope tables are rebuilt in float32 (the upstream buffers are
bfloat16) so the comparison measures the port, not bfloat16 rounding.
"""
import importlib.util, json, math, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("llama", repo / "fish_speech/models/text2semantic/llama.py")
llama = importlib.util.module_from_spec(spec); spec.loader.exec_module(llama)

torch.manual_seed(7)
cfg = llama.DualARModelArgs(
    vocab_size=160, n_layer=2, n_head=4, dim=32, intermediate_size=48, n_local_heads=2, head_dim=8,
    rope_base=1_000_000, norm_eps=1e-6, max_seq_len=32, tie_word_embeddings=True, attention_qk_norm=True,
    codebook_size=16, num_codebooks=4, semantic_begin_id=100, semantic_end_id=115,
    scale_codebook_embeddings=True, norm_fastlayer_input=True, n_fast_layer=2, fast_dim=32, fast_n_head=4,
    fast_n_local_heads=2, fast_head_dim=8, fast_intermediate_size=48, fast_attention_qk_norm=False,
    use_gradient_checkpointing=False,
)
model = llama.DualARTransformer(cfg).eval()
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p) * 0.2)
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))

def fp32_freqs(n, dim, base):
    f = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    ang = torch.outer(torch.arange(n).float(), f)
    return torch.stack([torch.cos(ang), torch.sin(ang)], -1)
model.freqs_cis = fp32_freqs(cfg.max_seq_len, cfg.head_dim, cfg.rope_base)
model.fast_freqs_cis = fp32_freqs(cfg.num_codebooks, cfg.fast_head_dim, cfg.rope_base)
model.setup_caches(1, cfg.max_seq_len, dtype=torch.float32)

# Prompt: 3 text tokens, then 4 semantic frames (row 0 = begin + code0, rows 1..4 = codes).
T = 7
tokens = [5, 17, 42] + [100 + c for c in (3, 9, 0, 15)]
codes = torch.randint(0, cfg.codebook_size, (cfg.num_codebooks, 4))
codes[0] = torch.tensor([3, 9, 0, 15])
inp = torch.zeros(1, cfg.num_codebooks + 1, T, dtype=torch.long)
inp[0, 0] = torch.tensor(tokens)
inp[0, 1:, 3:] = codes

with torch.no_grad():
    res = llama.BaseTransformer.forward_generate(model, inp, return_all=True)
    slow_logits = res.logits[0]            # [T, vocab]
    hidden = res.hidden_states[0]          # [T, dim] (post-norm: norm_fastlayer_input)
    last = hidden[-1:].unsqueeze(0)        # [1, 1, dim]
    fast_logits = []
    model.forward_generate_fast(last, torch.tensor([0]))
    prev = model.fast_embeddings(torch.tensor([int(codes[0, -1])]))
    for k in range(1, cfg.num_codebooks):
        lg = model.forward_generate_fast(prev, torch.tensor([k]))
        fast_logits.append(lg[0, 0])
        prev = model.fast_embeddings(torch.tensor([int(torch.argmax(lg[0, 0]))]))
    fast_codes = [int(codes[0, -1])]
    # replay greedily so the C# side can feed the same previous codes
    fast_logits_t = torch.stack(fast_logits)

sd = model.state_dict()
ck = {}
for k, v in sd.items():
    if k.startswith("fast_"):
        if k.startswith("fast_layers."):  nk = "audio_decoder.layers." + k[len("fast_layers."):]
        elif k == "fast_embeddings.weight": nk = "audio_decoder.embeddings.weight"
        elif k == "fast_norm.weight": nk = "audio_decoder.norm.weight"
        elif k == "fast_output.weight": nk = "audio_decoder.output.weight"
        else: raise SystemExit(f"unmapped {k}")
    elif k.startswith("codebook_embeddings."): nk = "audio_decoder." + k
    else: nk = "text_model.model." + k
    ck[nk] = v.contiguous().float()
ck["ref.slow_logits"] = slow_logits.contiguous()
ck["ref.hidden"] = hidden.contiguous()
ck["ref.fast_logits"] = fast_logits_t.contiguous()
out.mkdir(parents=True, exist_ok=True)
save_file(ck, str(out / "s2_tiny.safetensors"))
(out / "s2_tiny.json").write_text(json.dumps({
    "tokens": tokens, "codes": codes.tolist(), "last_code0": int(codes[0, -1]),
    "fast_prev": [int(torch.argmax(fast_logits_t[i])) for i in range(cfg.num_codebooks - 2)],
}))
print("wrote", out)
