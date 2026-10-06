"""Writes a tiny random T5Gemma2 text encoder (released key layout, `text_encoder.*`) and the hidden states the official
Breeze compat implementation (`models/t5gemma2_compat.py`) produces for it, for T5Gemma2TextEncoderParityTests.

Usage: python text_encoder_reference.py <breeze-tts checkout> <output dir>   (needs torch, transformers 4.57, safetensors)
"""
import importlib.util, json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("t5g", repo / "models" / "t5gemma2_compat.py")
t5g = importlib.util.module_from_spec(spec); spec.loader.exec_module(t5g)

torch.manual_seed(3)
layers = 7
layer_types = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(layers)]
cfg = t5g.T5Gemma2TextConfig(
    vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=layers, num_attention_heads=4,
    num_key_value_heads=1, head_dim=8, query_pre_attn_scalar=8, sliding_window=8, layer_types=layer_types,
    rope_parameters={"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1_000_000},
                     "sliding_attention": {"rope_type": "default", "rope_theta": 10_000}},
    eoi_token_index=60, max_position_embeddings=64)
cfg._attn_implementation = "eager"
model = t5g.T5Gemma2TextEncoder(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * (0.3 if "norm" in name else 0.15))
tokens = torch.randint(0, 59, (1, 21)); tokens[0, 5] = 60
mask = torch.ones_like(tokens)
with torch.no_grad():
    res = model(input_ids=tokens, attention_mask=mask, position_ids=torch.arange(21)[None], output_hidden_states=True)
ck = {f"text_encoder.{k}": v.contiguous().float() for k, v in model.state_dict().items()}
ck["ref.last"] = res.last_hidden_state[0].contiguous()
for i, hs in enumerate(res.hidden_states[:-1]):
    if i >= 1: ck[f"ref.layer{i-1}"] = hs[0].contiguous()
ck["ref.embed"] = res.hidden_states[0][0].contiguous()
out.mkdir(parents=True, exist_ok=True)
save_file(ck, str(out / "t5gemma2_tiny.safetensors"))
(out / "t5gemma2_tiny.json").write_text(json.dumps({"tokens": tokens[0].tolist()}))
print("wrote", out, list(res.last_hidden_state.shape))
