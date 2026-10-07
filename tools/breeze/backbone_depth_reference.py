"""Writes a tiny random Breeze backbone + depth decoder (released key layout) and the activations the official classes
(`BreezeBackboneAdapter` for the Qwen3 backbone, `BreezeDepthDecoderForCausalLM`) produce, for BreezeTts2ModelParityTests.

Usage: python backbone_depth_reference.py <breeze-tts checkout> <output dir>   (needs torch, transformers 4.57, safetensors)
"""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from models.breeze import BreezeDepthDecoderForCausalLM
from models.breeze_backbone_factory import BreezeBackboneAdapter
from models.breeze_base_config import BreezeConfig, BreezeDepthDecoderConfig

torch.manual_seed(5)
NC, V, H, DH = 4, 19, 32, 16
depth_cfg = BreezeDepthDecoderConfig(
    num_codebooks=NC, backbone_hidden_size=H, vocab_size=V, hidden_size=DH, intermediate_size=24, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=NC + 1, rope_theta=500000,
    rope_scaling={"factor": 32.0, "high_freq_factor": 0.0078125, "low_freq_factor": 0.001953125,
                  "original_max_position_embeddings": 16, "rope_type": "llama3"},
    audio_embed_size=H)
depth_cfg._attn_implementation = "eager"
depth = BreezeDepthDecoderForCausalLM(depth_cfg).eval()

backbone_cfg = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False, "head_dim": 8, "hidden_act": "silu",
                "hidden_size": H, "intermediate_size": 48, "max_position_embeddings": 64, "model_type": "qwen3",
                "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "rope_scaling": None, "rope_theta": 1000000, "sliding_window": None, "tie_word_embeddings": True,
                "use_sliding_window": False, "vocab_size": 100}
cfg = BreezeConfig(num_codebooks=NC, vocab_size=V, text_vocab_size=100, hidden_size=H, intermediate_size=48,
                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
                   backbone_model_type="qwen3", backbone_config=backbone_cfg, audio_embed_size=H, max_position_embeddings=64)
cfg._attn_implementation = "eager"
backbone = BreezeBackboneAdapter.create_from_config(cfg).eval()
lm_head = torch.nn.Linear(H, V + 1, bias=False)

with torch.no_grad():
    for m in (depth, backbone, lm_head):
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if "norm" in name else 0.15))
    # the released checkpoint ties the backbone's audio embedding to the depth decoder's
    backbone.embed_tokens.embed_audio_tokens.weight.copy_(depth.model.embed_tokens.weight)

# backbone: 5 audio frames (embedded as the sum over codebooks) followed by 3 free "text" embeddings
codes = torch.randint(0, V - 3, (1, 5, NC))
text = torch.randn(1, 3, H)
with torch.no_grad():
    audio_embeds = backbone.embed_tokens(codes)
    inputs = torch.cat([text, audio_embeds], dim=1)
    position_ids = torch.arange(inputs.shape[1])[None]
    hidden = backbone(inputs_embeds=inputs, position_ids=position_ids, use_cache=False).last_hidden_state
    logits = lm_head(hidden[:, -1])

# depth decoder: [placeholder, t0, t1, t2] with the last backbone hidden state at position 0
tokens = torch.tensor([[0, 3, 7, 11]])
with torch.no_grad():
    depth_logits = depth(input_ids=tokens, backbone_last_hidden_state=hidden[:, -1], use_cache=False).logits

ck = {}
for k, v in backbone.state_dict().items():
    if k.startswith("embed_tokens"): continue                  # tied to depth_decoder.model.embed_tokens
    ck[f"backbone_model.{k}"] = v.contiguous().float()
for k, v in depth.state_dict().items():
    ck[f"depth_decoder.{k}"] = v.contiguous().float()
ck["lm_head.weight"] = lm_head.weight.detach().contiguous().float()
ck["ref.audio_embeds"] = audio_embeds[0].contiguous()
ck["ref.hidden"] = hidden[0].contiguous()
ck["ref.logits"] = logits[0].contiguous()
ck["ref.depth_logits"] = depth_logits[0].contiguous()
ck["ref.text"] = text[0].contiguous()
out.mkdir(parents=True, exist_ok=True)
save_file(ck, str(out / "breeze_core_tiny.safetensors"))
(out / "breeze_core_tiny.json").write_text(json.dumps({"codes": codes[0].tolist(), "depth_tokens": tokens[0].tolist()}))
print("wrote", out, list(hidden.shape), list(depth_logits.shape), sorted(k for k in ck if k.startswith("depth_decoder"))[:5])
