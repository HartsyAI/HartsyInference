"""Writes a tiny random Qwen3.5 text trunk (released key layout, `model.language_model.*` + `lm_head.weight`) and the
last hidden states the official transformers `Qwen3_5ForConditionalGeneration` text model produces for it, for
Qwen35HuggingFaceParityTests.

Usage: python backbone_reference.py <clef-flash release dir (config.json)> <output dir>   (needs transformers 5.10, torch, safetensors)
"""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

release, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.load(open(release / "config.json"))
t = cfg["text_config"]
t.update(hidden_size=64, intermediate_size=96, num_hidden_layers=8,
         layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
         head_dim=16, num_attention_heads=4, num_key_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
         linear_num_key_heads=2, linear_num_value_heads=4, vocab_size=80, max_position_embeddings=256,
         eos_token_id=1)
cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64, num_position_embeddings=16)
model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**{k: v for k, v in cfg.items() if k != "architectures"})).float().eval()
torch.manual_seed(11)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "visual" in name:
            continue
        if name.endswith("A_log"):
            p.copy_(torch.rand_like(p) * 1.5 - 0.5)
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "norm" in name:
            p.copy_(torch.randn_like(p) * 0.2 + (1.0 if name.endswith("linear_attn.norm.weight") else 0.0))
        else:
            p.copy_(torch.randn_like(p) * 0.12)
ids = torch.randint(0, 80, (1, 19))
with torch.no_grad():
    hidden = model.model.language_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]

out.mkdir(parents=True, exist_ok=True)
state = {k: v.contiguous() for k, v in model.state_dict().items() if "visual" not in k}
save_file(state, str(out / "qwen35_tiny.safetensors"))
save_file({"hidden": hidden.contiguous(), "input_ids": ids[0].to(torch.int32).contiguous()}, str(out / "qwen35_tiny_expected.safetensors"))
json.dump({"text_config": t}, open(out / "qwen35_tiny_config.json", "w"))
print(hidden.shape, hidden.abs().max())
