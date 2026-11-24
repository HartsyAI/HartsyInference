"""Writes a tiny random SenseNova-Loop text trunk (released key layout: `model.language_model.*`, with the FP32 mHC routers
randomised, since the shipped ones start at zero) and the final hidden states that the checkpoint's own `modeling_sensenova_loop.py`
produces for it, for SenseNovaLoopHuggingFaceParityTests.

Usage: python backbone_reference.py <SenseNova-Loop-9B dir holding config.json and modeling_sensenova_loop.py> <output dir>
Needs transformers 5.18.0, torch 2.11 (CPU is enough), safetensors. Run it on WSL or a pod; never in the Windows venv.
Reference: sensenova/SenseNova-Loop-9B @ 3cc220e9c14bd7a74813c1299765e22147b1a3d5.
Then: SENSENOVA_LOOP_TINY_EXPECTED=<output dir>/sensenova_loop_tiny_expected.safetensors dotnet test --filter SenseNovaLoopHuggingFaceParity

Not run yet: this machine has no Linux Python, so the script is written against the reference source but unexecuted.
"""
import importlib.util, json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

release, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("modeling_sensenova_loop", release / "modeling_sensenova_loop.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# The checkpoint code targets transformers 5.18; later releases renamed the output class it returns. Same fields, so alias it.
if not hasattr(mod.hf, "Qwen3_5ModelOutputWithPast"):
    mod.hf.Qwen3_5ModelOutputWithPast = mod.hf.BaseModelOutputWithPast

cfg = json.load(open(release / "config.json"))
t = cfg["text_config"]
t.update(hidden_size=64, intermediate_size=96, num_hidden_layers=12,
         layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 3,
         head_dim=16, num_attention_heads=4, num_key_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
         linear_num_key_heads=2, linear_num_value_heads=4, vocab_size=80, max_position_embeddings=256,
         eos_token_id=1, pad_token_id=0, mtp_num_hidden_layers=0)
# YaRN stays on, with an original length small enough that the frequency blend is not a no-op over the 4 rotary pairs.
t["rope_parameters"].update(original_max_position_embeddings=128, factor=2.0)
cfg.update(pad_token_id=0, eos_token_id=1, image_token_id=2, video_token_id=3, vision_start_token_id=4, vision_end_token_id=5)
cfg["loop_settings"] = dict(start=4, end=8, eval_count=2, counts=[2])
cfg["mhc"] = dict(start=4, end=8, fixed_identity=False, iterations=20, pre_scale=1.0, streams=2)
cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=64,
                            num_position_embeddings=16, deepstack_visual_indexes=[])
config = mod.SenseNovaLoopConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map")})
model = mod.SenseNovaLoopForConditionalGeneration(config).float().eval()
torch.manual_seed(11)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "visual" in name:
            continue
        if name.endswith("A_log"):
            p.copy_(torch.rand_like(p) * 1.5 - 0.5)
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "mhc_" in name:
            p.copy_(torch.randn_like(p) * (0.08 if name.endswith("weight") else 0.6))
        elif "norm" in name:
            p.copy_(torch.randn_like(p) * 0.2 + (1.0 if name.endswith("linear_attn.norm.weight") else 0.0))
        else:
            p.copy_(torch.randn_like(p) * 0.12)
ids = torch.randint(0, 80, (1, 19))
with torch.no_grad():
    lm = model.model.language_model
    hidden = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
    # Prefill of the first 11 tokens, then 8 single steps through a cache: the per-visit slots must give the same last position.
    cache = lm(input_ids=ids[:, :11], use_cache=True).past_key_values
    for i in range(11, 19):
        step = lm(input_ids=ids[:, i:i + 1], past_key_values=cache, use_cache=True)
        cache = step.past_key_values
    stepped_last = step.last_hidden_state[0, -1]

out.mkdir(parents=True, exist_ok=True)
state = {k: v.contiguous() for k, v in model.state_dict().items() if "visual" not in k}
save_file(state, str(out / "sensenova_loop_tiny.safetensors"))
save_file({"hidden": hidden.contiguous(), "stepped_last": stepped_last.contiguous(), "input_ids": ids[0].to(torch.int32).contiguous()},
          str(out / "sensenova_loop_tiny_expected.safetensors"))
json.dump(cfg, open(out / "sensenova_loop_tiny_config.json", "w"))
print(hidden.shape, hidden.abs().max(), (hidden[-1] - stepped_last).abs().max())
