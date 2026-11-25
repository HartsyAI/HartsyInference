"""Builds a tiny Maxwell-V1.0-Flash checkpoint from the release's own modeling code and dumps reference tensors for the C# tests.

Run in the CPU reference image (never the Windows venv):

    run.sh -v "<worktree>:/wt" -v "<dir with the two release files>:/hf:ro" -- python /wt/tools/maxwell-reference/dump_tiny.py \
        --hf-dir /hf --out /wt/tests/HartsyInference.LLM.Tests/Fixtures/Maxwell

<hf-dir> holds modeling_maxwell_moe.py and configuration_maxwell_moe.py from maxwell-model/Maxwell-V1-Flash at commit
5dd7c6f62165342b5ffc5f0c5b38c2198e82c751 (Apache-2.0). Everything runs in float32 on the CPU with saved inputs, not seeds.

The geometry is shrunk (block 4, 2 prefix + 2 local + 2 remote blocks, so the route equals dense causal attention below position 24 and the
indexer picks from up to 8 candidate blocks by position 39), and every parameter the release's _init_weights leaves at zero or one (expert_bias,
hc head, alphas, biases, norm weights) is randomized so those paths are exercised.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file

TINY = dict(
    vocab_size=64, hidden_size=32, intermediate_size=24, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
    attention_head_size=16, max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0, moe_intermediate_size=8,
    shared_expert_intermediate_size=12, num_experts=8, num_experts_per_tok=3, mlp_only_layers=[0],
    block_sparse_gqa_indexer_n_heads=2, block_sparse_gqa_indexer_head_dim=16, block_sparse_gqa_block_size=4,
    block_sparse_gqa_prefix_window_size=8, block_sparse_gqa_local_window_size=8, block_sparse_gqa_num_remote_blocks=2,
    block_sparse_gqa_query_chunk_size=8, num_residual_streams=4, mhc_sinkhorn_iterations=20, mtp_num_layers=1,
    block_sparse_gqa_layer_freq=[1] * 5, block_sparse_gqa_total_support_blocks=6, block_sparse_gqa_topk_blocks=2,
    block_sparse_gqa_group_reduction="per_group_max",
    bos_token_id=1, eos_token_id=2, pad_token_id=3,
)
SEQ_LEN, PREFILL = 40, 30


def load_model_classes(hf_dir):
    package = tempfile.mkdtemp()
    target = os.path.join(package, "maxwell_hf")
    os.makedirs(target)
    for name in ("modeling_maxwell_moe.py", "configuration_maxwell_moe.py"):
        shutil.copy(os.path.join(hf_dir, name), target)
    open(os.path.join(target, "__init__.py"), "w").close()
    sys.path.insert(0, package)
    config_module = importlib.import_module("maxwell_hf.configuration_maxwell_moe")
    model_module = importlib.import_module("maxwell_hf.modeling_maxwell_moe")
    return config_module.MaxwellMoeConfig, model_module.MaxwellMoeForCausalLM


def randomize(model, generator):
    """Fills every parameter with a seeded draw the checkpoint could plausibly hold; the stored values are what the C# side reads."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            leaf = name.split(".")
            if name.endswith("expert_bias"):
                p.copy_(torch.rand(p.shape, generator=generator) * 0.02 - 0.01)
            elif leaf[-1] in ("alpha_pre", "alpha_post", "alpha_res"):
                p.copy_(0.2 + 0.8 * torch.rand(p.shape, generator=generator))
            elif leaf[-1] == "hc_head_scale":
                p.copy_(0.5 + torch.rand(p.shape, generator=generator))
            elif leaf[-1] in ("bias", "hc_head_base"):
                p.copy_(0.4 * torch.randn(p.shape, generator=generator))
            elif "layernorm" in name or leaf[-2] in ("enorm", "hnorm") or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=generator))
            elif name.endswith("hc_head_fn") or name.endswith("mapping_proj.weight"):
                p.copy_(0.15 * torch.randn(p.shape, generator=generator))
            else:
                p.copy_(0.12 * torch.randn(p.shape, generator=generator))


def save_npy(out, name, tensor):
    np.save(os.path.join(out, name + ".npy"), tensor.detach().cpu().float().numpy() if tensor.is_floating_point() else tensor.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config_class, model_class = load_model_classes(args.hf_dir)
    config = config_class(**TINY)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = model_class(config).float().eval()
    generator = torch.Generator().manual_seed(1234)
    randomize(model, generator)

    input_ids = torch.randint(4, TINY["vocab_size"], (1, SEQ_LEN), generator=generator, dtype=torch.long)
    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, output_mtp_logits=True, return_dict=True)

        step_logits = []
        out = model(input_ids=input_ids[:, :PREFILL], use_cache=True, return_dict=True)
        cache = out.past_key_values
        prefill_logits = out.logits
        for pos in range(PREFILL, SEQ_LEN):
            out = model(input_ids=input_ids[:, pos:pos + 1], past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            step_logits.append(out.logits[:, -1])

    decode = torch.stack(step_logits, dim=1)
    drift = (decode - full.logits[:, PREFILL:]).abs().max().item()
    prefill_drift = (prefill_logits - full.logits[:, :PREFILL]).abs().max().item()
    print(f"cached decode vs full forward, max |diff|: decode {drift:.3e}, prefill {prefill_drift:.3e}")

    # The same weights with room for every block: positions below the dense limit must agree, later ones must not, or the fixture misses the indexer.
    dense_limit = (TINY["block_sparse_gqa_prefix_window_size"] + TINY["block_sparse_gqa_local_window_size"]) // TINY["block_sparse_gqa_block_size"]
    dense_limit = (dense_limit + TINY["block_sparse_gqa_num_remote_blocks"]) * TINY["block_sparse_gqa_block_size"]
    wide = config_class(**{**TINY, "block_sparse_gqa_num_remote_blocks": 100})
    wide._attn_implementation = "eager"
    wide_model = model_class(wide).float().eval()
    wide_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        wide_logits = wide_model(input_ids=input_ids, use_cache=False, return_dict=True).logits
    gap = (wide_logits - full.logits).abs().amax(dim=(0, 2))
    print(f"dense limit {dense_limit}: max |logit diff| vs all-blocks attention below it {gap[:dense_limit].max().item():.3e}, "
          f"at or above it {gap[dense_limit:].max().item():.3e}")
    assert gap[:dense_limit].max().item() < 1e-5 and gap[dense_limit:].max().item() > 1e-3, "the fixture does not exercise remote selection"

    state = {k: v.detach().contiguous().float() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(args.out, "model.safetensors"))
    cfg = {k: v for k, v in config.to_dict().items() if not k.startswith("_") and k not in ("transformers_version",)}
    cfg["model_type"] = "maxwell_moe"
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    save_npy(args.out, "input_ids", input_ids.to(torch.int32))
    save_npy(args.out, "logits", full.logits)
    save_npy(args.out, "mtp_logits", full.mtp_logits)
    save_npy(args.out, "prefill_logits", prefill_logits)
    save_npy(args.out, "decode_logits", decode)
    # hidden_states[i] is the 4-stream residual before layer i; the last entry is the normed single stream the LM head reads.
    for i, h in enumerate(full.hidden_states):
        save_npy(args.out, f"hidden_{i}", h)
    print("wrote", sorted(os.listdir(args.out))[:6], "...", len(os.listdir(args.out)), "files")


if __name__ == "__main__":
    main()
