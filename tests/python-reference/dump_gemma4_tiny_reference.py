#!/usr/bin/env python3
"""Dumps a tiny-config Gemma 4 reference (weights, input ids, all hidden states) from ComfyUI's
comfy/text_encoders/gemma4.py so the C# Gemma4TextEncoder can be checked layer by layer.

The tiny config keeps the real per-layer-type geometry that the C# port has to reproduce: a 6-layer cycle
of 5 sliding + 1 global, different head dims per kind (16 / 32), GQA on both, k_eq_v + partial rotary on
the global layer. Two cases are dumped: one whose sequence fits inside the sliding window (the LTX shape,
where the window mask never fires) and one that exceeds it (window mask + the reference's expand_kv path).

Run with ComfyUI's venv, which is the only interpreter here with torch:
  "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/bin/python3" \
      tests/python-reference/dump_gemma4_tiny_reference.py --comfy-root <path> --output <dir>
"""
import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import torch


def shim_comfy_kitchen():
    """The venv's comfy_kitchen predates the pinned ComfyUI checkout; stub the probes it is missing so the
    import chain reaches gemma4.py. Only availability flags are stubbed, never a compute path."""
    try:
        import comfy_kitchen
    except ImportError:
        return
    for probe in ("int8_attention_is_available", "fp8_attention_is_available"):
        if not hasattr(comfy_kitchen, probe):
            setattr(comfy_kitchen, probe, lambda: False)


def build_config(gemma4, window: int):
    # MUST stay a @dataclass: Gemma4Config's generated __init__ assigns every annotated field on the instance,
    # so a plain subclass's annotated overrides are silently shadowed by the parent's defaults.
    @dataclass
    class TinyConfig(gemma4.Gemma4Config):
        vocab_size: int = 100
        hidden_size: int = 72
        intermediate_size: int = 128
        # Two full 6-layer cycles, so a GLOBAL layer's raw output is a capturable hidden state (index 6)
        # rather than only reachable through the final norm.
        num_hidden_layers: int = 12
        num_attention_heads: int = 2
        num_key_value_heads: int = 1
        rms_norm_eps: float = 1e-6
        rope_theta = [1000000.0, 10000.0]
        head_dim = 16
        global_head_dim = 32
        num_global_key_value_heads = 1
        attention_k_eq_v = True
        vision_bidirectional = False
        rms_norm_add = False
        mlp_activation = "gelu_pytorch_tanh"
        qkv_bias = False
        q_norm = "gemma3"
        k_norm = "gemma3"
        sliding_attention = [window, window, window, window, window, False]
        partial_rotary_factor: float = 0.25
        final_norm: bool = True
        lm_head: bool = False
        hidden_size_per_layer_input: int = 0
        num_kv_shared_layers: int = 0
        use_double_wide_mlp: bool = False
        vision_config = None
        audio_config = None

    return TinyConfig()


def randomize(model, seed: int):
    """Random but deliberately well-conditioned. Gemma 4 amplifies relative f32 error roughly 2x per layer,
    so wide norm / layer_scalar spreads push the reference's OWN f32-vs-f64 distance past 1e-4 by layer 12 and
    no correct port can match it. Keeping both within +-0.3 of 1.0 holds that floor near 2e-6 while staying far
    enough from 1.0 that a dropped scale or norm still fails loudly."""
    generator = torch.Generator().manual_seed(seed)
    # state_dict(), NOT named_buffers(): the RoPE inv_freq buffers are non-persistent, so they are absent here
    # but present there — randomizing them silently replaces the rotary tables with noise.
    for name, param in model.state_dict().items():
        if not torch.is_floating_point(param):
            continue
        if name.endswith("layer_scalar"):
            # Never leave this at 1.0: a port that drops the multiply entirely would still pass.
            value = 1.0 + (torch.rand(param.shape, generator=generator) * 2 - 1) * 0.3
        elif "norm" in name:
            # Gemma 4 stores RMS scales directly (verified against real bf16 bytes), so centre them on 1.
            value = 1.0 + (torch.rand(param.shape, generator=generator) * 2 - 1) * 0.3
        else:
            value = torch.randn(param.shape, generator=generator) * 0.05
        with torch.no_grad():
            param.copy_(value.to(param.dtype))


def dump_case(gemma4, out_dir: str, name: str, window: int, seq_len: int, seed: int):
    torch.manual_seed(seed)
    config = build_config(gemma4, window)

    import comfy.ops
    model = gemma4.Gemma4Transformer(config, device=torch.device("cpu"), dtype=torch.float32,
                                     ops=comfy.ops.disable_weight_init)
    model.eval()
    randomize(model, seed)

    generator = torch.Generator().manual_seed(seed + 1)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), generator=generator)

    with torch.no_grad():
        _, intermediate = model(input_ids, intermediate_output="all",
                                final_layer_norm_intermediate=False, dtype=torch.float32)

    case_dir = os.path.join(out_dir, name)
    weight_dir = os.path.join(case_dir, "weights")
    # Wipe first: a stale file from an earlier config would sit next to the new ones and read as valid.
    if os.path.isdir(weight_dir):
        shutil.rmtree(weight_dir)
    os.makedirs(weight_dir, exist_ok=True)

    shapes = {}
    state = dict(model.state_dict())
    for key, tensor in state.items():
        array = tensor.detach().to(torch.float32).contiguous().numpy()
        safe = key.replace("/", "_")
        array.tofile(os.path.join(weight_dir, f"model.{safe}.bin"))
        shapes[f"model.{key}"] = list(array.shape)

    input_ids.numpy().astype(np.int32).tofile(os.path.join(case_dir, "input_ids.bin"))
    states = intermediate.detach().to(torch.float32).contiguous().numpy()
    states.tofile(os.path.join(case_dir, "hidden_states.bin"))

    meta = {
        "name": name,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "num_global_key_value_heads": config.num_global_key_value_heads,
        "head_dim": config.head_dim,
        "global_head_dim": config.global_head_dim,
        "sliding_window": window,
        "partial_rotary_factor": config.partial_rotary_factor,
        "rms_norm_eps": config.rms_norm_eps,
        "global_rope_theta": config.rope_theta[0],
        "sliding_rope_theta": config.rope_theta[1],
        "seq_len": seq_len,
        "hidden_states_shape": list(states.shape),
        "weight_shapes": shapes,
    }
    with open(os.path.join(case_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[{name}] states {states.shape} window={window} seq={seq_len} -> {case_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True, help="Directory containing the 'comfy' package.")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "gemma4_tiny_reference"))
    args = parser.parse_args()

    sys.path.insert(0, args.comfy_root)
    shim_comfy_kitchen()
    import comfy.text_encoders.gemma4 as gemma4

    os.makedirs(args.output, exist_ok=True)
    dump_case(gemma4, args.output, "within_window", window=16, seq_len=8, seed=1234)
    dump_case(gemma4, args.output, "beyond_window", window=4, seq_len=12, seed=4321)


if __name__ == "__main__":
    main()
