"""Reference values for the DINOv2-giant image encoder Paint 2.1 and the Hunyuan3D shape models condition on, produced by Hugging Face transformers on the CPU.

Builds `transformers.Dinov2Model` with the `facebook/dinov2-giant` configuration (width 1536, 40 layers, 24 heads, SwiGLU FFN, 518 px, patch 14) and loads the F16 copy of its weights that
the Hunyuan3D shape checkpoints bundle (`conditioner.main_image_encoder.model.*`, the same tensor names as the Hugging Face model; widened to float32), then runs a fixed random
input through it:

    python dump_dinov2_giant.py --weights <hunyuan3d2/hunyuan3d-dit-v2-0/model.fp16.safetensors> --out <fixture.safetensors>

Fixture keys: `pixels` [1, 3, 518, 518], `last_hidden_state` [1, 1370, 1536] (after the final LayerNorm) and `prenorm` [1, 1370, 1536] (the last block's output, `hidden_states[-1]`).
"""
import argparse
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Dinov2Config, Dinov2Model

PREFIX = "conditioner.main_image_encoder.model."


def remap(state):
    """The bundled tensors use the original Hugging Face names; transformers 5 renamed the attention and SwiGLU projections: query/key/value -> q/k/v_proj, output.dense -> o_proj,
    weights_in (the two halves of the SwiGLU input, `silu(first) * second`) -> gate_proj and up_proj, weights_out -> down_proj."""
    out = {}
    for key, value in state.items():
        key = (key.replace("attention.attention.query", "attention.q_proj").replace("attention.attention.key", "attention.k_proj")
               .replace("attention.attention.value", "attention.v_proj").replace("attention.output.dense", "attention.o_proj").replace("mlp.weights_out", "mlp.down_proj"))
        if ".mlp.weights_in." in key:
            half = value.shape[0] // 2
            out[key.replace("weights_in", "gate_proj")], out[key.replace("weights_in", "up_proj")] = value[:half].clone(), value[half:].clone()
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    config = Dinov2Config(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, mlp_ratio=4, use_swiglu_ffn=True, image_size=518, patch_size=14,
                          layerscale_value=1.0, qkv_bias=True, layer_norm_eps=1e-6, hidden_act="gelu")
    model = Dinov2Model(config).eval()
    with safe_open(args.weights, framework="pt") as handle:
        state = remap({k[len(PREFIX):]: handle.get_tensor(k).float() for k in handle.keys() if k.startswith(PREFIX)})
    print(model.load_state_dict(state, strict=True))
    pixels = torch.randn(1, 3, 518, 518).clamp(-2.2, 2.6)
    start = time.time()
    with torch.no_grad():
        out = model(pixel_values=pixels, output_hidden_states=True)
    print(f"forward {time.time() - start:.1f} s, last_hidden_state {tuple(out.last_hidden_state.shape)}")
    save_file({"pixels": pixels, "last_hidden_state": out.last_hidden_state.contiguous(), "prenorm": out.hidden_states[-1].contiguous()}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
