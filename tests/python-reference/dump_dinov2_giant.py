"""Reference values for the DINOv2-giant image encoder Paint 2.1 and the Hunyuan3D shape models condition on, produced by Hugging Face transformers on the CPU.

Builds `transformers.Dinov2Model` with the `facebook/dinov2-giant` configuration (width 1536, 40 layers, 24 heads, SwiGLU FFN, 518 px, patch 14) and loads the F16 copy of its weights that
the Hunyuan3D shape checkpoints bundle (`conditioner.main_image_encoder.model.*`, the same tensor names as the Hugging Face model; widened to float32), then runs a fixed random
input through it:

    python dump_dinov2_giant.py --weights <hunyuan3d2/hunyuan3d-dit-v2-0/model.fp16.safetensors> --out <fixture.safetensors>

`--variant large` builds the `facebook/dinov2-large` configuration instead (width 1024, 24 layers, 16 heads, the plain GELU FFN; what Hunyuan3D-2.1 and Omni condition on) and reads the Hugging Face file
`facebook/dinov2-large/model.safetensors` directly (`--prefix ""`, the default for that variant):

    python dump_dinov2_giant.py --variant large --weights <dinov2-large/model.safetensors> --out <fixture.safetensors>

Fixture keys: `image_u8` [512, 512, 3] (uint8) and `pixels_preprocessed` [1, 3, 518, 518] (torchvision's resize, crop and normalisation of it, the Hunyuan3D-2.1 and Omni path), `pixels` [1, 3, 518, 518], `last_hidden_state` [1, 1370, width] (after the final LayerNorm) and `prenorm` [1, 1370, width] (the last block's output, `hidden_states[-1]`).
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
    parser.add_argument("--variant", choices=("giant", "large"), default="giant")
    parser.add_argument("--prefix", default=None, help="the key prefix of the encoder in the weights file (default: the Hunyuan3D shape checkpoint's for giant, none for large)")
    args = parser.parse_args()
    prefix = args.prefix if args.prefix is not None else (PREFIX if args.variant == "giant" else "")
    torch.manual_seed(0)
    if args.variant == "giant":
        config = Dinov2Config(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, mlp_ratio=4, use_swiglu_ffn=True, image_size=518, patch_size=14,
                              layerscale_value=1.0, qkv_bias=True, layer_norm_eps=1e-6, hidden_act="gelu")
    else:
        config = Dinov2Config(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, mlp_ratio=4, use_swiglu_ffn=False, image_size=518, patch_size=14,
                              layerscale_value=1.0, qkv_bias=True, layer_norm_eps=1e-6, hidden_act="gelu")
    model = Dinov2Model(config).eval()
    with safe_open(args.weights, framework="pt") as handle:
        state = remap({k[len(prefix):]: handle.get_tensor(k).float() for k in handle.keys() if k.startswith(prefix)})
    print(model.load_state_dict(state, strict=True))
    pixels = torch.randn(1, 3, 518, 518).clamp(-2.2, 2.6)
    start = time.time()
    with torch.no_grad():
        out = model(pixel_values=pixels, output_hidden_states=True)
    print(f"forward {time.time() - start:.1f} s, last_hidden_state {tuple(out.last_hidden_state.shape)}")
    # The preprocessing of Hunyuan3D-2.1 and Omni's `DinoImageEncoder`: Resize(518, bilinear, antialias) and CenterCrop(518) of the [0, 1] image, then ImageNet normalisation, on a 512 px image.
    import torchvision.transforms as T

    image = (torch.rand(512, 512, 3, generator=torch.Generator().manual_seed(1)) * 255).to(torch.uint8)
    image[100:300, 150:400] = torch.tensor([200, 30, 90], dtype=torch.uint8)  # an edge with colour, so the sampling position and filter matter
    transform = T.Compose([T.Resize(518, T.InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(518), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    preprocessed = transform((image.permute(2, 0, 1).float() / 255.0)[None])
    save_file({"pixels": pixels, "last_hidden_state": out.last_hidden_state.contiguous(), "prenorm": out.hidden_states[-1].contiguous(), "image_u8": image.contiguous(),
               "pixels_preprocessed": preprocessed.contiguous()}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
