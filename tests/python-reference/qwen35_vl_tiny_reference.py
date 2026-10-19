"""
Builds the tiny Qwen3.5 image-input fixture for Qwen35MultimodalEncoderParityTests: a random Qwen3.5 trunk and vision
tower (released key layout, model.language_model.* + model.visual.*), one 256x256 image, a prompt with a Picture-1 block,
and the final hidden state transformers' Qwen3_5Model returns for it (vision tower, placeholder splice, get_rope_index,
interleaved M-RoPE).

The trunk uses partial rotary 0.75 over a 32-wide head (12 frequency pairs) with mrope_section [5, 4, 3], so the
interleave bound cuts the width axis off early and every M-RoPE axis is exercised. The image is preprocessed the way
Qwen2VLImageProcessor does at its native size (normalize with mean/std 0.5, duplicate the frame across the temporal
patch, merge-block patch order), so the port's processor sees an image that needs no resize.

Usage: python qwen35_vl_tiny_reference.py <output dir>
  writes <dir>/encoder/{config.json, model.safetensors} (the checkpoint folder) and <dir>/expected.safetensors
Requires: torch, transformers >= 5.10 (Qwen3_5), safetensors. A torchvision or torchao that does not match the
installed torch breaks transformers' import; both are masked here because neither is used.
"""
import json
import sys
from pathlib import Path

for _broken in ("torchvision", "torchao"):
    sys.modules[_broken] = None

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

VOCAB = 80
VISION_START, IMAGE_PAD, VISION_END = 76, 77, 78
IMAGE = 256
PATCH, MERGE, TEMPORAL = 16, 2, 2

text_config = {
    "model_type": "qwen3_5_text", "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 96,
    "num_hidden_layers": 8, "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
    + ["full_attention"], "full_attention_interval": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "hidden_act": "silu", "rms_norm_eps": 1e-6, "max_position_embeddings": 512,
    "attn_output_gate": True, "tie_word_embeddings": False, "partial_rotary_factor": 0.75,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.75,
                        "mrope_section": [5, 4, 3], "mrope_interleaved": True},
    "mtp_num_hidden_layers": 0, "eos_token_id": 1,
}
vision_config = {
    "model_type": "qwen3_5_vision", "depth": 1, "hidden_size": 32, "intermediate_size": 48, "num_heads": 2,
    "in_channels": 3, "patch_size": PATCH, "spatial_merge_size": MERGE, "temporal_patch_size": TEMPORAL,
    "out_hidden_size": 64, "num_position_embeddings": 16, "deepstack_visual_indexes": [],
    "hidden_act": "gelu_pytorch_tanh",
}


def pixel_values(image):
    # image [3, H, W] in [0, 1] -> HF Qwen2VLImageProcessor flatten_patches at native size.
    x = (image - 0.5) / 0.5
    x = x[None].repeat(TEMPORAL, 1, 1, 1)                                            # [T, C, H, W]
    gh, gw = IMAGE // PATCH, IMAGE // PATCH
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH), torch.tensor([[1, gh, gw]])


def main():
    out = Path(sys.argv[1])
    folder = out / "encoder"
    folder.mkdir(parents=True, exist_ok=True)
    config = Qwen3_5Config(text_config=text_config, vision_config=vision_config, image_token_id=IMAGE_PAD,
                           vision_start_token_id=VISION_START, vision_end_token_id=VISION_END, tie_word_embeddings=False)
    model = Qwen3_5ForConditionalGeneration(config).float().eval()
    torch.manual_seed(20261019)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.rand_like(p) * 1.5 - 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "visual" in name and "norm" in name:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + torch.randn_like(p) * 0.1)
            elif "norm" in name:
                p.copy_(torch.randn_like(p) * 0.2 + (1.0 if name.endswith("linear_attn.norm.weight") else 0.0))
            else:
                p.copy_(torch.randn_like(p) * 0.12)
    image_u8 = torch.randint(0, 256, (3, IMAGE, IMAGE), dtype=torch.uint8)
    image = image_u8.float() / 255.0
    pixels, grid = pixel_values(image)
    pads = (IMAGE // PATCH // MERGE) ** 2
    ids = torch.tensor([[5, 9, 30, VISION_START] + [IMAGE_PAD] * pads + [VISION_END, 12, 33, 40, 2, 61]])
    types = (ids == IMAGE_PAD).to(torch.int32)
    text_ids = torch.tensor([[5, 9, 30, 12, 33, 40, 2, 61, 17, 44]])
    with torch.no_grad():
        hidden = model.model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types,
                             attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
        text_hidden = model.model(input_ids=text_ids, attention_mask=torch.ones_like(text_ids),
                                  use_cache=False).last_hidden_state[0]
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(folder / "model.safetensors"))
    save_file({
        "image_u8": image_u8.contiguous(),
        "input_ids": ids[0].to(torch.int32).contiguous(),
        "hidden": hidden.contiguous(),
        "text_ids": text_ids[0].to(torch.int32).contiguous(),
        "text_hidden": text_hidden.contiguous(),
    }, str(out / "expected.safetensors"))
    json.dump({"text_config": text_config, "vision_config": vision_config, "image_token_id": IMAGE_PAD,
               "vision_start_token_id": VISION_START, "vision_end_token_id": VISION_END}, open(folder / "config.json", "w"), indent=1)
    print("hidden", tuple(hidden.shape), float(hidden.abs().max()), "| text", tuple(text_hidden.shape))
    print("keys:", len(state), "visual:", sum(1 for k in state if "visual" in k))


if __name__ == "__main__":
    main()
