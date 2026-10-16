"""
Dumps a tiny FLUX.3 Klein transformer and its outputs from ComfyUI's own module, for Flux3ParityTests.

ComfyUI's comfy/ldm/flux3/model.py is the reference: it loads the released files as they are, where diffusers'
port renames every module. The model is built at hidden 128 (one 128-wide head, as the real head width), one
per-mode block per stream and two joint blocks, with a fused qkv so the C# converter's quant-aware split is on
the path. txt_in is narrowed to CONTEXT inputs
(the math does not depend on its width; the real model reads 20480). Weights are rounded to F16 and the
reference runs in F32 on those rounded values, so the fixture is exact input for the C# side.

comfy.model_management.in_training is set so QKNorm and modulated_norm take their plain-torch branches; the
inference branches are comfy_kitchen fusions of the same math (rms_rope, adaln).

Cases (keys prefixed in the output file):
  guided.*   flow_map=False, text only, sigma 0.7
  edit.*     flow_map=False, two reference latents (2x2 and 1x2 grids)
  flowmap.*  flow_map=True (ref_time_in), schedule [1.0, 0.7, 0.3, 0.0], sigma 0.7 -> target 0.3

Usage: python -I dump_flux3_reference.py <ComfyUI checkout> <output .safetensors>
Requires: torch, safetensors, einops, ComfyUI's requirements (psutil, ...).
"""
import sys

import torch
from safetensors.torch import save_file

HIDDEN = 128
DEPTH = 1
SINGLES = 2
CONTEXT = 64
TEXT_TOKENS = 5
GRID = (2, 3)
SEED = 20261015


def build(comfy_root: str, flow_map: bool, state: dict | None):
    sys.path.insert(0, comfy_root)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.model_management
    import comfy.ops
    from comfy.ldm.flux3.model import Flux3

    comfy.model_management.in_training = True
    model = Flux3(hidden_size=HIDDEN, num_heads=HIDDEN // 128, depth=DEPTH, depth_single_blocks=SINGLES,
                  fused_qkv=True, flow_map=flow_map, dtype=torch.float32, device="cpu",
                  operations=comfy.ops.disable_weight_init)
    model.txt_in = comfy.ops.disable_weight_init.Linear(CONTEXT, HIDDEN, bias=False, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if state is not None and name in state:
                param.copy_(state[name].float())
                continue
            if name.endswith("_norm.weight"):
                value = 1.0 + 0.1 * torch.randn(param.shape, generator=generator)
            else:
                value = torch.randn(param.shape, generator=generator) / param.shape[-1] ** 0.5
            param.copy_(value.half().float())
    return model.eval()


def forward(model, x, sigma, context, refs=None, sample_sigmas=None):
    options = {} if sample_sigmas is None else {"sample_sigmas": sample_sigmas}
    with torch.no_grad():
        return model._forward(x, torch.tensor([sigma]), context=context, ref_latents=refs, transformer_options=options)


def packed(t):
    return t.flatten(2).transpose(1, 2).contiguous()


def main():
    comfy_root, output = sys.argv[1], sys.argv[2]
    generator = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(1, 128, *GRID, generator=generator)
    context = torch.randn(1, TEXT_TOKENS, CONTEXT, generator=generator)
    refs = [torch.randn(1, 128, 2, 2, generator=generator), torch.randn(1, 128, 1, 2, generator=generator)]

    guided = build(comfy_root, flow_map=False, state=None)
    shared = {k: v.detach().clone() for k, v in guided.state_dict().items()}
    flowmap = build(comfy_root, flow_map=True, state=shared)

    tensors = {}
    for name, value in flowmap.state_dict().items():
        tensors[f"weights.{name}"] = value.half().contiguous()
    tensors["inputs.latent"] = packed(x)
    tensors["inputs.context"] = context.contiguous()
    tensors["inputs.ref0"] = packed(refs[0])
    tensors["inputs.ref1"] = packed(refs[1])
    tensors["guided.velocity"] = packed(forward(guided, x, 0.7, context))
    tensors["edit.velocity"] = packed(forward(guided, x, 0.7, context, refs))
    schedule = torch.tensor([1.0, 0.7, 0.3, 0.0])
    tensors["flowmap.velocity"] = packed(forward(flowmap, x, 0.7, context, sample_sigmas=schedule))
    txt_states, _ = guided.preprocess_conds(context)
    tensors["guided.txt_states"] = txt_states.contiguous()
    save_file(tensors, output)
    for key in sorted(k for k in tensors if not k.startswith("weights.")):
        t = tensors[key].float()
        print(f"{key}: {tuple(t.shape)} mean={t.mean():.6f} std={t.std():.6f}")


if __name__ == "__main__":
    main()
