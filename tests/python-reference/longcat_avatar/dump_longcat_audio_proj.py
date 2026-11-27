"""Reference dump for the LongCat-Video-Avatar audio projection (AudioProjModel, modules/avatar/blocks.py @ 58eb336).

Runs the pinned reference class on a tiny geometry with seeded random weights and inputs, and writes the weights, the inputs and the output as
raw F32 files plus meta.json (the names and shapes the C# parity test reads with LongCatReferenceDump.Load).

Run in the hartsy-ref:cpu image only (never the Windows host). Assemble the pinned source (longcat_video/ at 58eb336) into a scratch folder, then:

    python dump_longcat_audio_proj.py --repo /src --out /out/longcat_avatar_reference_tensors

Stubbed: diffusers (ConfigMixin and ModelMixin only; the projection uses nothing else from it). Nothing here loads weights from the checkpoint.
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch


def install_diffusers_stub():
    diffusers = types.ModuleType("diffusers")

    class ConfigMixin:
        pass

    class ModelMixin(torch.nn.Module):
        pass

    diffusers.ConfigMixin = ConfigMixin
    diffusers.ModelMixin = ModelMixin
    sys.modules["diffusers"] = diffusers


def load_blocks(repo):
    path = os.path.join(repo, "longcat_video", "modules", "avatar", "blocks.py")
    spec = importlib.util.spec_from_file_location("longcat_avatar_blocks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tensor(out, meta, name, array):
    array = np.ascontiguousarray(array.detach().cpu().numpy().astype(np.float32)) if torch.is_tensor(array) else np.ascontiguousarray(array.astype(np.float32))
    array.tofile(os.path.join(out, name + ".f32"))
    meta["tensors"][name] = list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Root of the pinned LongCat-Video source (contains longcat_video/).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    install_diffusers_stub()
    blocks = load_blocks(args.repo)

    # Tiny geometry: the DiT uses seq_len 5, seq_len_vf 5 + 4 - 1 = 8, 12 blocks, 768 channels, 512 intermediate, 768 output, 32 tokens.
    dims = dict(seq_len=5, seq_len_vf=8, blocks=3, channels=6, intermediate_dim=16, output_dim=10, context_tokens=4)
    first_frames, latter_frames = 1, 3

    torch.manual_seed(args.seed)
    model = blocks.AudioProjModel(**dims).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.2)

    first = torch.randn(1, first_frames, dims["seq_len"], dims["blocks"], dims["channels"])
    latter = torch.randn(1, latter_frames, dims["seq_len_vf"], dims["blocks"], dims["channels"])
    with torch.no_grad():
        out = model(first, latter)  # [1, F, context_tokens, output_dim]

    os.makedirs(args.out, exist_ok=True)
    meta = {"dims": dims, "frames": {"first": first_frames, "latter": latter_frames}, "tensors": {}}
    for key, value in model.state_dict().items():
        write_tensor(args.out, meta, "audio_proj." + key, value)
    write_tensor(args.out, meta, "inputs.first", first.reshape(first_frames, -1))
    write_tensor(args.out, meta, "inputs.latter", latter.reshape(latter_frames, -1))
    write_tensor(args.out, meta, "output", out.reshape(-1, dims["output_dim"]))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps({"output_shape": list(out.shape), "tensors": len(meta["tensors"])}))


if __name__ == "__main__":
    main()
