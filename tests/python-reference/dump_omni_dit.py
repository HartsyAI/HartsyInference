"""Reference values for the Hunyuan3D-Omni DiT: upstream's own `HunYuanDiTPlain` on the CPU in float32 with the released `model/pytorch_model.bin`.

Runs `hy3dshape/models/denoisers/hunyuandit.py` (and `moe_layers.py`) of a checkout of Tencent-Hunyuan/Hunyuan3D-Omni (pinned in docs/Research/HUNYUAN3D_OMNI_ARCHITECTURE.md) with the
`model/config.json` parameters. The DiT is the 2.1 architecture with a longer cross-attention context: the context here is 1370 DINO-like tokens, then `--control` control tokens, then the
10 signal tokens (random values; the content does not matter for the DiT, the length does). Safe and dependency-light, like the other dumps:

* the `.bin` file is read by the restricted unpickler of `dump_paint21_unet.py` (nothing else in the file can run), never by `torch.load`;
* the model is built on the `meta` device and the 12.2 GB of float32 weights are adopted without a copy (`load_state_dict(assign=True)`), so the host holds the weights once;
* `timm`, `diffusers` and `einops` are not needed beyond the stand-ins below, and the package `__init__` files (which pull in training and rendering code) are not run.

    python -I dump_omni_dit.py --hy3dshape-src <checkout>/hy3dshape --folder <hunyuan3d-omni> --out <fixture.npz>

The fixture holds `cond` (the context), `noise` (the latent set), `sigmas`, `velocity_cond_<n>` for each sigma, `velocity_uncond_0` (the first 1370 context tokens zeroed, control and signal
tokens kept: the guidance branch of the Omni pipeline), the block outputs `block<L>` of the first forward, and `t_token` and `x_embedded`.
"""
import argparse
import importlib.util
import logging
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class _GELUProj(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):  # diffusers.models.attention.FeedForward, activation_fn="gelu"
    def __init__(self, dim, dropout=0.0, activation_fn="gelu", final_dropout=False, inner_dim=None, bias=True):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        self.net = nn.ModuleList([_GELUProj(dim, inner_dim, bias=bias), nn.Dropout(dropout), nn.Linear(inner_dim, dim, bias=bias)])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


def rearrange(t, pattern, **axes):
    """The one pattern upstream's DiT uses: 'b n h d -> b h n d' (and its inverse)."""
    left, right = [side.strip() for side in pattern.split("->")]
    names = left.split()
    return t.permute([names.index(n) for n in right.split()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hy3dshape-src", required=True, help="the hy3dshape package directory of the Omni checkout")
    parser.add_argument("--folder", required=True, help="the tencent/Hunyuan3D-Omni folder (model/config.json, model/pytorch_model.bin)")
    parser.add_argument("--out", required=True, help=".npz path")
    parser.add_argument("--sigmas", default="0.35,0.7")
    parser.add_argument("--control", type=int, default=7, help="control tokens between the DINO tokens and the 10 signal tokens")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    logging.disable(logging.CRITICAL)

    stub("timm"), stub("timm.models"), stub("timm.models.vision_transformer", PatchEmbed=None, Attention=None, Mlp=None)
    stub("diffusers"), stub("diffusers.models"), stub("diffusers.models.attention", FeedForward=FeedForward)
    stub("einops", rearrange=rearrange)
    for name, sub in (("hy3dshape", ""), ("hy3dshape.models", "models"), ("hy3dshape.models.denoisers", os.path.join("models", "denoisers"))):
        package = types.ModuleType(name)
        package.__path__ = [os.path.join(args.hy3dshape_src, sub)]
        sys.modules[name] = package

    class Timer:
        def __init__(self, name=None):
            pass

        def __call__(self, fn):
            return fn

    stub("hy3dshape.models.utils", logger=logging.getLogger("hy3dshape"), synchronize_timer=Timer, smart_load_model=None)
    from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain

    import json
    with open(os.path.join(args.folder, "model", "config.json"), encoding="utf-8") as handle:
        params = json.load(handle)["params"]
    reader = load_module(os.path.join(HERE, "dump_paint21_unet.py"), "dump_paint21_unet")
    # The reader of dump_paint21_unet.py copies every storage twice (bytes, then bytearray) and clones every tensor, which more than doubles the host memory of a 12 GB state dict while it loads. This version
    # reads each storage straight into its tensor (the unpickler is still the restricted one) and keeps views, so the weights exist once.
    def read_state(path):
        import zipfile
        with zipfile.ZipFile(path) as archive:
            root = archive.namelist()[0].split("/")[0]

            def storage(key, dtype, numel):
                tensor = torch.empty(numel, dtype=dtype)
                with archive.open(f"{root}/data/{key}") as handle:
                    view = memoryview(tensor.numpy()).cast("B")
                    done = 0
                    while done < len(view):
                        got = handle.readinto(view[done:done + (1 << 24)])
                        if not got:
                            raise EOFError(key)
                        done += got
                return tensor

            with archive.open(f"{root}/data.pkl") as handle:
                return reader.SafeUnpickler(handle, storage).load()

    reader._rebuild_tensor = lambda storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None: torch.as_strided(storage, tuple(size), tuple(stride), offset)

    start = time.time()
    state = read_state(os.path.join(args.folder, "model", "pytorch_model.bin"))
    print(f"read {len(state)} tensors in {time.time() - start:.0f} s", flush=True)
    with torch.device("meta"):
        dit = HunYuanDiTPlain(**params)
    print(dit.load_state_dict(state, strict=True, assign=True), flush=True)
    del state
    dit.eval()
    print(f"loaded in {time.time() - start:.0f} s", flush=True)

    out = {}
    generator = torch.Generator().manual_seed(args.seed)
    context = torch.randn(1, params["text_len"] + args.control + 10, params["context_dim"], generator=generator)
    latents = torch.randn(1, params["input_size"], params["in_channels"], generator=generator)
    out["cond"], out["noise"] = context.numpy(), latents.numpy()

    captured = {}
    for index in (0, 10, 15, 20):
        dit.blocks[index].register_forward_hook(lambda m, i, o, index=index: captured.__setitem__(f"block{index}", o.detach().numpy().copy()))
    dit.x_embedder.register_forward_hook(lambda m, i, o: captured.__setitem__("x_embedded", o.detach().numpy().copy()))
    dit.t_embedder.register_forward_hook(lambda m, i, o: captured.__setitem__("t_token", o.detach().numpy().copy()))

    sigmas = [float(s) for s in args.sigmas.split(",")]
    out["sigmas"] = np.array(sigmas, dtype=np.float32)
    for n, sigma in enumerate(sigmas):
        t0 = time.time()
        with torch.no_grad():
            v = dit(latents, torch.tensor([sigma]), {"main": context})
        out[f"velocity_cond_{n}"] = v.numpy()
        if n == 0:
            out.update(captured)
        print(f"sigma {sigma}: {time.time() - t0:.0f} s", flush=True)

    uncond = context.clone()
    uncond[:, : params["text_len"]] = 0
    t0 = time.time()
    with torch.no_grad():
        out["velocity_uncond_0"] = dit(latents, torch.tensor([sigmas[0]]), {"main": uncond}).numpy()
    print(f"uncond: {time.time() - t0:.0f} s; total {time.time() - start:.0f} s", flush=True)
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
