"""Reference values for the Hunyuan3D-Paint 2.1 UNet, produced by upstream's own modules on the CPU.

Runs `hunyuanpaintpbr/unet/{modules,attn_processor}.py` of a checkout of Tencent-Hunyuan/Hunyuan3D-2.1 (`hy3dpaint`, pinned in
docs/Research/HUNYUAN3D_PAINT_ARCHITECTURE.md) with the released UNet and VAE weights on fixed random inputs. Three things differ from a normal
run, all to keep the dump safe, CPU-only and dependency-free:

* The `.bin` files are read by a restricted unpickler that allows only `OrderedDict` and torch's tensor rebuild call (nothing else in the file
  can be executed), not by `torch.load`.
* The one `.to("cuda:0")` in the material attention processor is removed from the source text before it is executed (a no-op on the CPU).
* `einops` is not installed here; `rearrange` is replaced by a small implementation of the patterns upstream uses. The package `__init__`
  files (which import the whole pipeline) are skipped: only `unet/modules.py` and `unet/attn_processor.py` are loaded.

    python dump_paint21_unet.py --upstream <hy3dpaint checkout>/hunyuanpaintpbr --folder <hunyuan3d-paint21> --out <fixture.safetensors>

The fixture holds the inputs and these outputs (names are the tensor keys): `dino_proj`, `ref.<block>` (the features the reference stream
writes), `out_full` (reference scale 1, DINO on) and `out_uncond` (reference scale 0, DINO off), each `[materials * views, 4, H, H]`,
plus the VAE decode of `vae_latent`. Times are printed.

`--encode-image <png>` adds the VAE ENCODE that the pipeline runs on the reference image and on the normal and position maps (`encode_images` of
`hunyuanpaintpbr/pipeline.py`): `enc_image` (the picture resized to the view size and the first position map, `[2, 3, S, S]` in [0, 1]), `enc_mean` and
`enc_std` (the posterior's mean and standard deviation times the scaling factor). Upstream draws `posterior.sample()`, that is `mean + std * noise`; the
C# encoder returns the mean, so `enc_std` is also the size of the noise upstream adds.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # upstream code touches "cuda" in places; this reference is CPU only

import argparse
import collections
import importlib.util
import json
import math
import pickle
import re
import sys
import time
import types
import zipfile

import torch

DTYPES = {"FloatStorage": torch.float32, "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16}


class SafeUnpickler(pickle.Unpickler):
    """Allows only what a flat torch state_dict needs; any other global raises instead of running."""

    def __init__(self, handle, storages):
        super().__init__(handle)
        self._storages = storages

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in DTYPES:
            return name
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid
        if kind != "storage" or storage_type not in DTYPES:
            raise pickle.UnpicklingError(f"unsupported storage {pid[:2]}")
        return self._storages(key, DTYPES[storage_type], numel)


def _rebuild_tensor(storage, offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return torch.as_strided(storage, tuple(size), tuple(stride), offset).clone()


def read_state_dict(path):
    with zipfile.ZipFile(path) as archive:
        root = archive.namelist()[0].split("/")[0]

        def storage(key, dtype, numel):
            raw = archive.read(f"{root}/data/{key}")
            return torch.frombuffer(bytearray(raw), dtype=dtype, count=numel)

        with archive.open(f"{root}/data.pkl") as handle:
            return SafeUnpickler(handle, storage).load()


def rearrange(x, pattern, **axes):
    """The subset of einops.rearrange that upstream's UNet uses: names, parenthesised groups, no ellipsis."""
    left, right = [side.strip() for side in pattern.split("->")]

    def parse(side):
        groups = []
        for match in re.finditer(r"\(([^)]*)\)|(\S+)", side):
            groups.append(match.group(1).split() if match.group(1) is not None else [match.group(2)])
        return groups

    lgroups, rgroups = parse(left), parse(right)
    sizes = dict(axes)
    for group, dim in zip(lgroups, x.shape):
        unknown = [n for n in group if n not in sizes]
        known = math.prod(sizes[n] for n in group if n in sizes)
        if len(unknown) > 1:
            raise ValueError(f"cannot infer {unknown} in {pattern}")
        if unknown:
            sizes[unknown[0]] = dim // known
    flat_left = [n for g in lgroups for n in g]
    t = x.reshape([sizes[n] for n in flat_left])
    flat_right = [n for g in rgroups for n in g]
    t = t.permute([flat_left.index(n) for n in flat_right])
    return t.reshape([math.prod(sizes[n] for n in g) for g in rgroups])


def load_upstream(root):
    """Loads attn_processor.py and modules.py as hunyuanpaintpbr.unet.* without running the package __init__ files."""
    einops = types.ModuleType("einops")
    einops.rearrange = rearrange
    sys.modules["einops"] = einops
    for name in ("hunyuanpaintpbr", "hunyuanpaintpbr.unet"):
        package = types.ModuleType(name)
        package.__path__ = [os.path.join(root, *name.split(".")[1:])]
        sys.modules[name] = package
    modules = {}
    for name in ("attn_processor", "modules"):
        full = f"hunyuanpaintpbr.unet.{name}"
        path = os.path.join(root, "unet", f"{name}.py")
        source = open(path, encoding="utf-8").read()
        # The material-wise attention moves its input to "cuda:0" unconditionally (attn_processor.py, PoseRoPE-free MDA path); on the CPU it is a no-op.
        source = source.replace('.to("cuda:0")', "")
        module = types.ModuleType(full)
        module.__file__ = path
        module.__package__ = "hunyuanpaintpbr.unet"
        sys.modules[full] = module
        exec(compile(source, path, "exec"), module.__dict__)
        modules[name] = module
    return modules["modules"]


def modern_vae_name(key):
    """The VAE pickle uses the pre-0.14 diffusers names for the mid-block attention; current diffusers expects to_q, to_k, to_v and to_out.0."""
    if ".attentions." not in key:
        return key
    for old, new in {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.", ".proj_attn.": ".to_out.0."}.items():
        key = key.replace(old, new)
    return key


def position_maps(views, size):
    """Position maps in [0, 1] on a unit ball seen from `views` azimuths; the background is exactly 1 (white), as upstream's masks expect."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    r2 = xs**2 + ys**2
    inside = r2 < 0.8**2
    z = torch.sqrt((1 - r2 / 0.8**2).clamp(min=0))
    maps = []
    for v in range(views):
        a = 2 * math.pi * v / views
        px = xs / 0.8 * math.cos(a) + z * math.sin(a)
        pz = -xs / 0.8 * math.sin(a) + z * math.cos(a)
        pos = torch.stack([px, -ys / 0.8, pz], dim=0) * 0.5 + 0.5
        pos = (pos.clamp(0, 0.98) * 255).round() / 255
        pos = torch.where(inside.unsqueeze(0), pos, torch.ones_like(pos))
        maps.append(pos)
    return torch.stack(maps).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--timestep", type=float, default=500.0)
    parser.add_argument("--aux-only", action="store_true", help="skip the UNet; dump only the voxel indices and the VAE decode")
    parser.add_argument("--encode-image", help="a picture to encode with the VAE (also dumps the posterior of the first position map)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from safetensors.torch import save_file

    modules = load_upstream(args.upstream)
    config = json.load(open(os.path.join(args.folder, "unet", "config.json")))
    h, n, materials = args.latent, args.views, 2
    out = {}
    pos_maps = position_maps(n, h * 8)
    if not args.aux_only:
        t0 = time.time()
        unet = UNet2DConditionModel.from_config(config)
        wrapped = modules.UNet2p5DConditionModel(unet)
        wrapped.unet.conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, kernel_size=3, padding=1)
        state = {k: v.float() for k, v in read_state_dict(os.path.join(args.folder, "unet", "diffusion_pytorch_model.bin")).items()}
        missing, unexpected = wrapped.load_state_dict(state, strict=False)
        print(f"loaded UNet in {time.time() - t0:.1f} s: {len(state)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
        if missing or unexpected:
            # A partly loaded upstream model would produce a quietly wrong oracle.
            raise SystemExit(f"UNet weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        wrapped.eval()

        sample = torch.randn(1, materials, n, 4, h, h)
        normal = torch.randn(1, n, 4, h, h)
        position = torch.randn(1, n, 4, h, h)
        dino = torch.randn(1, 257, 1536)
        ref_latents = torch.randn(1, 1, 4, h, h)
        text = torch.stack([wrapped.unet.learned_text_clip_albedo, wrapped.unet.learned_text_clip_mr], dim=0).unsqueeze(0)  # [1, 2, 77, 1024]
        out.update(sample=sample, normal=normal, position=position, position_maps=pos_maps, dino=dino, ref_latents=ref_latents)
        t = torch.tensor(args.timestep)

        def run(ref_scale, use_dino):
            cond = {
                "embeds_normal": normal,
                "embeds_position": position,
                "position_maps": pos_maps.clone(),
                "dino_hidden_states": dino if use_dino else torch.zeros_like(dino),
                "ref_latents": ref_latents,
                "ref_scale": ref_scale,
                "mva_scale": 1.0,
                "cache": {},  # the forward fills this dict (voxel indices, projected DINO tokens, reference features)
            }
            start = time.time()
            result = wrapped(sample, t, text, **cond)[0]
            print(f"forward (ref_scale {ref_scale}, dino {use_dino}): {time.time() - start:.1f} s")
            return result, cond

        full, cond = run(1.0, True)
        out["out_full"] = full
        out["dino_proj"] = cond["cache"]["dino_hidden_states_proj"].reshape(-1)
        for name, tensor in cond["cache"]["condition_embed_dict"].items():
            out[f"ref.{name}"] = tensor.reshape(-1)
        uncond, _ = run(0.0, False)
        out["out_uncond"] = uncond

    # Upstream's own voxel indices for the same position maps (the half-precision pooling included).
    voxel = modules.calc_multires_voxel_idxs(pos_maps.clone(), grid_resolutions=[h, h // 2, h // 4, h // 8], voxel_resolutions=[h * 8, h * 4, h * 2, h])
    for tokens, entry in voxel.items():
        out[f"voxel.{tokens}"] = entry["voxel_indices"].float()
    out["position_maps"] = pos_maps
    vae = AutoencoderKL.from_config(json.load(open(os.path.join(args.folder, "vae", "config.json"))))
    vae_state = {k: v.float() for k, v in read_state_dict(os.path.join(args.folder, "vae", "diffusion_pytorch_model.bin")).items()}
    # The pickle uses the pre-0.14 diffusers names for the mid-block attention.
    vae_state = {modern_vae_name(k): v for k, v in vae_state.items()}
    vae_missing, vae_unexpected = vae.load_state_dict(vae_state, strict=False)
    if vae_missing or vae_unexpected:
        raise SystemExit(f"VAE weights do not match: missing {vae_missing[:5]}, unexpected {vae_unexpected[:5]}")
    print("vae load: all keys matched")
    vae.eval()
    vae_latent = torch.randn(2, 4, h, h)
    start = time.time()
    decoded = vae.decode(vae_latent / vae.config.scaling_factor).sample
    print(f"vae decode: {time.time() - start:.1f} s, scaling {vae.config.scaling_factor}")
    out["vae_latent"] = vae_latent
    out["vae_decoded"] = decoded
    if args.encode_image:
        import numpy as np
        from PIL import Image

        size = h * 8
        picture = Image.open(args.encode_image).convert("RGB").resize((size, size), Image.BICUBIC)
        reference = torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0).permute(2, 0, 1)
        batch = torch.stack([reference, pos_maps[0, 0]])
        start = time.time()
        posterior = vae.encode((batch - 0.5) * 2.0).latent_dist
        scaling = vae.config.scaling_factor
        print(f"vae encode: {time.time() - start:.1f} s; posterior std / |mean| (scaled) = {(posterior.std * scaling).abs().mean() / (posterior.mean * scaling).abs().mean():.4f}")
        out["enc_image"] = batch
        out["enc_mean"] = posterior.mean * scaling
        out["enc_std"] = posterior.std * scaling
    save_file({k: v.detach().float().contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items() if not k.startswith("ref.")})


if __name__ == "__main__":
    main()
