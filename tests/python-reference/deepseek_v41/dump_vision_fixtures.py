"""Dump the UNMODIFIED upstream vision tower and aligner (inference/vision.py) on a small seeded config into fixtures/vision_tower.json.

Records every parameter under its checkpoint key, upstream's 2-D rotary tables and a rotation of random vectors, then for several patch grids the stage
values of the tower (patch embedding, each block's output stream, the final norm) and of the aligner (the folded rows, the hidden activation, the output).
The grids are odd on purpose so the aligner has to zero-pad; the aligner is also run on its own over more grids. Float32 throughout. vision.py imports only
torch, so no tilelang kernel shim is needed.
Usage: python dump_vision_fixtures.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
"""
import argparse
import json
import os
import sys
import types

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))

CONFIG: dict = dict(vision_patch_size=2, vision_dim=32, vision_n_heads=2, vision_inter_dim=48, vision_n_layers=2, vision_rope_theta=10000.0,
                    vision_downsample_ratio=3, dim=24)
TOWER_GRIDS: list = [(5, 7), (4, 3), (1, 4)]
ALIGNER_GRIDS: list = [(2, 5), (4, 7), (7, 2), (3, 6)]
# (grid height, grid width, rope dim, theta, heads): rope dim is half a head, so 8 matches CONFIG and 32 is the real checkpoint's
ROPE_CASES: list = [(5, 7, 8, 10000.0, 2), (3, 5, 32, 10000.0, 1), (4, 1, 8, 500.0, 2)]


def flat(t: torch.Tensor) -> list:
    # nine significant digits round-trip a float32 exactly and keep the fixture small
    return [float(f"{float(v):.9g}") for v in t.detach().float().reshape(-1)]


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import vision

    torch.manual_seed(1357)
    torch.set_default_dtype(torch.float32)
    args = types.SimpleNamespace(**CONFIG)
    vit = vision.ViT(args)
    aligner = vision.Aligner(args)
    for module, prefix in ((vit, "vision."), (aligner, "aligner.")):
        for name, prm in module.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
                prm.data.copy_(1.0 + 0.1 * torch.randn_like(prm))
            elif name.endswith("bias"):
                prm.data.normal_(0, 0.1)
            else:
                prm.data.normal_(0, 0.25)
    vit.eval()
    aligner.eval()

    params = {f"vision.{n}": prm for n, prm in vit.named_parameters()}
    params.update({f"aligner.{n}": prm for n, prm in aligner.named_parameters()})

    taps: dict = {}
    vit.patch_embed.register_forward_hook(lambda _m, _i, out: taps.__setitem__("patch_embed", out))
    for index, block in enumerate(vit.blocks):
        block.register_forward_hook(lambda _m, _i, out, index=index: taps.__setitem__(f"block.{index}", out))
    vit.norm.register_forward_hook(lambda _m, _i, out: taps.__setitem__("norm", out))
    aligner.w1.register_forward_pre_hook(lambda _m, inp: taps.__setitem__("unfold", inp[0]))
    aligner.w2.register_forward_pre_hook(lambda _m, inp: taps.__setitem__("hidden", inp[0]))

    p_size = CONFIG["vision_patch_size"]
    tower_cases = []
    for n_h, n_w in TOWER_GRIDS:
        taps.clear()
        patches = torch.rand(n_h * n_w, 3, p_size, p_size) * 2 - 1
        with torch.inference_mode():
            features = vit(patches, n_h, n_w)
            out = aligner(features, n_h, n_w)
        stages = {k: flat(v) for k, v in taps.items()}
        tower_cases.append({"gridHeight": n_h, "gridWidth": n_w, "patches": flat(patches), "stages": stages, "output": flat(out)})

    aligner_cases = []
    for n_h, n_w in ALIGNER_GRIDS:
        taps.clear()
        features = torch.randn(n_h * n_w, CONFIG["vision_dim"])
        with torch.inference_mode():
            out = aligner(features, n_h, n_w)
        aligner_cases.append({"gridHeight": n_h, "gridWidth": n_w, "features": flat(features), "unfold": flat(taps["unfold"]),
                              "hidden": flat(taps["hidden"]), "output": flat(out)})

    rope_cases = []
    for n_h, n_w, rope_dim, theta, heads in ROPE_CASES:
        cos, sin = vision.get_vision_cos_sin(n_h, n_w, rope_dim, theta)
        x = torch.randn(n_h * n_w, heads, 2 * rope_dim)
        rope_cases.append({"gridHeight": n_h, "gridWidth": n_w, "ropeDim": rope_dim, "theta": theta, "heads": heads, "cos": flat(cos), "sin": flat(sin),
                           "x": flat(x), "rotated": flat(vision.apply_rotary(x, cos, sin))})

    # The fixture must be able to see the mistakes that matter: each mutation below has to move the output well past the C# tolerance.
    n_h, n_w = TOWER_GRIDS[0]
    patches = torch.tensor(tower_cases[0]["patches"]).reshape(n_h * n_w, 3, p_size, p_size)
    with torch.inference_mode():
        base = aligner(vit(patches, n_h, n_w), n_h, n_w)
        base_folded = taps["unfold"].clone()
        original_cos_sin, original_rotary = vision.get_vision_cos_sin, vision.apply_rotary

        def swapped(h, w, dim, theta):
            cos, sin = original_cos_sin(h, w, dim, theta)
            q = dim // 2
            return torch.cat([cos[..., q:], cos[..., :q]], -1), torch.cat([sin[..., q:], sin[..., :q]], -1)

        def interleaved(x, cos, sin):
            half = x.shape[-1] // 2
            c, s = cos.repeat_interleave(2, -1)[..., :x.shape[-1]], sin.repeat_interleave(2, -1)[..., :x.shape[-1]]
            x1, x2 = x[..., 0::2], x[..., 1::2]
            rot = torch.stack([-x2, x1], -1).flatten(-2)
            return x * c + rot * s

        mutations = {}
        vision.get_vision_cos_sin = swapped
        mutations["rows and columns swapped in the rotary"] = rel_diff(aligner(vit(patches, n_h, n_w), n_h, n_w), base)
        vision.get_vision_cos_sin = original_cos_sin
        vision.apply_rotary = interleaved
        mutations["interleaved instead of half-split rotation"] = rel_diff(aligner(vit(patches, n_h, n_w), n_h, n_w), base)
        vision.apply_rotary = lambda x, cos, sin: x
        mutations["no rotation"] = rel_diff(aligner(vit(patches, n_h, n_w), n_h, n_w), base)
        vision.apply_rotary = original_rotary
        assert rel_diff(aligner(vit(patches, n_h, n_w), n_h, n_w), base) == 0.0, "the upstream functions were not restored"
        # a kernel-major instead of channel-major fold: permute the folded rows' columns before the first projection
        c, r = CONFIG["vision_dim"], CONFIG["vision_downsample_ratio"]
        wrong = base_folded.reshape(-1, c, r * r).transpose(1, 2).reshape(base_folded.shape)
        mutations["kernel-major fold"] = rel_diff(aligner.w2(torch.nn.functional.gelu(aligner.w1(wrong))), base)
    for name, diff in mutations.items():
        print(f"sensitivity: {name}: output moves by relL2 {diff:.3f}")
        assert diff > 0.01, f"the fixture cannot see: {name}"

    out = {"config": CONFIG, "towerGrids": TOWER_GRIDS,
           "params": {n: flat(prm) for n, prm in params.items()}, "shapes": {n: list(prm.shape) for n, prm in params.items()},
           "ropeCases": rope_cases, "towerCases": tower_cases, "alignerCases": aligner_cases}
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "vision_tower.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
