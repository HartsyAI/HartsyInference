"""Reference dump for the SeedVR2 preprocessing chain (Part A2 parity).

Runs ByteDance's OWN transform classes (imported from a SeedVR checkout) — AreaResize(bicubic,
antialias default) -> clamp -> DivisibleCrop(16) -> Normalize(0.5,0.5) -> cut_videos padding —
on seeded synthetic uint8 video tensors, and saves input/output pairs to one safetensors file.

Usage:
    <seedvr2-venv-python> dump_seedvr2_preprocess_reference.py <path-to-SeedVR-checkout> <out.safetensors>

The C# side (SeedVr2PreprocessParityTests, env var SEEDVR2_PRE_REF) reads the dump. CPU, fp32,
deterministic. Sizes cover: upscale (the dominant restore path), downscale (>720p-area input),
non-divisible dims (DivisibleCrop both-side raggedness), odd frame counts (cut_videos padding),
t==1 stills, and a tiny input (bicubic border handling dominates).
"""
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, sys.argv[1])
from data.image.transforms.area_resize import AreaResize  # noqa: E402
from data.image.transforms.divisible_crop import DivisibleCrop  # noqa: E402

RES_H, RES_W = 1280, 720  # inference_seedvr2_3b.py defaults; max_area = res_h * res_w

# (frames, height, width) in pixels, uint8. Names must stay stable — C# keys off them.
CASES = [
    ("up_360p_t5", 5, 360, 640),
    ("up_240p_t1", 1, 240, 320),
    ("up_odd_t7", 7, 480, 704),
    ("down_1080p_t5", 5, 1082, 1920),
    ("tiny_t3", 3, 37, 53),
    ("nondiv_t9", 9, 721, 1279),
    ("exact_720p_t5", 5, 720, 1280),
]


def cut_videos(videos: torch.Tensor) -> torch.Tensor:
    """Inference-script padding (sp_size=1): repeat last frame until (t-1) % 4 == 0."""
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4:
        padding = [videos[:, -1].unsqueeze(1)] * (4 - t + 1)
        return torch.cat([videos, *padding], dim=1)
    if (t - 1) % 4 == 0:
        return videos
    padding = [videos[:, -1].unsqueeze(1)] * (4 - ((t - 1) % 4))
    return torch.cat([videos, *padding], dim=1)


def main() -> None:
    resize = AreaResize(max_area=float(RES_H * RES_W), downsample_only=False)
    crop = DivisibleCrop((16, 16))
    out = {}
    gen = torch.Generator().manual_seed(20260801)
    for name, t, h, w in CASES:
        video_u8 = torch.randint(0, 256, (t, 3, h, w), generator=gen, dtype=torch.uint8)
        video = video_u8.float() / 255.0                       # read_video convention, TCHW
        x = resize(video)
        x = torch.clamp(x, 0.0, 1.0)
        x = crop(x)
        x = (x - 0.5) / 0.5                                    # Normalize(0.5, 0.5)
        x = x.permute(1, 0, 2, 3).contiguous()                 # t c h w -> c t h w
        x = cut_videos(x)
        out[f"{name}.input"] = video_u8
        out[f"{name}.output"] = x.float()
        print(f"{name}: in ({t},3,{h},{w}) -> out {tuple(x.shape)}")
    save_file(out, sys.argv[2])
    print(f"saved {len(out)} tensors -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
