"""Tiny glm5_next vision tower reference: a random tower in the real `model.visual.*` key layout and its outputs on saved patch rows.

Offline only. Needs CPU torch, safetensors, numpy and transformers from the pinned commit (see glm5_next_reference.py):

    python tools/parity/glm5_next_vision_reference.py --out tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/tiny_vision

Written to --out: config.json (vision_config only), model.safetensors (model.visual.* keys), patches.npy (float32 [N, 3*T*P*P], the processor's
flattened patch rows in merge-block order), grid_thw.npy (int32 [G, 3]), tower.npy (float32 [N / merge^2, out_hidden]: the tower's downsampled rows),
merged.npy (the merger output: the rows that replace the image placeholders), meta.json (including the rope_parameters transformers filled in).
The grids are one image (t 1) and one two-frame clip (t 2), so per-frame attention segments and the repeated positions are both exercised.
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

PIN = "52bac5d0e0da56e26289c072f306877ff5870262"
GRIDS = [(1, 4, 6), (2, 4, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import transformers
    from transformers.models.glm5_next import modeling_glm5_next as m
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig

    gen = torch.Generator().manual_seed(args.seed)
    cfg = Glm5NextVisionConfig(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, patch_size=14, temporal_patch_size=2,
                               spatial_merge_size=2, out_hidden_size=24, projection_intermediate_size=48, swiglu_limit=0.5)
    cfg._attn_implementation = "eager"
    tower = m.Glm5NextVisionModel(cfg).float().eval()
    with torch.no_grad():
        for name, p in tower.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * (0.35 if p.ndim > 1 else 0.2))

    grid = torch.tensor(GRIDS, dtype=torch.long)
    n = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    patch_dim = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    patches = torch.randn(n, patch_dim, generator=gen)
    with torch.no_grad():
        out = tower(patches, grid_thw=grid)

    save_file({"model.visual." + k: v.detach().float().contiguous() for k, v in tower.state_dict().items()}, os.path.join(args.out, "model.safetensors"))
    vision = {k: v for k, v in cfg.to_dict().items() if k not in ("rope_parameters", "architectures", "transformers_version")}
    vision["model_type"] = "glm5_next_vision"
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"vision_config": vision}, f, indent=1)
    np.save(os.path.join(args.out, "patches.npy"), patches.numpy())
    np.save(os.path.join(args.out, "grid_thw.npy"), grid.numpy().astype(np.int32))
    np.save(os.path.join(args.out, "tower.npy"), out.last_hidden_state.numpy())
    np.save(os.path.join(args.out, "merged.npy"), out.pooler_output.numpy())
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"transformers": transformers.__version__, "pinned_commit": PIN, "seed": args.seed, "grids": GRIDS,
                   "rope_parameters": cfg.rope_parameters}, f, indent=1)
    print("rope_parameters:", cfg.rope_parameters, "tower", tuple(out.last_hidden_state.shape), "merged", tuple(out.pooler_output.shape))


if __name__ == "__main__":
    main()
