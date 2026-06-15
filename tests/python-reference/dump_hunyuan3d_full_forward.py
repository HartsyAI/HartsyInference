"""
Dumps a Hunyuan3D-2 shape forward pass (DINOv2 conditioning → flow-match DiT → ShapeVAE occupancy) for a
deterministic synthetic input. Source of truth for the C# layer-by-layer diff (the project's 🔧→✅ loop).

Output: tests/python-reference/hunyuan3d_reference_tensors/full_forward/
  inputs/image.bin              [1, 3, H, W]      F32   (preprocessed conditioning image, seed 42)
  inputs/cond_tokens.bin        [1, S_img, Dcond] F32   (DINOv2 last_hidden_state)
  inputs/latent.bin             [1, N, C]         F32   (initial VecSet noise, seed 42)
  inputs/timestep.bin           [1]               F32
  layers/dit_block_<i>.bin      [1, N, Width]     F32   (per-DiT-block output)
  layers/vae_occupancy.bin      [1, P]            F32   (occupancy at a fixed probe point set)
  output_velocity.bin           [1, N, C]         F32
  index.json                    ordered [{name, file, shape}]

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_hunyuan3d_full_forward.py

NOTE: the exact reference module + key layout are VALIDATION-GATED. Fill in CKPT_PATH and the two TODOs
(model construction + forward hooks) against the downloaded `tencent/Hunyuan3D-2` checkpoint when running
the numeric pass; the dump layout below is what diff_hunyuan3d_layers.py expects.
"""
import os
import json
import numpy as np
import torch

# TODO[VG]: point at the local Hunyuan3D-2 checkpoint dir (DiT + ShapeVAE + DINOv2).
CKPT_PATH = "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/3D/Hunyuan3D-2"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "tests/python-reference/hunyuan3d_reference_tensors/full_forward")
os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)

index = []


def save(name, t, subdir="layers"):
    arr = t.float().detach().cpu().contiguous().numpy()
    rel = f"{subdir}/{name}.bin"
    arr.tofile(os.path.join(OUT_DIR, rel))
    index.append({"name": name, "file": rel, "shape": list(arr.shape)})


def main():
    torch.manual_seed(42)

    # TODO[VG]: construct the reference pipeline from CKPT_PATH, e.g. the hy3dgen repo's
    #   from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    #   pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(CKPT_PATH)
    # then expose: pipe.conditioner (DINOv2), pipe.model (DiT), pipe.vae (ShapeVAE).
    raise SystemExit(
        "Fill in the reference model construction (TODO[VG]) before running. "
        "The dump/index layout this script writes is already aligned with the C# Hunyuan3DDebugDump tags."
    )

    # --- Reference shape, kept as documentation of the intended dump (unreachable until TODO filled) ---
    # img = preprocess(load_test_image())                      # [1,3,H,W]
    # save("image", img, "inputs")
    # cond = pipe.conditioner(img)                             # [1, S_img, Dcond]
    # save("cond_tokens", cond, "inputs")
    # latent = torch.randn(1, N, C)                            # VecSet noise
    # save("latent", latent, "inputs")
    # t = torch.tensor([0.5]); save("timestep", t, "inputs")
    # hooks: for i, blk in enumerate(pipe.model.blocks):
    #            blk.register_forward_hook(lambda m, i_, o, i=i: save(f"dit_block_{i}", o))
    # v = pipe.model(latent, cond, t); save("output_velocity", v, ".")
    # occ = pipe.vae.decode_points(latent, FIXED_PROBE_POINTS)  # [1, P]
    # save("vae_occupancy", occ)
    # with open(os.path.join(OUT_DIR, "index.json"), "w") as f: json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
