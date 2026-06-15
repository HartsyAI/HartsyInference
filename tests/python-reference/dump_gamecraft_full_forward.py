"""
Dumps a Hunyuan-GameCraft forward pass (camera/Plücker → CameraNet → 33-ch composite → HunyuanVideo MM-DiT)
for a deterministic synthetic input. Source of truth for the C# layer-by-layer diff (the project's 🔧→✅ loop).

Output: tests/python-reference/gamecraft_reference_tensors/full_forward/
  inputs/composite.bin       [1, 33, T_lat, H_lat, W_lat] F32  (noisy16 + history16 + mask1)
  inputs/txt.bin             [1, L, 4096]  F32  (Llava token-refiner output upstream)
  inputs/pooled.bin          [1, 768]      F32  (CLIP-L pooled)
  inputs/camera_tokens.bin   [1, S_img, 3072] F32  (CameraNet output added to img tokens)
  inputs/timestep.bin        [1]           F32
  layers/double_block_<i>.bin  [1, S_img, 3072] F32  (per dual-stream block image output)
  layers/single_block_<i>.bin  [1, S_img, 3072] F32
  output_velocity.bin          [1, 16, T_lat, H_lat, W_lat] F32
  index.json

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_gamecraft_full_forward.py

NOTE: the reference module + the original→diffusers block key remap are VALIDATION-GATED. Fill in CKPT_PATH and
the TODOs against the real `tencent/Hunyuan-GameCraft-1.0` weights (load via the official repo, or via the C#
PytorchPickleLoader + a torch shim) when running the numeric pass; the dump layout below is what
diff_gamecraft_layers.py and the C# Hunyuan-GameCraft debug dump align to.
"""
import os
import json
import numpy as np
import torch

# TODO[VG]: local GameCraft checkpoint (mp_rank_00_model_states.pt) + its config.
CKPT_PATH = "/home/kalebbroo/Desktop/Projects/HartsyInference/Models/Interactive/Hunyuan-GameCraft-1.0"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "tests/python-reference/gamecraft_reference_tensors/full_forward")
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
    raise SystemExit(
        "Fill in the GameCraft reference construction (TODO[VG]) before running. The dump/index layout this "
        "script writes already matches the C# HunyuanVideoDit debug tags (double_block_<i>, single_block_<i>, "
        "camera_tokens, output_velocity)."
    )
    # Reference shape (unreachable until TODO filled):
    # model = load_gamecraft(CKPT_PATH)                      # HYVideoDiffusionTransformer + camera_in
    # composite, txt, pooled, plucker, t = synth_inputs()
    # cam = model.camera_in(plucker); save("camera_tokens", cam, "inputs")
    # save("composite", composite, "inputs"); save("txt", txt, "inputs"); save("pooled", pooled, "inputs")
    # hooks dump double_blocks/single_blocks image outputs
    # v = model(composite, txt, pooled, t, cam); save("output_velocity", v, ".")
    # json.dump(index, open(os.path.join(OUT_DIR, "index.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
