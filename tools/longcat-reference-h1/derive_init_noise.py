"""Derive the reference's initial noise for an ai2v run from its step-0 dump, so the engine can start from the same noise.

The reference saves call00_step00_latent (the latents after the scheduler step) and call00_step00_noise (the model output the
scheduler received), and call00_step0{0,1}_t (the timesteps). The pinned FlowMatchEuler step is
    x_next = x + (sigma_next - sigma) * output,  with sigma = timestep / 1000 (timesteps = sigmas * num_train_timesteps).
So for step 0:
    x_init[:, :, 1:] = latent0[:, :, 1:] - ((t1 - t0) / 1000) * noise0[:, :, 1:]
Frame 0 is the conditioning frame: the reference overwrites it (latents[:, :, :num_cond] = cond) and so does the engine, so
the frame-0 values of the derived file are never read. They are written as zeros.

Checks printed: the timestep spacing, and the mean and std of the derived noise frames (the seeded noise is about N(0, 1)).
The formula assumes sigma = t / 1000; confirm it against the reference's scheduler source (scheduling_flow_match_euler_discrete.py,
FlowMatchEulerDiscreteScheduler.set_timesteps and step) before using the file. Usage:
    python derive_init_noise.py <reference dump dir> <output.npy>
"""
import os
import sys

import numpy as np


def main(dump_dir, out_path):
    t0 = float(np.load(os.path.join(dump_dir, "call00_step00_t.npy")))
    t1 = float(np.load(os.path.join(dump_dir, "call00_step01_t.npy")))
    latent0 = np.load(os.path.join(dump_dir, "call00_step00_latent.npy")).astype(np.float64)
    noise0 = np.load(os.path.join(dump_dir, "call00_step00_noise.npy")).astype(np.float64)
    assert latent0.shape == noise0.shape and latent0.ndim == 5 and latent0.shape[0] == 1, (latent0.shape, noise0.shape)
    dt = (t1 - t0) / 1000.0
    # Precision of the saved tensors: a value whose low 16 float32 bits are zero is on the bf16 grid; the step casts its result to the
    # model output's dtype, so a half-precision noise_pred leaves latent0 on that grid and the derived init inherits the rounding.
    for name, arr in (("latent0", latent0), ("noise0", noise0)):
        bits = arr.astype(np.float32).view(np.uint32)
        print(f"{name}: share on bf16 grid {np.mean((bits & 0xFFFF) == 0):.4f}")
    x = np.zeros_like(latent0)
    x[:, :, 1:] = latent0[:, :, 1:] - dt * noise0[:, :, 1:]
    frames = x[:, :, 1:]
    print(f"t0={t0!r} t1={t1!r} dt={dt!r} shape={x.shape}")
    print(f"rounding bound: |derived - exact| <= |dt| * |noise0| * eps_noise + eps_latent * |latent0|, eps = 2^-8 for bf16 (about 4e-3 relative)")
    print(f"noise frames 1..: mean={frames.mean():.4f} std={frames.std():.4f} (seeded noise is about 0 and 1)")
    if not (abs(frames.mean()) < 0.05 and 0.9 < frames.std() < 1.1):
        raise SystemExit("derived noise is not N(0, 1)-like: the sigma formula or the dump is wrong; do not use this file")
    np.save(out_path, x.astype(np.float32))
    print(f"wrote {out_path} as float32 [1, C, T, H, W]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
