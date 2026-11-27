"""Per-step dump for the reference denoise loops (H1_STEP_DUMP_DIR). Off unless the variable is set; read-only when on.

Each denoise call starts with step index 0 and gets the next call number n. For each step it writes, as float32 .npy:
  call{n:02d}_step{i:02d}_t.npy      the timestep (float)
  call{n:02d}_step{i:02d}_noise.npy  the model output noise_pred, after the sign flip the pipeline applies before the scheduler
  call{n:02d}_step{i:02d}_latent.npy the latents after the scheduler step
Tensors are copied to the host only when the variable is set, so the default path does no work and no sync.
"""

import os

import numpy as np

_CALL = [-1]


def maybe_dump(i, t, noise_pred, latents):
    directory = os.environ.get("H1_STEP_DUMP_DIR")
    if not directory:
        return
    if i == 0:
        _CALL[0] += 1
    os.makedirs(directory, exist_ok=True)
    tag = f"call{max(_CALL[0], 0):02d}_step{i:02d}"
    t_value = float(t.item()) if hasattr(t, "item") else float(t)
    np.save(os.path.join(directory, f"{tag}_t.npy"), np.asarray(t_value, dtype=np.float32))
    np.save(os.path.join(directory, f"{tag}_noise.npy"), noise_pred.detach().float().cpu().numpy())
    np.save(os.path.join(directory, f"{tag}_latent.npy"), latents.detach().float().cpu().numpy())
