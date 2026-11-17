"""Shared setup for the Idol / Budget3D reference dumps (CPU, float32, upstream code imported from clones, never copied).

Environment:
  IDOL_ROOT       clone of github.com/speridlabs/idol-sdk at 956439f5e664a7e8c5db25a8055e72d6365aafa5
  BUDGET3D_ROOT   clone of github.com/BudgetAny/Budget3D at 95bbeab46063375d560f9b20fa7b9d9fedb078f7
  IDOL_MODELS     the engine's Models/3D folder (holds idol/, idol-turbo/, budget3d-idol/, budget3d-idol-turbo/)
"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import numpy as np
import torch
from safetensors.torch import load_file, save_file

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
torch.backends.cudnn.enabled = False

IDOL_ROOT = os.environ["IDOL_ROOT"]
BUDGET3D_ROOT = os.environ["BUDGET3D_ROOT"]
MODELS = os.environ["IDOL_MODELS"]
for p in (IDOL_ROOT, BUDGET3D_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

IDOL_DIR = os.path.join(MODELS, "idol")
POS_PE_PATH = os.path.join(IDOL_DIR, "pos_pe_f32_1x8192x3.bin")
POS_PE_SHA256 = "884df4db664a680c100d032c9855397ced6afa28f8984efbee6b77f6ba3efe6b"


def idol_config():
    with open(os.path.join(IDOL_DIR, "speridlabs", "config.json")) as f:
        return json.load(f)


def pos_pe():
    """Idol's 8192 Sobol positions [1, 8192, 3]; equal to SobolEngine(3, scramble=True, seed=123).draw(8192)."""
    raw = open(POS_PE_PATH, "rb").read()
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == POS_PE_SHA256, "pos_pe file does not match its pin"
    return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).reshape(1, 8192, 3)


def f32(state):
    return {k: v.float() for k, v in state.items()}


def strip(state, prefix):
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def set_fp32(model):
    """Undo Idol's fp16 cast of the block Linears: everything float32 on the CPU."""
    model.float()
    for m in model.modules():
        if hasattr(m, "dtype") and isinstance(getattr(m, "dtype"), torch.dtype):
            m.dtype = torch.float32
    return model


def idol_denoiser(state):
    """Idol's LatentSeqMMFlowModel, float32, with ``state`` (native keys, no pos_pe) and the pinned pos_pe."""
    from idol.models.dit import LatentSeqMMFlowModel
    model = set_fp32(LatentSeqMMFlowModel(**idol_config()["denoiser"]["args"]))
    state = dict(f32(state))
    state["pos_pe"] = pos_pe()
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def budget3d_model(ckpt_path, turbo=False):
    """Budget3DModel (float32, adapters as released) loaded from a BudgetAny file; turbo injects the phase LoRA first."""
    from budget3d.models.denoiser import Budget3DModel
    from budget3d.models.load import ADAPTER_DEFAULTS
    args = {**ADAPTER_DEFAULTS, **idol_config()["denoiser"]["args"]}
    model = Budget3DModel(**args)
    model.bf16_blocks = False
    for m in model.modules():
        if hasattr(m, "bf16_blocks"):
            m.bf16_blocks = False
    state = f32(load_file(ckpt_path))
    if turbo:
        from budget3d.models.turbo import PhaseState, bind_phase, inject
        st = PhaseState(allow_uniform_fastpath=True)
        inject(model, st)
        bind_phase(model, st)
    with torch.no_grad():
        model.pos_pe.copy_(pos_pe())
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:6]
    bad = [k for k in missing if k not in ("pos_pe", "perm_to_octree", "inv_perm", "cube_to_cells", "cell_ctr", "cube_ctr")]
    assert not bad, bad[:6]
    return model.float().bind_geometry().eval().requires_grad_(False)


def save(out_dir, name, tensors, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    # clone: safetensors refuses tensors that share storage (a view and its source, e.g. a transposed copy)
    tensors = {k: (v.detach().contiguous().float() if v.is_floating_point() else v.detach().contiguous()).clone()
               for k, v in tensors.items()}
    save_file(tensors, os.path.join(out_dir, name + ".safetensors"),
              metadata={k: str(v) for k, v in (meta or {}).items()})
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in tensors.items()), flush=True)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t
        print(f"[{self.label}] {self.seconds:.1f} s", flush=True)
