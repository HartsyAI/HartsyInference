"""Dump ACE-Step 1.5 CFG guidance references — APG (momentum, 4-call sequence), ADG, and DCW —
using the UPSTREAM modules verbatim (cloned at /tmp/acestep15), so the C# AceStep15Guidance port is
verified against the real implementation, not a re-derivation. Pure math on fixed random tensors:
no checkpoint, no GPU, tiny memory.

Writes .bin files (same <i32 ndim><i32 dims...><f32 data> layout as acestep15_ref) to
/tmp/ace15cfg; copy into tests/python-reference/acestep15_ref/ (cfg_*.bin).
"""
import os, struct, sys
import numpy as np
import torch

sys.path.insert(0, '/tmp/acestep15')
from acestep.models.common.apg_guidance import apg_forward, adg_forward, cfg_forward, MomentumBuffer
from acestep.models.common.dcw_correction import DCWCorrector

OUT = '/tmp/ace15cfg'
os.makedirs(OUT, exist_ok=True)

def w(name, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(f'{OUT}/cfg_{name}.bin', 'wb') as f:
        f.write(struct.pack('<i', arr.ndim))
        for s in arr.shape: f.write(struct.pack('<i', s))
        f.write(arr.tobytes())

rng = np.random.RandomState(4242)
T, C = 40, 64
G = 7.0

# ── APG: 4 sequential guided steps sharing one MomentumBuffer (the momentum path is stateful) ──
mb = MomentumBuffer()
apg_conds, apg_unconds, apg_outs = [], [], []
for k in range(4):
    cond = torch.tensor(rng.randn(1, T, C).astype(np.float32))
    uncond = torch.tensor(rng.randn(1, T, C).astype(np.float32) * 0.9)
    out = apg_forward(pred_cond=cond, pred_uncond=uncond, guidance_scale=G, momentum_buffer=mb, dims=[1])
    apg_conds.append(cond.numpy()[0]); apg_unconds.append(uncond.numpy()[0]); apg_outs.append(out.numpy()[0])
w('apg_cond', np.stack(apg_conds)); w('apg_uncond', np.stack(apg_unconds)); w('apg_out', np.stack(apg_outs))

# ── ADG: single call at t=0.6 ──
xt = torch.tensor(rng.randn(1, T, C).astype(np.float32))
c2 = torch.tensor(rng.randn(1, T, C).astype(np.float32))
u2 = torch.tensor(rng.randn(1, T, C).astype(np.float32) * 0.8)
adg = adg_forward(latents=xt, noise_pred_cond=c2, noise_pred_uncond=u2, sigma=0.6, guidance_scale=G)
w('adg_xt', xt.numpy()[0]); w('adg_cond', c2.numpy()[0]); w('adg_uncond', u2.numpy()[0]); w('adg_out', adg.numpy()[0])

# ── plain CFG ──
w('cfgf_out', cfg_forward(c2, u2, G).numpy()[0])

# ── DCW: modes double/low/high at t=0.75 (even T) and odd T=41 for the padding path ──
for tag, T2 in (('even', 40), ('odd', 41)):
    x = torch.tensor(rng.randn(1, T2, C).astype(np.float32))
    y = torch.tensor(rng.randn(1, T2, C).astype(np.float32))
    w(f'dcw_{tag}_x', x.numpy()[0]); w(f'dcw_{tag}_y', y.numpy()[0])
    for mode in ('double', 'low', 'high'):
        corr = DCWCorrector(enabled=True, mode=mode, scaler=0.05, high_scaler=0.02, wavelet='haar')
        out = corr.apply(x.clone(), y, 0.75)
        w(f'dcw_{tag}_{mode}', out.numpy()[0])

print('wrote', len(os.listdir(OUT)), 'bins to', OUT)
