"""Is our Na3d window rule the same as comfy_kitchen.na3d's, in the regime where the window actually slides?

Stage 0 of the decoder has H,W clamped to the full axis (dense) and matched the reference; stage 1 is the first
place H,W genuinely slide, and it diverges. This compares comfy_kitchen.na3d against a numpy transcription of
IBackend.Na3dReference on exactly that shape.
"""
import numpy as np
import torch
import comfy_kitchen


def window_start(i, length, k):
    s = i - k // 2
    last = length - k
    if s < 0:
        s = 0
    if s > last:
        s = last
    return s


def our_na3d(q, k, v, kt, kh, kw, scale=1.0):
    B, T, H, W, Hd, D = q.shape
    kt, kh, kw = min(kt, T), min(kh, H), min(kw, W)
    out = np.zeros_like(q)
    for b in range(B):
        for it in range(T):
            t0 = window_start(it, T, kt)
            for ih in range(H):
                h0 = window_start(ih, H, kh)
                for iw in range(W):
                    w0 = window_start(iw, W, kw)
                    kk = k[b, t0:t0 + kt, h0:h0 + kh, w0:w0 + kw]     # [kt,kh,kw,Hd,D]
                    vv = v[b, t0:t0 + kt, h0:h0 + kh, w0:w0 + kw]
                    kk = kk.reshape(-1, Hd, D)
                    vv = vv.reshape(-1, Hd, D)
                    qq = q[b, it, ih, iw]                              # [Hd,D]
                    s = np.einsum('hd,nhd->hn', qq, kk) * scale
                    s = s - s.max(axis=-1, keepdims=True)
                    e = np.exp(s)
                    e = e / e.sum(axis=-1, keepdims=True)
                    out[b, it, ih, iw] = np.einsum('hn,nhd->hd', e, vv)
    return out


def run(T, H, W, heads, D, kernel, label):
    rng = np.random.RandomState(0)
    shape = (1, T, H, W, heads, D)
    q = rng.randn(*shape).astype(np.float32)
    k = rng.randn(*shape).astype(np.float32)
    v = rng.randn(*shape).astype(np.float32)
    ref = comfy_kitchen.na3d(torch.from_numpy(q), torch.from_numpy(k),
                             torch.from_numpy(v), list(kernel), None, 1.0).numpy()
    ours = our_na3d(q, k, v, *kernel)
    num = np.sqrt(((ref - ours) ** 2).sum())
    den = np.sqrt((ref ** 2).sum())
    print(f"{label:42s} kernel={kernel} dims=({T},{H},{W})  relL2 {num/den:.3e}")


# Stage 0 regime: H,W shorter than the kernel -> window covers the whole axis (this one matched end-to-end).
run(5, 4, 4, 2, 8, (3, 7, 7), "stage-0 regime (H,W clamped, dense)")
# Stage 1 regime: H,W longer than the kernel -> the window genuinely slides (this one diverged).
run(5, 8, 8, 2, 8, (3, 7, 7), "stage-1 regime (H,W slide)")
# Isolate each axis.
run(8, 2, 2, 2, 8, (3, 1, 1), "T slides only")
run(2, 8, 2, 2, 8, (1, 7, 1), "H slides only")
run(2, 2, 8, 2, 8, (1, 1, 7), "W slides only")
run(2, 8, 2, 2, 8, (1, 3, 1), "H slides, odd kernel 3")
run(2, 9, 2, 2, 8, (1, 4, 1), "H slides, EVEN kernel 4")
