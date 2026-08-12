import numpy as np, torch, itertools, struct
from na_eager import na3d

CASES = [
    # (B, T, H, W, heads, headDim, kt, kh, kw)
    (1, 4, 5, 6, 2, 8, 3, 3, 3),
    (1, 3, 3, 3, 1, 4, 3, 5, 5),   # kernel larger than an axis
    (1, 1, 7, 9, 2, 4, 3, 7, 7),   # degenerate temporal axis
    (2, 5, 4, 4, 3, 8, 5, 3, 3),   # batch > 1
    (1, 6, 6, 6, 1, 4, 11, 11, 11),# kernel exceeds every axis
]
g = torch.Generator().manual_seed(1234)
with open("cases.bin", "wb") as f:
    f.write(struct.pack("<i", len(CASES)))
    for (B,T,H,W,NH,HD,kt,kh,kw) in CASES:
        q = torch.randn(B,T,H,W,NH,HD, generator=g, dtype=torch.float32)
        k = torch.randn(B,T,H,W,NH,HD, generator=g, dtype=torch.float32)
        v = torch.randn(B,T,H,W,NH,HD, generator=g, dtype=torch.float32)
        out = na3d(q, k, v, [kt,kh,kw], None, 1.0)
        f.write(struct.pack("<9i", B,T,H,W,NH,HD,kt,kh,kw))
        for t_ in (q,k,v,out):
            f.write(t_.contiguous().numpy().astype("<f4").tobytes())
        print(f"case B{B} T{T} H{H} W{W} NH{NH} HD{HD} k({kt},{kh},{kw}) out absmax {out.abs().max():.5f}")
print("wrote cases.bin")
