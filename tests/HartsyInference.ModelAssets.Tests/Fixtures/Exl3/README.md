# EXL3 decode fixture

A 256x384 (in x out) 2-bit MCG EXL3 weight: 2 input Hadamard blocks by 3 output blocks, so block boundaries,
tile order and both rotation axes are exercised. It is synthetic on purpose (random trellis words, random-sign
`suh`/`svh` in [0.5, 2]), because the real checkpoint is 100+ GB and the decode is data independent.

Files (all little endian, row-major):

| file | content |
| --- | --- |
| `trellis.i16` | `[16, 24, 32]` int16, the packed trellis (`[in/16, out/16, 16*K]`, K=2) |
| `suh.f16` / `svh.f16` | `[256]` / `[384]` fp16 sign-scale vectors |
| `w_hat.f16` | `[256, 384]` fp16, exllamav3 `reconstruct_tile<2,1>` output (trellis stage only, before any rotation) |
| `w_fused.f16` | `[256, 384]` fp16, exllamav3 `reconstruct_had_tile<2,1>` output (the full `W`, its fused fp16 butterfly kernel) |

## Provenance

The expected values are exllamav3's own device code, not this repo's. exllamav3 @ `d3739fd`
(`exllamav3/exllamav3_ext/quant/exl3_dq.cuh` and `hadamard_inner.cuh`) was wrapped in two `extern "C"` kernels and compiled with
`nvcc 13.0 -arch=sm_80 -ptx` (the full torch extension cannot be built with that toolchain), then launched on an RTX 3060 by
`gen_fixture.py` (`python gen_fixture.py 256 384 20260930 <outdir>`). `oracle.py` is a numpy port of the tile decode plus an fp64 dense
Hadamard; it reproduces `w_hat.f16` bit for bit and puts the fused kernel within 1.2e-3 of max|W| of the fp64 result, which is fp16 butterfly rounding.

The MCG codebook is `x = state * 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60; w = half(x.lo) + half(x.hi)`.
