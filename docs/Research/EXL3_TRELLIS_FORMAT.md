# EXL3 trellis layout (2 bpw, MCG)

The layout `Exl3Codec` and `dequant_exl3_2bit_to_bf16` implement, confirmed against exllamav3 @ `d3739fd` (`exl3_dq.cuh`, `hadamard_inner.cuh`), the real
`sfxnz/DeepSeek-V4.1-Flash-EXL3` headers and the committed fixture (`tests/HartsyInference.ModelAssets.Tests/Fixtures/Exl3/`). Only 2 bits with the MCG
codebook is supported; any other width throws naming the key and shape.

## Tensors per expert matrix

| key suffix | dtype, shape | meaning |
|---|---|---|
| `.trellis` | I16 `[in/16, out/16, 16*K]`, K=2 so `[.., .., 32]` | one 16x16 tile = 64 bytes; expert `w1`/`w3` are `[320, 144, 32]` (in 5120, out 2304), `w2` is `[144, 320, 32]` |
| `.suh` | F16 `[in]` | input sign-scale |
| `.svh` | F16 `[out]` | output sign-scale |
| `.mcg` | I32 scalar | must be `0xCBAC1FED` (checked at decode: it is data, not header) |

`W[in, out] = diag(suh) * H(in, 128-blocks) * W_hat * H(out, 128-blocks) * diag(svh)`. `H` is the Sylvester natural-order 128-point Hadamard scaled by
`0.08838834764831845` (`1/sqrt(128)`) on each side, so both dims must be multiples of 128. The engine stores `M[o, i] = W[i, o]` (rows = out).

## Tile decode

- A tile's 64 bytes are 16 little-endian uint32 words read MSB-first as a 512-bit tail-biting ring. Value `p` (0..255, stored order) uses the 16-bit window ending at bit `(p+1)*2`.
- MCG: `x = state * 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60; value = half(x.lo16) + half(x.hi16)` rounded to fp16.
- Stored position `p` is `lane = p / 8, e = p % 8`: input row `(lane & 3)*2 + (e & 1) + ((e >> 1) & 1)*8`, output column `(lane >> 2) + (e >> 2)*8` (the mma fragment order).

## Rotation order (the host/device contract)

Per 128x128 block, F32: in-axis butterfly (`h = 1..64`, `(a+b, a-b)`), `v * S * suh[i]`, out-axis butterfly, `v * S * svh[o]`. The device kernel repeats
this with `__fmul_rn` so BF16 equals the host F32 rounded once. exllamav3's fused kernel runs its butterflies in fp16, so it differs by fp16 rounding:
1.1e-7 of max|W| between our F32 and an fp64 oracle, 1.2e-3 between our F32 and the fused kernel.

## Limits

- Slicing: input-column windows on 128 boundaries only. A partial output-row window is refused, because the trellis is tile-row major and a row window is not contiguous.
- Non-routed weights in the EXL3 checkpoint are FP8 E4M3 + F8_E8M0 32x32 scales; `lm_head` is FP8 with a `[129280, 160]` U8 1x32 scale.
