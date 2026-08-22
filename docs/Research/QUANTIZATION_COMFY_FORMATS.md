# ComfyUI quantization formats (`comfy_quant` / comfy-kitchen)

> **Stub-shaped by design.** `int8_tensorwise` is built and real-weight verified (MiniMax-H3, all five DiT
> builds load — see `MODEL_STATUS_VIDEO.md`); `nvfp4` is verified bit-exact for GPU-dequant but the resident
> kernel path is not yet exercised by a full generation. This is cross-cutting format/math reference shared by
> LTX-2.5 and MiniMax-H3, not a single model's bring-up log — kept in full since it's already reference-only
> (no narrative walkthrough to strip).

ComfyUI has its own quantization family, unrelated to GGUF (see
[`QUANTIZATION_DIFFUSION.md`](QUANTIZATION_DIFFUSION.md) for that one). It is what the **official**
Lightricks LTX 2.5 and Comfy-Org MiniMax-H3 quantized releases ship, so supporting a model's official
small build means supporting this, not a community GGUF repack.

The runtime lives in the `comfy_kitchen` wheel (`comfy_kitchen/tensor/*.py` for layouts,
`backends/eager/quantization.py` for the reference math, `backends/cuda` + `backends/triton` for the fast
paths). ComfyUI itself only declares the table (`comfy/quant_ops.py` `QUANT_ALGOS`) and does the loading
(`comfy/ops.py`). The **eager backend is the authority** — the CUDA and Triton backends are checked
against it upstream, so parity against eager is parity against ComfyUI.

## Summary

| `format` | Storage | Companions | HartsyInference |
|---|---|---|---|
| `float8_e4m3fn` / `float8_e5m2` | fp8 | `weight_scale`, `input_scale` | supported — `Tensor.Fp8ScaleFactor` + native fp8 GEMM |
| `int8_tensorwise` | int8 | `weight_scale` (± ConvRot) | supported — resident, `CudaBackend` int8 IMMA path. **Verified on real weights**: relL2 5.1e-8–2.7e-7 vs the eager reference (F32 activations), and a real MiniMax-H3 generation |
| `nvfp4` | U8, 2×E2M1/byte | `weight_scale` (E4M3, blocked), `weight_scale_2`, `pre_quant_scale` | supported — resident, dequant-in-kernel. **Verified on real weights**: GPU dequant bit-exact vs the host reference on `qwen3vl_32b_minimax_h3_nvfp4_awq` layers. Not yet exercised by a full generation |
| `mxfp8` | fp8 | `weight_scale` (E8M0, block 32) | dequantized to BF16 at load (`Mxfp8Codec`); not resident |
| `convrot_w4a4` | int8 carrying int4 | `weight_scale` | not supported — no official release uses it |
| `svdquant_w4a4`, `awq_w4a16` | — | — | present in comfy-kitchen but **absent from ComfyUI's `QUANT_ALGOS`**; not reachable from a checkpoint |

Resident nvfp4 is a **VRAM** win, not a speed one: neither GPU here has FP4 tensor cores, so the GEMM still
runs in F16 off a transiently dequantized weight. It is opt-in per caller (`residentNvfp4`), because the
eager unpack is what the CPU and Vulkan backends need and AWQ layers carrying `pre_quant_scale` must take
it regardless.

## Data layouts / formats

### The per-layer descriptor

Every quantized Linear carries a `{prefix}.comfy_quant` U8 tensor holding UTF-8 JSON:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256, "per_row": true}
```

Recognized keys: `format`, `convrot`, `convrot_groupsize`, `linear_dtype`, and
`full_precision_matrix_mult` (ComfyUI's `_full_precision_mm_config` — the layer opts out of the
quantized GEMM entirely). `convrot`/`convrot_groupsize` may also appear nested under a `params` object;
ComfyUI reads both spellings.

**The per-layer blob is authoritative.** Files usually also carry a
`__metadata__._quantization_metadata` mirror of every layer's descriptor, but re-quants of the same model
skip different layer sets — the official `Lightricks/LTX-2.5` build and `DmitryDB/LTX-2.5-ComfyUI-Quants`
disagree — so a loader keyed on the file-level copy will apply one layer's format to another's weight.

### `int8_tensorwise`

```
{p}.weight        I8   [N, K]   row-major, already ConvRot-rotated when convrot is set
{p}.weight_scale  F32  [N, 1]   per-output-row dequant scale (a scalar means per-tensor)
{p}.bias          BF16 [N]      never quantized
```

No `input_scale` is stored — activations are quantized dynamically per row at runtime
(`scale = rowabsmax/127`, round-to-nearest, clamp to ±127).

### ConvRot

`convrot` is an orthogonal Hadamard rotation applied per contiguous `convrot_groupsize` slice of the
**input** dim. `H = kron(h4, …, h4) / √size` where

```
h4 = [[ 1,  1,  1, -1],
      [ 1,  1, -1,  1],
      [ 1, -1,  1,  1],
      [-1,  1,  1,  1]]
```

so `size` must be a power of **four** (256 = 4⁴ in every shipped checkpoint). `H` is symmetric *and*
orthogonal, therefore `H·H = I`.

- Offline the quantizer stores `W_rot = W @ Hᵀ`, then row-quantizes.
- Online the consumer owes `x_rot = x @ H`, and `x_rot · W_rotᵀ = x·H·H·Wᵀ = x·Wᵀ`.

Because `H` is symmetric, "`@ H`" and "`@ Hᵀ`" are the same operation — a transpose bug here is invisible
to a test, so build `H` from the 4×4 above rather than trusting a direction.

The quantizer applies ConvRot **only when `in_features % 256 == 0`**; other layers in the same file are
quantized unrotated, with the same per-row scale. `convrot_groupsize` absent or `convrot: false` is
normal, not an error.

Rotation factors into `log₄(size)` radix-4 stages, so no matrix need be materialized. With the `h4`
above, every stage is `out[t] = (sum − 2·v[3−t]) / 2` over its four lanes, and the ½ per stage
accumulates to the `1/√size` normalization.

### `nvfp4`

```
{p}.weight           U8       [N, K/2]   two E2M1 nibbles per byte, HIGH nibble = even element
{p}.weight_scale     F8_E4M3  [pR, pC]   one scale per 16 input elements, NVIDIA blocked/swizzled layout
{p}.weight_scale_2   F32      scalar
{p}.pre_quant_scale  F32      [K]        optional; multiplies the ACTIVATION
```

E2M1 by nibble: `0, 0.5, 1, 1.5, 2, 3, 4, 6` for 0–7 and their negatives for 8–15. The swizzle is
`comfy.float.to_blocked`; `BlockScaleSwizzle.SwizzledIndex` inverts it. `pre_quant_scale` exists because
the AWQ per-input-channel scale was migrated into the preceding RMSNorm for most layers, but `o_proj` and
`down_proj` have no such host: `x·Wᵀ = (x⊙s)·(W/s)ᵀ`.

## Key numbers / constants

- ConvRot group size in every shipped checkpoint: **256**. Valid sizes are powers of four.
- Activation quant: per row, `absmax/127`, round-to-nearest, clamp ±127. Weights the same, per output row.
- comfy-kitchen pads the IMMA GEMM to `round_up(max(m, 32), 32)` rows and slices the result back; Turing
  additionally rejects some skinny-N shapes, so it aligns N to 32 there (8 elsewhere).
- `TensorWiseINT8Layout.MIN_SM_VERSION` is (7, 5) — INT8 tensor cores from Turing on.
- nvfp4 group size 16; mxfp8 group size 32.

## What the official releases ship

`Lightricks/LTX-2.5` (gated: the tree API answers but file downloads 401):

| File | Size |
|---|---|
| `diffusion_models/ltx-2.5-22b-{dev,distilled}-transformer-bf16` | 42.02 GB |
| `diffusion_models/ltx-2.5-22b-{dev,distilled}-transformer-comfy-int8-convrot` | 21.50 GB |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4` | 18.72 GB |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16` | 26.26 GB |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot` | 15.37 GB |

The dev DiT has **1440** quantized matrices, all `int8_tensorwise` + convrot/256. The Gemma 4 TE has
**328**; `model.embed_tokens` is deliberately left BF16, so no int8 embedding-gather path is needed.
Left dense in both: `adaln_single.*`, `*.to_gate_logits`, q/k norms, patchify/proj_out.

`Comfy-Org/MiniMax-H3` (ungated) ships five DiT builds per task; the pruned int8 build is
`minimax_h3_fl2va_pruned_int8_convrot.safetensors` at 20.97 GB with **200** quantized matrices, same
format and group size.

Ungated mirrors when the official file is gated: `DmitryDB/LTX-2.5-ComfyUI-Quants` (int8, int8+convrot
and nvfp4 rebuilds with their build reports and layer policies published alongside — a useful cross-check
on which layers a policy skips) and `dummy9996/LTX-2.5-22b-ungate` for BF16 reference tensors.

## Reference implementations

- **comfy-kitchen** `tensor/int8.py` (`TensorWiseINT8Layout`), `tensor/int8_utils.py` (`_build_hadamard`,
  `_rotate_weight`, `_rotate_activation`), `tensor/nvfp4.py`, `backends/eager/quantization.py`
  (`quantize_int8_rowwise`, `quantize_int8_convrot_weight`, `int8_linear`, `mm_int8`). Vendored locally
  at `Models/bench-comfy/pylibs/comfy_kitchen` (0.2.26).
- **ComfyUI** `comfy/quant_ops.py` (the `QUANT_ALGOS` table), `comfy/ops.py` (`comfy_quant` loading,
  `_full_precision_mm_config`, the per-format scale views). Vendored at `Models/bench-comfy/ComfyUI`.
- **comfy-quants** (`github.com/Comfy-Org/comfy-quants`) — the exporter and its per-model layer policies.
- Our dump script: `tests/python-reference/int8_convrot_reference.py`.

## Differences between implementations

- ComfyUI's `int8_linear` **dequantizes the weight** whenever the layout's fast path declines (a
  transposed operand, per-row scales on an `mm` RHS). We take a dequant fallback in the same places, plus
  `full_precision_matrix_mult`, but **not** for small `m` — comfy-kitchen pads there and so do we, since
  dequantizing a 28672×5376 weight on every short prompt would dominate a text encoder.
- comfy-kitchen rotates the activation in the activation's own dtype (BF16 in ComfyUI). We rotate in the
  engine's F16/F32 activation dtype, so parity is relL2, not bit-exact.
- The eager reference clamps int8 to [−128, 127]; with `scale = absmax/127` nothing reaches −128, so a
  ±127 clamp is equivalent.
- Release notes for LTX 2.5 claimed `audio_ff_bias` and `use_prompt_adaln_single` changes that the tensor
  keys disprove — see [`LTX_2_5.md`](LTX_2_5.md). Trust tensor keys over release notes here generally.

## Implementation notes

- `DType.F4E2M1` exists precisely so a packed nvfp4 weight can be labelled `[N, K]` rather than U8
  `[N, K/2]`. `LinearImpl` derives `k` from `weight.Shape[1]`, so a U8-labelled packed weight silently
  runs the whole GEMM at half the true K.
- The int32 IMMA accumulator is 4 bytes per output element. At video token counts an unchunked `m·n·4`
  buffer runs to gigabytes, so the resident int8 path chunks over rows; ComfyUI chunks its rescale for
  the same reason.
- Attaching the companions to the weight `Tensor` (`Tensor.QuantInfo`) rather than to a per-model linear
  wrapper is what lets one backend branch serve every model that loads such a checkpoint.

### Three latent inconsistencies found while building this, all currently unreachable

Recorded because each is invisible in normal operation and expensive to re-derive, not because any is a bug
to fix today. Confirm the premise still holds before acting on one.

- **Two E4M3 decode conventions coexist.** `CheckpointConvertUtils.E4M3Table` decodes `0x7F`/`0xFF` as **NaN**
  (torch `float8_e4m3fn` semantics); `Tensor.CastTo`, `Nvfp4ResidentCodec`, `Nvfp4Linear` and
  `dequant_nvfp4_to_f16.cu` decode them as **±480**. So `Nvfp4Codec.DequantExpertSlice` — the shipped GPT-OSS
  MoE path — disagrees with the resident path on exactly those two bytes. **Unreachable today**: comfy-kitchen
  clamps to `F8_E4M3_MAX = 448` (`float_utils.py`), so its quantizer never emits `0x7F` as a block scale or an
  fp8 weight. It surfaced as a `relL2 = NaN` on an all-256-scale-bytes synthetic case, which is the only way to
  hit it. Do not "unify" these without first checking which callers depend on which.
- **`Tensor.Reshape` drops `Fp8ScaleFactor`** (and `QuantInfo`). `Nvfp4Linear.Load` reshapes its block scale to
  the rank-3 bank shape *before* `DequantBf16Core` reads `blockScale.Fp8ScaleFactor`, so a non-1 factor would be
  silently ignored there while the resident path (which reads the un-reshaped tensor) honours it. Latent only
  because the factor is 1 in every checkpoint inspected.
- **Per-weight quant scale caches are freed wholesale, not per weight.** `_int8RowScaleDevice`,
  `_nvfp4ScaleDevice` and `_w8a8WeightCache` are all released by `FreeW8A8Cache` (Dispose /
  `FreeAllDeviceMemory` / `FreePreloadedWeights`) and not by `FreeWeights`. Since they are keyed by `Tensor`,
  a resident quantized weight that is freed and dropped keeps its small device scale buffer *and* its managed
  `Tensor` alive until backend disposal. Consistent across all three by design; the cost is bounded (one
  `[N]` float vector per weight).
