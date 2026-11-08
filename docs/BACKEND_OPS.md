# Backend op contracts

The semantics of the generic `IBackend` ops that replaced model-named ones. Each section is normative: the host
reference named in it is the executable form, every accelerator kernel is checked against that reference with
`tools/OpParity` ([VERIFICATION.md](VERIFICATION.md)), and a kernel that disagrees is wrong whatever a model's output
looks like. Code comments point here instead of restating a contract.

## Strided views

`StridedView` (`src/HartsyInference.Core/Backends/Layout/`) is a window onto a tensor's storage: one to eight extents,
a non-negative element stride per axis and an element offset. A stride of 0 broadcasts the axis. Views compose
without touching data, exactly as NumPy's `as_strided`:

| Call | Result |
|---|---|
| `StridedView.Of(shape)` | the whole tensor, row-major |
| `Reshape(dims)` | new extents over a contiguous view, same elements |
| `Select(axis, i)` | fixes an axis and drops it |
| `Slice(axis, start, length, step)` | keeps `length` positions from `start`, every `step`-th |
| `Permute(order)` | reorders axes |
| `Broadcast(axis, count)` / `Unsqueeze(axis)` | inserts a stride-0 axis |
| `VectorAlong(dims, axis)` | a vector broadcast over every other axis (a per-channel bias) |

A `StridedCopySpec` pairs a destination view with a source view of the same extents: element `i` of the destination
reads element `i` of the source. `StridedCopySpec.Gather(view)` packs the destination row-major from element 0;
`StridedCopySpec.Scatter(view)` packs the source. `StridedCopies` names the common shapes: `SliceAxis`,
`ScatterAxis` and `RepeatInterleave` (the GQA key/value head expansion).

## CopyStrided

```
output[dst(i)] = input[src(i)]   for every multi-index i of the spec's extents
```

Reference: `StridedCopyReference.Copy`. Contract: `StridedCopyContract.ValidateCopy`.

- A pure move of bits. Both tensors have the same dtype, of 1, 2, 4 or 8 bytes per element; quantized dtypes are refused.
- Every element either view addresses lies inside its tensor's allocation.
- The destination addresses every element at most once: an axis of extent above 1 has a positive stride, and each
  axis's stride exceeds the span of the axes with smaller strides. A broadcast destination is refused.
- Elements of the output the destination does not address keep their contents, so a scatter into a slot of a larger
  tensor is in place. When the destination addresses every element of the output (a permutation of the whole
  tensor), a backend may write a fresh buffer instead.
- The output does not share storage with the input.

The contract collapses axes before a kernel sees them: axes of extent 1 are dropped and adjacent axes both views walk
contiguously are merged, so a head split arrives as two or three axes, not five.

## AccumulateStrided

```
output[dst(i)] = round(float(output[dst(i)]) + float(input[src(i)]))
```

Reference: `StridedCopyReference.Accumulate`. Contract: `StridedCopyContract.ValidateAccumulate`.

- F32, F16 or BF16, the same on both sides, or an F32 input into a BF16 output (a per-channel bias added to BF16
  storage). The add is F32; the result rounds to the storage dtype once, and bfloat16 rounds to nearest even
  (`BFloat16Bits.FromFloat`, the CUDA `__float2bfloat16` rounding).
- The same geometry rules as `CopyStrided`. Because the destination addresses each element once, no two adds touch
  one element and the result does not depend on execution order; no atomics are needed.
- Always in place.

## Pad

```
output[o] = input[clamp/reflect(o - before)]   or zero bits outside the input (PadMode.Zero)
```

Reference: `StridedCopyReference.Pad`. Contract: `PadContract.Validate`.

- `PadSpec` gives every axis a `before` and `after` count and one `PadMode` for all padded positions: `Zero`
  (`F.pad(mode="constant", value=0)`), `Replicate` (the nearest edge element) or `Reflect` (mirrored about the edge
  element, which is not repeated; each count stays below the axis extent).
- Output extent of every axis is the input extent plus its two counts. Any 1, 2, 4 or 8-byte dtype, the same on
  both sides. The whole output is written.

## Pixel shuffles over CopyStrided

Depth-to-space, space-to-depth, patchify and unpatchify are permutations, so `PixelShuffles`
(`src/HartsyInference.Core/Backends/Layout/`) builds them as `StridedView` pairs and issues one `CopyStrided`; no backend
has a shuffle kernel of its own. A `ShuffleSpec` names the geometry:

| Field | Meaning |
|---|---|
| `PackedLayout`, `UnpackedLayout` | `ChannelsFirst` (`[B, C, T, H, W]`) or `Tokens` (`[B·T·H·W, C]` rows) on each side |
| `Batch`, `Frames`, `Height`, `Width`, `Channels` | the packed grid and the unpacked channel count |
| `FactorT`, `FactorH`, `FactorW` | sub-positions per packed cell on each axis |
| `Order` | how a packed channel nests channel, time, height and width (`ChannelOrder.CTHW`, `CWHT`, `HWTC`, `HWCT`, `CHWT`) |
| `ChannelRepeat` | packed channels per stored channel (`repeat_interleave` before the shuffle; the repeat must not straddle an axis) |
| `DropLeadingFrames` | unpacked frames discarded from the front (a causal shuffle's duplicated frames) |
| `DropDuplicateFrame` | unpacked frame index 1 dropped and later frames shifted down (SeedVR2) |
| `ChunkStartFrame`, `UnpackedFrames` | this call's position in an unpacked tensor several calls fill |

`DepthToSpace(backend, unpacked, packed, spec)` writes `unpacked[b, c, t·fT + iT, h·fH + iH, w·fW + iW] = packed[b, order(c, iT, iH, iW) / repeat, t, h, w]`,
with the frame drops applied to `t·fT + iT`. A dropped frame splits the move into at most three copies; every other
spec is one. `SpaceToDepth` is the inverse for a spec without repeats, drops or chunks. `UnpatchifyTokens` and
`PatchifyTokens` are the DiT token grid (`TokenGrid` spec) with the two packings the DiT families use.

## Rotary

```
x0' = x0·c0 − x1·s0        x1' = x1·c1 + x0·s1        for every pair (x0, x1) with entries (c0, s0), (c1, s1)
```

Reference: `RotaryReference.Apply`. Contract: `RotaryContract.Validate`. Each product and each sum is rounded once
(no fused multiply-add), so every backend reproduces the reference bit for bit.

A `RotarySpec` is three views over the canonical axes `[B, H, S1, S2, S3, P, 2]` (batch, head, up to three sequence
axes, pair, element of the pair): `X` says which two activation elements form a pair, `Cos` and `Sin` say which table
entries each element reads. That one shape expresses every rotation the models use:

| Variation | How the views express it |
|---|---|
| interleaved `(2i, 2i+1)` or rotate-half `(i, i+half)` pairs | the pair axis and the pair-element stride of `X` (`RotaryViews.Create` with `RotaryPairing`) |
| partial rotation, a rotated slice at an offset | the pair axis cut from `[dimOffset, dimOffset + rotaryDim)` |
| one angle per pair, per element, or duplicated per pair (`RopeTableLayout`) | the table's pair-element stride: 0, `half`, or 0 over a step-2 pair axis |
| tables per position, per batch, per head, per grid axis | broadcast (stride 0) table axes |
| token-major, head-major, packed rows, 3-D grids | the activation base view (`RotaryViews.TokenMajor`, `HeadMajor`, `Rows`, `Grid`, `Explicit`) |
| a captured decode step | `AtDevicePosition(handle, slot, rowStride)`: the table offsets grow by `position[slot] · rowStride` at run time |

Rules: activations F32, F16 or BF16 with F32 tables; the three views share their extents; the pair axis has
extent 2 and a non-zero activation stride; the activation view addresses every element once (the injectivity rule of
`CopyStrided`); every view stays inside its tensor; a device position needs a positive row stride and host-indexed
tables otherwise. In place on `X`.

`RmsNormRotary(output, input, normWeight, eps, cos, sin, spec)` is the fused form of a full-row RMS norm over
token-major rows, the rotation `spec` describes over those rows, and a token-major or head-major (four-axis output)
emit. The default composes `RmsNorm`, `Rotary` and `Permute0213`; CUDA fuses the three for
`RotaryViews.SplitRows` (rotate-half per head, one angle per pair, per-head lanes), the LTX-2 geometry its kernel runs.

## QKV decode epilogue

```
q' = rotate(norm(q), position)      k' = rotate(norm(k), position)      kCache[h, position] = k'_h      vCache[h, position] = v_h
```

Reference: `QkvDecodeReference.Apply` (with the position given, since the host has no device position buffer). Contract:
`QkvDecodeContract.Validate`. One `QkvDecodeCall` is a decode step's whole attention input path after the projection:
the q, k and v sections (`QkvSection`: a tensor and the element the section starts at, so `[q | k | v]`, `[q | k]` + v or
three tensors are the same call, built by `Fused`, `QkPlusV` and `Separate`), the norm (`QkvNormKind`: none, per head with a
`[headDim]` weight, or full width with a `[heads·headDim]` weight, `WithNorm`), the rotation (the tables `[rows, headDim]`,
the pairing and the rotated width, as `RotaryCall.DecodeStep` reads them) and the caches `[1, kvHeads, maxSeq, headDim]`.
The position is the query offset of the device position buffer; a slot at or past the cache length writes nothing.

Rules: sections F32 inside their tensors; q output F32 `[1, heads, 1, headDim]`; both caches F32 or both F16 (an F16 cache
stores k and v rounded to nearest and saturated at ±65504), one shape; tables F32 with at least as many rows as the cache
has slots; the rotated width even and at most the head dim, and 0 with interleaved pairing (an interleaved partial width
comes from identity table rows); norm weights of the width the norm kind reads; the q output overlaps no section. The
sections are read only. The norm is `x · rsqrt(mean(x²) + eps) · w` over each head or each full section; the rotation is
the rotary formula.

The default composes the call from `CopyStrided`, `RmsNorm`, `Rotary` and `KvCacheAppendDev` (one k temporary, one v
temporary for a packed v, plus the norm's inputs); CUDA runs one kernel per norm kind, each bit-identical to the
composition it replaces.

## Execution requirements

Every implementation of these ops:

- allocates nothing on the device beyond the output buffer (an in-place destination reuses the output's buffer);
- adds no host synchronization and reads no host memory of a device-resident operand;
- is graph-capture safe: the geometry is passed by value as kernel parameters or push constants, never through
  constant memory or a device-side spec buffer;
- is one launch per call.

## Backend coverage

| Op | CPU | CUDA | Vulkan |
|---|---|---|---|
| `CopyStrided` | reference | `layout_copy.ptx` (`copy_strided_u8/u16/u32/u64`) | `copy_strided_u16/u32/u64`; 1-byte elements take the reference |
| `AccumulateStrided` | reference | `accumulate_strided_f32/f16/bf16` | `accumulate_strided_f32/f16`; BF16 takes the reference |
| `Pad` | reference | `pad_u8/u16/u32/u64` | `pad_strided_u16/u32/u64`; 1-byte elements take the reference |
| `Rotary` | reference | `rotary.ptx` (`rotary_f32/f16/bf16`) | `rotary_f32/f16`; BF16 takes the reference |
| `QkvDecodeEpilogue` | no device position buffer | `lm_qkv_rope_scatter_f32/f16kv`, `lm_qknorm_rope_scatter_f32/f16kv`, `lm_qknorm_full_rope_scatter_f32/f16kv` | composed; F32 caches only |

The Vulkan shaders index with 32-bit words, so a view reaching past 2³² − 1 elements is refused there.

## What these replaced

The Oasis and Matrix-Game 3 head splits and merges, the Matrix-Game 3 keyboard K/V broadcast and mouse-window concat,
the Wan 2.2 VAE QKV split and token transpose, the GQA `RepeatKvHeads`, the KV `SliceTimeRange`, the CausalConv3d frame
extract, frame write, bias fill and tap accumulate, Ideogram 4's `ScatterRowsAfter` and the SeedVR2 bottom-right pad
are all one `CopyStrided`, `AccumulateStrided` or `Pad` call at their call sites; `tools/OpParity --op layout` holds
one case per former call shape. `PixelShuffle2d`, `PixelShuffle3d`, `SeedVr2PixelShuffle`, `Ltx25PixelShuffle`, `DupUp3dVae`,
`UnpatchifyVae`, `UnpatchifyTokens`, `OasisUnpatchify` and `PatchifyTokens` are `PixelShuffles` calls (`--op shuffle`).
`ApplyRope`, `ApplyRopeSingle`, `ApplyRopeSingleHeadMajor`, both `ApplyRopeInterleaved` forms, `WanRopeInterleaved`,
`WanRopeInterleavedPerHead`, `OasisRopeInterleaved`, `Mg3RopeBatched`, `Ltx2SplitRope`, `Ltx25NaRope3d` and
`RopeApplyDecodeStep` are `Rotary` calls built with `RotaryCall`'s factories or `RotaryViews`; `Ltx2QkNormRopeHeadMajor`
and `Ltx2QkNormRopeTokenMajor` are `RmsNormRotary` (`--op rotary`). `QkvRopeScatterDecodeStep`, `RopeScatterKvDecodeStep`,
`QkRopeScatterVDecodeStep`, `QkvNormRopeScatterDecodeStep`, `QkNormRopeScatterVDecodeStep` and `QkNormFullRopeScatterDecodeStep`
are `QkvDecodeEpilogue` calls built with `QkvDecodeCall`'s factories (`--op qkvdecode`).
