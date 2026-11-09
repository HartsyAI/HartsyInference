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

## NormModulate

```
y[r, d] = (norm(x[r, :])[d] · w[d] + b[d]) · s[m(r), d] + t[m(r), d]        every absent term dropped
```

Reference: `NormModulateReference.Apply`. Contract: `NormModulateContract.Validate`. A `NormModulateCall` names the row
axis (`NormAxis.LastDim`, or `NormAxis.Channel` for axis 1 of `[N, C, ...]`), the norm (`NormKind`), the norm's learned
affine `w` and `b` (`WithAffine`) and the adaLN modulation (`ModulationSpec`, `WithModulation`): a scale `s` and shift `t`
with a `ScaleMode` (`Raw`: `v · s`; `OnePlus`: `v · (1 + s)`) and the row each activation row reads:

| Variation | How the spec expresses it |
|---|---|
| one vector for every row, or `[B, D]` over each batch of `[B, S, D]` | `ModulationSpec.Broadcast(x, s, t, mode)`: rows share a modulation row in equal runs |
| one modulation row per token, gathered | `ModulationSpec.RowIndexed(x, s, t, rowIndex, mode)` with an I32 index |
| scale and shift cut from one packed `[mods, k · D]` projection | `ModulationSpec.Packed(mod, rowStride, scaleOffset, shiftOffset, rowsPerMod, mode)` |
| explicit addressing | the constructor: row stride, offsets, rows per modulation row (0 shares row 0) |

Norms: `LayerNorm` is `(x − mean) · rsqrt(var + eps)` with the biased variance, `RmsNorm` is `x · rsqrt(mean(x²) + eps)`,
`L2Norm` is `x · sqrt(n) / max(‖x‖₂, eps)` (Wan 2.2's `RMS_norm`), `None` applies the affine and modulation alone. The
reference accumulates the sums in F64; the kernels reduce in F32 (the channel kernel on CUDA in F64), so a norm is held to a
tolerance while `None` is bit-identical.

Rules: activations F32, F16 or BF16, the input and output dtypes independent and of equal element count; vectors F32 of
the row width; a positive epsilon for a norm; a modulation along the last axis only, its vectors covering every modulation
row the rows read, the index one I32 entry per row; the output overlaps no operand.

`LayerNormNoAffine` and `AffineBroadcastLastDim` remain on `IBackend` as fixed spellings of this call (a LayerNorm with
nothing else; no norm with a raw broadcast modulation).

## ChunkActivate

```
outputs[c][r, d] = act_c(x[r, c · width + d])        act ∈ {identity, 1 + x, tanh, sigmoid}
```

Reference: `ChunkActivateReference.Apply`. Cuts an F32 `[rows, count · width]` projection into `count` F32 outputs
`[rows, width]` through one `ChunkActivation` each, at most eight chunks: an adaLN modulation projection split into its
scales and gates in one pass. tanh and sigmoid follow each backend's math library, so the gates are held to a tolerance.

## Glu

```
out[r, i] = act(gate[r, i]) · value[r, i]        gate and value the two halves of input row r, in the order the spec says
```

Reference: `GluReference.Apply`. Contract: `GluContract.Validate`. A `GluSpec` names the gate function (`GluActivation`:
`Silu`, `GeluTanh`, `GeluErf`, or `Situ`, whose gate is `β · tanh(g/β) · sigmoid(g)` and whose value side is soft-capped to
`β_lin · tanh(u/β_lin)`, with `β` and `β_lin` in `GluSpec.SituBeta` and `SituLinearBeta`, defaulting to `SituDefaults`) and
which half is the gate (`GluGateHalf.First` for `[gate | up]` projections, `Second` for `[value | gate]`). The presets
`GluSpec.SwiGlu`, `GeGluFirst`, `GeGlu`, `GeGluErf` and `Situ` are the forms the models use. The CUDA and Vulkan kernels
compile in the defaults and refuse other SiTU betas.
Input and output are F32, F16 or BF16 alike; the output's last extent is half the input's. The sigmoid takes the sign-aware
form; the tanh GELU on CUDA keeps the fast-math `2 · sigmoid(2x) − 1` form of the kernels it replaced. `GluEmitQ8` is the same
op with a Q8_1 sidecar of the output where a backend keeps one.

## GroupScale

```
out[r, group] = f(gates[offset + r · stride + group]) · in[r, group]        or  out[r, group] += that, a zero gate skipped
```

Reference: `GroupScaleReference.Apply`. Contract: `GroupScaleContract.Validate`. A `GroupScaleSpec` cuts each row into
groups of `GroupWidth` elements (0 for the whole row) and names the gate of row `r`, group `g` by its stride and offset, the
function the gate passes through (`GateFunction.Identity` or `TwoSigmoid`) and whether the scaled value is written or
accumulated. `PerRow()` is token masking by a `[rows]` vector, `PerGroup(headDim, heads, TwoSigmoid)` LTX-2's per-head
output gate, `Column(experts, expert, accumulate: true)` one expert of a dense routing table added into the combine. The
output may be the input (in place); gates are F32 or the activation's dtype.

## LinearFused

A projection with a `LinearFusion` folded around it: a `GroupScale` of the input before the GEMM (`PrologueGates` and
`Prologue`) and a `LinearActivation` of the output after it. `LinearFusion.Gelu` and `LinearFusion.HeadGated(logits, heads,
headDim)` are the two forms the models use. The default (`IBackend.ComposeLinearFused`) runs the group scale on a copy of
the input, the GEMM and the activation; CUDA folds the GELU into its resident int8 dequant epilogue and the F16 head gate
into the activation quantize that precedes the GEMM, and composes everything else.

## MoeRoute dense table

`MoeRoute` takes an optional `dense` output `[T, E]` that receives every token's weights at its expert ids and zeros
elsewhere, for a combine that runs every expert densely and scales each one's rows with `GroupScale.Column`. The host
reference writes it from the same routing pass; CUDA and Vulkan write it in the route kernel.

## GuidedEulerStep

```
guided[i] = g · cond[i] + (1 − g) · uncond[i]          GuidanceForm.Lerp
          = uncond[i] + g · (cond[i] − uncond[i])      GuidanceForm.Extrapolate
ratio     = clamp(‖cond‖₂ / (‖guided‖₂ + eps), Min, Max)   over the whole tensor (Global) or each last-dimension row (PerRow); 0 when the denominator is 0
z[i]     += guided[i] · ratio · delta                   in place; ratio is 1 for GuidanceNorm.None
```

Reference: `GuidedEulerReference.Apply`. Contract: `GuidedEulerContract.Validate`. A `GuidedEulerSpec` carries the guidance,
the delta, the blend form and the `GuidanceNorm`; `GuidedEulerSpec.Lerp(g, delta)` is the flow-match samplers' step (both
predictions the same tensor and `g` 1 is an unguided Euler step; a zero-filled latent and delta 1 leave the blended logits),
`GlobalNorm(g, delta, min)` Lance's renormalized step and `RowNorm(g, delta)` Lumina 2's per-row normalized step. The two
blend forms are one value in exact arithmetic and differ in the last bit in F32, so a model keeps the form its reference uses.
The norms sum in F64. A zero delta leaves the latent untouched. F32; the predictions may be one tensor but neither may share
storage with the latent.

## TopKGuidedLogits

```
threshold = the K-th largest cond[r, ·]
out[r, i] = cond[r, i] < threshold ? −∞ : uncond[r, i] + (cond[r, i] − uncond[r, i]) · g
```

Reference: `TopKGuidedReference.Apply`. Contract: `TopKGuidedContract.Validate`. Ties at the threshold stay candidates. F32
rows of `[rows, width]` with `1 ≤ topK ≤ width`; the −∞ entries are the contract, so a sampler that follows draws only from the
conditional top-K (MiniMax Music 3's semantic draw).

## Row index ops

```
GatherRows      output[m] = input[indices[m]]
ScatterAddRows  output[indices[m]] += weights[m] · input[m]     weights optional (1 without); each destination row named at most once
GatherAddRows   h[r] += table[indices[r]]                       for every row of h
```

Reference: `RowIndexReference.Gather`, `ScatterAdd`, `GatherAdd`. Contract: `RowIndexContract.ValidateGather`, `ValidateScatterAdd`,
`ValidateGatherAdd`. Rows are the last extent of the indexed tensor; F32 rows, I32 index, F32 weights. The index is bounds-checked
(and, for the scatter-add, checked for repeats) when its host bytes are current (`Tensor.HostBytesCurrent`); an index that lives on
the device is the caller's to keep in range and, for the scatter-add, free of repeats, because the kernels add without atomics.
`GatherRows` and `ScatterAddRows` also take host spans, which become host tensors for the call (`IndexTensor.Rows`, `Values`).
`TryGatherRowsResident` is the residency probe beside them: a gather that runs only when the input is already on the device.

## GridScatter, GridGather

```
cell(v)              = (d · Height + h) · Width + w       (d, h, w) = coords[v, FirstCoordColumn .. +2] of a CoordStride-wide row
GridScatter          grid[c, cell(v)] = features[v, c]    in place; every other cell keeps its bytes; each cell written at most once
GridGather           features[v, c] = grid[c, cell(v)]
```

Reference: `GridReference.Scatter`, `Gather`. Contract: `GridContract.Validate`. A `GridSpec` names the grid extents and how a
coordinate row is read; `GridSpec.Cube(r, 4, 1)` is TRELLIS's `[batch, d, h, w]` table over an `r³` grid. The grid holds
`channels · Depth · Height · Width` F32 elements (`[1, C, D, H, W]` or `[C, D, H, W]`), features `[voxels, channels]`, coordinates
I32. Coordinates are range-checked (and, for the scatter, checked for repeated cells) when their host bytes are current.

## NeighborhoodAttention

```
out[b, t, h, w, head, :] = softmax_j(scale · q · k_j) · v_j     j over the window [t0, t0 + kT) × [h0, h0 + kH) × [w0, w0 + kW)
start(i, n, k)           = clamp(i − k / 2, 0, n − k)             the window slides inward at the borders (NATTEN's rule)
```

Reference: `NeighborhoodAttentionReference.Apply`. Contract: `NeighborhoodAttentionContract.Validate` (`WindowStart` is the
border rule). F32 `[batch, T, H, W, heads, headDim]` q, k and v; a `NeighborhoodWindow` extent longer than its axis collapses to
the axis; the output is the q grid or its flat `[tokens, heads · headDim]` form, so a caller whose next op is row-wise allocates
it flat and skips a reshape. CUDA runs the per-query kernel, or the query-tiled kernels at head width 64 unless
`numerics.neighborhoodAttentionTiled` is off (the tiled path is an online softmax over the tile's union window, a different
summation order).

## FakeQuantizeUniform

```
out[i] = low + trunc((clamp(in[i], low, high) − low) / (high − low) · (levels − 1)) / (levels − 1) · (high − low)
```

Reference: `FakeQuantizeReference.Apply`. Contract: `FakeQuantizeContract.Validate`. F32; `low < high`, at least two levels.
DIAMOND's pixel quantization is `(−1, 1, 256)`.

## FourierEmbed

```
row = [ x₀ … x_{D−1} (IncludeInput) | sin(x_c · 2^b) for c, then b | cos(x_c · 2^b), the same order ]      width = (IncludeInput ? D : 0) + 2 · D · bands
```

Reference: `FourierEmbedReference.Apply`. Contract: `FourierEmbedContract.Validate`. A `FourierEmbedSpec` names the input
dimensions, the bands (at most 30, so `2^b` stays an exact integer) and whether the coordinates lead the row. F32 `[count, D]`
coordinates. Hunyuan3D's query embedding is `(3, 8, IncludeInput: true)`.

## Exp, Reciprocal

```
out[i] = e^in[i]        out[i] = 1 / in[i]
```

Reference: `ElementwiseMathReference.Exp` and `.Reciprocal`. Contract: `ElementwiseMathContract.Validate`. Input and output are
each F32 or BF16, with equal element counts (shapes may differ); the value is read as F32, mapped in F32 and rounded once to the
output dtype, BF16 to nearest even. The division is IEEE (`±0` gives `±∞`). `exp` follows each backend's math library, so it
is held to a tolerance. The output shares no storage with the input. SANA-WM's Gated-DeltaNet decay rate and denominator use them.

## FrameDeltaScan

```
per head h, frame t:  G_t = Σ_s β k kᵀ   U_t = Σ_s β k vᵀ   S ← d_t·(S − G_t·S) + U_t
F_t = S after frame t (forward, inclusive)      R_t = S from frames after t (reverse, exclusive)
out[n = t·spatial + s, h, :] = q[n, h, :] · (F_t + R_t)
```

Reference: `FrameDeltaScanReference.Run`; validation `FrameDeltaScanReference.Validate`. Token-major F32 `query` and `key`
`[N, heads·keyDim]`, `value` `[N, heads·valueDim]` or null for a value of 1 (`valueDim` 1: a linear-attention denominator),
F32 `beta [heads, frames, spatial]` and `decay [heads, frames]`. A frame's tokens update the state together (the SANA-WM
Gated-DeltaNet, main branch numerator and denominator and camera branch). The CUDA build computes every (head, frame) Gram
and injection with two batched GEMMs over the token-major operands in place (row stride `heads·dim`), runs the two
recurrences as parallel blocks of one kernel, and reads the tokens out with a third batched GEMM; the GEMMs run at the
backend's F32 GEMM precision (TF32 by default). It is the documented exception to the execution requirements below: it
allocates its intermediates (the β-scaled keys, Grams, injections and the two state sets), is six launches, and uploads
its GEMM pointer arrays per call, so it is not graph-capture safe.

## DepthwiseConvGlu

```
d = depthwise_k×k(act(input)) + bias        output = d[..., :H] · SiLU(d[..., H:])        act = SiLU or identity
```

Reference: `DepthwiseConvGluReference.Run`. Channels-last F32 or BF16 `input [frames, height, width, 2H]` (value channels,
then gate), F32 tap-major `taps [k, k, 2H]`, optional F32 `bias [2H]`, output `[frames, height, width, H]` in the input
dtype. Stride 1, zero padding `k/2`. With `inputSilu` the convolution reads `SiLU(input)`, so the activation of the GEMM that
produced the input needs no pass of its own (SANA-WM's GLUMBConv). The convolution output is never stored; the product is
rounded once. CUDA: one thread per output element, or, for BF16 3×3 with whole 8-channel groups and 4-pixel runs, one
thread per 8 channels of an image row that loads each input column once per run.

## UcpeTransform

```
first headDim/2 of each head:  y[4g..4g+3] = M_n · x[4g..4g+3]          (per-token 4×4 matrix)
last headDim/2:                (x0, x1) → (x0·c − x1·s, x1·c + x0·s)   (angle at the pair's even index)
```

Reference: `UcpeTransformReference.Run`. F32 token-major `input` and `output [N, heads·headDim]`, `matrices [N, 4, 4]`
row-major, `cos` and `sin [N, headDim/2]` (negate `sin` for the inverse rotation). SANA-WM's camera branch (UCPE) applies
it to q, k, v and its readout; it replaces a gather, a batched 4×4 GEMM, a scatter, a copy and a rotary.

## DepthwiseConvFrames

```
out[t, s, c] = Σ_j weight[c, j] · input[t + j − pad, s, c]      (zero outside the clip)
```

Reference: `DepthwiseConvFramesReference.Run`. F32 token-major `[frames, spatial, channels]` in and out, F32 `weight` of
`channels·taps` values. A depthwise conv along the frame axis without transposing to channel-first: the SANA-WM key paths'
bidirectional temporal conv (a symmetric 7-tap kernel).

## RmsNormRelu

```
out = max(x · rsqrt(mean(x²) + eps) · w, 0) · scale      per row of weight.ElementCount values
```

The default composes `RmsNorm`, `LeakyRelu(0)` and `Scale`; CUDA fuses them into one warp-per-row kernel for F32. The output
may be shaped differently from the input (same element count), as the SANA-WM camera keys are.

## Execution requirements

Every implementation of these ops:

- allocates nothing on the device beyond the output buffer (an in-place destination reuses the output's buffer);
- adds no host synchronization and reads no host memory of a device-resident operand;
- is graph-capture safe: the geometry is passed by value as kernel parameters or push constants, never through
  constant memory or a device-side spec buffer;
- is one launch per call, except the globally normalized `GuidedEulerStep`, which is three (the partial sums, the
  ratio, the update), as the op it replaced was.

## Backend coverage

| Op | CPU | CUDA | Vulkan |
|---|---|---|---|
| `CopyStrided` | reference | `layout_copy.ptx` (`copy_strided_u8/u16/u32/u64`) | `copy_strided_u16/u32/u64`; 1-byte elements take the reference |
| `AccumulateStrided` | reference | `accumulate_strided_f32/f16/bf16` | `accumulate_strided_f32/f16`; BF16 takes the reference |
| `Pad` | reference | `pad_u8/u16/u32/u64` | `pad_strided_u16/u32/u64`; 1-byte elements take the reference |
| `Rotary` | reference | `rotary.ptx` (`rotary_f32/f16/bf16`) | `rotary_f32/f16`; BF16 takes the reference |
| `QkvDecodeEpilogue` | no device position buffer | `lm_qkv_rope_scatter_f32/f16kv`, `lm_qknorm_rope_scatter_f32/f16kv`, `lm_qknorm_full_rope_scatter_f32/f16kv` | composed; F32 caches only |
| `NormModulate` | reference | `norm_modulate.ptx` (`norm_modulate_rows/flat/channel_<in>_<out>`: F32, F16, F32↔F16, BF16) | `norm_modulate_rows/flat_f32/f16/f32_f16/f16_f32`, `norm_modulate_channel_f32`; BF16 and a non-F32 channel norm take the reference |
| `ChunkActivate` | reference | `chunk_activate_f32` | `chunk_activate` |
| `Glu` | reference | `glu_gate.ptx` (`glu_f32/f16/bf16`) | `glu_f32/f16`; BF16 takes the reference |
| `GroupScale` | reference | `group_scale_f32`, `_f16_f32`, `_f16_f16`, `_bf16_f32`, `_bf16_bf16` | `group_scale_f32/f16/f16_f16`; BF16 takes the reference |
| `LinearFused` | composed | int8 fused epilogues where eligible, else composed | composed |
| `GuidedEulerStep` | reference | `guided_euler.ptx` (`guided_euler_f32`; `_reduce/_finalize/_scaled` for the global norm; `_rows` per row) | `guided_euler`; the normalized forms take the reference |
| `TopKGuidedLogits` | reference | `topk_guided_logits.ptx` (`topk_guided_logits_f32`, rows up to 16448 wide) | reference |
| `GatherRows`, `ScatterAddRows`, `GatherAddRows` | reference | `row_index.ptx` (`gather_rows_f32`, `scatter_add_rows_f32`, `gather_add_rows_f32`) | reference (`TryGatherRowsResident` copies resident rows) |
| `GridScatter`, `GridGather` | reference | `grid_scatter_f32`, `grid_gather_f32` | reference |
| `NeighborhoodAttention` | reference | `neighborhood_attention.ptx` (`neighborhood_attention_f32`, `_tiled8x8_f32`, `_tiled4x8_f32`) | reference |
| `FakeQuantizeUniform` | reference | `point_ops.ptx` (`fake_quantize_uniform_f32`) | reference |
| `FourierEmbed` | reference | `fourier_embed_f32` | reference |
| `Exp`, `Reciprocal` | reference | `point_ops.ptx` (`elementwise_math_<in>_<out>`, F32 and BF16 each side) | reference |
| `FrameDeltaScan` | reference | `frame_scan.ptx` (`frame_scan_scale_keys_f32`, `frame_scan_recurrence_f32`, `_col_f32`, `frame_scan_sum_f32`) around cuBLAS batched GEMMs | reference |
| `DepthwiseConvGlu` | reference | `depthwise_glu.ptx` (`depthwise_glu_f32/bf16`, `depthwise_glu_bf16_vec`) | reference |
| `UcpeTransform` | reference | `ucpe.ptx` (`ucpe_transform_f32`) | reference |
| `DepthwiseConvFrames` | reference | `frame_conv.ptx` (`depthwise_conv_frames_f32`) | reference |
| `RmsNormRelu` | composed | `rms_relu.ptx` (`rms_norm_relu_f32`); other dtypes composed | composed |

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
are `QkvDecodeEpilogue` calls built with `QkvDecodeCall`'s factories (`--op qkvdecode`). `AffineBroadcastRowIndexed`,
`LayerNormModulate`, `RmsNormModulate`, `Ltx2RmsModulate`, `OasisAdaLn`, `WanRmsNormChannel` and `ChannelLayerNorm3d` are
`NormModulate` calls, `ModulationSplit4` is a `ChunkActivate` (`--op normmod`). `GluActivate`, `GluActivateEmitQ8`, `GeGlu`,
`GegluErf` and `SituGlu` are `Glu` calls; `MaskRows`, `Ltx2HeadGate` and `RowGatedAccumulate` are `GroupScale`; `LinearGelu`
and `LinearHeadGated` are `LinearFused`; `MoeTopKGate` is `MoeRoute` with its dense output (`--op gating`). `CfgEulerStep`, `CfgRenormEulerStep` and
`CfgNormalizedEulerStep` are `GuidedEulerStep` calls built with `GuidedEulerSpec`'s factories; `CfgTopKGuided` is
`TopKGuidedLogits` (`--op guidance`). The span-form `GatherRows` and `RowGather` are `GatherRows`; `ScatterAddWeightedRows`
and `RowScatterAdd` are `ScatterAddRows`; `IndexAddRows` is `GatherAddRows`; `SparseScatterToGrid` and `SparseGatherFromGrid`
are `GridScatter` and `GridGather` with a `GridSpec` (`--op rowindex`). `Na3d` is `NeighborhoodAttention` with a
`NeighborhoodWindow` (`--op neighborhood`); `PixelQuantize` is `FakeQuantizeUniform(−1, 1, 256)` and `FourierEmbed` takes a
`FourierEmbedSpec` (`--op point`).
