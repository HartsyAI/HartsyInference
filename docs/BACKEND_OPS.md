# Backend op contracts

The semantics of the generic `IBackend` ops that replaced model-named ones. Each section is normative: the host
reference named in it is the executable form, every accelerator kernel is checked against that reference with
`tools/OpParity` ([VERIFICATION.md](VERIFICATION.md)), and a kernel that disagrees is wrong whatever a model's output
looks like. Code comments point here instead of restating a contract.

## Strided views

`StridedView` (`src/HartsyInference.Core/Backends/Layout/`) is a window onto a tensor's storage: one to six extents,
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

The Vulkan shaders index with 32-bit words, so a view reaching past 2³² − 1 elements is refused there.

## What these replaced

The Oasis and Matrix-Game 3 head splits and merges, the Matrix-Game 3 keyboard K/V broadcast and mouse-window concat,
the Wan 2.2 VAE QKV split and token transpose, the GQA `RepeatKvHeads`, the KV `SliceTimeRange`, the CausalConv3d frame
extract, frame write, bias fill and tap accumulate, Ideogram 4's `ScatterRowsAfter` and the SeedVR2 bottom-right pad
are all one `CopyStrided`, `AccumulateStrided` or `Pad` call at their call sites; `tools/OpParity --op layout` holds
one case per former call shape.
