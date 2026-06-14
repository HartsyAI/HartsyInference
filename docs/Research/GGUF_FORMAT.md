# GGUF Format — Research Notes

## Summary

GGUF (GGML Universal File) is a binary format for storing quantized ML model weights, designed for efficient `mmap`-based loading and single-file deployment. It is the native format for llama.cpp (LLMs) and stable-diffusion.cpp (diffusion models). The current version is **v3**. The format consists of a fixed header, typed key-value metadata, tensor descriptors, alignment padding, and a contiguous tensor data blob. Quantization is block-based: weights are grouped into blocks of 32 (legacy types) or super-blocks of 256 (K-quant types), each carrying its own scale factors. Dequantization reconstructs floats by multiplying quantized integers by per-block scales and adding per-block minimums where applicable.

Key facts for HartsyInference:
- All integers are little-endian by default (v3 adds big-endian support but no flag marks it).
- Strings are length-prefixed (`uint64` length + UTF-8 bytes, no null terminator).
- Default alignment is 32 bytes; overridable via `general.alignment` metadata key.
- sd.cpp detects architecture via tensor name patterns, not dedicated metadata keys.
- Q4_K_M is not a single block type; it is a mixed-precision strategy using Q4_K and Q6_K blocks on different tensors.

---

## Detailed Findings

### 1. GGUF Version History

| Version | Key Change |
|---------|-----------|
| v1 | Initial release. Counts (`tensor_count`, `metadata_kv_count`) were `uint32`. |
| v2 | Changed all countable values (lengths, counts) from `uint32` to `uint64` to support larger models. |
| v3 | Added big-endian support. When big-endian, all values including metadata and tensors follow big-endian byte order. No explicit endianness flag exists; readers must detect or assume little-endian. |

The version field is `uint32` at offset 0x04. Only structural format changes warrant a version bump; metadata-only changes do not.

### 2. Header Structure (24 bytes fixed + variable metadata)

```
Offset  Size   Type       Field                Description
0x00    4      uint32     magic                0x47475546 = "GGUF" in ASCII
0x04    4      uint32     version              Must be 3 for current spec
0x08    8      uint64     tensor_count         Number of tensors in the file
0x10    8      uint64     metadata_kv_count    Number of metadata key-value pairs
0x18    var    kv[]       metadata_kv          Array of metadata_kv_count entries
```

The magic bytes in order are: `0x47 ('G'), 0x47 ('G'), 0x55 ('U'), 0x46 ('F')`.

### 3. String Type (`gguf_string_t`)

```c
struct gguf_string_t {
    uint64_t len;        // byte length of string
    char     string[len]; // UTF-8, NOT null-terminated
};
```

Keys must be valid ASCII, hierarchical with `.` separators, `lower_snake_case`, max 65535 bytes.

### 4. Metadata Value Types

The value type is encoded as a `uint32` preceding the value data:

| ID | Type Name | Size | Notes |
|----|-----------|------|-------|
| 0  | UINT8     | 1 byte | |
| 1  | INT8      | 1 byte | |
| 2  | UINT16    | 2 bytes | little-endian |
| 3  | INT16     | 2 bytes | little-endian |
| 4  | UINT32    | 4 bytes | little-endian |
| 5  | INT32     | 4 bytes | little-endian |
| 6  | FLOAT32   | 4 bytes | IEEE 754 |
| 7  | BOOL      | 1 byte | 0 = false, 1 = true |
| 8  | STRING    | variable | `gguf_string_t` (uint64 length + UTF-8 bytes) |
| 9  | ARRAY     | variable | uint32 element_type + uint64 count + count elements |
| 10 | UINT64    | 8 bytes | little-endian |
| 11 | INT64     | 8 bytes | little-endian |
| 12 | FLOAT64   | 8 bytes | IEEE 754 double |

A metadata KV entry is serialized as:
```
gguf_string_t  key
uint32         value_type
<value data>   value     (size depends on value_type)
```

For ARRAY type, the layout is:
```
uint32  element_type   (one of the types above, excluding ARRAY)
uint64  count
<element_type data>[count]
```

### 5. Tensor Descriptor (`gguf_tensor_info_t`)

```
Field            Type              Notes
name             gguf_string_t     Max 64 bytes. Tensor identifier.
n_dimensions     uint32            Currently max 4.
dimensions       uint64[n_dims]    Size of each axis.
type             uint32            ggml_type enum value (see below).
offset           uint64            Byte offset from start of tensor_data section.
                                   Must be a multiple of general.alignment (default 32).
```

After all tensor info entries, the file is padded with `0x00` bytes to the next alignment boundary. Then the tensor data section begins.

### 6. `ggml_type` Enum (Quantization Type IDs)

| ID | Name      | Block Size | Bits/Weight | Bytes/Block |
|----|-----------|-----------|-------------|-------------|
| 0  | F32       | 1         | 32          | 4           |
| 1  | F16       | 1         | 16          | 2           |
| 2  | Q4_0      | 32        | 4.5         | 18          |
| 3  | Q4_1      | 32        | 5.0         | 20          |
| 6  | Q5_0      | 32        | 5.5         | 22          |
| 7  | Q5_1      | 32        | 6.0         | 24          |
| 8  | Q8_0      | 32        | 8.5         | 34          |
| 9  | Q8_1      | 32        | 9.0         | 36          |
| 10 | Q2_K      | 256       | 3.125       | — |
| 11 | Q3_K      | 256       | 3.4375      | — |
| 12 | Q4_K      | 256       | 4.5         | 144         |
| 13 | Q5_K      | 256       | 5.5         | 176         |
| 14 | Q6_K      | 256       | 6.5625      | 210         |
| 15 | Q8_K      | 256       | 8.5         | — |
| 16 | IQ2_XXS   | 256       | 2.0625      | — |
| 17 | IQ2_XS    | 256       | 2.3125      | — |
| 18 | IQ3_XXS   | 256       | 3.0625      | — |
| 19 | IQ1_S     | 256       | 1.5         | — |
| 20 | IQ4_NL    | 32        | 4.5         | — |
| 21 | IQ3_S     | 256       | 3.4375      | — |
| 22 | IQ2_S     | 256       | 2.5         | — |
| 23 | IQ4_XS    | 256       | 4.25        | — |
| 24 | I8        | 1         | 8           | 1           |
| 25 | I16       | 1         | 16          | 2           |
| 26 | I32       | 1         | 32          | 4           |
| 27 | I64       | 1         | 64          | 8           |
| 28 | F64       | 1         | 64          | 8           |
| 29 | IQ1_M     | 256       | 1.75        | — |
| 30 | BF16      | 1         | 16          | 2           |
| 34 | TQ1_0     | 256       | 1.6875      | — |
| 35 | TQ2_0     | 256       | 2.0625      | — |
| 39 | MXFP4     | 32        | —           | — |
| 40 | COUNT     | —         | —           | — (sentinel) |

Note: IDs 4, 5, 31-33, 36-38 are unused/reserved. The COUNT sentinel marks the end of the enum.

### 7. File Layout Sequence

```
┌─────────────────────────────────────┐
│ Header (24 bytes)                   │
│   magic + version + counts          │
├─────────────────────────────────────┤
│ Metadata KV pairs (variable)        │
│   metadata_kv_count entries         │
├─────────────────────────────────────┤
│ Tensor Info Array (variable)        │
│   tensor_count entries              │
├─────────────────────────────────────┤
│ Alignment Padding (0x00 bytes)      │
│   Pad to next multiple of alignment │
├─────────────────────────────────────┤
│ Tensor Data (contiguous blob)       │
│   Each tensor at its stated offset  │
│   Offsets relative to this section  │
└─────────────────────────────────────┘
```

### 8. Alignment Rules

- Default alignment: **32 bytes** (defined by `GGUF_DEFAULT_ALIGNMENT`).
- Overridable via the `general.alignment` metadata key (must be a power of 2 and >= 8).
- Alignment formula: `aligned = offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT`
- Tensor data offsets are **relative to the start of the tensor data section**, not to the start of the file.
- Each tensor's data offset must be a multiple of the alignment value.

---

## Key Numbers / Constants

| Constant | Value | Notes |
|----------|-------|-------|
| GGUF_MAGIC | `0x47475546` | "GGUF" in little-endian uint32 |
| GGUF_VERSION | 3 | Current spec version |
| GGUF_DEFAULT_ALIGNMENT | 32 | Bytes; overridable via metadata |
| QK4_0 | 32 | Block size for Q4_0 |
| QK8_0 | 32 | Block size for Q8_0 |
| QK_K | 256 | Super-block size for all K-quants |
| K_SCALE_SIZE | 12 | Bytes for packed scales/mins in Q4_K/Q5_K |
| Max tensor name | 64 bytes | Per spec |
| Max metadata key | 65535 bytes | 2^16 - 1 |
| Max tensor dimensions | 4 | Current limit |

---

## Data Layouts / Formats

### Q4_0 Block Layout (18 bytes, 32 weights)

```c
#define QK4_0 32
typedef struct {
    ggml_half d;             // FP16 scale factor (2 bytes)
    uint8_t   qs[QK4_0 / 2]; // 16 bytes: 32 x 4-bit quantized values, packed as nibbles
} block_q4_0;
// Total: 2 + 16 = 18 bytes per 32 weights = 4.5 bits/weight
```

**Nibble packing**: Each byte in `qs[]` stores two 4-bit values. The **low nibble** (bits 0-3) stores the first value and the **high nibble** (bits 4-7) stores the second value. For the i-th weight:
- If `i < 16`: value is in `qs[i] & 0x0F` (low nibble)
- If `i >= 16`: value is in `(qs[i-16] >> 4) & 0x0F` (high nibble)

Raw nibble values are unsigned in [0, 15]. They are converted to signed by subtracting 8, giving a range of [-8, +7].

**Dequantization (Q4_0)**:
```
For each block b:
    d = float(b.d)                    // FP16 -> float
    For i = 0..15:
        q_lo = (b.qs[i] & 0x0F) - 8  // low nibble, signed
        q_hi = (b.qs[i] >> 4) - 8    // high nibble, signed
        output[i]    = d * q_lo
        output[i+16] = d * q_hi
```

This is **symmetric quantization** (no zero-point / minimum offset). The formula is: `w = d * (nibble - 8)`.

---

### Q8_0 Block Layout (34 bytes, 32 weights)

```c
#define QK8_0 32
typedef struct {
    ggml_half d;          // FP16 scale factor (2 bytes)
    int8_t    qs[QK8_0];  // 32 bytes: 32 x signed 8-bit quantized values
} block_q8_0;
// Total: 2 + 32 = 34 bytes per 32 weights = 8.5 bits/weight
```

**Dequantization (Q8_0)**:
```
For each block b:
    d = float(b.d)              // FP16 -> float
    For i = 0..31:
        output[i] = d * b.qs[i] // int8 * scale
```

This is the simplest quantization. Each int8 value is directly multiplied by the FP16 scale. Essentially lossless (~0.01 perplexity increase vs FP16).

---

### Q4_K Block Layout (144 bytes, 256 weights)

```c
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    ggml_half d;                     // FP16 super-block scale for quantized scales (2 bytes)
    ggml_half dmin;                  // FP16 super-block scale for quantized mins (2 bytes)
    uint8_t   scales[K_SCALE_SIZE];  // 12 bytes: 8 x (6-bit scale + 6-bit min), packed
    uint8_t   qs[QK_K / 2];         // 128 bytes: 256 x 4-bit quants packed as nibbles
} block_q4_K;
// Total: 2 + 2 + 12 + 128 = 144 bytes per 256 weights = 4.5 bits/weight
```

Q4_K is a **type-1 (asymmetric) K-quant** with 8 sub-blocks of 32 weights each. It uses **double quantization**: the per-sub-block scale and minimum values are themselves quantized to 6 bits.

**6-bit Scale Packing in the 12-byte `scales[]` array**:

The 12 bytes encode 8 scale values and 8 min values, each 6 bits. The packing uses a bit-interleaving pattern optimized for SIMD:

```
Byte  Contents (uppercase = scale bits, lowercase = min bits)
 0:   EEAAAAAA    (low 6 bits of scale[0], high 2 bits = bits 0-1 of scale[4])
 1:   FFBBBBBB    (low 6 bits of scale[1], high 2 bits = bits 0-1 of scale[5])
 2:   GGCCCCCC    (low 6 bits of scale[2], high 2 bits = bits 0-1 of scale[6])
 3:   HHDDDDDD    (low 6 bits of scale[3], high 2 bits = bits 0-1 of scale[7])
 4:   eeaaaaaa    (low 6 bits of min[0],   high 2 bits = bits 0-1 of min[4])
 5:   ffbbbbbb    (low 6 bits of min[1],   high 2 bits = bits 0-1 of min[5])
 6:   ggcccccc    (low 6 bits of min[2],   high 2 bits = bits 0-1 of min[6])
 7:   hhdddddd    (low 6 bits of min[3],   high 2 bits = bits 0-1 of min[7])
 8:   eeeeEEEE    (high 4 bits of scale[4] in upper nibble, high 4 bits of min[4] ... wait)
 9:   ffffFFFF    (bits 2-5 of scale[5] in upper, bits 2-5 of min[5] in lower)
10:   ggggGGGG
11:   hhhhHHHH
```

More precisely, the unpacking algorithm for sub-block `j` (0..7):

```
if j < 4:
    sc = scales[j] & 63                                   // low 6 bits
    m  = scales[j + 4] & 63                               // low 6 bits
else:
    sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)  // combine nibble + 2 high bits
    m  = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)  // combine nibble + 2 high bits
```

Where `sc` is the 6-bit quantized scale index (0..63) and `m` is the 6-bit quantized min index (0..63) for sub-block `j`.

**Dequantization (Q4_K)**:
```
For each super-block b:
    d_all  = float(b.d)       // super-block scale
    dmin   = float(b.dmin)    // super-block min scale

    For each sub-block j = 0..7:
        sc, m = unpack_scales(b.scales, j)   // 6-bit values as above
        block_scale = d_all * sc
        block_min   = dmin * m

        For i = 0..31:  (32 weights in this sub-block)
            global_idx = j * 32 + i
            if i < 16:
                q = b.qs[global_idx / 2 ...] & 0x0F     // low nibble
            else:
                q = (b.qs[(global_idx - 16) / 2 ...] >> 4) & 0x0F  // high nibble
            output[global_idx] = q * block_scale - block_min
```

The formula per weight is: **`w = q * (d * sc) - (dmin * m)`** where `q` is the unsigned 4-bit value [0..15].

Note: Some sources write this as `w = q * block_scale + block_min` with `block_min` being negative. The sign convention varies, but the effect is the same: an asymmetric affine mapping from [0,15] to the original value range.

---

### Q6_K Block Layout (210 bytes, 256 weights)

```c
typedef struct {
    uint8_t  ql[QK_K / 2];     // 128 bytes: lower 4 bits of each 6-bit quant
    uint8_t  qh[QK_K / 4];     //  64 bytes: upper 2 bits of each 6-bit quant
    int8_t   scales[QK_K / 16]; //  16 bytes: 16 x int8 scales (one per 16-weight block)
    ggml_half d;                //   2 bytes: FP16 super-block scale
} block_q6_K;
// Total: 128 + 64 + 16 + 2 = 210 bytes per 256 weights = 6.5625 bits/weight
```

Q6_K is a **type-0 (symmetric) K-quant** with 16 sub-blocks of 16 weights each. Scales are quantized to 8 bits (int8).

**6-bit Quant Reconstruction from `ql[]` and `qh[]`**:

Each weight has a 6-bit quantized value split across two arrays:
- `ql[]` stores the lower 4 bits (128 bytes = 256 nibble-pairs, 2 per byte)
- `qh[]` stores the upper 2 bits (64 bytes = 256 x 2-bit values, 4 per byte)

Reconstruction for weight `i` (0..255):
```
q_lo = (ql[i / 2] >> ((i % 2) * 4)) & 0x0F    // 4-bit low part
q_hi = (qh[i / 4] >> ((i % 4) * 2)) & 0x03    // 2-bit high part
q6   = q_lo | (q_hi << 4)                      // full 6-bit value [0..63]
q_signed = q6 - 32                              // center to [-32..+31]
```

**Dequantization (Q6_K)**:
```
For each super-block b:
    d_all = float(b.d)       // super-block scale

    For each sub-block j = 0..15:
        sc = b.scales[j]     // int8 scale for this 16-weight block
        block_scale = d_all * sc

        For i = 0..15:
            global_idx = j * 16 + i
            q6 = reconstruct_6bit(b.ql, b.qh, global_idx)
            q_signed = q6 - 32
            output[global_idx] = block_scale * q_signed
```

The formula per weight is: **`w = (d * scale_j) * (q6 - 32)`** where `q6` is the unsigned 6-bit value [0..63] and `scale_j` is the int8 per-block scale.

---

### Q4_K_M Mixed-Precision Strategy

Q4_K_M is **not** a single block type. It is a quantization strategy that assigns different block types to different tensors based on sensitivity:

| Tensor Type | Assigned Quant |
|-------------|---------------|
| Embeddings, output projections | Q6_K |
| Half of attention.wv tensors | Q6_K |
| Half of feed_forward.w2 tensors | Q6_K |
| All other tensors | Q4_K |

This mixed approach preserves quality on attention-critical tensors while aggressively compressing less sensitive ones.

---

## Algorithm Steps

### Reading a GGUF File (Parser Algorithm)

1. **Read header**: 24 bytes. Verify magic = `0x47475546`, version >= 2.
2. **Read metadata**: For `metadata_kv_count` entries:
   a. Read key as `gguf_string_t` (uint64 len + bytes).
   b. Read value type as `uint32`.
   c. Read value based on type (for ARRAY: read element_type + count + elements).
   d. If key == `general.alignment`, store alignment override.
3. **Read tensor info**: For `tensor_count` entries:
   a. Read name as `gguf_string_t`.
   b. Read `n_dimensions` as `uint32`.
   c. Read `dimensions` as `uint64[n_dimensions]`.
   d. Read `type` as `uint32` (ggml_type).
   e. Read `offset` as `uint64`.
4. **Compute tensor data start**: Current file position, padded to alignment boundary.
5. **Map tensor data**: Each tensor's absolute file position = `tensor_data_start + tensor.offset`.
6. **Dequantize on demand**: When a tensor is needed, read its raw bytes and apply the block-type-specific dequantization.

### Dequantizing a Tensor (General)

1. Determine the block type from the tensor's `ggml_type`.
2. Compute `n_blocks = total_elements / block_size`.
3. For each block, read the block struct and apply the type-specific formula.
4. Output float32 array of `total_elements` values.

---

## Reference Implementations

| Source | URL | License | Notes |
|--------|-----|---------|-------|
| GGUF spec (canonical) | [ggml-org/ggml/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) | MIT | Definitive format specification |
| llama.cpp ggml-common.h | [ggml-org/llama.cpp/.../ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h) | MIT | Block struct definitions |
| llama.cpp ggml-quants.c | ggml-org/llama.cpp ggml/src/ggml-cpu/quants.c | MIT | Reference dequantize_row_* functions |
| llama.cpp Tensor Encoding Wiki | [Tensor Encoding Schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes) | MIT | Scale packing diagrams |
| stable-diffusion.cpp | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | MIT | Diffusion model GGUF loading |
| ComfyUI-GGUF | [city96/ComfyUI-GGUF](https://deepwiki.com/city96/ComfyUI-GGUF) | — | Python dequantization + loader |
| k-quants PR | [llama.cpp PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684) | MIT | Original K-quant implementation |
| GGUF structural guide | [Malcolm Mill guide](https://malcolm-mill.github.io/LLM/gguf-file-structure-guide/) | — | Clear byte-offset tables |

**IMPORTANT**: dotLLM's GGUF implementation is GPLv3-licensed. Do NOT reference it for HartsyInference.

---

## Differences Between Implementations

### llama.cpp vs sd.cpp GGUF Usage

| Aspect | llama.cpp | sd.cpp |
|--------|-----------|--------|
| Architecture detection | Reads `general.architecture` metadata key | Detects via tensor name patterns (e.g., `double_blocks.*` for Flux, `joint_blocks.*` for SD3) |
| Supported quant types | All types including IQ-quants | Primarily f16, f32, q8_0, q5_0, q5_1, q4_0, q4_1 |
| K-quant support | Full (Q2_K through Q8_K) | Limited (some K-quant types cause overflow in activation dequantization for certain models) |
| Metadata keys | Rich: tokenizer data, architecture params, training config | Minimal: `general.architecture` sometimes set to "pig" for sd.cpp-converted files |
| Tensor naming | Standardized via TensorNameMap | Uses `convert_tensors_name()` to normalize across ckpt/safetensors/diffusers sources |

### sd.cpp Architecture Detection Patterns

| Architecture | Detection Pattern |
|-------------|------------------|
| Flux | `double_blocks.*` tensor names |
| SD3 | `joint_blocks.*` tensor names |
| WAN | `cross_attn.*.norm_k` tensor names |
| SDXL | Presence of `label_emb.0.0` tensor |
| SD 1.x/2.x | UNet tensor naming patterns |

### ComfyUI-GGUF Metadata Keys

ComfyUI-GGUF uses additional metadata keys not found in llama.cpp or sd.cpp:
- `general.architecture` — model type (flux, sd3, sdxl, etc.)
- `tokenizer.ggml.*` — SentencePiece tokenizer data for text encoders
- `comfy.gguf.orig_shape.*` — original tensor dimensions for reshaped Conv2D models (needed to restore shape after quantization-friendly reshaping)

### Legacy vs K-Quant Comparison

| Property | Legacy (Q4_0, Q8_0) | K-Quant (Q4_K, Q6_K) |
|----------|---------------------|----------------------|
| Block size | 32 weights | 256 weights (super-block) |
| Scale storage | 1 FP16 per 32 weights | 1 FP16 super-scale + quantized sub-scales per 256 weights |
| Scale precision | Full FP16 | 6-bit (Q4_K) or 8-bit (Q6_K) quantized |
| Zero-point/min | None (Q4_0) or FP16 (Q4_1) | 6-bit quantized min (Q4_K) or none (Q6_K) |
| Dequant complexity | Simple multiply | Unpack scales, then multiply + offset |
| Quality/size ratio | Lower | Higher (double quantization saves metadata bits) |

---

## Open Questions

- [ ] Exact nibble indexing for Q4_K `qs[]` array — verify against `dequantize_row_q4_K` in ggml-quants.c
- [ ] Q6_K bit reconstruction — verify `ql[]`/`qh[]` indexing against ggml-quants.c (may use complex interleaving)
- [ ] Whether sd.cpp GGUF files ever set `general.alignment` to non-default values
- [ ] Big-endian detection strategy — check if magic bytes are reversed (`0x46554747`)?

---

## Implementation Notes

### For HartsyInference.ModelHandler

1. **Parser structure**: Implement a streaming reader that reads the header, then lazily reads metadata and tensor info. Use `mmap` (or `MemoryMappedFile` in .NET) for the tensor data section to avoid loading all weights into memory.

2. **FP16 conversion**: `ggml_half` / `ggml_fp16_t` is IEEE 754 half-precision (binary16). In .NET 10, use `System.Half` for direct conversion. For older targets, use `BitConverter` with the standard FP16 bit layout (1 sign, 5 exponent, 10 mantissa).

3. **Alignment**: After reading all metadata KV pairs and tensor info entries, compute padding as `(alignment - (position % alignment)) % alignment` and skip that many bytes. This is where tensor data begins.

4. **Dequantization priority**: Implement in this order:
   - F32 / F16 / BF16 (trivial, just type conversion)
   - Q8_0 (simplest quantized: scale * int8)
   - Q4_0 (nibble unpacking + offset - 8)
   - Q6_K (6-bit reconstruction from ql/qh + int8 scales)
   - Q4_K (6-bit scale unpacking from 12-byte array + 4-bit quants)

5. **Block size constants**: Use `QK4_0 = 32`, `QK8_0 = 32`, `QK_K = 256` as compile-time constants. Always use `QK_K = 256` (the `GGML_QKK_64` mode with `QK_K = 64` is a special Qualcomm build variant and not used in standard GGUF files).

6. **SIMD optimization**: The K-quant scale packing is designed for SIMD-friendly byte-level access. Consider using `System.Runtime.Intrinsics` (AVX2/NEON) for the inner dequantization loops, especially for Q4_K and Q6_K which involve bit manipulation.

7. **sd.cpp compatibility**: Since sd.cpp detects architecture via tensor names rather than metadata, HartsyInference should implement pattern-based architecture detection as a fallback when `general.architecture` is missing or set to "pig".

8. **Tensor data computation**: For a tensor with dimensions `[d0, d1, ..., dn]` and type `T`:
   - `total_elements = d0 * d1 * ... * dn`
   - `n_blocks = total_elements / block_size(T)`
   - `byte_size = n_blocks * bytes_per_block(T)`

9. **Memory estimation**: For a model with `N` parameters:
   - Q4_0: ~`N * 4.5 / 8` bytes
   - Q8_0: ~`N * 8.5 / 8` bytes
   - Q4_K: ~`N * 4.5 / 8` bytes
   - Q6_K: ~`N * 6.5625 / 8` bytes
   - F16: ~`N * 2` bytes

10. **Error handling**: Validate magic, version >= 2, tensor offsets within file bounds, and that `tensor_count * minimum_block_bytes` does not exceed file size. Reject files with unknown ggml_type values rather than silently skipping tensors.
