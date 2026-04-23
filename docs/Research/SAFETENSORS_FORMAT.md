# Safetensors Format — Research Notes

## Summary

Safetensors is a simple, safe binary format for storing tensors, created by Hugging Face as a secure alternative to Python pickle. Unlike pickle, safetensors files cannot execute arbitrary code on load — they are pure data. The format consists of three contiguous sections: an 8-byte little-endian header length, a UTF-8 JSON header describing all tensors (name, dtype, shape, byte offsets), and a raw data blob containing packed tensor bytes in little-endian, C-contiguous (row-major) order.

The format has been security-audited by Trail of Bits (commissioned jointly by Hugging Face, EleutherAI, and Stability AI) with no critical code-execution flaws found ([audit blog post](https://huggingface.co/blog/safetensors-security-audit)). As of April 2026, safetensors became a PyTorch Foundation contributed project ([announcement](https://pytorch.org/blog/pytorch-foundation-announces-safetensors-as-newest-contributed-project-to-secure-ai-model-execution/)). It is now the default format for virtually all models on Hugging Face Hub.

## Detailed Findings

### Binary Layout

A `.safetensors` file has exactly three contiguous sections ([source: README format spec](https://github.com/huggingface/safetensors/blob/main/README.md)):

| File Offset | Size | Content |
|-------------|------|---------|
| `0` | 8 bytes | `N` — unsigned **little-endian 64-bit integer** giving the size of the JSON header in bytes |
| `8` | `N` bytes | UTF-8 JSON string — the header (tensor metadata) |
| `8 + N` | remainder of file | Raw tensor data blob (little-endian, C/row-major, packed) |

```
┌──────────┬──────────────────────────────────┬─────────────────────────┐
│ 8 bytes  │           N bytes                │    remaining bytes      │
│ u64 LE   │       JSON header (UTF-8)        │   tensor data blob      │
│ = N      │                                  │                         │
└──────────┴──────────────────────────────────┴─────────────────────────┘
```

Key rules from the specification:
- The header data **MUST** begin with `{` (0x7B)
- The header data **MAY** be trailing-padded with whitespace (0x20)
- The header is padded to a **multiple of 8 bytes** to ensure the data blob starts at an 8-byte-aligned file offset ([source: Rust `tensor.rs`, `next_multiple_of(N_LEN)` where `N_LEN = 8`](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs))
- All numeric data is **little-endian**
- Tensor data is **C-contiguous** (row-major), no striding — all tensors must be contiguous before serialization
- The data blob must be **entirely indexed** by tensor offsets — no holes, no trailing unaccounted bytes

### JSON Header Schema

The header is a flat JSON object. Each key is either a tensor name or the special `__metadata__` key ([source: README](https://github.com/huggingface/safetensors/blob/main/README.md), [HuggingFace docs](https://huggingface.co/docs/safetensors/v0.3.2/en/metadata_parsing)):

```json
{
  "__metadata__": {
    "format": "pt",
    "custom_key": "custom_value"
  },
  "model.layer.0.weight": {
    "dtype": "F16",
    "shape": [768, 3072],
    "data_offsets": [0, 4718592]
  },
  "model.layer.0.bias": {
    "dtype": "F16",
    "shape": [768],
    "data_offsets": [4718592, 4720128]
  }
}
```

#### Tensor entry fields

| Field | Type | Description |
|-------|------|-------------|
| `dtype` | string | Dtype identifier (see dtype table below) |
| `shape` | array of integers | Tensor dimensions. `[]` = scalar (0-rank). A dimension of `0` is legal (empty tensor, stores no data). |
| `data_offsets` | `[begin, end]` | Byte offsets **relative to the start of the data blob** (NOT absolute file offsets). Byte size = `end - begin`. |

#### `__metadata__` key

- **Optional**
- Must be a **string-to-string map** — all values must be strings, not arbitrary JSON ([source: huggingface.js types](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts))
- Common keys: `"format"` (`"pt"`, `"tf"`, `"flax"`, `"numpy"`), custom user metadata

#### Additional rules
- **Duplicate tensor names are disallowed**
- The JSON subset is implicitly defined by `serde_json` — exotic JSON representations may be rejected in future versions

### TypeScript Type Reference

From [huggingface.js](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts):

```typescript
interface TensorInfo {
  dtype: Dtype;
  shape: number[];
  data_offsets: [number, number];
}

type SafetensorsFileHeader = Record<string, TensorInfo> & {
  __metadata__?: Record<string, string>;
};
```

## Key Numbers / Constants

| Constant | Value | Source |
|----------|-------|--------|
| Header length prefix | 8 bytes, unsigned LE u64 | [README](https://github.com/huggingface/safetensors/blob/main/README.md) |
| Max header size | **100,000,000 bytes (100 MB)** | [tensor.rs `MAX_HEADER_SIZE`](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs) |
| Header padding alignment | 8 bytes (0x20 whitespace) | [tensor.rs `next_multiple_of(N_LEN)`](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs) |
| Byte order | Little-endian | [README](https://github.com/huggingface/safetensors/blob/main/README.md) |
| Memory order | C / row-major | [README](https://github.com/huggingface/safetensors/blob/main/README.md) |

## Data Layouts / Formats

### Complete Dtype Catalog

From the Rust `Dtype` enum in [`tensor.rs`](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs) and [docs.rs](https://docs.rs/safetensors/latest/safetensors/tensor/enum.Dtype.html). The enum is `#[non_exhaustive]` — more variants may be added.

| Dtype String | Bits | Bytes | Description | Relevant to SharpInference |
|-------------|------|-------|-------------|---------------------------|
| `BOOL` | 8 | 1 | Boolean | Rare in model weights |
| `U8` | 8 | 1 | Unsigned 8-bit integer | Image data, masks |
| `I8` | 8 | 1 | Signed 8-bit integer | INT8 quantized weights |
| `F8_E5M2` | 8 | 1 | FP8, 5-bit exponent, 2-bit mantissa | Emerging quantization |
| `F8_E4M3` | 8 | 1 | FP8, 4-bit exponent, 3-bit mantissa | Emerging quantization |
| `F8_E8M0` | 8 | 1 | FP8, 8-bit exponent, 0-bit mantissa | MX scaling format |
| `F8_E4M3FNUZ` | 8 | 1 | FP8, no negative zero, no infinity | AMD variant |
| `F8_E5M2FNUZ` | 8 | 1 | FP8, no negative zero, no infinity | AMD variant |
| `I16` | 16 | 2 | Signed 16-bit integer | Rare |
| `U16` | 16 | 2 | Unsigned 16-bit integer | Rare |
| `F16` | 16 | 2 | IEEE 754 half-precision float | **Primary weight format** |
| `BF16` | 16 | 2 | Brain floating point 16 | **Common weight format** |
| `I32` | 32 | 4 | Signed 32-bit integer | Indices, token IDs |
| `U32` | 32 | 4 | Unsigned 32-bit integer | Rare |
| `F32` | 32 | 4 | IEEE 754 single-precision float | **Common weight format** |
| `C64` | 64 | 8 | Complex (two F32s) | Rare in inference |
| `F64` | 64 | 8 | IEEE 754 double-precision float | Rare in inference |
| `I64` | 64 | 8 | Signed 64-bit integer | Rare |
| `U64` | 64 | 8 | Unsigned 64-bit integer | Rare |
| `F4` | 4 | 0.5 | MX FP4 (sub-byte) | Experimental |
| `F6_E2M3` | 6 | 0.75 | MX FP6 variant (sub-byte) | Experimental |
| `F6_E3M2` | 6 | 0.75 | MX FP6 variant (sub-byte) | Experimental |

**Total: 22 dtype variants** as of latest source.

**Priority for SharpInference**: `F32`, `F16`, `BF16` cover ~99% of diffusion model weights. `I8` for quantized models. `I32`/`I64` for embedding indices and metadata tensors.

**Sub-byte dtype warning** (from README): "Some smaller than 1 byte dtypes appeared, which make alignment tricky. Non-traditional APIs might be required." The library errors on non-byte-aligned reads (`nbits % 8 != 0`).

### Multi-Shard Format

Large models are split across multiple `.safetensors` files with an accompanying index file ([source: HuggingFace Hub conventions](https://huggingface.co/docs/transformers/models), [huggingface_hub Python source](https://github.com/huggingface/huggingface_hub/blob/v0.34.4/src/huggingface_hub/utils/_safetensors.py)).

#### Shard file naming convention

```
model-00001-of-00006.safetensors
model-00002-of-00006.safetensors
...
model-00006-of-00006.safetensors
model.safetensors.index.json       ← index file
```

Also seen with other prefixes: `diffusion_pytorch_model-00001-of-00002.safetensors`.

#### Index JSON schema (`model.safetensors.index.json`)

```json
{
  "metadata": {
    "total_size": 28966928384
  },
  "weight_map": {
    "lm_head.weight": "model-00006-of-00006.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00006.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00006.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00006.safetensors"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `metadata.total_size` | integer | Total bytes of all tensor data across all shards |
| `metadata.total_parameters` | string or integer | Optional, total parameter count |
| `weight_map` | `Record<tensor_name, shard_filename>` | Maps every tensor name to the shard file containing it |

Key details:
- Each shard file is a **valid standalone `.safetensors` file** with its own header
- A non-sharded model uses a single `model.safetensors` with no index JSON
- The index file is the **only way** to know which shard contains a given tensor — there is no pattern to infer it from the tensor name
- All shard files must be in the **same directory** as the index file

TypeScript definition from [huggingface.js](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts):

```typescript
interface SafetensorsIndexJson {
  dtype?: string;
  metadata?: { total_parameters?: string | number } & Record<string, string>;
  weight_map: Record<TensorName, FileName>;
}
```

## Algorithm Steps

### Parsing a single `.safetensors` file

```
1. Read 8 bytes from offset 0 → interpret as unsigned LE u64 → header_size
2. Validate: header_size ≤ 100,000,000 (100 MB)
3. Validate: file_size ≥ 8 + header_size
4. Read header_size bytes from offset 8 → UTF-8 string → parse as JSON object
5. Extract __metadata__ if present (validate: all values are strings)
6. For each remaining key in JSON:
   a. Extract dtype (string), shape (int[]), data_offsets ([begin, end])
   b. Validate dtype is a known enum value
   c. Compute expected_bytes = product(shape) × dtype_byte_size
   d. Validate: end - begin == expected_bytes
   e. Validate: begin == previous tensor's end offset (contiguous, no gaps)
   f. Validate: no duplicate tensor names
7. Validate: last tensor's end offset == file_size - 8 - header_size
   (data blob is fully covered, no holes, no trailing data)
8. Tensor data for tensor T is at file offset: 8 + header_size + T.data_offsets[0]
```

### Loading a sharded model

```
1. Check for model.safetensors.index.json in model directory
2. If not found → load single model.safetensors file (non-sharded)
3. If found → parse index JSON
4. Build lookup: tensor_name → shard_filename from weight_map
5. Determine unique set of shard files to open
6. For each shard file:
   a. Parse header (steps 1-7 above)
   b. Memory-map or read the file
7. To access tensor T:
   a. Look up shard_filename from weight_map
   b. Get tensor metadata from that shard's header
   c. Read data at: 8 + shard_header_size + data_offsets[0] in that shard file
```

## Reference Implementations

| Implementation | Location | Notes |
|---------------|----------|-------|
| **Rust (reference)** | [safetensors/src/tensor.rs](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs) | Canonical parser. Contains `Dtype` enum, `MAX_HEADER_SIZE`, all validation logic. Read `SafeTensors::deserialize()` and `HashSet`-based duplicate key detection. |
| **Rust (slicing)** | [safetensors/src/slice.rs](https://github.com/huggingface/safetensors/blob/main/safetensors/src/slice.rs) | Sub-byte dtype alignment checks (`MisalignedSlice` error). |
| **Python bindings** | [safetensors Python package](https://github.com/huggingface/safetensors/tree/main/bindings/python) | Wraps the Rust implementation via PyO3. `safe_open()` API for lazy/mmap'd loading. |
| **huggingface_hub** | [huggingface_hub/utils/_safetensors.py](https://github.com/huggingface/huggingface_hub/blob/v0.34.4/src/huggingface_hub/utils/_safetensors.py) | Shard index parsing logic, `weight_map` handling. |
| **huggingface.js** | [parse-safetensors-metadata.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts) | TypeScript types for both single-file and index JSON schemas. Clean type definitions useful as reference. |
| **HuggingFace docs** | [Metadata Parsing](https://huggingface.co/docs/safetensors/v0.3.2/en/metadata_parsing) | Official documentation on reading header metadata. |
| **Trail of Bits audit** | [Blog post](https://huggingface.co/blog/safetensors-security-audit) | Security audit results, polyglot file prevention. |

## Differences Between Implementations

| Aspect | Rust (reference) | Python | Notes |
|--------|-----------------|--------|-------|
| Duplicate key handling | Enforced via `HashSet` — error on duplicate | Inherited from Rust (PyO3 binding) | C# should use `Dictionary` — will naturally reject duplicates |
| Memory mapping | `SafeTensors::deserialize()` works on `&[u8]` — caller handles mmap | `safe_open(framework="pt", device="cpu")` handles mmap internally | C# should use `MemoryMappedFile` |
| Tensor ordering on write | Sorted by descending dtype alignment | Same (wraps Rust) | Reader must NOT assume any order |
| Sub-byte dtype support | Full support with `MisalignedSlice` error | Same | SharpInference can defer — no diffusion models use sub-byte dtypes yet |
| Header padding | Pads to multiple of 8 with 0x20 | Same | Reader must tolerate trailing whitespace in header |

## Open Questions

- [ ] Whether any popular diffusion model uses `C64`, `F64`, `U16`, `U32`, or `U64` dtypes (likely not — can defer support)
- [ ] Exact behavior when `__metadata__` contains non-string values — reference implementation likely rejects, but some third-party writers may produce them

## Implementation Notes

### Recommended C# approach

1. **Use `MemoryMappedFile`** for loading — avoid reading entire multi-GB files into memory. Memory-map the file and access tensor data via `MemoryMappedViewAccessor` or unsafe pointers. This aligns with the project's "zero-allocation hot paths" pillar.

2. **Parse header with `System.Text.Json`** — use `JsonDocument` for low-allocation parsing. The header is at most 100 MB but typically <5 MB for even the largest models.

3. **Dtype enum** — create a C# enum mapping the dtype strings. For Phase 1, only `F32`, `F16`, `BF16`, `I8`, `I32`, and `I64` are needed. The full 22-dtype set can be added incrementally.

4. **Shard loading** — implement shard support from the start. T5-XXL (used by SD3 and Flux) ships as multiple shards. The `model.safetensors.index.json` loader should build a `Dictionary<string, (string shardFile, long begin, long end)>` for O(1) tensor lookup.

5. **Validation** — implement all validation rules from Section 6 above. The format is safety-critical and wrong offsets = corrupt weights = silent inference failures. Validate:
   - Header size ≤ 100 MB
   - Offsets are contiguous (no gaps)
   - Data blob is fully covered (no trailing bytes)
   - Shape × dtype size matches declared byte range
   - No duplicate tensor names

6. **BF16 handling** — .NET has `System.Half` (F16) but no native `BFloat16`. Implement BF16→F32 conversion: `float value = BitConverter.Int32BitsToSingle(bf16_bits << 16)`. This is exact — BF16 is just truncated F32.

7. **Endianness** — safetensors is always little-endian. On little-endian platforms (all x86/x64, most ARM), no byte-swapping is needed. Use `BitConverter.IsLittleEndian` as a guard.

8. **Sub-byte dtypes** — defer. No diffusion, audio, or vision model uses F4/F6 dtypes today. Register the dtype strings in the enum but throw `NotSupportedException` on load.

9. **File offset calculation** — tensor data for tensor `T` in file `F` is at: `file_offset = 8 + header_size + T.data_offsets[0]`. This is the critical formula for mmap-based access.

10. **Thread safety** — `MemoryMappedFile` views can be accessed from multiple threads safely for read-only access, which is the inference case.
