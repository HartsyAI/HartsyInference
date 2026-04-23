# im2col CPU — Research Notes

---

## Summary

The im2col (image-to-column) transformation converts a convolution operation into a general matrix multiplication (GEMM) by rearranging input image patches into columns of a matrix. Each column represents one flattened receptive field (all input channels for one spatial patch). The filter weights are reshaped into rows, and a single GEMM call produces the entire convolution output. This approach trades memory (the im2col buffer is K^2 times larger than the input per channel) for the ability to use highly optimized BLAS/SIMD GEMM routines. For SharpInference.Cpu, im2col + GEMM is the primary Conv2D strategy, with special-case optimizations for 1x1 kernels (skip im2col entirely) and depthwise convolution (per-channel im2col + element-wise multiply or batched small GEMMs).

---

## Detailed Findings

### 1. Core Concept

Convolution slides a kernel over an input feature map, computing element-wise multiply-accumulate at each position. The im2col trick "unrolls" every patch the kernel touches into a column vector, stacks all columns side by side into a matrix, then multiplies by the reshaped filter matrix. The result is the output feature map in matrix form, which is reshaped back to the spatial output tensor.

**Why it works:** Convolution is a linear operation. Each output pixel is a dot product between the filter and an input patch. Collecting all patches as matrix columns and all filters as matrix rows converts N dot products into one matrix multiply, which BLAS libraries execute at near-peak FLOP/s on modern CPUs via SIMD, cache tiling, and micro-kernels.

### 2. NCHW Data Layout Assumption

SharpInference uses NCHW (batch, channels, height, width) tensor layout, matching PyTorch and ONNX conventions. The im2col transformation and GEMM dimensions below assume NCHW throughout.

### 3. Output Dimension Formulas

For input `[N, C_in, H_in, W_in]` with kernel `[C_out, C_in, K_h, K_w]`:

```
H_out = floor((H_in + 2*pad_h - dilation_h*(K_h - 1) - 1) / stride_h) + 1
W_out = floor((W_in + 2*pad_w - dilation_w*(K_w - 1) - 1) / stride_w) + 1
```

### 4. Memory Expansion Factor

The im2col buffer has shape `[C_in * K_h * K_w, H_out * W_out]`. For a 3x3 kernel, this is 9x the spatial size of one channel, times C_in channels. For typical diffusion UNet layers (e.g., C_in=320, 64x64 spatial, 3x3 kernel), the buffer is `320*9 * 4096 = 11,796,480` floats = ~45 MB per sample. This is significant but manageable with buffer pooling.

### 5. 1x1 Convolution Optimization

When K_h = K_w = 1, stride = 1, padding = 0, dilation = 1, the im2col matrix is identical to the input reshaped as `[C_in, H_out * W_out]`. No data copying is needed -- the convolution becomes a direct GEMM of filter `[C_out, C_in]` against reshaped input `[C_in, H*W]`, producing output `[C_out, H*W]`. This is a critical fast path since 1x1 convolutions are heavily used in diffusion models (pointwise projections in attention, channel mixing).

### 6. Depthwise Convolution

In depthwise convolution (groups = C_in = C_out), each channel is convolved independently with its own single-channel filter. Two strategies:

- **Per-channel im2col + batched GEMV:** For each channel c, extract the im2col patch matrix of shape `[K_h*K_w, H_out*W_out]`, multiply by filter vector `[1, K_h*K_w]`. This produces C_in independent GEMV calls. Overhead: many small GEMV calls.
- **Direct loop with SIMD:** For depthwise, it is often faster to avoid im2col entirely and use a direct convolution loop with SIMD vectorization across the spatial (W_out) dimension. This avoids the memory expansion entirely.

For SharpInference, the recommended approach is direct SIMD for depthwise (common kernel sizes 3x3, 5x5) and im2col+GEMM for grouped convolution where groups > 1 but groups < C_in.

### 7. col2im (Reverse Operation)

col2im converts the column matrix back to image format. It is needed for:
- **Transposed convolution (ConvTranspose2d):** Used in VAE decoder upsampling. The transposed convolution is implemented as: reshape output gradient to column form, multiply by transposed filter, then col2im to accumulate into the input-shaped tensor.
- **Gradient computation** (if training is ever supported).

The critical detail: when stride > 1 or patches overlap, col2im must **accumulate** (+=) values at overlapping positions, not overwrite them. This is because a single input element contributes to multiple output patches.

### 8. Cache Optimization Strategies

Research literature identifies several approaches to improve cache behavior:

- **Fused im2col + packing:** Instead of building the full im2col matrix and then feeding it to GEMM (which re-reads it), construct the im2col matrix block-by-block directly into the GEMM packing buffer. This keeps data in L2 cache. This is how high-performance BLAS implementations (OpenBLAS, BLIS) handle it internally when given the im2col matrix.
- **YaConv approach:** Pack input image into an L3-resident buffer, preload chunks into L1, and compute against all L2-resident filter elements before moving to the next image chunk. Achieves ~24% speedup over naive im2col.
- **MEC (Memory-Efficient Convolution):** Reduces the lowered matrix size by ~54% by exploiting horizontal overlap between adjacent patches and splitting one large GEMM into multiple smaller parallel GEMMs.
- **Tile blocking:** Divide the output spatial dimensions into tiles that fit in L2 cache. For each tile, only the corresponding im2col columns are materialized, keeping the working set small.

For SharpInference's initial implementation, the practical approach is: build the im2col buffer in tiles that fit in L2 (typically 256 KB - 1 MB per core on modern x86), then call GEMM per tile. This balances simplicity with good cache behavior.

---

## Key Numbers / Constants

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| L1d cache | 32-48 KB per core | Data cache, fastest |
| L2 cache | 256 KB - 1.25 MB per core | Main target for tile sizing |
| L3 cache | 1.5-4 MB per core (shared) | Filter matrix should fit here |
| AVX2 vector width | 256 bits = 8 floats | Minimum SIMD target |
| AVX-512 vector width | 512 bits = 16 floats | Optional fast path |
| NEON vector width | 128 bits = 4 floats | ARM fallback |
| Typical diffusion kernel sizes | 3x3, 1x1 | 3x3 dominates UNet, 1x1 for projections |
| Diffusion channel counts | 320, 640, 1280 | SD 1.5 UNet channel progression |
| im2col expansion factor (3x3) | 9x per channel | K_h * K_w = 9 |
| im2col expansion factor (1x1) | 1x (no-op) | Skip im2col entirely |
| Typical H_out * W_out (64x64 latent) | 4096 | Spatial output positions |
| im2col buffer size (320ch, 3x3, 64x64) | ~45 MB (float32) | 320 * 9 * 4096 * 4 bytes |

---

## Data Layouts / Formats

### Input Tensor (NCHW)
```
Memory: [n=0,c=0,h=0,w=0], [n=0,c=0,h=0,w=1], ..., [n=0,c=0,h=0,w=W-1],
        [n=0,c=0,h=1,w=0], ..., [n=0,c=0,h=H-1,w=W-1],
        [n=0,c=1,h=0,w=0], ...,
        ...

Stride: [C*H*W, H*W, W, 1]
```

### im2col Output Matrix
```
Shape: [C_in * K_h * K_w,  H_out * W_out]

Row index = c * K_h * K_w + kh * K_w + kw    (flattened filter position across all input channels)
Col index = oh * W_out + ow                    (flattened output spatial position)

Each column = one complete receptive field (all channels, all kernel positions) for one output pixel.
Each row = one filter tap position across all output spatial locations.
```

### Filter Matrix (reshaped for GEMM)
```
Shape: [C_out,  C_in * K_h * K_w]

Row index = output channel
Col index = c * K_h * K_w + kh * K_w + kw   (same ordering as im2col rows)
```

### GEMM Operation
```
Output[C_out, H_out*W_out] = Filter[C_out, C_in*K_h*K_w] x im2col[C_in*K_h*K_w, H_out*W_out]

Then reshape output to [C_out, H_out, W_out].
Add bias (if present) broadcast across spatial dimensions.
```

### Grouped Convolution Layout

For G groups with C_in/G input channels per group and C_out/G output channels per group:
```
For each group g in [0, G):
    im2col_g shape: [(C_in/G) * K_h * K_w,  H_out * W_out]
    filter_g shape: [C_out/G,  (C_in/G) * K_h * K_w]
    output_g shape: [C_out/G,  H_out * W_out]

    output_g = filter_g x im2col_g
```

---

## Algorithm Steps

### im2col Forward

**Step 1: Compute output dimensions**
```
H_out = (H_in + 2*pad_h - dilation_h*(K_h-1) - 1) / stride_h + 1
W_out = (W_in + 2*pad_w - dilation_w*(K_w-1) - 1) / stride_w + 1
```

**Step 2: Check for 1x1 fast path**
```
if K_h == 1 && K_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0:
    // No im2col needed. Input is already [C_in, H*W].
    // Proceed directly to GEMM with input pointer.
    skip to Step 5
```

**Step 3: Allocate im2col buffer**
```
col_buffer = TensorPool.Rent(C_in * K_h * K_w * H_out * W_out * sizeof(float))
```

**Step 4: Fill im2col buffer (Caffe-style)**
```pseudocode
function im2col_cpu(input, C_in, H_in, W_in, K_h, K_w,
                     pad_h, pad_w, stride_h, stride_w,
                     dilation_h, dilation_w, col_buffer):

    col_idx = 0   // pointer into col_buffer

    for c in [0, C_in):
        for kh in [0, K_h):
            for kw in [0, K_w):
                // This (c, kh, kw) triple identifies one row of the im2col matrix.
                // We fill H_out * W_out entries (one per output spatial position).

                for oh in [0, H_out):
                    ih = oh * stride_h - pad_h + kh * dilation_h

                    for ow in [0, W_out):
                        iw = ow * stride_w - pad_w + kw * dilation_w

                        if ih >= 0 AND ih < H_in AND iw >= 0 AND iw < W_in:
                            col_buffer[col_idx] = input[c * H_in * W_in + ih * W_in + iw]
                        else:
                            col_buffer[col_idx] = 0.0   // zero-padding
                        col_idx++
```

**Step 5: GEMM**
```pseudocode
// A = filter reshaped to [C_out, C_in * K_h * K_w]  (no copy needed, just reinterpret)
// B = col_buffer [C_in * K_h * K_w, H_out * W_out]  (or input reshaped for 1x1)
// C = output [C_out, H_out * W_out]

GEMM(A, B, C, M=C_out, N=H_out*W_out, K=C_in*K_h*K_w, alpha=1.0, beta=0.0)
```

**Step 6: Add bias**
```pseudocode
if bias is not null:
    for co in [0, C_out):
        for i in [0, H_out * W_out):
            output[co * H_out * W_out + i] += bias[co]
```

**Step 7: Return buffer**
```
TensorPool.Return(col_buffer)
```

### col2im (Reverse for Transposed Convolution)

```pseudocode
function col2im_cpu(col_buffer, C_in, H_in, W_in, K_h, K_w,
                     pad_h, pad_w, stride_h, stride_w,
                     dilation_h, dilation_w, output):

    // Initialize output to zeros (critical -- we accumulate)
    memset(output, 0, C_in * H_in * W_in * sizeof(float))

    col_idx = 0

    for c in [0, C_in):
        for kh in [0, K_h):
            for kw in [0, K_w):
                for oh in [0, H_out):
                    ih = oh * stride_h - pad_h + kh * dilation_h

                    for ow in [0, W_out):
                        iw = ow * stride_w - pad_w + kw * dilation_w

                        if ih >= 0 AND ih < H_in AND iw >= 0 AND iw < W_in:
                            output[c * H_in * W_in + ih * W_in + iw] += col_buffer[col_idx]
                            // NOTE: += not = because patches overlap
                        col_idx++
```

### Transposed Convolution using col2im

```pseudocode
function conv_transpose_2d(input, filter, bias, stride, padding, dilation):
    // input:  [N, C_in, H_in, W_in]
    // filter: [C_in, C_out, K_h, K_w]  (note: transposed filter shape)

    // Step 1: GEMM to get col_buffer
    // A = filter^T reshaped to [C_out * K_h * K_w, C_in]
    // B = input reshaped to [C_in, H_in * W_in]
    // C = col_buffer [C_out * K_h * K_w, H_in * W_in]
    GEMM(A^T, B, C, ...)

    // Step 2: col2im to convert col_buffer back to spatial output
    col2im_cpu(col_buffer, C_out, H_out, W_out, K_h, K_w,
               pad_h, pad_w, stride_h, stride_w,
               dilation_h, dilation_w, output)

    // Step 3: Add bias
    add_bias(output, bias)
```

---

## Reference Implementations

### 1. Caffe (Canonical)

- **Source:** [BVLC/caffe im2col.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp)
- **Language:** C++ with templates for float/double
- **Key design choices:**
  - Separate `pad_h`/`pad_w`, `stride_h`/`stride_w`, `dilation_h`/`dilation_w` parameters (supports non-square)
  - Uses `is_a_ge_zero_and_a_lt_b()` helper that casts int to unsigned for branchless bounds checking
  - Loop order: channel -> kernel_row -> kernel_col -> output_row -> output_col (row-major fill of im2col matrix)
  - Also provides `im2col_nd_core_cpu()` for N-dimensional convolution
  - Companion `col2im_cpu()` with identical loop structure but using `+=` accumulation

### 2. ONNX Runtime

- **Source:** [onnxruntime/core/providers/cpu/nn/conv.cc](https://github.com/microsoft/onnxruntime)
- **Language:** C++
- **Key design choices:**
  - Uses MlasConv API which internally handles im2col + GEMM fusion
  - Pre-allocated workspace buffer sized to the largest convolution in the model
  - Supports grouped convolution by iterating groups and adjusting pointers
  - im2row variant used (rows instead of columns) for better CPU cache locality

### 3. PyTorch (ATen)

- **Source:** aten/src/ATen/native/cpu/ConvolutionKernel.cpp
- **Language:** C++
- **Key design choices:**
  - Dispatches to MKL-DNN (oneDNN) when available for best x86 performance
  - Falls back to im2col + GEMM for generic path
  - Uses `at::native::im2col` which closely mirrors Caffe's implementation
  - 1x1 convolution fast path skips im2col

### 4. The Indirect Convolution Algorithm (Google/XNNPACK)

- **Source:** [Dukhan 2019, arXiv:1907.02129](https://arxiv.org/abs/1907.02129)
- **Key insight:** Instead of copying data into an im2col buffer, maintain an indirection buffer of pointers to the start of each input row. The GEMM micro-kernel follows pointers to read input data in place. Eliminates the O(K^2) memory expansion entirely.
- **Trade-off:** Requires a modified GEMM micro-kernel. The indirection buffer itself is small (just pointers) and can be precomputed once.

---

## Differences Between Implementations

| Aspect | Caffe | ONNX Runtime | PyTorch | XNNPACK |
|--------|-------|-------------|---------|---------|
| im2col variant | im2col (columns) | im2row (rows) | im2col | Indirect (no copy) |
| Buffer strategy | Per-call allocation | Pre-allocated workspace | Temporary tensor | Indirection buffer (tiny) |
| 1x1 fast path | No | Yes | Yes | N/A (always indirect) |
| Grouped conv | Separate loop | Pointer offset per group | Separate loop | Native support |
| Non-square kernel | Yes (separate h/w params) | Yes | Yes | Yes |
| Dilation support | Yes | Yes | Yes | Yes |
| SIMD in im2col | No (relies on GEMM) | No (relies on MLAS) | No (relies on MKL) | Yes (in micro-kernel) |
| Cache optimization | None (naive fill) | Workspace reuse | MKL-DNN handles it | Fused with GEMM |
| Memory overhead | K^2 x input size | Pre-sized max buffer | K^2 x input size | ~0 (pointers only) |

**im2col vs im2row:** im2row fills the lowered matrix in row-major order (each row = one output position, each column = one filter tap). On CPU, im2row tends to have better spatial locality since GEMM implementations access the B matrix in column-major order. On GPU, im2col is preferred. For SharpInference CPU, **im2col (Caffe-style) is recommended** for initial implementation due to simplicity and extensive documentation; im2row can be explored later if profiling shows cache misses in the GEMM B-matrix access.

---

## Open Questions

- [ ] **Whether to implement im2row instead of im2col for better CPU cache behavior:** Benchmark both approaches with the project's SIMD GEMM implementation.
- [ ] **Whether to implement the indirect convolution algorithm for zero-copy:** This would eliminate the im2col buffer entirely but requires modifying the GEMM micro-kernel. Consider as a Phase 2 optimization.
- [ ] **Optimal tile size for L2 across different CPU architectures:** Profile on Intel (256 KB L2), AMD (512 KB - 1 MB L2), and Apple Silicon (128 KB perf core L2) to determine if a single tile size works or if runtime detection is needed.

---

## Implementation Notes

### Recommended Implementation Order

1. **im2col_cpu with 1x1 fast path** -- handles the two most common cases (3x3 and 1x1 kernels)
2. **GEMM integration** -- wire up to the existing SIMD GEMM from SharpInference.Cpu
3. **Grouped convolution** -- loop over groups with pointer offsets
4. **Depthwise convolution** -- direct SIMD loop (skip im2col for this case)
5. **col2im for transposed convolution** -- needed for VAE decoder
6. **Cache tiling** -- split large im2col into L2-sized tiles

### Buffer Management

Use `TensorPool` from `SharpInference.Core.Tensors`:
```csharp
// Rent im2col workspace
nuint bufferSize = (nuint)(cIn * kH * kW * hOut * wOut * sizeof(float));
NativeBuffer colBuffer = pool.Rent(bufferSize);

try
{
    Im2Col(input, colBuffer.Pointer, cIn, hIn, wIn, kH, kW, ...);
    Gemm(filterPtr, colBuffer.Pointer, outputPtr, cOut, hOut * wOut, cIn * kH * kW);
}
finally
{
    pool.Return(colBuffer);
}
```

### SIMD Optimization in im2col Itself

The im2col loop's inner loop (over `ow`) is a candidate for SIMD gather operations when stride=1 and dilation=1: the source addresses are contiguous in memory (`input[c * H * W + ih * W + iw]` for sequential `iw`), so a simple `Vector256.Load` / `Vector256.Store` can copy 8 floats at once. When stride > 1 or dilation > 1, the source addresses are non-contiguous and require scalar access or AVX2 gather instructions (`Avx2.GatherVector256`), which are slower but still faster than scalar on large enough spans.

### Bounds Check Optimization (from Caffe)

Caffe uses this trick to avoid branching on padding checks:
```csharp
// Instead of: if (ih >= 0 && ih < H_in)
// Use: if ((uint)ih < (uint)H_in)
// Negative values wrap to large unsigned, failing the < check.
```
This eliminates one comparison per iteration and can prevent branch mispredictions in the padding boundary region.

### Non-Square Kernel Handling

All parameters (kernel size, padding, stride, dilation) should be stored as `(int h, int w)` tuples. The API should accept either a single int (applied to both dimensions) or a tuple:
```csharp
public static void Conv2D(
    Tensor input, Tensor filter, Tensor output,
    (int h, int w) padding,
    (int h, int w) stride,
    (int h, int w) dilation,
    int groups = 1)
```

### Performance Expectations

For a 320-channel 3x3 Conv2D at 64x64 spatial resolution (common in SD 1.5 UNet):
- im2col buffer: 320 * 9 * 4096 = 11.8M floats = 45 MB
- GEMM: [320, 2880] x [2880, 4096] = 3.77 GFLOP
- At ~100 GFLOP/s (single-core AVX2 FP32 throughput): ~38 ms per convolution
- With 4-core parallelism over batch/spatial tiles: ~10 ms target

For 1x1 convolutions (skip im2col):
- GEMM only: [C_out, C_in] x [C_in, H*W]
- No memory expansion, significantly faster

---

## Sources

- [Caffe im2col.cpp (BVLC)](https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp) -- canonical reference implementation
- [Caffe im2col.cu (GPU)](https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu) -- GPU kernel showing same algorithm structure
- [im2col Convolution (OpenGenus)](https://iq.opengenus.org/im2col/) -- clear visual explanation of the transformation
- [The Indirect Convolution Algorithm (Dukhan 2019)](https://arxiv.org/abs/1907.02129) -- zero-copy alternative to im2col
- [MEC: Memory-efficient Convolution (Cho 2017)](https://arxiv.org/abs/1706.06873) -- 54% memory reduction technique
- [YaConv: Convolution with Low Cache Footprint](https://dl.acm.org/doi/10.1145/3570305) -- 24% speedup via cache-aware tiling
- [Anatomy of a High-Speed Convolution (Sahni)](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/) -- detailed walkthrough of optimized CPU convolution
- [Im2win: Memory Efficient Convolution on SIMD Architectures (Lu 2023)](https://arxiv.org/abs/2306.14316) -- im2win variant for SIMD
- [Characterizing the Implicit Convolution Algorithm (Zhou 2021)](https://arxiv.org/abs/2110.03901) -- analysis of implicit vs explicit im2col
- [Speeding up Convolutions](https://scocoyash.github.io/speeding-up-convolutions/) -- survey of convolution optimization strategies
- [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) -- parameter definitions and non-square kernel support
- [Transposed Convolution (D2L)](https://www.d2l.ai/chapter_computer-vision/transposed-conv.html) -- transposed convolution explanation
- [ArrayPool in .NET (Adam Sitnik)](https://adamsitnik.com/Array-Pool/) -- buffer pooling patterns for .NET
- [p-im2col: Flexibly Controlled Memory Overhead (IEEE)](https://ieeexplore.ieee.org/abstract/document/9650846) -- partial im2col for memory-constrained scenarios
- [Improving Convolution via Cache Hierarchy Tiling](https://dl.acm.org/doi/10.1145/3559009.3569678) -- tile blocking for cache optimization
